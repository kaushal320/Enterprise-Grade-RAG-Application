#!/usr/bin/env python3
"""
CI RAGAS eval gate.

In-process RAG quality gate for the CI pipeline. Unlike evals/pipeline.py and
evals/guardrails_eval.py, this script does NOT call the live /query HTTP API.
It imports the compiled LangGraph agent directly (app.agents.graph.rag_agent)
and drives it with the same initial state app.main uses, then scores the first
N golden RAG samples with RAGAS Faithfulness + AnswerRelevancy.

Judging uses the separate JUDGE_GROQ key (falls back to JUDGE_GROQ, then
GROQ_API_KEY) so the production key is never exhausted by eval runs. Answer
Relevancy embeddings go through the Jina OpenAI-compatible endpoint using the
configured JINA_API_KEY — no local sentence-transformers download in CI.

Exit codes:
    0  both metric averages >= threshold (default 0.6) — merge OK
    1  either average < 0.6, a sample fails to run, or a required secret is missing

Run:
    python evals/ci_ragas_gate.py
    python evals/ci_ragas_gate.py --golden path/to/dataset.json --samples 3 --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so `app` is importable from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

if os.getenv("LOGFIRE_TOKEN"):
    import logfire

    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="ci_ragas_gate")

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.agents.graph import rag_agent

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_SAMPLES = 3
DEFAULT_THRESHOLD = 0.6

# gpt-oss-20b (and the old llama-3.1-8b-instant) are too weak as NLI judges:
# they return "unsupported" for every claim even when the context explicitly
# contains it, collapsing Faithfulness to 0.0. Use the 120b model so verdicts
# are actually grounded.
JUDGE_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JINA_BASE_URL = "https://api.jina.ai/v1"
JINA_EMBEDDING_MODEL = "jina-embeddings-v3"

CONTENT_PREFIX = "CONTENT: "  # prefix retrieve_node prepends to every document

# Judge inputs are trimmed (mirroring evals/metrics.py) to stay inside the
# Groq 70b on_demand tokens-per-day cap (~100k): fewer samples + only the
# top-2 contexts at 300 chars each keeps a full gate run near ~17k tokens.
CONTEXT_TRUNCATE = 300  # chars per context chunk passed to the judge
CONTEXT_LIMIT = 2       # max context chunks passed to the judge

# The 70b judge (and the agent's 70b planner/responder) all draw on the same
# Groq budget. Interleave calls so no 60s TPM window overflows — Faithfulness
# alone makes two judge calls per sample (statements + verdicts).
DELAY_BETWEEN_SAMPLES = 8  # seconds between agent runs
DELAY_BETWEEN_JUDGE_CALLS = 12  # seconds between per-sample RAGAS judge calls
JUDGE_RETRIES = 4  # attempts per RAGAS score on transient API errors


def load_golden(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_agent_question(question: str, thread_id: str) -> tuple[str, list[str]]:
    """
    Drive the compiled LangGraph agent in-process (no HTTP) with the same
    initial state app.main uses. Returns (answer, retrieved_contexts) with the
    "CONTENT: " prefix stripped so RAGAS sees clean passages.
    """
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "current_query": question,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    config = {"configurable": {"thread_id": thread_id}}

    output = rag_agent.invoke(initial_state, config=config)
    answer = (output.get("final_answer") or "").strip()
    documents = output.get("documents") or []
    contexts = [
        doc.split(CONTENT_PREFIX, 1)[-1] if doc.startswith(CONTENT_PREFIX) else doc
        for doc in documents
    ]
    return answer, contexts


def build_judge():
    """RAGAS judge LLM on the JUDGE_GROQ key via Groq's OpenAI-compatible API."""
    api_key = (
        os.getenv("JUDGE_GROQ_KEY")
        or os.getenv("JUDGE_GROQ")
        or os.getenv("GROQ_API_KEY")
    )
    if not api_key:
        raise RuntimeError("JUDGE_GROQ_KEY (or JUDGE_GROQ/GROQ_API_KEY) must be set.")
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    llm.model_args["max_tokens"] = 4096
    return llm


def build_embeddings():
    """Answer Relevancy embeddings through the Jina OpenAI-compatible endpoint."""
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY must be set for RAGAS answer relevancy.")
    client = AsyncOpenAI(api_key=api_key, base_url=JINA_BASE_URL)
    model = os.getenv("JINA_EMBEDDING_MODEL") or JINA_EMBEDDING_MODEL
    return OpenAIEmbeddings(client=client, model=model)


def _clean(value) -> float:
    """Normalize a RAGAS score; NaN (no statements/verdicts) counts as 0.0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(score) else score


def _score_with_retry(metric, kwargs: dict, label: str) -> float:
    """Score one sample, retrying on transient API errors. Returns 0.0 on failure."""
    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            return _clean(metric.score(**kwargs).value)
        except Exception as exc:
            if attempt == JUDGE_RETRIES:
                print(f"  ❌ {label} failed after {JUDGE_RETRIES} attempts: {exc}")
                return 0.0
            print(f"  ⚠️ {label} attempt {attempt} failed ({exc}); retrying...")
            time.sleep(2 ** attempt)


def score_samples(samples: list, judge_llm, embeddings) -> list[dict]:
    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)

    rows: list[dict] = []
    n = len(samples)

    for i, sample in enumerate(samples):
        question = sample["question"]
        print(f"\n🧪 Sample {i + 1}/{n}: {question[:80]}")

        try:
            answer, contexts = run_agent_question(
                question, thread_id=f"ci_ragas_{i}"
            )
        except Exception as exc:
            print(f"  ❌ Agent run failed: {exc}")
            rows.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                }
            )
            time.sleep(DELAY_BETWEEN_SAMPLES)
            continue

        if not answer:
            print("  ❌ No answer produced — scoring as 0.0.")
            rows.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": contexts,
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                }
            )
            time.sleep(DELAY_BETWEEN_SAMPLES)
            continue

        print(f"  ✅ answer ({len(answer)} chars), {len(contexts)} retrieved contexts")

        if not contexts:
            print("  ⚠️ No retrieved contexts — faithfulness scored as 0.0.")
            fth_score = 0.0
        else:
            # Trim contexts for the judge (budget + metrics.py parity): the
            # top chunks hold the answer's evidence, and Faithfulness's verdict
            # call is the gate's largest token consumer.
            judge_contexts = [c[:CONTEXT_TRUNCATE] for c in contexts[:CONTEXT_LIMIT]]
            fth_score = _score_with_retry(
                faithfulness,
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": judge_contexts,
                },
                "Faithfulness",
            )
        time.sleep(DELAY_BETWEEN_JUDGE_CALLS)

        rel_score = _score_with_retry(
            answer_relevancy,
            {"user_input": question, "response": answer},
            "AnswerRelevancy",
        )
        time.sleep(DELAY_BETWEEN_JUDGE_CALLS)

        print(f"  📊 faithfulness={fth_score:.3f}  answer_relevancy={rel_score:.3f}")
        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "faithfulness": fth_score,
                "answer_relevancy": rel_score,
            }
        )

        if i < n - 1:
            time.sleep(DELAY_BETWEEN_SAMPLES)

    return rows


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Path to the golden dataset JSON (default: evals/golden_dataset.json).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of leading RAG samples to evaluate (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum allowed per-metric average (default: 0.6).",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_golden(args.golden)
    except FileNotFoundError:
        print(f"[FAIL] Golden dataset not found: {args.golden}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"[FAIL] Golden dataset is not valid JSON: {exc}")
        return 1

    rag_samples = dataset.get("rag_samples") or []
    if not rag_samples:
        print("[FAIL] Golden dataset contains no 'rag_samples'.")
        return 1
    samples = rag_samples[: args.samples]

    print(
        f"🚀 Running in-process LangGraph agent + RAGAS on the first "
        f"{len(samples)} golden RAG samples (no HTTP)..."
    )

    try:
        judge_llm = build_judge()
        embeddings = build_embeddings()
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    rows = score_samples(samples, judge_llm, embeddings)

    fth_avg = sum(r["faithfulness"] for r in rows) / len(rows)
    rel_avg = sum(r["answer_relevancy"] for r in rows) / len(rows)

    print("\n" + "=" * 72)
    print("RAGAS EVAL GATE — SUMMARY")
    print("=" * 72)
    for r in rows:
        print(
            f"  {r['question'][:70]:<72} "
            f"fth={r['faithfulness']:.3f} rel={r['answer_relevancy']:.3f}"
        )
    print(f"\n  Average Faithfulness:     {fth_avg:.3f} (min {args.threshold:.3f})")
    print(f"  Average Answer Relevancy: {rel_avg:.3f} (min {args.threshold:.3f})")

    ok = fth_avg >= args.threshold and rel_avg >= args.threshold
    if ok:
        print("\n[PASS] RAGAS EVAL GATE — both metric averages meet the threshold.")
        return 0
    print(
        f"\n[FAIL] RAGAS EVAL GATE — a metric average is below {args.threshold:.3f}. "
        "Fix retrieval/answers before merging."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
