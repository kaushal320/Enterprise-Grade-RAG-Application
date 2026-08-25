# 📊 Enterprise RAG — Evaluation & Quality Results

This document describes the evaluation suite, the CI quality gates, and the
current quality results for the Enterprise-Grade RAG application. It is the
canonical reference for **how** the system is measured and **what** the latest
numbers are.

---

## 1. Evaluation Suite Overview

The eval suite lives in `evals/` and spans two execution models:

| Module | Execution | What it measures |
| :--- | :--- | :--- |
| `offline_gate.py` | Static (stdlib only, no API/LLM) | Golden dataset integrity — well-formedness + internal coherence before any live run |
| `pipeline.py` | Live (HTTP `/query`) | **Phase 1** — captures each sample's actual response, retrieved contexts, and tools called |
| `metrics.py` | Live (HTTP `/query`) | **Phase 2** — RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness) + Tool Correctness |
| `guardrails_eval.py` | Live (HTTP `/query`) | Guardrail correctness — TP / TN / FP / FN, precision + recall |
| `ci_ragas_gate.py` | In-process (compiled LangGraph agent, no HTTP) | CI gate — Faithfulness + Answer Relevancy on the first N golden samples |
| `app.py` | Streamlit UI | Orchestrates phases 1–2 and the guardrail eval against a running backend |

The golden dataset (`evals/golden_dataset.json`) is the single source of truth
for all of the above.

### Golden Dataset

| Section | Count | Description |
| :--- | :--- | :--- |
| `rag_samples` | 15 | Technical Q/A pairs across 5 domains (`parallel_work_queue`, `cronjobs`, `job_management`, `monitor_job`, `pods_autoscale`) with a `reference` answer, `relevant_contexts`, and `expected_tools` |
| `guardrails_samples` | 6 | Attack/behaviour cases (jailbreak, off-topic, legit) with an `expected_blocked` label |

Each `rag_samples` entry carries the expected-tool contract used for Tool
Correctness (`expected_tools`) plus `actual_*` fields that the live phases fill
in-place.

---

## 2. Quality Metrics

| Metric | Definition (RAGAS) | Notes |
| :--- | :--- | :--- |
| **Faithfulness** | Share of the answer's claims **supported by the retrieved contexts** (NLI) | Judge-dependent — needs a strong NLI judge (see §6) |
| **Answer Relevancy** | How well the answer addresses the question | Uses an embedding similarity step |
| **Context Precision** | Are relevant chunks ranked high? | Requires a reference |
| **Context Recall** | Do the retrieved contexts contain the reference facts? | Requires a reference |
| **Answer Correctness** | Combined factual + semantic match vs. reference | Requires a reference |
| **Tool Correctness** | Jaccard match of expected vs. actual tools called | No LLM calls — deterministic |

Guardrails are measured separately with **precision** (of the calls the guardrail
blocked, how many should have been) and **recall** (of the attacks it should have
blocked, how many it did).

---

## 3. CI Quality Gates

`.github/workflows/ci.yml` runs on every push/PR to `main`:

| Job | Runs | Needs secrets? | Gate logic |
| :--- | :--- | :--- | :--- |
| `lint-and-syntax` | `ruff check app/ ui/ tests/` | No | Static lint |
| `unit-tests` | `pytest tests/ -v` | No (mock keys) | All tests pass |
| `eval-gate` | `python evals/offline_gate.py --golden evals/golden_dataset.json` | No | Dataset is well-formed & coherent |
| `ragas-eval-gate` | `python evals/ci_ragas_gate.py --golden evals/golden_dataset.json` | **Yes** | Faithfulness **and** Answer Relevancy averages ≥ **0.6** |
| `docker-build-check` | Docker buildx (backend + UI) | No | Images compile |

`docker-build-check` `needs: [lint-and-syntax, unit-tests, eval-gate, ragas-eval-gate]`
— a failing RAG quality gate blocks the image builds.

### `ragas-eval-gate` details

- **In-process agent**: `evals/ci_ragas_gate.py` imports `app.agents.graph.rag_agent`
  directly and drives it with the same initial state `app.main` uses — **no HTTP**.
- **Samples**: the first **3** golden RAG samples (kept small to stay inside the
  Groq `llama-3.3-70b-versatile` on-demand daily token cap).
- **Judge**: RAGAS Faithfulness + Answer Relevancy scored by
  `llama-3.3-70b-versatile` on the `JUDGE_GROQ_KEY` (falls back to `JUDGE_GROQ`,
  then `GROQ_API_KEY`) so production keys are never exhausted by eval runs.
- **Embeddings**: Answer Relevancy embeddings via the Jina OpenAI-compatible
  endpoint (`JINA_API_KEY`).
- **Threshold**: exits `1` if either metric average < `0.6`.
- **Secrets**: `GROQ_API_KEY`, `JUDGE_GROQ_KEY`, `QDRANT_API_KEY`, `QDRANT_URL`,
  `QDRANT_COLLECTION_NAME`, `JINA_API_KEY`, `PORTKEY_API_KEY`.

---

## 4. Observed Results

### 4.1 CI RAGAS gate — full 5-sample run (2026-08-10)

Recorded after the routing + judge model fixes (see §6), via the in-process
agent on the ChatGroq fallback path. Scored with the `llama-3.3-70b-versatile`
judge over **full** retrieved contexts.

| # | Question | Faithfulness | Answer Relevancy |
| :-- | :--- | :---: | :---: |
| 1 | How do you start Redis for a Kubernetes work queue? | 1.000 | 1.000 |
| 2 | What does the parallelism field do in the Kubernetes job-wq-2 manifest? | 1.000 | 0.987 |
| 3 | How do you fill the Redis work queue with tasks using the CLI? | 0.889 | 1.000 |
| 4 | What is the difference between HPA and VPA in Kubernetes? | 1.000 | 0.840 |
| 5 | How do you install the Metrics Server for Kubernetes pod autoscaling? | 1.000 | 0.997 |
| | **Average** | **0.978** | **0.965** |

✅ Both averages ≥ `0.6` — **gate passed**.

### 4.2 Regression baseline (same day, before fixes)

The same gate previously failed with `Faithfulness = 0.000` across all samples,
for two distinct root causes:

1. **Weak routing LLM**: the planner's ChatGroq fallback used
   `llama-3.1-8b-instant`, which classified first-turn technical questions as
   `CONVERSATIONAL` — the graph skipped retrieval entirely (`0` retrieved
   contexts → Faithfulness `0.0`).
2. **Weak NLI judge**: even with contexts retrieved, the `llama-3.1-8b-instant`
   judge rejected *every* claim as unsupported, even when the evidence was
   verbatim in the context.

Both were fixed by promoting the planner fallback **and** the RAGAS judge to
`llama-3.3-70b-versatile` (commit `12b5dab`).

### 4.3 Budget note

The Groq `llama-3.3-70b-versatile` on-demand tier is capped at **100k tokens /
day**. A full 5-sample gate run costs ~39k tokens, so the gate is configured to
run **3 samples** with the judge receiving only the **top-2 contexts × 300
chars** (mirroring `evals/metrics.py`), bringing a run to ~17k tokens and
allowing 5+ CI runs/day. Re-verification of the trimmed gate is pending a daily
token-budget reset.

### 4.4 Phase 1 + 2 / guardrail results

The full Phase 1 + 2 and guardrail numbers are produced by the Streamlit eval
app (`streamlit run evals/app.py`) against a **running backend** on `:8080`.
Results populate the live UI per run; no persistent result file is written yet.
Run the suite locally to (re)generate the RAGAS and guardrail matrices.

---

## 5. Running the Evaluations Locally

```bash
# 0. Requirements
pip install -r requirements-prod.txt
pip install ragas==0.4.3 langchain-google-vertexai "langchain-community<0.4.0"

# 1. Static dataset gate (no secrets, offline)
python evals/offline_gate.py --golden evals/golden_dataset.json

# 2. CI RAGAS gate (in-process agent, needs live secrets)
python evals/ci_ragas_gate.py --golden evals/golden_dataset.json

# 3. Full eval suite (requires the backend running on :8080)
uvicorn app.main:app --port 8080
streamlit run evals/app.py
```

The Streamlit eval UI drives `pipeline.py` (Phase 1), `metrics.py` (Phase 2),
and `guardrails_eval.py` (guardrails), and renders per-sample scores and
averages.

---

## 6. Model & Budget Notes (important)

- **Judge model**: `llama-3.3-70b-versatile` is required for Faithfulness.
  `llama-3.1-8b-instant` collapses the NLI verdicts to all-unsupported and must
  not be used as a RAGAS judge.
- **Planner fallback**: the planner's ChatGroq fallback also uses
  `llama-3.3-70b-versatile` so intent routing is reliable when the Portkey
  gateway is not configured or its call fails. In CI, `PORTKEY_API_KEY` is set
  but `GROQ_SLUG` is not, so the Portkey target name cannot be built and the
  ChatGroq fallback path runs.
- **Groq tier**: the on-demand tier caps `llama-3.3-70b-versatile` at ~100k
  tokens/day. The gate is trimmed to fit; upgrading the tier is the lever for
  larger sample sets.
- **`langchain-community < 0.4.0`**: required pin — `ragas==0.4.3` imports the
  legacy `langchain_community.chat_models.vertexai` path removed in 0.4.0.
