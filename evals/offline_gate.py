#!/usr/bin/env python3
"""
Offline static eval gate.

Deterministic, secrets-free CI gate over evals/golden_dataset.json.

Unlike evals/pipeline.py, evals/guardrails_eval.py, and evals/metrics.py, this
script does NOT call the live /query API or any LLM. It validates that the
golden dataset is well-formed and internally coherent, so that live eval runs
are never fed a broken dataset (e.g. an empty reference, a typo'd expected
tool, or a guardrail case whose expected_blocked contradicts its declared
attack class).

Runs on the Python stdlib alone (no pip installs needed in CI).

Exit codes:
    0  all checks pass — dataset is safe to eval against
    1  any check fails — dataset must be fixed before merging

Run:
    python evals/offline_gate.py
    python evals/offline_gate.py --golden path/to/dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Tool taxonomy shared with the planner node / detect_tool() in evals.pipeline.
KNOWN_TOOLS = {"retrieve_documents", "direct_answer", "guardrails"}
# Guardrail test-class taxonomy used by evals.guardrails_eval and evals.app.
GUARDRAIL_TYPES = {"jailbreak", "off_topic", "legit"}
# Anything shorter than this is noise, not evidence.
MIN_CONTEXT_LEN = 15

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_dataset.json"


def _label(sample: dict, kind: str) -> str:
    return f"{kind}_sample[{sample.get('id', '?')}]"


def _require(sample: dict, key: str, label: str, failures: list) -> None:
    if key not in sample:
        failures.append(f"{label} missing required field {key!r}.")


def _require_nonempty(sample: dict, key: str, label: str, failures: list) -> None:
    _require(sample, key, label, failures)
    val = sample.get(key)
    if val is not None and not str(val).strip():
        failures.append(f"{label} field {key!r} must be non-empty.")


def check_rag_samples(samples: list, failures: list) -> None:
    """Each golden Q&A pair must be complete and use only known tools."""
    seen_ids: set = set()
    for s in samples:
        label = _label(s, "rag")

        _require(s, "id", label, failures)
        if "id" in s:
            if s["id"] in seen_ids:
                failures.append(f"{label} duplicate id {s['id']!r}.")
            seen_ids.add(s["id"])

        _require_nonempty(s, "domain", label, failures)
        _require_nonempty(s, "question", label, failures)
        _require_nonempty(s, "reference", label, failures)

        contexts = s.get("relevant_contexts")
        if not isinstance(contexts, list) or not contexts:
            failures.append(f"{label} 'relevant_contexts' must be a non-empty list.")
        else:
            for i, ctx in enumerate(contexts):
                if not isinstance(ctx, str) or len(ctx.strip()) < MIN_CONTEXT_LEN:
                    failures.append(
                        f"{label} relevant_contexts[{i}] shorter than "
                        f"{MIN_CONTEXT_LEN} chars: {str(ctx)[:40]!r}"
                    )

        tools = s.get("expected_tools")
        if not isinstance(tools, list) or not tools:
            failures.append(f"{label} 'expected_tools' must be a non-empty list.")
        else:
            for t in tools:
                if t not in KNOWN_TOOLS:
                    failures.append(
                        f"{label} unknown expected tool {t!r} "
                        f"(known: {sorted(KNOWN_TOOLS)})."
                    )


def check_guardrails_samples(samples: list, failures: list) -> None:
    """Each case must be well-formed and its expected_blocked must match its type."""
    seen_ids: set = set()
    has_block = False
    has_pass = False
    for s in samples:
        label = _label(s, "guardrails")

        _require(s, "id", label, failures)
        if "id" in s:
            if s["id"] in seen_ids:
                failures.append(f"{label} duplicate id {s['id']!r}.")
            seen_ids.add(s["id"])

        _require_nonempty(s, "input", label, failures)
        _require(s, "description", label, failures)

        blocked = s.get("expected_blocked")
        if not isinstance(blocked, bool):
            failures.append(f"{label} 'expected_blocked' must be a boolean, got {blocked!r}.")
            continue  # can't reason about coherence without a valid label

        gtype = s.get("type")
        if gtype not in GUARDRAIL_TYPES:
            failures.append(
                f"{label} unknown guardrail type {gtype!r} "
                f"(known: {sorted(GUARDRAIL_TYPES)})."
            )
            continue

        # Policy coherence: jailbreaks/off-topic requests must be blocked,
        # legitimate enterprise questions must pass through.
        if blocked and gtype == "legit":
            failures.append(
                f"{label} type={gtype!r} but expected_blocked=True — "
                "legitimate questions must not be labeled as blocked."
            )
        if not blocked and gtype != "legit":
            failures.append(
                f"{label} type={gtype!r} but expected_blocked=False — "
                f"{gtype} inputs must be labeled as blocked."
            )

        if blocked:
            has_block = True
        else:
            has_pass = True

    if not has_block:
        failures.append("No guardrails sample expects a block — adversarial cases are missing.")
    if not has_pass:
        failures.append("No guardrails sample expects a pass — legitimate cases are missing.")


def check_dataset(dataset: dict, failures: list) -> None:
    if "rag_samples" not in dataset or not isinstance(dataset["rag_samples"], list):
        failures.append("Missing or invalid 'rag_samples' list.")
        return
    if "guardrails_samples" not in dataset or not isinstance(dataset["guardrails_samples"], list):
        failures.append("Missing or invalid 'guardrails_samples' list.")
        return

    if not dataset["rag_samples"]:
        failures.append("'rag_samples' is empty — no golden Q&A pairs to eval.")
    if not dataset["guardrails_samples"]:
        failures.append("'guardrails_samples' is empty — no guardrail test cases to eval.")

    check_rag_samples(dataset["rag_samples"], failures)
    check_guardrails_samples(dataset["guardrails_samples"], failures)


def _print_report(failures: list) -> None:
    if not failures:
        print("[PASS] OFFLINE EVAL GATE — golden dataset is well-formed and coherent.")
        return
    print("[FAIL] OFFLINE EVAL GATE — fix the golden dataset before merging:\n")
    for i, failure in enumerate(failures, 1):
        print(f"  {i}. {failure}")
    print(f"\n{len(failures)} violation(s) found.")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Path to the golden dataset JSON (default: evals/golden_dataset.json).",
    )
    args = parser.parse_args(argv)

    try:
        dataset = json.loads(args.golden.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[FAIL] Golden dataset not found: {args.golden}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"[FAIL] Golden dataset is not valid JSON: {exc}")
        return 1

    failures: list = []
    check_dataset(dataset, failures)
    _print_report(failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
