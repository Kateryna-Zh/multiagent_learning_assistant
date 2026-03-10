"""CLI entry point for RAG evaluation.

Usage:
    uv run python -m eval.run_eval                      # full evaluation
    uv run python -m eval.run_eval --retrieval-only      # skip LLM judge
    uv run python -m eval.run_eval --samples 5           # first N samples
"""

from __future__ import annotations

import argparse

from eval.report import print_summary, save_json
from eval.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM-as-judge metrics (fast, retrieval metrics only)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples",
    )
    args = parser.parse_args()

    print("Starting RAG evaluation...")
    if args.retrieval_only:
        print("Mode: retrieval-only (no LLM judge calls)")
    if args.samples:
        print(f"Limiting to first {args.samples} samples")

    results = run_evaluation(
        retrieval_only=args.retrieval_only,
        max_samples=args.samples,
    )

    print_summary(results)
    save_json(results)


if __name__ == "__main__":
    main()
