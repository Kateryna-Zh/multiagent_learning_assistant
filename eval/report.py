"""Console table and JSON output for evaluation results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def print_summary(results: dict) -> None:
    """Print a formatted summary table to the console."""
    summary = results["summary"]
    samples = results["samples"]

    print("\n" + "=" * 70)
    print("RAG EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {summary.get('total_samples', len(samples))}")
    print()

    # Retrieval metrics
    print("RETRIEVAL METRICS")
    print("-" * 40)
    for key in ["context_precision", "context_recall", "hit_rate", "mrr"]:
        mean_key = f"retrieval_{key}_mean"
        min_key = f"retrieval_{key}_min"
        if mean_key in summary:
            line = f"  {key:<22} mean={summary[mean_key]:.3f}"
            if min_key in summary:
                line += f"  min={summary[min_key]:.3f}"
            print(line)

    # Grounding metrics
    has_grounding = any("grounding" in s for s in samples)
    if has_grounding:
        print()
        print("GROUNDING METRICS")
        print("-" * 40)
        for metric in ["faithfulness", "correctness"]:
            mean_key = f"{metric}_mean"
            if mean_key in summary:
                line = f"  {metric:<22} mean={summary[mean_key]:.2f}/5"
                if f"{metric}_min" in summary:
                    line += f"  min={summary[f'{metric}_min']}"
                if f"{metric}_max" in summary:
                    line += f"  max={summary[f'{metric}_max']}"
                print(line)

        if "token_f1_mean" in summary:
            line = f"  {'token_f1':<22} mean={summary['token_f1_mean']:.3f}"
            if "token_f1_min" in summary:
                line += f"  min={summary['token_f1_min']:.3f}"
            print(line)

    # Worst-performing samples
    print()
    print("WORST-PERFORMING SAMPLES")
    print("-" * 40)

    # By retrieval precision
    retrieval_samples = [
        (s["id"], s["retrieval"]["context_precision"], s["question"])
        for s in samples
        if "retrieval" in s and s.get("category") != "out_of_scope"
    ]
    retrieval_samples.sort(key=lambda x: x[1])
    for sid, score, question in retrieval_samples[:3]:
        print(f"  [{sid}] precision={score:.2f} — {question[:50]}")

    # By faithfulness (if available)
    faith_samples = [
        (s["id"], s["grounding"]["faithfulness"]["score"], s["question"])
        for s in samples
        if "grounding" in s and "faithfulness" in s.get("grounding", {})
    ]
    if faith_samples:
        faith_samples.sort(key=lambda x: x[1])
        print()
        for sid, score, question in faith_samples[:3]:
            print(f"  [{sid}] faithfulness={score}/5 — {question[:50]}")

    print("=" * 70)


def save_json(results: dict) -> Path:
    """Write full results to a timestamped JSON file in eval/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    return output_path
