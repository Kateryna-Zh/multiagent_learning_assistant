"""Integration tests for RAG evaluation pipeline.

Requires Ollama and ChromaDB running locally.
"""

import pytest

from eval.runner import run_evaluation


@pytest.mark.integration
def test_retrieval_precision_above_threshold():
    """Retrieval context_precision mean should be >= 0.6."""
    results = run_evaluation(retrieval_only=True, max_samples=5)
    summary = results["summary"]
    assert summary["retrieval_context_precision_mean"] >= 0.6, (
        f"Context precision too low: {summary['retrieval_context_precision_mean']:.3f}"
    )


@pytest.mark.integration
def test_retrieval_hit_rate_above_threshold():
    """Hit rate should be >= 0.8 for in-scope questions."""
    results = run_evaluation(retrieval_only=True, max_samples=5)
    # Filter out out-of-scope samples
    in_scope = [
        s for s in results["samples"] if s.get("category") != "out_of_scope"
    ]
    if in_scope:
        mean_hit = sum(s["retrieval"]["hit_rate"] for s in in_scope) / len(in_scope)
        assert mean_hit >= 0.8, f"Hit rate too low: {mean_hit:.3f}"


@pytest.mark.integration
def test_faithfulness_above_threshold():
    """Faithfulness mean score should be >= 3.0/5."""
    results = run_evaluation(retrieval_only=False, max_samples=3)
    summary = results["summary"]
    assert "faithfulness_mean" in summary, "No faithfulness scores computed"
    assert summary["faithfulness_mean"] >= 3.0, (
        f"Faithfulness too low: {summary['faithfulness_mean']:.2f}/5"
    )


@pytest.mark.integration
def test_full_pipeline_produces_all_metrics():
    """Full pipeline should produce retrieval + grounding metrics."""
    results = run_evaluation(retrieval_only=False, max_samples=2)
    summary = results["summary"]

    assert "retrieval_context_precision_mean" in summary
    assert "retrieval_hit_rate_mean" in summary
    assert "faithfulness_mean" in summary
    assert "total_samples" in summary
    assert summary["total_samples"] == 2
