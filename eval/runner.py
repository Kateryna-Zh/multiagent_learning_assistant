"""Evaluation runner — orchestrates retrieve → generate → judge → aggregate."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.prompts.tutor import TUTOR_SYSTEM_PROMPT, TUTOR_USER_PROMPT
from app.rag.retriever import get_retriever
from app.utils.llm_helpers import invoke_llm
from eval.metrics.grounding import score_correctness, score_faithfulness, token_f1
from eval.metrics.retrieval import context_precision, context_recall, hit_rate, mrr

DATASET_PATH = Path(__file__).parent / "dataset.json"


def load_dataset(path: Path | None = None) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    p = path or DATASET_PATH
    with open(p) as f:
        return json.load(f)


def _format_rag_context(docs) -> str:
    """Format retrieved documents into context string.

    Replicates logic from app/tools/retrieve_context.py:39-55.
    """
    chunks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source") or ""
        source_name = os.path.basename(source) if source else ""
        if source_name:
            chunks.append(f"Source: {source_name}\n{doc.page_content.strip()}")
        else:
            chunks.append(doc.page_content.strip())
    return "\n\n".join(chunks).strip()


def _generate_answer(question: str, rag_context: str) -> str:
    """Generate an answer using tutor prompts, replicating tutor_node logic."""
    system = TUTOR_SYSTEM_PROMPT.format(rag_context=rag_context)
    user = TUTOR_USER_PROMPT.format(user_input=question)
    prompt = f"{system}\n\n{user}"
    return invoke_llm(prompt)


def evaluate_sample(
    sample: dict,
    retriever,
    retrieval_only: bool = False,
) -> dict:
    """Run full evaluation pipeline for a single sample.

    Returns a dict with question, retrieval metrics, and optionally grounding metrics.
    """
    question = sample["question"]
    expected_sources = sample.get("expected_sources", [])
    ground_truth = sample.get("ground_truth", "")

    # Step 1: Retrieve
    docs = retriever.invoke(question)

    # Step 2: Format context
    rag_context = _format_rag_context(docs)

    # Step 3: Retrieval metrics (deterministic)
    result = {
        "id": sample["id"],
        "question": question,
        "category": sample.get("category", ""),
        "num_docs_retrieved": len(docs),
        "retrieval": {
            "context_precision": context_precision(docs, expected_sources),
            "context_recall": context_recall(docs, expected_sources),
            "hit_rate": hit_rate(docs, expected_sources),
            "mrr": mrr(docs, expected_sources),
        },
    }

    if retrieval_only:
        return result

    # Step 4: Generate answer
    generated_answer = _generate_answer(question, rag_context)
    result["generated_answer"] = generated_answer
    result["rag_context_preview"] = rag_context[:300] + "..." if len(rag_context) > 300 else rag_context

    # Step 5: Grounding metrics
    faithfulness = score_faithfulness(rag_context, generated_answer)
    grounding: dict = {"faithfulness": faithfulness}

    if ground_truth:
        correctness = score_correctness(ground_truth, generated_answer)
        grounding["correctness"] = correctness
        grounding["token_f1"] = token_f1(generated_answer, ground_truth)

    result["grounding"] = grounding

    return result


def aggregate(samples: list[dict]) -> dict:
    """Compute aggregate statistics across all evaluated samples."""
    summary: dict = {}

    # Retrieval metrics
    retrieval_keys = ["context_precision", "context_recall", "hit_rate", "mrr"]
    for key in retrieval_keys:
        values = [s["retrieval"][key] for s in samples if "retrieval" in s]
        if values:
            summary[f"retrieval_{key}_mean"] = sum(values) / len(values)
            summary[f"retrieval_{key}_min"] = min(values)

    # Grounding metrics (only if present)
    for metric in ["faithfulness", "correctness"]:
        scores = [
            s["grounding"][metric]["score"]
            for s in samples
            if "grounding" in s and metric in s.get("grounding", {})
        ]
        if scores:
            summary[f"{metric}_mean"] = sum(scores) / len(scores)
            summary[f"{metric}_min"] = min(scores)
            summary[f"{metric}_max"] = max(scores)

    # Token F1
    f1_values = [
        s["grounding"]["token_f1"]
        for s in samples
        if "grounding" in s and "token_f1" in s.get("grounding", {})
    ]
    if f1_values:
        summary["token_f1_mean"] = sum(f1_values) / len(f1_values)
        summary["token_f1_min"] = min(f1_values)

    summary["total_samples"] = len(samples)
    return summary


def run_evaluation(
    retrieval_only: bool = False,
    max_samples: int | None = None,
    dataset_path: Path | None = None,
) -> dict:
    """Run the full evaluation pipeline.

    Returns {"samples": [...], "summary": {...}}.
    """
    dataset = load_dataset(dataset_path)
    if max_samples is not None:
        dataset = dataset[:max_samples]

    retriever = get_retriever()
    results = []

    for i, sample in enumerate(dataset):
        print(f"[{i + 1}/{len(dataset)}] Evaluating: {sample['question'][:60]}...")
        result = evaluate_sample(sample, retriever, retrieval_only=retrieval_only)
        results.append(result)

    summary = aggregate(results)
    return {"samples": results, "summary": summary}
