"""LLM-as-judge grounding metrics and deterministic token F1."""

from __future__ import annotations

import json
import re
from collections import Counter

from app.utils.llm_helpers import invoke_llm
from eval.prompts import CORRECTNESS_JUDGE_PROMPT, FAITHFULNESS_JUDGE_PROMPT


def _parse_judge_response(raw: str) -> dict:
    """Extract score and reasoning from LLM judge output.

    Tries JSON parsing first, then regex fallback for llama3.2 reliability.
    """
    # Try extracting a JSON object (replicates _extract_json_object pattern)
    start = raw.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(raw)):
            if raw[idx] == "{":
                depth += 1
            elif raw[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw[start : idx + 1])
                        if "score" in data:
                            return {
                                "score": int(data["score"]),
                                "reasoning": str(data.get("reasoning", "")),
                            }
                    except (json.JSONDecodeError, ValueError):
                        break

    # Regex fallback
    score_match = re.search(r'"score"\s*:\s*(\d)', raw)
    if score_match:
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
        return {
            "score": int(score_match.group(1)),
            "reasoning": reasoning_match.group(1) if reasoning_match else "",
        }

    return {"score": 3, "reasoning": "Failed to parse judge response"}


def score_faithfulness(rag_context: str, generated_answer: str) -> dict:
    """Score whether the generated answer is faithful to the RAG context.

    Returns {"score": int 1-5, "reasoning": str}.
    """
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        rag_context=rag_context,
        generated_answer=generated_answer,
    )
    raw = invoke_llm(prompt)
    return _parse_judge_response(raw)


def score_correctness(ground_truth: str, generated_answer: str) -> dict:
    """Score whether the generated answer captures key points from reference.

    Returns {"score": int 1-5, "reasoning": str}.
    """
    prompt = CORRECTNESS_JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        generated_answer=generated_answer,
    )
    raw = invoke_llm(prompt)
    return _parse_judge_response(raw)


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference strings.

    Provides a bias-free baseline metric that doesn't rely on LLM judgment.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = sum((pred_counts & ref_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
