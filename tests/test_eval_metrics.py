"""Unit tests for deterministic evaluation metrics (no external services)."""

import pytest
from unittest.mock import MagicMock

from eval.metrics.retrieval import context_precision, context_recall, hit_rate, mrr
from eval.metrics.grounding import token_f1, _parse_judge_response


# ── Helpers ──────────────────────────────────────────────────────────

def _make_doc(source: str) -> MagicMock:
    """Create a mock LangChain Document with source metadata."""
    doc = MagicMock()
    doc.metadata = {"source": f"/some/path/{source}"}
    doc.page_content = f"Content from {source}"
    return doc


# ── context_precision ────────────────────────────────────────────────

class TestContextPrecision:
    def test_all_relevant(self):
        docs = [_make_doc("langchain.md"), _make_doc("langchain.md")]
        assert context_precision(docs, ["langchain.md"]) == 1.0

    def test_none_relevant(self):
        docs = [_make_doc("other.md"), _make_doc("other.md")]
        assert context_precision(docs, ["langchain.md"]) == 0.0

    def test_mixed(self):
        docs = [_make_doc("langchain.md"), _make_doc("other.md")]
        assert context_precision(docs, ["langchain.md"]) == 0.5

    def test_empty_docs(self):
        assert context_precision([], ["langchain.md"]) == 0.0

    def test_case_insensitive(self):
        docs = [_make_doc("LangChain.md")]
        assert context_precision(docs, ["langchain.md"]) == 1.0


# ── context_recall ───────────────────────────────────────────────────

class TestContextRecall:
    def test_all_sources_found(self):
        docs = [_make_doc("langchain.md"), _make_doc("langgraph.md")]
        assert context_recall(docs, ["langchain.md", "langgraph.md"]) == 1.0

    def test_partial_recall(self):
        docs = [_make_doc("langchain.md")]
        assert context_recall(docs, ["langchain.md", "langgraph.md"]) == 0.5

    def test_no_expected_sources(self):
        docs = [_make_doc("langchain.md")]
        assert context_recall(docs, []) == 1.0

    def test_no_docs(self):
        assert context_recall([], ["langchain.md"]) == 0.0


# ── hit_rate ─────────────────────────────────────────────────────────

class TestHitRate:
    def test_hit(self):
        docs = [_make_doc("other.md"), _make_doc("langchain.md")]
        assert hit_rate(docs, ["langchain.md"]) == 1.0

    def test_miss(self):
        docs = [_make_doc("other.md")]
        assert hit_rate(docs, ["langchain.md"]) == 0.0

    def test_no_expected(self):
        assert hit_rate([_make_doc("any.md")], []) == 1.0


# ── mrr ──────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_position(self):
        docs = [_make_doc("langchain.md"), _make_doc("other.md")]
        assert mrr(docs, ["langchain.md"]) == 1.0

    def test_second_position(self):
        docs = [_make_doc("other.md"), _make_doc("langchain.md")]
        assert mrr(docs, ["langchain.md"]) == 0.5

    def test_third_position(self):
        docs = [_make_doc("a.md"), _make_doc("b.md"), _make_doc("langchain.md")]
        assert mrr(docs, ["langchain.md"]) == pytest.approx(1 / 3)

    def test_no_match(self):
        docs = [_make_doc("other.md")]
        assert mrr(docs, ["langchain.md"]) == 0.0

    def test_no_expected(self):
        assert mrr([_make_doc("any.md")], []) == 1.0


# ── token_f1 ─────────────────────────────────────────────────────────

class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert token_f1("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        f1 = token_f1("the cat sat", "the cat ran")
        # overlap=2 (the, cat), precision=2/3, recall=2/3, f1=2/3
        assert f1 == pytest.approx(2 / 3)

    def test_empty_prediction(self):
        assert token_f1("", "hello world") == 0.0

    def test_empty_reference(self):
        assert token_f1("hello world", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("Hello World", "hello world") == 1.0


# ── _parse_judge_response ────────────────────────────────────────────

class TestParseJudgeResponse:
    def test_valid_json(self):
        raw = '{"score": 4, "reasoning": "Good answer"}'
        result = _parse_judge_response(raw)
        assert result["score"] == 4
        assert result["reasoning"] == "Good answer"

    def test_json_with_surrounding_text(self):
        raw = 'Here is my evaluation:\n{"score": 3, "reasoning": "Decent"}\nDone.'
        result = _parse_judge_response(raw)
        assert result["score"] == 3

    def test_regex_fallback(self):
        raw = 'score is "score": 5 and "reasoning": "Perfect"'
        result = _parse_judge_response(raw)
        assert result["score"] == 5

    def test_completely_unparseable(self):
        raw = "I think the answer is good"
        result = _parse_judge_response(raw)
        assert result["score"] == 3  # default fallback

    def test_json_with_newlines(self):
        raw = '{\n  "score": 2,\n  "reasoning": "Missing key points"\n}'
        result = _parse_judge_response(raw)
        assert result["score"] == 2
        assert "Missing" in result["reasoning"]
