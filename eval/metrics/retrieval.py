"""Deterministic retrieval metrics operating on LangChain Document lists."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def _source_name(doc: Document) -> str:
    """Extract basename from document metadata source field."""
    source = doc.metadata.get("source") or ""
    return os.path.basename(source) if source else ""


def context_precision(docs: list[Document], expected_sources: list[str]) -> float:
    """Fraction of retrieved chunks that come from expected source files."""
    if not docs:
        return 0.0
    expected = {s.lower() for s in expected_sources}
    relevant = sum(1 for d in docs if _source_name(d).lower() in expected)
    return relevant / len(docs)


def context_recall(docs: list[Document], expected_sources: list[str]) -> float:
    """Fraction of expected sources represented in retrieved chunks."""
    if not expected_sources:
        return 1.0
    expected = {s.lower() for s in expected_sources}
    found = {_source_name(d).lower() for d in docs} & expected
    return len(found) / len(expected)


def hit_rate(docs: list[Document], expected_sources: list[str]) -> float:
    """Binary — did at least one relevant chunk appear?"""
    if not expected_sources:
        return 1.0
    expected = {s.lower() for s in expected_sources}
    for d in docs:
        if _source_name(d).lower() in expected:
            return 1.0
    return 0.0


def mrr(docs: list[Document], expected_sources: list[str]) -> float:
    """Mean Reciprocal Rank — 1/position of first relevant chunk."""
    if not expected_sources:
        return 1.0
    expected = {s.lower() for s in expected_sources}
    for i, d in enumerate(docs):
        if _source_name(d).lower() in expected:
            return 1.0 / (i + 1)
    return 0.0
