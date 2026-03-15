# RAG Evaluation Workflow

**Status:** Done
**Priority:** Medium
**Created:** 2026-03-15

## Context

The learning assistant uses RAG (ChromaDB + Ollama) for the tutor and quiz agents, but has no way to measure retrieval quality or response grounding. Without metrics, there is no feedback loop for tuning retrieval parameters (chunk size, overlap, top-k) or prompt changes.

RAGAS was evaluated and rejected — `llama3.2` struggles with RAGAS's internal JSON parsing, and it pulls heavy transitive dependencies. Instead, we build lightweight custom metrics reusing existing `invoke_llm()` and `get_retriever()` infrastructure.

## Goal

Add an evaluation workflow that quantifies retrieval quality and answer grounding with:
- Deterministic retrieval metrics (no LLM calls, fast).
- LLM-as-judge grounding metrics (faithfulness, correctness).
- A deterministic token F1 baseline (bias-free).
- CLI with `--retrieval-only` for fast iteration without LLM overhead.

## Requirements

- R1: Evaluation dataset of 15-20 curated Q&A pairs sourced from existing KB files, with expected source annotations.
- R2: Four deterministic retrieval metrics: context precision, context recall, hit rate, MRR.
- R3: Two LLM-as-judge metrics: faithfulness (1-5) and correctness (1-5), with robust JSON parsing + regex fallback.
- R4: Deterministic token F1 as a bias-free baseline (same model generates and judges).
- R5: CLI entry point with `--retrieval-only` (skip LLM judge) and `--samples N` (subset evaluation).
- R6: Console summary table + timestamped JSON output in `eval/results/` (gitignored).
- R7: Unit tests for all deterministic metrics and judge response parsing. Integration tests with threshold assertions.
- R8: No new dependencies — reuse existing `langchain`, `chromadb`, `langchain-ollama`.

## File Structure

```
eval/
    __init__.py                   # Package marker
    __main__.py                   # `python -m eval.run_eval` entry
    run_eval.py                   # CLI argument parsing, orchestration
    runner.py                     # Pipeline: retrieve → generate → judge → aggregate
    prompts.py                    # FAITHFULNESS_JUDGE_PROMPT, CORRECTNESS_JUDGE_PROMPT
    report.py                     # Console table + JSON persistence
    dataset.json                  # 19 curated Q&A ground truth pairs
    metrics/
        __init__.py               # Package marker
        retrieval.py              # Deterministic retrieval metrics
        grounding.py              # LLM-as-judge + token F1
    results/                      # Gitignored output directory
tests/
    test_eval_metrics.py          # Unit tests (no services)
    test_rag_eval.py              # Integration tests (Ollama + ChromaDB)
```

## Tasks

### Task 1: Create evaluation dataset

**File:** `eval/dataset.json`

**Changes:**
1. Create 19 manually curated entries from KB files with this schema:
   ```json
   {
     "id": "langchain_001",
     "question": "What is LangChain?",
     "ground_truth": "LangChain is a Python framework...",
     "expected_sources": ["langchain.md"],
     "category": "explain"
   }
   ```
2. Distribution: 7 from `langchain.md`, 6 from `langgraph.md`, 4 from `python_interview.md`, 2 negative/out-of-scope (empty `ground_truth`, empty `expected_sources`, category `"out_of_scope"`).
3. Ground truth must be manually written from KB content (not LLM-generated).

**Source files to read:** `kb/langchain.md`, `kb/langgraph.md`, `kb/python_interview.md`

**Acceptance criteria:**
- 19 entries with all required fields.
- At least 2 out-of-scope entries with empty `expected_sources`.
- Ground truth matches actual KB content.

### Task 2: Create judge prompts

**File:** `eval/prompts.py`

**Changes:**
1. `FAITHFULNESS_JUDGE_PROMPT` — given `{rag_context}` and `{generated_answer}`, scores 1-5 whether every claim in the answer is supported by the context.
2. `CORRECTNESS_JUDGE_PROMPT` — given `{ground_truth}` and `{generated_answer}`, scores 1-5 whether the answer captures key points from the reference.
3. Both request JSON output: `{"score": <int>, "reasoning": "<brief>"}`.
4. Use double-brace escaping for JSON template literals in f-string-safe prompts.

**Acceptance criteria:**
- Two prompt templates with `{rag_context}`, `{generated_answer}`, `{ground_truth}` placeholders.
- Clear 1-5 scoring rubric in each prompt.
- JSON output format specified.

### Task 3: Create deterministic retrieval metrics

**File:** `eval/metrics/retrieval.py`

**Changes:**
Create four pure functions operating on LangChain `Document` lists:

| Function | Signature | What it measures |
|---|---|---|
| `context_precision(docs, expected_sources)` | `-> float` | Fraction of retrieved chunks from expected source files |
| `context_recall(docs, expected_sources)` | `-> float` | Fraction of expected sources represented in retrieved chunks |
| `hit_rate(docs, expected_sources)` | `-> float` | Binary — at least one relevant chunk? (1.0 or 0.0) |
| `mrr(docs, expected_sources)` | `-> float` | 1/position of first relevant chunk |

Implementation details:
- Extract source via `doc.metadata.get("source")` + `os.path.basename()` — same pattern as `app/tools/retrieve_context.py:42-43`.
- Case-insensitive source matching.
- Return 0.0 for empty docs, 1.0 for empty `expected_sources` (vacuously true).
- Use `TYPE_CHECKING` guard for `Document` import to avoid runtime dependency.

**Acceptance criteria:**
- All four functions are pure (no LLM calls, no I/O).
- Edge cases handled: empty docs, empty expected_sources, case insensitivity.

### Task 4: Create LLM-as-judge grounding metrics

**File:** `eval/metrics/grounding.py`

**Changes:**
1. `score_faithfulness(rag_context, generated_answer) -> dict` — uses `invoke_llm()` with `FAITHFULNESS_JUDGE_PROMPT`. Returns `{"score": int, "reasoning": str}`.
2. `score_correctness(ground_truth, generated_answer) -> dict` — uses `invoke_llm()` with `CORRECTNESS_JUDGE_PROMPT`. Returns `{"score": int, "reasoning": str}`.
3. `token_f1(prediction, reference) -> float` — deterministic token-level F1 using `Counter` overlap. Case-insensitive, whitespace-split tokenization.
4. `_parse_judge_response(raw) -> dict` — JSON extraction with brace-depth matching, regex fallback `"score"\s*:\s*(\d)`, default score 3 when unparseable.

**Reuse:** `invoke_llm()` from `app/utils/llm_helpers.py`, prompts from `eval/prompts.py`.

**Acceptance criteria:**
- Judge functions return `{"score": int, "reasoning": str}`.
- `_parse_judge_response` handles: valid JSON, JSON with surrounding text, regex fallback, completely unparseable (default 3).
- `token_f1` returns 0.0 for empty inputs, 1.0 for exact match.

### Task 5: Create evaluation runner

**File:** `eval/runner.py`

**Changes:**
1. `load_dataset(path?) -> list[dict]` — loads `eval/dataset.json`.
2. `evaluate_sample(sample, retriever, retrieval_only=False) -> dict` — per-sample pipeline:
   - Retrieve: `retriever.invoke(question)`.
   - Format context: replicate `app/tools/retrieve_context.py:39-55` logic (source basename + page_content).
   - Compute retrieval metrics (deterministic).
   - If not `retrieval_only`: generate answer using tutor prompts (`TUTOR_SYSTEM_PROMPT` + `TUTOR_USER_PROMPT` → `invoke_llm()`), then compute grounding metrics.
3. `aggregate(samples) -> dict` — mean/min/max per metric across all samples.
4. `run_evaluation(retrieval_only=False, max_samples=None, dataset_path=None) -> dict` — orchestrator returning `{"samples": [...], "summary": {...}}`.

**Reuse points:**
- `app/rag/retriever.py::get_retriever()` — same retriever as production.
- `app/utils/llm_helpers.py::invoke_llm()` — for answer generation and judging.
- `app/prompts/tutor.py::TUTOR_SYSTEM_PROMPT, TUTOR_USER_PROMPT` — identical prompts to production.

**Acceptance criteria:**
- `retrieval_only=True` skips all LLM calls (generation + judging).
- `max_samples` limits dataset size.
- Output dict includes per-sample retrieval metrics and (optionally) grounding metrics.
- Aggregate summary includes mean and min for all metric types.

### Task 6: Create report output

**File:** `eval/report.py`

**Changes:**
1. `print_summary(results) -> None` — console output with:
   - Retrieval metrics table (mean, min).
   - Grounding metrics table (mean, min, max) — only if present.
   - Token F1 (mean, min) — only if present.
   - Worst-performing samples (bottom 3 by precision, bottom 3 by faithfulness).
   - Filters out `out_of_scope` category from worst-performing list.
2. `save_json(results) -> Path` — writes to `eval/results/eval_YYYYMMDD_HHMMSS.json`.

**Acceptance criteria:**
- Console output is human-readable with section headers and alignment.
- JSON output includes full results (samples + summary).
- `eval/results/` directory created automatically if missing.

### Task 7: Create CLI entry point

**Files:** `eval/run_eval.py`, `eval/__main__.py`

**Changes:**
1. `eval/run_eval.py`: `argparse` CLI with `--retrieval-only` (store_true) and `--samples N` (int, optional). Calls `run_evaluation()`, then `print_summary()` and `save_json()`.
2. `eval/__main__.py`: delegates to `run_eval.main()` for `python -m eval.run_eval` usage.

**CLI commands:**
```bash
uv run python -m eval.run_eval                      # full evaluation
uv run python -m eval.run_eval --retrieval-only      # skip LLM judge (fast)
uv run python -m eval.run_eval --samples 5           # first N samples only
```

**Acceptance criteria:**
- `--retrieval-only` produces results with no LLM calls (~3 LLM calls per sample x 30s timeout = significant time savings).
- `--samples 5` evaluates only first 5 entries.
- Progress printed to console: `[1/20] Evaluating: What is LangChain?...`

### Task 8: Add tests

**File:** `tests/test_eval_metrics.py` (unit, no services)

**Changes:**
Test classes with the following coverage:
- `TestContextPrecision` — 5 tests: all relevant, none relevant, mixed, empty docs, case insensitive.
- `TestContextRecall` — 4 tests: all found, partial, no expected sources, no docs.
- `TestHitRate` — 3 tests: hit, miss, no expected.
- `TestMRR` — 5 tests: positions 1/2/3, no match, no expected.
- `TestTokenF1` — 6 tests: exact match, no overlap, partial, empty prediction, empty reference, case insensitive.
- `TestParseJudgeResponse` — 5 tests: valid JSON, JSON with surrounding text, regex fallback, completely unparseable, JSON with newlines.

Uses `MagicMock` for `Document` objects with `.metadata = {"source": "/some/path/file.md"}`.

**File:** `tests/test_rag_eval.py` (integration, needs Ollama + ChromaDB)

**Changes:**
All marked `@pytest.mark.integration`:
- `test_retrieval_precision_above_threshold` — precision mean >= 0.6 (5 samples, retrieval-only).
- `test_retrieval_hit_rate_above_threshold` — hit rate >= 0.8 for in-scope questions (5 samples).
- `test_faithfulness_above_threshold` — faithfulness mean >= 3.0/5 (3 samples, full pipeline).
- `test_full_pipeline_produces_all_metrics` — validates all metric keys present (2 samples).

**Acceptance criteria:**
- Unit tests pass without external services: `uv run pytest tests/test_eval_metrics.py -v`
- Integration tests pass with local services: `uv run pytest -m integration tests/test_rag_eval.py -v`
- 28+ individual test cases total.

### Task 9: Update project files

**Changes:**
1. Add `eval/results/` to `.gitignore`.
2. Add eval CLI commands to `CLAUDE.md` (already done — see Commands section).
3. No new dependencies needed.

**Acceptance criteria:**
- `eval/results/*.json` files are gitignored.
- `CLAUDE.md` documents eval CLI usage.

## Dependencies

- Ollama running locally with `llama3.2` model (generation + judging).
- ChromaDB populated (`uv run python scripts/ingest_kb.py`).
- No new Python packages.

## Known Limitations

- **Self-evaluation bias:** Same model (`llama3.2`) generates and judges. Mitigated by deterministic metrics (retrieval precision/recall/hit_rate/MRR, token_f1) that provide bias-free baselines. Future improvement: add `EVAL_JUDGE_MODEL` config for a separate judge model.
- **Small dataset:** 20 samples across a 4-file KB is appropriate for initial measurement but not statistically robust.
- **Faithfulness default:** When `_parse_judge_response` fails, it returns score 3 (midpoint). This avoids penalizing parsing failures but may mask issues.

## Verification

| # | Command | Expected |
|---|---------|----------|
| 1 | `uv run python -m eval.run_eval --retrieval-only` | Retrieval metrics compute without error, console table printed |
| 2 | `uv run python -m eval.run_eval --samples 3` | Full pipeline on 3 samples, JSON saved to `eval/results/` |
| 3 | `uv run python -m eval.run_eval` | Full 19-sample run, check JSON output |
| 4 | `uv run pytest tests/test_eval_metrics.py -v` | All 28+ unit tests pass |
| 5 | `uv run pytest -m integration tests/test_rag_eval.py -v` | Integration tests pass with thresholds met |

## Non-Goals

- No RAGAS dependency.
- No multi-model judge configuration (future work).
- No automatic CI integration (manual runs only).
- No changes to production RAG pipeline or prompts.
