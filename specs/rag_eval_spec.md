# RAG Evaluation Workflow

## Context

The learning assistant uses RAG (ChromaDB + Ollama) for the tutor and quiz agents, but has no way to measure retrieval quality or response grounding. This plan adds an evaluation workflow to quantify how well retrieval works and whether generated answers stay faithful to the retrieved context.

## Approach: Custom Metrics (not RAGAS)

RAGAS has compatibility issues with local Ollama models (llama3.2 struggles with RAGAS's internal JSON parsing requirements) and pulls heavy transitive dependencies. Instead, we build lightweight custom metrics reusing existing `invoke_llm()` and `get_retriever()` infrastructure.

## New Files

```
eval/
    __init__.py
    dataset.json              # 15-20 curated Q&A ground truth pairs
    prompts.py                # Judge prompt templates
    metrics/
        __init__.py
        retrieval.py          # Deterministic retrieval metrics
        grounding.py          # LLM-as-judge faithfulness + correctness
    runner.py                 # Orchestrator: retrieve → generate → judge → aggregate
    report.py                 # Console table + JSON output
    run_eval.py               # CLI entry point (__main__)
    results/                  # gitignored output directory
tests/
    test_eval_metrics.py      # Unit tests for deterministic metrics
    test_rag_eval.py          # Integration tests (threshold assertions)
```

## Step 1: Create evaluation dataset (`eval/dataset.json`)

15-20 manually curated entries from the KB files:

```json
[
  {
    "id": "langchain_001",
    "question": "What is LangChain?",
    "ground_truth": "LangChain is a Python framework for building LLM-powered applications...",
    "expected_sources": ["langchain.md"],
    "category": "explain"
  }
]
```

Distribution: ~7 from `langchain.md`, ~6 from `langgraph.md`, ~4 from `python_interview.md`, ~2 negative/out-of-scope questions.

**Source files to read**: `kb/langchain.md`, `kb/langgraph.md`, `kb/python_interview.md`

## Step 2: Create judge prompts (`eval/prompts.py`)

Two prompt templates following the pattern in `app/prompts/tutor.py`:

- **`FAITHFULNESS_JUDGE_PROMPT`** — given `{rag_context}` and `{generated_answer}`, scores 1-5 whether every claim in the answer is supported by the context
- **`CORRECTNESS_JUDGE_PROMPT`** — given `{ground_truth}` and `{generated_answer}`, scores 1-5 whether the answer captures the key points from the reference

Both request JSON output: `{"score": <int>, "reasoning": "<brief>"}`

## Step 3: Create deterministic retrieval metrics (`eval/metrics/retrieval.py`)

Pure functions operating on LangChain `Document` lists, no LLM calls:

| Metric | What it measures |
|---|---|
| **Context precision** | Fraction of retrieved chunks from expected source files |
| **Context recall** | Fraction of expected sources represented in retrieved chunks |
| **Hit rate** | Binary — did at least one relevant chunk appear? |
| **MRR** | Position of first relevant chunk (1/rank) |

Uses `doc.metadata.get("source")` + `os.path.basename()` — same pattern as `app/tools/retrieve_context.py:42-43`.

## Step 4: Create LLM-as-judge metrics (`eval/metrics/grounding.py`)

Two functions using `invoke_llm()` from `app/utils/llm_helpers.py`:

- `score_faithfulness(rag_context, generated_answer) -> {"score": int, "reasoning": str}`
- `score_correctness(ground_truth, generated_answer) -> {"score": int, "reasoning": str}`

JSON extraction: use `_extract_json_object()` pattern from `app/utils/llm_parse.py:30-45`, with regex fallback for `"score":\s*(\d)` when JSON fails (important for llama3.2 reliability).

Also includes a deterministic `token_f1(prediction, reference) -> float` as a bias-free baseline.

## Step 5: Create evaluation runner (`eval/runner.py`)

Orchestrates the full pipeline per sample:

1. **Retrieve** — `get_retriever().invoke(question)` (reuses `app/rag/retriever.py`)
2. **Format context** — replicate logic from `app/tools/retrieve_context.py:39-55`
3. **Generate answer** — replicate `tutor_node` logic: `TUTOR_SYSTEM_PROMPT` + `TUTOR_USER_PROMPT.format(...)` → `invoke_llm()` (reuses `app/prompts/tutor.py`, `app/utils/llm_helpers.py`)
4. **Compute retrieval metrics** — deterministic, no LLM
5. **Compute grounding metrics** — LLM-as-judge calls
6. **Aggregate** — mean/min/max per metric across all samples

Returns `{"samples": [...], "summary": {...}}`.

**Key reuse points**:
- `app/rag/retriever.py` :: `get_retriever()` — same retriever as production
- `app/utils/llm_helpers.py` :: `invoke_llm()` — for generation and judging
- `app/prompts/tutor.py` :: `TUTOR_SYSTEM_PROMPT`, `TUTOR_USER_PROMPT` — identical prompts

## Step 6: Create report output (`eval/report.py`)

- **Console**: summary table with mean scores + worst-performing samples flagged
- **JSON**: full results written to `eval/results/eval_YYYYMMDD_HHMMSS.json`

## Step 7: Create CLI entry point (`eval/run_eval.py`)

```bash
uv run python -m eval.run_eval                      # full evaluation
uv run python -m eval.run_eval --retrieval-only      # skip LLM judge (fast)
uv run python -m eval.run_eval --samples 5           # first N samples only
```

`--retrieval-only` is critical for fast iteration on retrieval params without waiting for LLM calls (~3 LLM calls per sample × 30s timeout = slow).

## Step 8: Add tests

**`tests/test_eval_metrics.py`** (unit, no services):
- Test each retrieval metric function with mock `Document` objects
- Test `token_f1` with known inputs
- Test judge response parsing with various LLM output formats

**`tests/test_rag_eval.py`** (integration, needs Ollama + ChromaDB):
- Threshold assertions: retrieval precision >= 0.6, faithfulness mean >= 3.0/5
- Marked `@pytest.mark.integration` per existing convention

## Step 9: Update project files

- Add `eval/results/` to `.gitignore`
- Add eval CLI commands to `CLAUDE.md`
- No new dependencies needed — everything uses existing `langchain`, `chromadb`, `langchain-ollama`

## Known Limitations

- **Self-evaluation bias**: same model (llama3.2) generates and judges — deterministic metrics (retrieval, token_f1) provide bias-free baselines. Future: add `EVAL_JUDGE_MODEL` config for a separate judge model.
- **Small dataset**: 15-20 samples for a 4-file KB is appropriate for initial measurement but not statistically robust.

## Verification

1. `uv run python -m eval.run_eval --retrieval-only` — confirms retrieval metrics compute without error
2. `uv run python -m eval.run_eval --samples 3` — confirms full pipeline (retrieve + generate + judge) on 3 samples
3. `uv run python -m eval.run_eval` — full run, check JSON output in `eval/results/`
4. `uv run pytest tests/test_eval_metrics.py -v` — unit tests pass
5. `uv run pytest -m integration tests/test_rag_eval.py -v` — integration tests pass (needs local services)
