# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run dev server (auto-reload)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run dev server with logging
uv run uvicorn app.main:app --reload --log-level info --access-log

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_router.py -v           # single test file
uv run pytest tests/test_router.py::test_name -v # single test
uv run pytest -m integration tests/              # integration tests only (need local services)

# Lint
uv run ruff check app/ tests/

# Ingest knowledge base into ChromaDB
uv run python scripts/ingest_kb.py              # append
uv run python scripts/ingest_kb.py --rebuild    # delete & recreate

# Initialize PostgreSQL schema
psql -U postgres -d learning_assistant -f db/init.sql

# MCP connectivity check
uv run python -m app.cli.mcp_check --message "hello" --debug
uv run python -m app.cli.mcp_check --message "hello" --cleanup   # deletes test rows

# Interactive chat CLI
uv run python -m app.cli.chat_cli --url http://127.0.0.1:8000/chat

# RAG evaluation
uv run python -m eval.run_eval                      # full evaluation
uv run python -m eval.run_eval --retrieval-only      # skip LLM judge (fast)
uv run python -m eval.run_eval --samples 5           # first N samples only
```

## Architecture

Multi-agent learning assistant using **LangGraph** for orchestration, **Ollama** for local LLMs, **PostgreSQL** for persistence, and **ChromaDB** for RAG.

### Graph Flow

```
START → router → (conditional routing) → [context tools | specialist] → format_response → END
```

The **router** (`app/agents/router_agent.py`) classifies each user message into an intent (PLAN, EXPLAIN, QUIZ, LOG_PROGRESS, REVIEW, LATEST) and sets routing flags (`needs_rag`, `needs_web`, `needs_db`).

**Routing logic** (`app/graph/routing.py`):
- `needs_db=true` → `db` agent (except PLAN without SAVE_PLAN → `planner`)
- `needs_rag=true` → `retrieve_context` (ChromaDB) → specialist
- `needs_web=true` → `web_search` (Tavily) → specialist
- Otherwise → specialist directly based on intent

**Specialists**: planner, tutor, quiz, research, db — all flow to `format_response` → END.

### Key Patterns

- **Dual DB backend**: MCP server (primary, via `pg-mcp-server` over stdio) with psycopg2 fallback. Controlled by `DB_BACKEND` env var. Backend selection in `app/db/repository_factory.py`.
- **MCP lifecycle**: Started/stopped via FastAPI lifespan in `app/main.py`. Long-lived session managed by `app/mcp/manager.py`. The `MCPClient` (`app/mcp/client.py`) handles cross-thread async via `run_coroutine_threadsafe` since the graph runs in a `ThreadPoolExecutor`.
- **Repository pattern**: `MCPRepository` and `PsycopgRepository` share the same method signatures. `mcp_repository.py` converts `%s` placeholders to `$1` dollar params or inlines them when `MCP_SUPPORTS_PARAMS=false`.
- **Tool-calling agent**: `db_agent` binds LangChain `StructuredTool`s to the LLM for DB operations, with an intent-driven fallback path if tool-calling returns nothing.
- **Plan draft flow**: Planner creates a draft → stored in `_PLAN_DRAFTS` (in-memory, keyed by session_id) → user confirms → router sets `PLAN/SAVE_PLAN` → db agent writes to DB.
- **Session state**: `_SESSION_CACHE` in `main.py` persists `last_intent`, `last_db_context`, and `quiz_state` across turns.
- **Shared utilities**: `app/utils/llm_helpers.py` (`invoke_llm()` used by all agents), `app/utils/constants.py` (compiled regexes, stopwords, thresholds), `app/db/row_extract.py` (canonical MCP row extraction shared by `mcp_repository` and `mcp_check`).
- **LLM output parsing**: `app/utils/llm_parse.py` extracts JSON from LLM text, validates against Pydantic schemas, and retries with a correction prompt on failure.

### State Model

`GraphState` (`app/models/state.py`) is a TypedDict. Key fields:
- `messages` (with `add_messages` reducer), `user_input`, `intent`, `sub_intent`
- `needs_rag/web/db` — routing flags
- `rag_context`, `web_context`, `db_context` — retrieved data
- `plan_draft`, `plan_confirmed`, `quiz_state` — multi-turn state

### Database Schema

Defined in `db/init.sql`: sessions, messages, topics, study_plan, plan_items, quiz_attempts, flashcards.

## Configuration

All config via `.env` (see `.env.example`). Loaded by Pydantic `BaseSettings` in `app/config.py`.

Key settings: `DB_BACKEND` (mcp|psycopg2), `PG_PORT` (default 5433), `MCP_SUPPORTS_PARAMS` (false for pg-mcp-server), `MCP_FALLBACK_TO_PSYCOPG2` (true).

Timeout settings: `OLLAMA_TIMEOUT_SECONDS` (30), `CHAT_TIMEOUT_SECONDS` (15, graph execution), `DB_TOOL_TIMEOUT_SECONDS` (4).

## Testing

- Unit tests run without external services; integration tests (marked `@pytest.mark.integration`) need PostgreSQL, Ollama, and ChromaDB running.
- Tests use `FakeRepo` for repository mocking and `monkeypatch` for LLM stubbing.
- Graph runs in a `ThreadPoolExecutor` (single worker) with `CHAT_TIMEOUT_SECONDS` enforced at the executor level.

## Notable Directories

- `kb/` — Markdown knowledge base files ingested into ChromaDB for RAG.
- `eval/` — RAG evaluation framework: dataset, metrics, runner, and CLI (see README.md § RAG Evaluation).
- `specs/` — Design documents and refactoring specs (useful context for understanding past decisions).
- `chroma_data/` — ChromaDB persistence (gitignored).

## Prerequisites

- PostgreSQL on port 5433 with `learning_assistant` database
- Ollama running locally with `llama3.2` and `nomic-embed-text` models
- Node.js/npx for MCP server (`pg-mcp-server`)
- Tavily API key for web search (LATEST intent)
