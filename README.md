# multiagent_learning_assistant

## Setup

```bash
uv sync
```

## MCP Checks (Best Practices)

Purpose: Verify MCP server connectivity and DB read/write without relying on agent logic.

Scope: Local-only checks. The CLI writes test rows unless `--cleanup` is used.

Prerequisites:
- Local Postgres running and schema applied from `db/init.sql`.
- MCP server configured via `.env` (see below).
- This project uses [`pg-mcp-server`](https://github.com/ericzakariasson/pg-mcp-server) as the PostgreSQL MCP server.
- `MCP_ALLOW_WRITE_OPS=true` for write tests.

Required env keys:
- `DB_BACKEND=mcp`
- `MCP_SERVER_COMMAND` / `MCP_SERVER_ARGS`
- `MCP_DATABASE_URL`
- `MCP_ALLOW_WRITE_OPS`
- `MCP_SUPPORTS_PARAMS` (for `pg-mcp-server`, set `false`)

Commands:
- CLI check (read/write):
```bash
uv run python -m app.cli.mcp_check --message "hello from mcp test"
```
- CLI check with cleanup:
```bash
uv run python -m app.cli.mcp_check --message "hello from mcp test" --cleanup
```
- MCP health endpoint:
```bash
curl http://localhost:8000/health/mcp
```

Success criteria:
- CLI exit code `0`
- Output includes `message_present: True`
- `/health/mcp` returns `{ "ok": true, ... }`

Troubleshooting:
- Use `--debug` to print raw MCP payloads and tool schema:
```bash
uv run python -m app.cli.mcp_check --message "hello" --debug
```
- If you see parameter errors, confirm:
  - `MCP_QUERY_KEY=sql`
  - `MCP_SUPPORTS_PARAMS=false`
  - `MCP_ALLOW_WRITE_OPS=true`

Data hygiene:
- Use `--cleanup` if you do not want test rows persisted.
- If needed, delete test rows manually from `messages` and `sessions`.

Security notes:
- Keep MCP local-only and never expose it publicly.
- Pin MCP server version in `.env`/`.env.example` (e.g., `pg-mcp-server@x.y.z`).
- Do not commit `.env` with credentials.

Versioning:
- Update this section whenever MCP server version or tool schema changes.

## PostgreSQL Access (MCP + psycopg2 fallback)

Overview:
- Default DB backend is MCP (`DB_BACKEND=mcp`).
- MCP server is started/stopped with FastAPI lifespan.
- If MCP fails, the app logs a warning and falls back to psycopg2 (configurable).

Key modules:
- MCP lifecycle: `app/mcp/manager.py`
- MCP client wrapper: `app/mcp/client.py`
- MCP repository: `app/db/mcp_repository.py`
- psycopg2 repository: `app/db/repository.py`
- backend selection: `app/db/repository_factory.py`
- MCP row extraction: `app/db/row_extract.py`
- DB tools + registry: `app/tools/db_tools.py`, `app/tools/tool_registry.py`

Config (MCP):
- `MCP_SERVER_COMMAND` / `MCP_SERVER_ARGS` (pin version in args)
- `MCP_DATABASE_URL`
- `MCP_ALLOW_WRITE_OPS=true` (required for write tests)
- `MCP_TOOL_NAME=query`
- `MCP_QUERY_KEY=sql`
- `MCP_SUPPORTS_PARAMS=false` (pg-mcp-server uses only `sql`)
- `MCP_FALLBACK_TO_PSYCOPG2=true`

Config (psycopg2 fallback):
- `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`
- `PG_POOL_MIN`, `PG_POOL_MAX`

Runtime behavior:
- MCP is the primary path for reads/writes.
- If MCP is unavailable or returns errors, fallback logs:
  - `MCP DB read failed, falling back to psycopg2`
  - `MCP DB write failed, falling back to psycopg2`

Notes:
- pg-mcp-server does not accept query params; when `MCP_SUPPORTS_PARAMS=false`, SQL is inlined safely for local use.
- MCP server used by this project: [`pg-mcp-server`](https://github.com/ericzakariasson/pg-mcp-server).
- Keep MCP local-only and never expose it publicly.

## Agents, Routing, and Orchestration (LangGraph + LangChain)

This project uses LangGraph for orchestration and LangChain for model calls and MCP client sessions. There is no LCEL pipe (`|`) usage; agents use a shared `invoke_llm()` helper (`app/utils/llm_helpers.py`) and parse outputs with Pydantic.

### App Flow

Entry point:
- `app/graph/builder.py` builds the LangGraph state machine.

Routing logic:
- `app/agents/router_agent.py` classifies intent and sets routing flags.
- `app/graph/routing.py` maps intent/flags to the next node.

Execution path (simplified):
- Router -> optional context tools (`retrieve_context`, `web_search`) -> specialist agent -> `format_response`.
- DB reads/writes for plans and progress are handled by `db_agent` (tool-calling executor).
- Quiz flow can round-trip to `db_agent` to persist results, then returns to `format_response`.

### Agentic vs Deterministic

This system is a hybrid: LLM-driven where judgment is needed, deterministic where safety and consistency matter.

Agentic (LLM-driven):
- Router intent classification (`router_agent.py`).
- Plan drafting (`planner_agent.py`).
- DB tool-calling decisions (`db_agent.py`) when tools are returned by the model.
- Quiz generation, scoring, and feedback (`quiz_agent.py`).

Deterministic guardrails:
- Quiz fast-path routing for numbered A/B/C/D answers (skips LLM routing when a quiz is in progress).
- Plan save gating: only save when intent is `PLAN` with `SAVE_PLAN` and a `plan_draft` exists.
- DB fallback path if tool-calling returns nothing (intent-driven rules for REVIEW/LOG_PROGRESS).
- Error normalization into user-facing messages (`validation_error`, `conflict`, `not_found`, `permission_denied`, `db_error`).
- Strict top-level tool input validation (unexpected fields are rejected; nested extras are stripped).
- Quiz post-save loop prevention and wrong-answer cleanup.
- Research agent summaries are deterministic (formatting `web_context` without LLM summarization).

### Agents and Responsibilities

Router agent:
- File: `app/agents/router_agent.py`
- Uses the router prompt (`app/prompts/router.py`) to classify intents like PLAN, REVIEW, LOG_PROGRESS.
- Sets `needs_db` and `sub_intent` so the graph knows which node to run next.
- Fast-path: if a quiz is in progress and the user replies with a numbered A/B/C/D answer, routing skips the LLM and goes straight to QUIZ evaluation.

Planner agent:
- File: `app/agents/planner_agent.py`
- Generates a plan draft with structured JSON validated by `app/schemas/planner.py`.
- Returns a Markdown summary to the user and a structured `plan_draft` for saving.
- Drafts are cached per session; the router only confirms saving when intent is `PLAN` with `SAVE_PLAN`.

DB agent (tool-calling executor):
- File: `app/agents/db_agent.py`
- Executes DB actions using native Ollama tool calling with an intent-based fallback.
- Tools are defined in `app/tools/db_tools.py` and cover list/create/update operations.
- Formats responses and handles duplicate titles using `created_at` for disambiguation.
- Progress updates (`update_item_status`) are plan-scoped. If no plan is specified, the tool-calling model may guess a plan ID; if the item is not in that plan, the update returns `not_found`.
- Errors are normalized into user-facing messages: `validation_error`, `conflict`, `not_found`, `permission_denied`, `db_error`.

Tutor, Quiz, Research agents:
- Files: `app/agents/tutor_agent.py`, `app/agents/quiz_agent.py`, `app/agents/research_agent.py`
- Tutor uses RAG context for explanations.
- Quiz flow details:
- Starts when the router selects intent `QUIZ` and routes to `quiz` (or when a follow-up answer is detected and short-circuits to QUIZ).
- If the topic is related to retrieved context, the agent uses RAG context to generate questions.
- Detailed agentic quiz process:
- Retrieve prior wrong questions from the DB (`db_agent` → `quiz_pre_fetch`) to seed the session.
- Generate quiz questions (optionally grounded in RAG context when the topic is related).
- Present questions and capture user answers.
- Score answers and compute feedback (`quiz_feedback`).
- Build a `quiz_save` payload containing wrong answers and correct retries.
- If there are wrong answers, persist results to the DB (`db_agent` → `quiz_post_save`), which saves new wrong answers and deletes previously wrong answers now answered correctly.
- Return to `format_response`; cache `quiz_state` and mark `quiz_results_saved` to avoid loops.
- Research flow is routed by the router intent.
- Research agent is currently deterministic: it formats and summarizes `web_context` results without an LLM to reduce hallucinations.
- Future option: re-enable LLM summarization for a more agentic research flow (with tests/guardrails).

### Parsing and Schema Validation

Parsing and shared utilities:
- `app/utils/llm_parse.py` — LLM output parsing and validation with Pydantic schemas; includes sanitization and JSON extraction for robustness.
- `app/utils/llm_helpers.py` — shared `invoke_llm()` helper used by all agents for model invocation.
- `app/utils/constants.py` — shared regex patterns (answer parsing, quiz fast-path detection) and magic values.

Schemas:
- Router: `app/schemas/router.py`
- Planner: `app/schemas/planner.py`

### MCP Integration in Agents

MCP client usage:
- `db_agent` uses repository backends (MCP or psycopg2) to execute tools.
- MCP session parameters come from `.env` and `app/config.py`.

DB execution flow:
- Planner creates a draft.
- On confirmation, DB agent writes the plan via tools and formats the response.

### RAG (ChromaDB Knowledge Base)

RAG overview:
- Retrieval uses ChromaDB (`app/rag/retriever.py`) with Ollama embeddings.
- Knowledge base files live in `kb/` and are ingested via `app/rag/ingest.py`.
- The retriever queries the `knowledge_base` collection in `./chroma_data`.
- Retrieval uses MMR with `k=6` and `fetch_k=12`.
- The `retrieve_context` tool populates `rag_context` for tutor/quiz flows.

Current KB topics:
- `kb/langchain.md`
- `kb/langgraph.md`
- `kb/links.md`
- `kb/python_interview.md`

### RAG Evaluation

The project includes a custom evaluation framework for measuring retrieval quality and response grounding. It uses lightweight custom metrics instead of RAGAS (which has compatibility issues with local Ollama models).

Commands:
```bash
uv run python -m eval.run_eval                      # full evaluation
uv run python -m eval.run_eval --retrieval-only      # skip LLM judge (fast)
uv run python -m eval.run_eval --samples 5           # first N samples only
```

Evaluation dataset:
- 19 curated Q&A pairs in `eval/dataset.json`, sourced from KB files.
- Distribution: 7 langchain, 6 langgraph, 4 python_interview, 2 out-of-scope (negative) questions.

Retrieval metrics (deterministic, no LLM calls):
- **Context precision** — fraction of retrieved chunks from expected source files.
- **Context recall** — fraction of expected sources represented in retrieved chunks.
- **Hit rate** — binary, did at least one relevant chunk appear.
- **MRR** — reciprocal rank of first relevant chunk.

Grounding metrics (LLM-as-judge):
- **Faithfulness** — scores 1-5 whether every claim in the answer is supported by the retrieved context.
- **Correctness** — scores 1-5 whether the answer captures key points from the reference answer.
- **Token F1** — deterministic token-level overlap as a bias-free baseline.

Pipeline per sample: retrieve (ChromaDB) → generate (tutor prompts + Ollama) → judge (LLM-as-judge) → aggregate.

Key modules:
- `eval/dataset.json` — ground truth Q&A pairs.
- `eval/prompts.py` — judge prompt templates.
- `eval/metrics/retrieval.py` — deterministic retrieval metrics.
- `eval/metrics/grounding.py` — LLM-as-judge + token F1.
- `eval/runner.py` — orchestrator.
- `eval/report.py` — console summary table + JSON output to `eval/results/`.
- `eval/run_eval.py` — CLI entry point.

Reuses existing infrastructure: `get_retriever()`, `invoke_llm()`, tutor prompts. No new dependencies.

Tests:
- `tests/test_eval_metrics.py` — unit tests for deterministic metrics (no services needed).
- `tests/test_rag_eval.py` — integration tests with threshold assertions (needs Ollama + ChromaDB).
