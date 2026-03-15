# Quiz Agent — RAG-Grounded Quiz Generation

**Status:** Done
**Priority:** High
**Created:** 2026-03-15

## Context

When a user asks for a quiz (e.g., "quiz me on LangChain"), the quiz agent generates questions purely from LLM general knowledge. The RAG retrieval pipeline already exists and is used by the tutor agent, but the quiz flow bypasses it — the router does not set `needs_rag=true` for QUIZ intent, and the quiz agent does not consume `rag_context`.

## Goal

Make quiz generation grounded in the Knowledge Base (ChromaDB) so questions are relevant to content the user has actually ingested, while preserving fallback to LLM-only generation for topics outside the KB.

## Requirements

- R1: Router must set `needs_rag=true` for QUIZ intent when the topic is covered by the KB.
- R2: Quiz generation prompts must accept and prioritize KB context as the primary source for questions.
- R3: Quiz agent must read `rag_context` from graph state and pass it into prompt formatting.
- R4: When `rag_context` is empty or irrelevant, quiz generation must fall back to LLM general knowledge (graceful degradation).
- R5: Quiz scoring/answering flow must remain unchanged (`needs_rag=false` on fast-path).
- R6: Existing graph topology must support the `retrieve_context → quiz` flow without modification.

## Tasks

### Task 1: Update router prompt for QUIZ + RAG routing

**File:** `app/prompts/router.py`

**Changes:**
1. Add rule to `ROUTER_SYSTEM_PROMPT`: "For QUIZ intent, set `needs_rag = true` when the quiz topic is covered by the KB."
2. Add clarifying rule: "For QUIZ, ONLY set `needs_rag = true` if the topic explicitly matches KB scope or a KB filename is mentioned."
3. Add two examples:
   - `"Quiz me on LangChain"` → `needs_rag: true` (topic in KB)
   - `"Quiz me on React"` → `needs_rag: false` (topic not in KB)

**Acceptance criteria:**
- Router prompt contains explicit QUIZ + RAG guidance.
- Examples cover both in-KB and out-of-KB quiz topics.
- No changes needed in `router_agent.py` — it already passes through the LLM's `needs_rag` decision.

### Task 2: Update quiz prompts to accept RAG context

**File:** `app/prompts/quiz.py`

**Changes:**
1. Update `QUIZ_GENERATE_SYSTEM_PROMPT` to instruct the LLM:
   - Use provided KB context as the primary source for questions when available and relevant.
   - If KB context does not match the requested topic, ignore it and use own knowledge.
2. Update `QUIZ_GENERATE_USER_PROMPT` to include `{rag_context}` placeholder:
   ```
   Topic: {user_input}

   Knowledge base context:
   {rag_context}

   Previously wrong questions (must include in quiz):
   {wrong_questions}
   ```
3. Add `QUIZ_RAG_RELEVANCE_SYSTEM_PROMPT` and `QUIZ_RAG_RELEVANCE_USER_PROMPT` for a two-pass relevance filter that decides whether retrieved context actually matches the quiz topic.

**Acceptance criteria:**
- `QUIZ_GENERATE_USER_PROMPT` includes `{rag_context}` placeholder.
- System prompt instructs LLM to prioritize KB context when relevant.
- Relevance-check prompts exist for filtering irrelevant RAG results.

### Task 3: Update quiz agent to consume RAG context

**File:** `app/agents/quiz_agent.py`

**Changes:**
1. Read `rag_context` from `state.get("rag_context", "")` (same pattern as `tutor_node`).
2. Pass `rag_context` into `QUIZ_GENERATE_USER_PROMPT.format(...)`.
3. Add `_check_rag_relevance()` function implementing two-pass relevance filtering:
   - Fast path: substring match of topic name in RAG context.
   - Slow path: LLM-as-judge relevance check using `QUIZ_RAG_RELEVANCE_*` prompts.
   - Returns original `rag_context` if relevant, empty string if not.
4. Pass `rag_context` to `_retry_regenerate_mcq_only()` so retries are also KB-grounded.

**Acceptance criteria:**
- `quiz_node()` reads `rag_context` from state at line ~44.
- `_generate_quiz()` receives and formats `rag_context` into the prompt.
- `_check_rag_relevance()` filters irrelevant context before generation.
- `_retry_regenerate_mcq_only()` includes `rag_context` in regeneration prompt.
- When `rag_context` is empty, the flow works identically to before (no branching needed).

### Task 4: Verify graph topology (no code changes)

**Files:** `app/graph/builder.py`, `app/graph/routing.py`

**Verification:**
1. `route_after_router()` routes to `retrieve_context` when `needs_rag=true` — confirmed at `routing.py:28`.
2. `retrieve_context` has a conditional edge to `route_to_specialist`.
3. `route_to_specialist()` maps QUIZ intent to `"quiz"` node — confirmed at `routing.py:53`.
4. `route_after_db()` supports pre-quiz DB fetch → RAG → quiz flow — confirmed at `routing.py:62-71`.

**Acceptance criteria:**
- No code changes required.
- Graph supports: `router → retrieve_context → quiz → format_response → END`.

## Flow After Changes

### KB-grounded quiz
```
User: "Quiz me on LangChain"
  → router (intent=QUIZ, needs_rag=true)
  → retrieve_context (queries ChromaDB, populates rag_context)
  → quiz (_check_rag_relevance → _generate_quiz with KB context)
  → format_response → END
```

### Non-KB quiz (fallback)
```
User: "Quiz me on React"
  → router (intent=QUIZ, needs_rag=false)
  → quiz (_generate_quiz with empty rag_context, LLM-only)
  → format_response → END
```

### Quiz scoring (unchanged)
```
User: "1:A, 2:B, 3:C"
  → router (fast-path, needs_rag=false)
  → quiz (_handle_scoring)
  → format_response → END
```

## Dependencies

- ChromaDB must be running and populated (`uv run python scripts/ingest_kb.py`).
- No new Python packages required.
- No database schema changes.

## Verification

| # | Test | Expected |
|---|------|----------|
| 1 | Send "quiz me on LangChain" | Server logs show `RETRIEVE HIT`; quiz questions reference KB content |
| 2 | Send "quiz me on React" | Quiz generates normally from LLM knowledge (no RAG) |
| 3 | Answer the quiz (e.g., "1:A, 2:B") | Scoring works correctly, no regression |
| 4 | `uv run pytest tests/ -v` | All existing tests pass |

## Non-Goals

- No changes to quiz scoring or evaluation logic.
- No changes to `router_agent.py` runtime code.
- No new dependencies or DB schema changes.
