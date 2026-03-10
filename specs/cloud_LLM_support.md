# Cloud LLM Support (Local-First)

## Context

The app is currently hardwired to local Ollama via `app/llm/ollama_client.py` and `app/utils/llm_helpers.py`. We want optional cloud LLM support while keeping local-only as the default to avoid token spend. The design must be minimal, reuse existing call sites, and fail fast if a cloud provider is selected without required config.

## Approach

Introduce a small provider router that returns a LangChain `BaseChatModel` (and later embeddings if needed). Keep all agent call sites unchanged by routing through `invoke_llm()` and `get_chat_model()`. Default provider remains `ollama`.

Key goals:
- Local-first: `LLM_PROVIDER=ollama` by default.
- Explicit opt-in for cloud providers.
- Clear validation errors when required env vars are missing.
- Small, isolated change footprint.

## Modified or Created Files

Created:
- `app/llm/provider.py` (provider router)
- `app/llm/providers/openai.py` (cloud provider adapter)
- `app/llm/providers/anthropic.py` (cloud provider adapter)

Modified:
- `app/config.py` (new provider settings)
- `app/utils/llm_helpers.py` (import `get_chat_model()` from router)
- `app/llm/ollama_client.py` (optional: keep as-is or move into providers/ollama.py)
- `README.md` (document env vars and cloud opt-in)
- `CLAUDE.md` (optional: dev notes)

## Step 1: Add provider settings (`app/config.py`)

Add fields with safe defaults:
- `llm_provider: str = "ollama"`
- `llm_model: str = ""` (cloud default model, empty means "not set")
- `llm_api_key: str = ""`
- `llm_base_url: str = ""` (optional, provider-specific)

Keep existing Ollama settings unchanged.

## Step 2: Create provider router (`app/llm/provider.py`)

This module chooses which provider adapter to use.

Function signatures:
```py
from langchain_core.language_models.chat_models import BaseChatModel

def get_chat_model() -> BaseChatModel:
    ...

def get_embeddings():
    ...
```

Behavior:
- If `settings.llm_provider == "ollama"`, return `providers.ollama.get_chat_model()`
- If `settings.llm_provider == "openai"`, validate `llm_api_key` and `llm_model`, then return `providers.openai.get_chat_model()`
- If `settings.llm_provider == "anthropic"`, validate `llm_api_key` and `llm_model`, then return `providers.anthropic.get_chat_model()`
- Else: raise `ValueError("Unknown LLM_PROVIDER: ...")`

Add a warning log when `llm_provider != "ollama"` to prevent accidental token spend.

## Step 3: Provider adapters

### Ollama adapter

Either:
- keep `app/llm/ollama_client.py` and import it from `provider.py`, or
- move logic into `app/llm/providers/ollama.py` and re-export if needed.

Function signatures (unchanged):
```py
from langchain_ollama import ChatOllama, OllamaEmbeddings

def get_chat_model() -> ChatOllama:
    ...

def get_embeddings() -> OllamaEmbeddings:
    ...
```

### OpenAI adapter (`app/llm/providers/openai.py`)

Function signature:
```py
from langchain_openai import ChatOpenAI

def get_chat_model(model: str, api_key: str, base_url: str | None = None) -> ChatOpenAI:
    ...
```

Notes:
- If `base_url` is set, pass it through (useful for proxies).
- This module should not read env vars directly; use validated values from `provider.py`.

### Anthropic adapter (`app/llm/providers/anthropic.py`)

Function signature:
```py
from langchain_anthropic import ChatAnthropic

def get_chat_model(model: str, api_key: str, base_url: str | None = None) -> ChatAnthropic:
    ...
```

## Step 4: Update `invoke_llm()` wiring (`app/utils/llm_helpers.py`)

Replace:
- `from app.llm.ollama_client import get_chat_model`

With:
- `from app.llm.provider import get_chat_model`

Function signature stays the same:
```py
def invoke_llm(prompt: str, llm=None) -> str:
    ...
```

This keeps all agents unchanged.

## Step 5: Document environment variables

Update `README.md` to include:
- `LLM_PROVIDER=ollama|openai|anthropic`
- `LLM_MODEL` and `LLM_API_KEY` required for cloud
- `LLM_BASE_URL` optional

Document the safety behavior:
- local-only default
- cloud use requires explicit provider selection + env vars
- app raises clear error if required vars missing

## Step 6: Verification

Manual checks (no cloud usage):
1. Default local run unchanged (Ollama only).
2. Set `LLM_PROVIDER=openai` without `LLM_API_KEY` -> app raises clear error at startup or first LLM call.
3. Set `LLM_PROVIDER=anthropic` with missing `LLM_MODEL` -> error.

Optional:
- Add a test that monkeypatches `settings.llm_provider` to verify routing and error handling.

## Non-Goals

- No changes to agent logic or prompts.
- No multi-provider fallback logic.
- No cloud embeddings unless explicitly added later.
