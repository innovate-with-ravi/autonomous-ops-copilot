"""
Model layer for the Autonomous Customer Ops Copilot.

Design goal: every node in the graph calls `get_llm(role)` and never
instantiates a provider client directly. That means swapping providers,
adding fallbacks, or moving to Nebius Token Factory later is a one-line
change here -- nothing in nodes.py, agent.py, or the subgraphs needs to
change.

Current setup: `Google Gemini as primary`, OpenAi as automatic fallback
`add groq as OpenAi credits are exhausted`
(via LangChain's `.with_fallbacks()`), while Nebius billing is blocked.
Rate limits / token-exceeded errors on one provider will silently retry
on the next provider in the chain.

---
To bring Nebius online later: add a "nebius" entry to PROVIDER_REGISTRY
(already stubbed below) and prepend it to whichever ROLE_CONFIG chains
you want backed by Nemotron.
"""

import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

try:
    from langchain_groq import ChatGroq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False


# ---------------------------------------------------------------------------
# Provider registry: provider name -> factory(model_name) -> chat model
# ---------------------------------------------------------------------------

def _openai(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0,
    )


def _google(model: str):
    if not _HAS_GOOGLE:
        raise ImportError(
            "langchain-google-genai not installed. "
            "Run: pip install langchain-google-genai"
        )
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
    )


def _groq(model: str):
    if not _HAS_GROQ:
        raise ImportError(
            "langchain-groq not installed. "
            "Run: pip install langchain-groq"
        )
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
    )


def _nebius(model: str):
    # Stub for later: Nebius Token Factory is OpenAI-compatible, so this
    # reuses ChatOpenAI with a different base_url and key.
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("NEBIUS_API_KEY environment variable not set.")
    return ChatOpenAI(
        model=model,
        base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
        api_key=api_key,
        temperature=0,
    )

"""A dict which stores callable func corresponding to provider"""
PROVIDER_REGISTRY = {
    "openai": _openai,
    "google": _google,
    "groq": _groq,
    "nebius": _nebius,
}


# ---------------------------------------------------------------------------
# Role -> ordered [(provider, model), ...] fallback chain.
#
# First entry is primary; the rest are fallbacks tried in order if the
# primary raises (rate limit, timeout, quota exceeded, etc).
#
# Roles map to where they're used in the graph:
#   "router"    -> supervisor's intent classification (cheap, fast, high volume)
#   "reasoning" -> billing subgraph, anything needing judgment calls
#   "default"   -> everything else (FAQ subgraph, general chit-chat)
# ---------------------------------------------------------------------------

"""A dict which stores a list of (provider,model) corresponding to role"""
ROLE_CONFIG = {
    "router": [
        ("google", "gemini-3-flash-preview"),
        ("groq", "llama-3.1-8b-instant"),
        ("openai", "gpt-4o-mini"),
    ],
    "reasoning": [
        ("google", "gemini-3-flash-preview"),
        ("groq", "llama-3.3-70b-versatile"),
        ("openai", "gpt-4o"),
    ],
    "default": [
        ("google", "gemini-3-flash-preview"),
        ("groq", "llama-3.3-70b-versatile"),
        ("openai", "gpt-4o-mini"),
    ],
}

# Once Nebius billing is sorted, this is the only edit needed to bring
# Nemotron into the mix, e.g.:
#
# ROLE_CONFIG["router"] = [
#     ("nebius", "nvidia/nemotron-3-nano-30b-a3b"),
#     ("google", "gemini-2.0-flash"),
# ]
# ROLE_CONFIG["reasoning"] = [
#     ("nebius", "nvidia/nemotron-3-ultra-...."),
#     ("google", "gemini-2.0-flash"),
# ]


@lru_cache(maxsize=None)
def get_llm(role: str = "default"):
    """
    Return a chat model for the given role, wired with automatic
    fallback across providers.

    ---
    Cached per role so repeated calls to `get_llm()` within a graph run reuse the same
    client instance (chatObject returned by get_llm()) instead of re-instantiating on every node call.
    """
    if role not in ROLE_CONFIG:
        raise ValueError(
            f"Unknown role '{role}'. Known roles: {list(ROLE_CONFIG)}"
        )

    chain = ROLE_CONFIG[role]
    primary_provider, primary_model = chain[0]
    primary_provider_func = PROVIDER_REGISTRY[primary_provider]
    llm = primary_provider_func(primary_model)

    fallbacks = []
    for provider, model in chain[1:]:
        try:
            fallbacks.append(PROVIDER_REGISTRY[provider](model))
        except Exception:
            # If fallback provider is missing API key, skip it safely
            continue

    if fallbacks:
        llm = llm.with_fallbacks(fallbacks)

    return llm
