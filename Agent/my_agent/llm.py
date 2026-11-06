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
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False


# ---------------------------------------------------------------------------
# Provider registry: provider name -> factory(model_name) -> chat model
# ---------------------------------------------------------------------------

def _openai(model: str):
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )


def _google(model: str):
    if not _HAS_GOOGLE:
        raise ImportError(
            "langchain-google-genai not installed. "
            "Run: pip install langchain-google-genai"
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0,
    )


def _nebius(model: str):
    # Stub for later: Nebius Token Factory is OpenAI-compatible, so this
    # reuses ChatOpenAI with a different base_url and key.
    return ChatOpenAI(
        model=model,
        base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
        api_key=os.environ["NEBIUS_API_KEY"],
        temperature=0,
    )

"""A dict which stores callable func corresponding to provider"""
PROVIDER_REGISTRY = {
    "openai": _openai,
    "google": _google,
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
        ("google", "gemini-3.7-flash"),
        ("openai", "gpt-4o-mini"),
    ],
    "reasoning": [
        ("google", "gemini-3.1-pro-preview"),
        ("openai", "gpt-4o"),
    ],
    # one heavy llm is also needed
    "default": [
        ("google", "gemini-3.7-flash"),
        ("openai", "gpt-4o-mini"),
    ],
}

# Once Nebius billing is sorted, this is the only edit needed to bring
# Nemotron into the mix, e.g.:
#
# ROLE_CONFIG["router"] = [
#     ("nebius", "nvidia/nemotron-3-nano-30b-a3b"),
#     ("openai", "gpt-4o-mini"),
# ]
# ROLE_CONFIG["reasoning"] = [
#     ("nebius", "nvidia/nemotron-3-ultra-...."),
#     ("openai", "gpt-4o"),
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
    # chain = 
    # [
    # 0 -    ("google", "gemini-2.0-flash"),
    # 1 -    ("openai", "gpt-4o-mini"),
    # ]
    primary_provider, primary_model = chain[0]
    primary_provider_func = PROVIDER_REGISTRY[primary_provider]
    llm = primary_provider_func(primary_model)

    fallbacks = [
        PROVIDER_REGISTRY[provider](model)
        for provider, model in chain[1:]
    ]
    if fallbacks:
        llm = llm.with_fallbacks(fallbacks)

    return llm
