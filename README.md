# Autonomous Customer Ops Copilot

Multi-agent LangGraph support system: supervisor routes to FAQ / orders /
billing subgraphs, refunds pause for human approval via `interrupt()`,
conversations persist across sessions, and an ops dashboard can replay
and fork past runs.

Built for the **Nebius x NVIDIA Global AI Hackathon** (Best Apps and
Agents track) and as a full-scope LangGraph portfolio project.

## Current status: Phase 1 (single-agent MVP)

- One agent node with tool calling (`get_order_status`, `search_faq`)
- `InMemorySaver` checkpointing -- multi-turn memory works within a thread
- Streaming via `stream_mode="messages"`

Not yet built: Store/long-term memory (Phase 2), supervisor + subgraphs
(Phase 3), interrupts (Phase 4), fault tolerance (Phase 5), Postgres
(Phase 6), frontend (Phase 7+).

## Model layer: provider-agnostic, Nebius pending

Nebius Token Factory billing (card verification) is currently blocked,
so this project runs on a **fallback chain across OpenAI and Google
Gemini** in the meantime -- see `my_agent/llm.py`.

Every node calls `get_llm(role)` rather than instantiating a provider
client directly, so bringing Nemotron/Nebius online later is a
one-line config change in `ROLE_CONFIG`, not a rewrite. `_nebius()` is
already stubbed in `llm.py`.

Per the hackathon's own guidance for this track: reach for
**Nemotron 3 Ultra** for heavy reasoning (the `"reasoning"` role,
currently used by the billing subgraph), and **Nano or Super** for
fast/everyday calls (the `"router"` and `"default"` roles).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and GOOGLE_API_KEY
```

## Run it

Terminal smoke test (proves multi-turn memory + streaming work):

```bash
python -m my_agent.agent
```

Or via LangGraph Studio, for visual step-through debugging:

```bash
langgraph dev
```

## Repo shape

```
support-ops/
├── my_agent/
│   ├── llm.py         # multi-provider factory w/ fallback (see above)
│   ├── state.py        # OpsState (Phase 1), OpsContext stub (Phase 2)
│   ├── tools.py         # mock order lookup + FAQ search
│   ├── agent.py          # Phase 1 graph assembly + compile()
│   └── subgraphs/         # empty -- Phase 3
├── tests/                   # empty -- Phase 9
├── frontend/                 # empty -- Phase 7
├── langgraph.json
├── requirements.txt
├── .env.example
└── README.md
```

## Next steps

1. Run the terminal smoke test, confirm multi-turn memory + streaming work
2. Phase 2: add `InMemoryStore` + `OpsContext` for cross-thread memory
3. Phase 3: split into supervisor + FAQ/orders/billing subgraphs
4. When Nebius billing is resolved: fill in `NEBIUS_API_KEY` in `.env`
   and add Nemotron entries to `ROLE_CONFIG` in `my_agent/llm.py`
