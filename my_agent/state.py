"""
State schemas for the Autonomous Customer Ops Copilot.

Phase 1 scope: a single MessagesState is enough -- the graph is one
node calling a `model with tools`, checkpointed with InMemorySaver.

Left as separate, clearly-labeled sections so Phase 2 (Store + Context)
and Phase 3 (per-subgraph state) can be added without restructuring
what's already here.
"""

from dataclasses import dataclass
from langgraph.graph import MessagesState

# ---------------------------------------------------------------------------
# Phase 1: parent graph state
# ---------------------------------------------------------------------------

class OpsState(MessagesState):
    """
    Extends the built-in MessagesState (which already gives us a
    `messages` key with the add_messages reducer) with copilot-specific
    fields.
    """
    # Populated once the supervisor exists (Phase 3). Harmless to have
    # now -- nodes that don't set it just leave it as None.
    current_intent: str | None


# ---------------------------------------------------------------------------
# Phase 2 (not wired in yet): runtime context injected per-invocation,
# used to scope long-term memory in the Store by user_id.
# ---------------------------------------------------------------------------

@dataclass
class OpsContext:
    user_id: str
    thread_id: str
