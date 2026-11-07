"""
Phase 1 graph: a single agent node with tool calling, checkpointed with
InMemorySaver so multi-turn conversations work with a thread_id.

Run this with `langgraph dev` (see langgraph.json) to get a working
graph you can inspect in LangSmith Studio, or invoke it directly with
`python -m my_agent.agent` for a quick terminal smoke test.

Phase 2 will add a Store for cross-thread memory.
Phase 3 will replace this single node with a supervisor + subgraphs.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from my_agent.llm import get_llm
from my_agent.state import OpsState
from my_agent.tools import ALL_TOOLS

# use \ when you write in next line but wants it to stay in same line after rendering with """ """
SYSTEM_PROMPT = """You are a helpful customer operations copilot for an \
e-commerce company. You can look up order status and answer FAQ questions. \
Be concise and friendly. If you don't have a tool for something (like \
processing a refund), say so honestly rather than making something up -- \
refund handling is coming in a later version of this agent."""

def agent_node(state: OpsState):
    llm = get_llm("default").bind_tools(ALL_TOOLS)
    messages = state["messages"]

    # Prepend the system prompt only if it's not already the first message
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages) # gives the final AI message
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(OpsState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph


# LangGraph dev provides persistence through its API runtime, so the exported
# graph must not provide a custom checkpointer.
graph = build_graph().compile()


# to run this use command: python -m my_agent.agent
if __name__ == "__main__":
    # Quick terminal smoke test: proves multi-turn memory + streaming
    # work before you touch the frontend.
    import uuid

    local_graph = build_graph().compile(checkpointer=InMemorySaver())
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Thread ID: {thread_id}")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break

        print("Copilot: ", end="", flush=True)
        for chunk, metadata in local_graph.stream(
            {"messages": [("user", user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if chunk.content:
                if isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block.get("text", ""), end="", flush=True)
        print("\n")


        # stream = graph.stream_events({
        # "messages": [{"role": "user", "content": user_input}],
        # }, version="v3")

        # for message in stream.messages:
        #     for token in message.text:
        #         print(token, end="", flush=True)

        # final_state = stream.output