"""
Mock backend tools for the copilot.

Phase 1: simple, reliable mocks so you can prove the agent + streaming
loop works before introducing complexity.

Phase 5 will make get_order_status / process_refund randomly throw or
hang, to exercise RetryPolicy / TimeoutPolicy / error_handler. Don't
add that yet -- get the happy path solid first.
"""

from langchain_core.tools import tool

_MOCK_ORDERS = {
    "ORD-1001": {"status": "shipped", "item": "Wireless Mouse", "eta": "2026-09-18"},
    "ORD-1002": {"status": "processing", "item": "Mechanical Keyboard", "eta": "2026-09-22"},
    "ORD-1003": {"status": "delivered", "item": "USB-C Hub", "eta": "2026-09-10"},
}

_MOCK_FAQ = {
    "return policy": "Items can be returned within 30 days of delivery for a full refund.",
    "shipping time": "Standard shipping takes 5-7 business days; express takes 2-3.",
    "change address": "You can change your shipping address within 1 hour of placing an order by contacting support.",
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer's order by order ID (e.g. ORD-1001)."""
    order = _MOCK_ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with ID {order_id}."
    return (
        f"Order {order_id}: {order['item']} is currently '{order['status']}', "
        f"expected by {order['eta']}."
    )


@tool
def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for an answer to a general question."""
    query_lower = query.lower()
    for topic, answer in _MOCK_FAQ.items():
        if topic in query_lower:
            return answer
    return (
        "I couldn't find a specific FAQ entry for that. "
        "Could you rephrase, or ask about return policy, shipping time, "
        "or changing an address?"
    )


ALL_TOOLS = [get_order_status, search_faq]
