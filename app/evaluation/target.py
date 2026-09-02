from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage

from app.core.agent import create_agent


@lru_cache(maxsize=1)
def get_evaluation_agent():
    """Build the Hybrid Agent once and reuse it for the whole experiment."""
    return create_agent()


def extract_tool_names(messages: list[Any]) -> list[str]:
    """Collect tool names from every AI message in an Agent trajectory."""
    return [
        tool_call["name"]
        for message in messages
        for tool_call in getattr(message, "tool_calls", [])
    ]


def run_hybrid_agent(inputs: dict) -> dict:
    """Adapt a LangSmith dataset example to the Hybrid Agent state shape."""
    result = get_evaluation_agent().invoke(
        {"messages": [HumanMessage(content=inputs["question"])]}
    )

    return {
        "answer": result["messages"][-1].content,
        "mode": result.get("mode"),
        "tools": extract_tool_names(result["messages"]),
    }
