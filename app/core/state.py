from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class DeliberativeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    current_step: int
    step_results: dict[int, str]
    original_question: str
    tool_rounds: int


class HybridState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: NotRequired[str]
    step_count: NotRequired[int]
    plan: NotRequired[list[str]]
    current_step: NotRequired[int]
    step_results: NotRequired[dict[int, str]]
    original_question: NotRequired[str]
    tool_rounds: NotRequired[int]
