from langchain_core.messages import AIMessage, HumanMessage

from app.evaluation.target import extract_tool_names


def test_extract_tool_names_reads_the_whole_trajectory() -> None:
    messages = [
        HumanMessage(content="question"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_tool", "args": {"query": "LangGraph"}, "id": "1"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator_tool", "args": {"expression": "1+1"}, "id": "2"}
            ],
        ),
    ]

    assert extract_tool_names(messages) == ["search_tool", "calculator_tool"]
