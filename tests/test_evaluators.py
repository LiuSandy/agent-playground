from app.evaluation.evaluators import (
    answer_correctness,
    route_correctness,
    tool_correctness,
)


def test_answer_correctness_is_case_insensitive() -> None:
    result = answer_correctness(
        {"answer": "LangGraph can orchestrate an Agent."},
        {"answer_keywords": ["langgraph", "agent"]},
    )

    assert result == {"key": "answer_correctness", "score": 1}


def test_answer_correctness_fails_when_keyword_is_missing() -> None:
    result = answer_correctness(
        {"answer": "The result is 175."},
        {"answer_keywords": ["176"]},
    )

    assert result["score"] == 0


def test_answer_correctness_rejects_missing_reference_keywords() -> None:
    result = answer_correctness({"answer": "Any answer"}, {})

    assert result["score"] == 0


def test_route_correctness_compares_hybrid_mode() -> None:
    result = route_correctness(
        {"mode": "deliberative"},
        {"expected_mode": "deliberative"},
    )

    assert result["score"] == 1


def test_tool_correctness_allows_repeated_expected_tool() -> None:
    result = tool_correctness(
        {"tools": ["search_tool", "search_tool"]},
        {"expected_tools": ["search_tool"]},
    )

    assert result["score"] == 1


def test_tool_correctness_rejects_unexpected_tool() -> None:
    result = tool_correctness(
        {"tools": ["search_tool", "calculator_tool"]},
        {"expected_tools": ["search_tool"]},
    )

    assert result["score"] == 0


def test_tool_correctness_accepts_no_tool_when_expected() -> None:
    result = tool_correctness(
        {"tools": []},
        {"expected_tools": []},
    )

    assert result["score"] == 1
