def answer_correctness(outputs: dict, reference_outputs: dict) -> dict:
    """Score whether the final answer contains every required keyword."""
    answer = str(outputs.get("answer", "")).lower()
    keywords = reference_outputs.get("answer_keywords", [])
    passed = bool(keywords) and all(
        str(keyword).lower() in answer for keyword in keywords
    )
    return {"key": "answer_correctness", "score": int(passed)}


def route_correctness(outputs: dict, reference_outputs: dict) -> dict:
    """Score whether the Hybrid Router selected the expected mode."""
    expected = reference_outputs.get("expected_mode")
    passed = expected is not None and outputs.get("mode") == expected
    return {"key": "route_correctness", "score": int(passed)}


def tool_correctness(outputs: dict, reference_outputs: dict) -> dict:
    """Score the set of tools used while allowing repeated valid calls."""
    actual = set(outputs.get("tools", []))
    expected_tools = reference_outputs.get("expected_tools")
    passed = expected_tools is not None and actual == set(expected_tools)
    return {"key": "tool_correctness", "score": int(passed)}
