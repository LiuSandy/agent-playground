from dataclasses import dataclass

import pytest

import scripts.run_evaluation as evaluation_module
from scripts.run_evaluation import calculate_averages, enforce_thresholds


@dataclass
class Feedback:
    key: str
    score: float | None


def test_calculate_averages_groups_feedback_keys() -> None:
    rows = [
        {
            "evaluation_results": {
                "results": [
                    Feedback("answer_correctness", 1),
                    Feedback("route_correctness", 1),
                ]
            }
        },
        {
            "evaluation_results": {
                "results": [
                    Feedback("answer_correctness", 0),
                    Feedback("route_correctness", 1),
                ]
            }
        },
    ]

    assert calculate_averages(rows) == {
        "answer_correctness": 0.5,
        "route_correctness": 1.0,
    }


def test_enforce_thresholds_passes_when_all_metrics_meet_gate() -> None:
    enforce_thresholds(
        {"answer_correctness": 1.0},
        {"answer_correctness": 0.85},
    )


def test_enforce_thresholds_fails_for_missing_metric() -> None:
    with pytest.raises(SystemExit, match="route_correctness"):
        enforce_thresholds({}, {"route_correctness": 0.90})


def test_run_experiment_loads_dotenv_before_evaluate(monkeypatch) -> None:
    events = []

    monkeypatch.setattr(
        evaluation_module,
        "load_dotenv",
        lambda: events.append("dotenv-loaded"),
    )

    def fake_evaluate(*args, **kwargs):
        assert events == ["dotenv-loaded"]
        return "experiment"

    monkeypatch.setattr(evaluation_module, "evaluate", fake_evaluate)

    assert evaluation_module.run_experiment() == "experiment"
