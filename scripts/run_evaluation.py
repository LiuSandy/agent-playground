from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv
from langsmith import evaluate

from app.evaluation.evaluators import (
    answer_correctness,
    route_correctness,
    tool_correctness,
)
from app.evaluation.target import run_hybrid_agent
from scripts.create_dataset import DATASET_NAME


THRESHOLDS = {
    "answer_correctness": 0.85,
    "route_correctness": 0.90,
    "tool_correctness": 0.90,
}


def calculate_averages(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Aggregate evaluator scores returned by a LangSmith experiment."""
    scores: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        feedback_items = row["evaluation_results"]["results"]
        for feedback in feedback_items:
            if feedback.score is not None:
                scores[feedback.key].append(float(feedback.score))

    return {
        key: sum(values) / len(values)
        for key, values in scores.items()
        if values
    }


def enforce_thresholds(
    averages: dict[str, float],
    thresholds: dict[str, float] = THRESHOLDS,
) -> None:
    """Fail the process when an expected metric is missing or below its gate."""
    failed = [
        key
        for key, threshold in thresholds.items()
        if averages.get(key, 0.0) < threshold
    ]
    if failed:
        raise SystemExit(f"Evaluation gate failed: {failed}; scores={averages}")


def run_experiment():
    load_dotenv()
    return evaluate(
        run_hybrid_agent,
        data=DATASET_NAME,
        evaluators=[
            answer_correctness,
            route_correctness,
            tool_correctness,
        ],
        experiment_prefix="hybrid-agent-baseline",
        max_concurrency=2,
        metadata={
            "agent_mode": "hybrid",
            "prompt_version": "v3",
            "dataset_version": "v1",
        },
    )


def main() -> None:
    rows = list(run_experiment())
    averages = calculate_averages(rows)
    enforce_thresholds(averages)
    print(f"Evaluation gate passed: {averages}")


if __name__ == "__main__":
    main()
