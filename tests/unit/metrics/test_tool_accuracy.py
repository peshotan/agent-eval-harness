"""Tests for deterministic tool-use metrics."""

from harness.metrics.tool_accuracy import ToolAccuracyInput, ToolAccuracyMetric
from harness.schemas import ToolCall, ToolExpectation


def _scores(context: ToolAccuracyInput) -> dict[str, float | None]:
    return {result.name: result.score for result in ToolAccuracyMetric().evaluate(context)}


def test_correct_calls_receive_full_scores() -> None:
    scores = _scores(
        ToolAccuracyInput(
            expected_tools=[
                ToolExpectation(name="lookup", required_arguments={"quarter": "Q3"}),
                ToolExpectation(name="average"),
            ],
            actual_calls=[
                ToolCall(
                    call_id="1",
                    name="lookup",
                    arguments={"quarter": "Q3", "cache": True},
                ),
                ToolCall(call_id="2", name="average"),
            ],
            known_tools={"lookup", "average"},
        )
    )

    assert scores == {
        "tool_precision": 1.0,
        "tool_recall": 1.0,
        "argument_accuracy": 1.0,
        "unknown_tool_rate": 1.0,
    }


def test_extra_unknown_call_reduces_precision_and_unknown_score() -> None:
    scores = _scores(
        ToolAccuracyInput(
            expected_tools=[ToolExpectation(name="lookup", required_arguments={"id": 7})],
            actual_calls=[
                ToolCall(call_id="1", name="lookup", arguments={"id": 7}),
                ToolCall(call_id="2", name="invented"),
            ],
            known_tools={"lookup"},
        )
    )

    assert scores["tool_precision"] == 0.5
    assert scores["tool_recall"] == 1.0
    assert scores["unknown_tool_rate"] == 0.5


def test_wrong_arguments_reduce_recall_and_argument_accuracy() -> None:
    scores = _scores(
        ToolAccuracyInput(
            expected_tools=[
                ToolExpectation(name="lookup", required_arguments={"quarter": "Q3", "year": 2024})
            ],
            actual_calls=[
                ToolCall(call_id="1", name="lookup", arguments={"quarter": "Q2", "year": 2024})
            ],
            known_tools={"lookup"},
        )
    )

    assert scores["tool_precision"] == 0.0
    assert scores["tool_recall"] == 0.0
    assert scores["argument_accuracy"] == 0.5


def test_no_expected_or_actual_tools_is_perfect() -> None:
    assert _scores(ToolAccuracyInput()) == {
        "tool_precision": 1.0,
        "tool_recall": 1.0,
        "argument_accuracy": 1.0,
        "unknown_tool_rate": 1.0,
    }
