"""Tests for observable trajectory metrics."""

from harness.metrics.trajectory_score import TrajectoryInput, TrajectoryMetric
from harness.schemas import MetricResult, ToolCall, ToolResult, TrajectoryStep, TrajectoryStepType


def _call(step: int, call_id: str, *, quarter: str = "Q3") -> TrajectoryStep:
    return TrajectoryStep(
        step=step,
        type=TrajectoryStepType.TOOL_CALL,
        tool_call=ToolCall(
            call_id=call_id,
            name="lookup",
            arguments={"quarter": quarter},
        ),
    )


def _result(step: int, call_id: str, *, success: bool = True) -> TrajectoryStep:
    return TrajectoryStep(
        step=step,
        type=TrajectoryStepType.TOOL_RESULT,
        tool_result=ToolResult(
            call_id=call_id,
            name="lookup",
            success=success,
            result={"sales": 10} if success else None,
            error=None if success else "database unavailable",
        ),
    )


def _by_name(context: TrajectoryInput) -> dict[str, MetricResult]:
    return {result.name: result for result in TrajectoryMetric().evaluate(context)}


def test_efficient_successful_trajectory_passes() -> None:
    results = _by_name(
        TrajectoryInput(
            steps=[_call(1, "1"), _result(2, "1")],
            optimal_steps=2,
            max_allowed_steps=3,
        )
    )

    assert results["trajectory_efficiency"].score == 1.0
    assert results["loop_detection"].score == 1.0
    assert results["tool_failure_rate"].score == 1.0


def test_duplicate_call_is_reported_as_loop() -> None:
    results = _by_name(
        TrajectoryInput(
            steps=[_call(1, "1"), _result(2, "1"), _call(3, "2")],
            optimal_steps=2,
            max_allowed_steps=2,
        )
    )

    loop = results["loop_detection"]
    assert loop.score == 0.0
    assert loop.details["repeated_steps"] == [1, 3]
    assert results["trajectory_efficiency"].score == 2 / 3


def test_tool_failure_rate_is_scored() -> None:
    results = _by_name(
        TrajectoryInput(
            steps=[_call(1, "1"), _result(2, "1", success=False)],
            optimal_steps=2,
        )
    )

    failure = results["tool_failure_rate"]
    assert failure.score == 0.0
    assert failure.details["rate"] == 1.0
