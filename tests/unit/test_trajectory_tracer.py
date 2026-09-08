"""Tests for observable trajectory recording."""

from harness.schemas import TrajectoryStepType
from harness.trajectory_tracer import TrajectoryTracer


def test_tracer_records_ordered_defensive_snapshot() -> None:
    tracer = TrajectoryTracer()
    call = tracer.record_tool_call(call_id="1", name="lookup", arguments={"id": 7})
    tracer.record_tool_result(call, result={"value": "found"})
    tracer.record_final_answer("Done")

    snapshot = tracer.steps
    snapshot.clear()

    assert [step.step for step in tracer.steps] == [1, 2, 3]
    assert [step.type for step in tracer.steps] == [
        TrajectoryStepType.TOOL_CALL,
        TrajectoryStepType.TOOL_RESULT,
        TrajectoryStepType.FINAL_ANSWER,
    ]


def test_tracer_records_failed_tool_result() -> None:
    tracer = TrajectoryTracer()
    call = tracer.record_tool_call(call_id="1", name="lookup")
    result = tracer.record_tool_result(call, error="not found")

    assert not result.success
    assert result.error == "not found"


def test_tracer_records_state_and_retry_metadata() -> None:
    tracer = TrajectoryTracer()
    tracer.record_state_transition("planning", metadata={"attempt": 1})
    tracer.record_retry("transient failure", metadata={"attempt": 2})

    assert tracer.steps[0].content == "planning"
    assert tracer.steps[1].metadata == {"attempt": 2}
