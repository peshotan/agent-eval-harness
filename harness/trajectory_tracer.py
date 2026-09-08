"""Recorder for observable, ordered agent execution events."""

from harness.schemas import (
    JsonValue,
    ToolCall,
    ToolResult,
    TrajectoryStep,
    TrajectoryStepType,
)


class TrajectoryTracer:
    """Build an immutable snapshot of observable trajectory steps."""

    def __init__(self) -> None:
        self._steps: list[TrajectoryStep] = []

    @property
    def steps(self) -> list[TrajectoryStep]:
        """Return a defensive copy of the recorded steps."""
        return list(self._steps)

    def record_tool_call(
        self,
        *,
        call_id: str,
        name: str,
        arguments: dict[str, JsonValue] | None = None,
    ) -> ToolCall:
        call = ToolCall(call_id=call_id, name=name, arguments=arguments or {})
        self._append(TrajectoryStepType.TOOL_CALL, tool_call=call)
        return call

    def record_tool_result(
        self,
        call: ToolCall,
        *,
        result: JsonValue = None,
        error: str | None = None,
    ) -> ToolResult:
        tool_result = ToolResult(
            call_id=call.call_id,
            name=call.name,
            success=error is None,
            result=result,
            error=error,
        )
        self._append(TrajectoryStepType.TOOL_RESULT, tool_result=tool_result)
        return tool_result

    def record_state_transition(
        self,
        state: str,
        *,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        self._append(
            TrajectoryStepType.STATE_TRANSITION,
            content=state,
            metadata=metadata or {},
        )

    def record_retry(self, reason: str, *, metadata: dict[str, JsonValue] | None = None) -> None:
        self._append(TrajectoryStepType.RETRY, content=reason, metadata=metadata or {})

    def record_final_answer(self, answer: str) -> None:
        self._append(TrajectoryStepType.FINAL_ANSWER, content=answer)

    def _append(
        self,
        step_type: TrajectoryStepType,
        *,
        tool_call: ToolCall | None = None,
        tool_result: ToolResult | None = None,
        content: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        self._steps.append(
            TrajectoryStep(
                step=len(self._steps) + 1,
                type=step_type,
                tool_call=tool_call,
                tool_result=tool_result,
                content=content,
                metadata=metadata or {},
            )
        )
