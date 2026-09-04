"""Metrics over observable agent trajectory events."""

import json
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from harness.metrics.base import Metric
from harness.schemas import (
    JsonValue,
    MetricResult,
    MetricStatus,
    TrajectoryStep,
    TrajectoryStepType,
)


class TrajectoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[TrajectoryStep] = Field(default_factory=list)
    optimal_steps: int = Field(ge=1)
    max_allowed_steps: int | None = Field(default=None, ge=1)


class TrajectoryMetric(Metric[TrajectoryInput]):
    def evaluate(self, context: TrajectoryInput) -> list[MetricResult]:
        actual_steps = len(context.steps)
        efficiency = min(1.0, context.optimal_steps / actual_steps) if actual_steps else 0.0
        within_limit = (
            actual_steps <= context.max_allowed_steps
            if context.max_allowed_steps is not None
            else efficiency == 1.0
        )
        repeated_steps = self._repeated_tool_call_steps(context.steps)
        tool_results = [
            step.tool_result
            for step in context.steps
            if step.type is TrajectoryStepType.TOOL_RESULT and step.tool_result is not None
        ]
        failed_calls = sum(not result.success for result in tool_results)
        failure_rate = failed_calls / len(tool_results) if tool_results else 0.0
        repeated_steps_json: list[JsonValue] = [step for step in repeated_steps]

        return [
            MetricResult(
                name="trajectory_efficiency",
                status=MetricStatus.PASSED if within_limit else MetricStatus.FAILED,
                score=efficiency,
                details={
                    "optimal_steps": context.optimal_steps,
                    "actual_steps": actual_steps,
                    "max_allowed_steps": context.max_allowed_steps,
                },
            ),
            MetricResult(
                name="loop_detection",
                status=MetricStatus.FAILED if repeated_steps else MetricStatus.PASSED,
                score=0.0 if repeated_steps else 1.0,
                details={
                    "loop_detected": bool(repeated_steps),
                    "repeated_steps": repeated_steps_json,
                },
            ),
            MetricResult(
                name="tool_failure_rate",
                status=MetricStatus.PASSED if failure_rate == 0.0 else MetricStatus.FAILED,
                score=1.0 - failure_rate,
                details={
                    "failed_tool_calls": failed_calls,
                    "total_tool_calls": len(tool_results),
                    "rate": failure_rate,
                },
            ),
        ]

    @staticmethod
    def _repeated_tool_call_steps(steps: list[TrajectoryStep]) -> list[int]:
        occurrences: dict[str, list[int]] = defaultdict(list)
        for step in steps:
            if step.type is not TrajectoryStepType.TOOL_CALL or step.tool_call is None:
                continue
            signature = json.dumps(
                {"name": step.tool_call.name, "arguments": step.tool_call.arguments},
                sort_keys=True,
                separators=(",", ":"),
            )
            occurrences[signature].append(step.step)

        return sorted(
            step_number
            for repeated in occurrences.values()
            if len(repeated) > 1
            for step_number in repeated
        )
