"""Deterministic tool-selection and argument metrics."""

from pydantic import BaseModel, ConfigDict, Field

from harness.metrics.base import Metric
from harness.schemas import JsonValue, MetricResult, MetricStatus, ToolCall, ToolExpectation


class ToolAccuracyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_tools: list[ToolExpectation] = Field(default_factory=list)
    actual_calls: list[ToolCall] = Field(default_factory=list)
    known_tools: set[str] = Field(default_factory=set)


def _argument_score(expectation: ToolExpectation, call: ToolCall) -> float:
    required = expectation.required_arguments
    if not required:
        return 1.0
    matches = sum(call.arguments.get(key) == value for key, value in required.items())
    return matches / len(required)


class ToolAccuracyMetric(Metric[ToolAccuracyInput]):
    def evaluate(self, context: ToolAccuracyInput) -> list[MetricResult]:
        unused_call_indexes = set(range(len(context.actual_calls)))
        exact_matches = 0
        argument_scores: list[float] = []

        for expectation in context.expected_tools:
            same_name = [
                index
                for index in unused_call_indexes
                if context.actual_calls[index].name == expectation.name
            ]
            if not same_name:
                argument_scores.append(0.0)
                continue

            best_index = max(
                same_name,
                key=lambda index: _argument_score(expectation, context.actual_calls[index]),
            )
            score = _argument_score(expectation, context.actual_calls[best_index])
            argument_scores.append(score)
            unused_call_indexes.remove(best_index)
            if score == 1.0:
                exact_matches += 1

        precision = self._safe_ratio(
            exact_matches,
            len(context.actual_calls),
            empty_value=1.0 if not context.expected_tools else 0.0,
        )
        recall = self._safe_ratio(exact_matches, len(context.expected_tools), empty_value=1.0)
        argument_accuracy = (
            sum(argument_scores) / len(argument_scores) if argument_scores else 1.0
        )
        known_tools = context.known_tools or {item.name for item in context.expected_tools}
        unknown_count = sum(call.name not in known_tools for call in context.actual_calls)
        unknown_rate = self._safe_ratio(unknown_count, len(context.actual_calls), empty_value=0.0)
        argument_scores_json: list[JsonValue] = [score for score in argument_scores]

        return [
            self._score_result("tool_precision", precision, {"exact_matches": exact_matches}),
            self._score_result("tool_recall", recall, {"exact_matches": exact_matches}),
            self._score_result(
                "argument_accuracy",
                argument_accuracy,
                {"per_expectation_scores": argument_scores_json},
            ),
            MetricResult(
                name="unknown_tool_rate",
                status=MetricStatus.PASSED if unknown_rate == 0.0 else MetricStatus.FAILED,
                score=1.0 - unknown_rate,
                details={"unknown_calls": unknown_count, "rate": unknown_rate},
            ),
        ]

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
        return numerator / denominator if denominator else empty_value

    @staticmethod
    def _score_result(
        name: str,
        score: float,
        details: dict[str, JsonValue],
    ) -> MetricResult:
        return MetricResult(
            name=name,
            status=MetricStatus.PASSED if score == 1.0 else MetricStatus.FAILED,
            score=score,
            details=details,
        )
