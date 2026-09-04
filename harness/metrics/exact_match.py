"""Exact-match evaluation."""

import json

from pydantic import BaseModel, ConfigDict

from harness.metrics.base import Metric
from harness.schemas import JsonValue, MetricResult, MetricStatus


class ExactMatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: JsonValue
    actual: JsonValue
    case_sensitive: bool = True
    normalize_whitespace: bool = False


def _normalize(value: JsonValue, *, case_sensitive: bool, whitespace: bool) -> JsonValue:
    if isinstance(value, str):
        normalized = " ".join(value.split()) if whitespace else value
        return normalized if case_sensitive else normalized.casefold()
    if isinstance(value, list):
        return [
            _normalize(item, case_sensitive=case_sensitive, whitespace=whitespace)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _normalize(item, case_sensitive=case_sensitive, whitespace=whitespace)
            for key, item in value.items()
        }
    return value


class ExactMatchMetric(Metric[ExactMatchInput]):
    def evaluate(self, context: ExactMatchInput) -> list[MetricResult]:
        expected = _normalize(
            context.expected,
            case_sensitive=context.case_sensitive,
            whitespace=context.normalize_whitespace,
        )
        actual = _normalize(
            context.actual,
            case_sensitive=context.case_sensitive,
            whitespace=context.normalize_whitespace,
        )
        matched = expected == actual

        return [
            MetricResult(
                name="exact_match",
                status=MetricStatus.PASSED if matched else MetricStatus.FAILED,
                score=1.0 if matched else 0.0,
                explanation=(
                    "Actual output matches expected output." if matched else "Outputs differ."
                ),
                details={
                    "expected": json.dumps(context.expected, sort_keys=True),
                    "actual": json.dumps(context.actual, sort_keys=True),
                },
            )
        ]
