"""Tests for structured-output evaluation."""

from harness.metrics.structured_output import StructuredOutputInput, StructuredOutputMetric
from harness.schemas import JsonValue, MetricResult, MetricStatus

SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"failure_rate": {"type": "number"}, "service": {"type": "string"}},
    "required": ["failure_rate", "service"],
    "additionalProperties": False,
}


def _by_name(output: str) -> dict[str, MetricResult]:
    results = StructuredOutputMetric().evaluate(
        StructuredOutputInput(output=output, output_schema=SCHEMA)
    )
    return {result.name: result for result in results}


def test_valid_json_that_matches_schema_passes() -> None:
    results = _by_name('{"failure_rate": 0.025, "service": "api"}')

    assert results["structured_parse_success"].status is MetricStatus.PASSED
    assert results["structured_schema_success"].score == 1.0
    assert results["structured_required_field_accuracy"].score == 1.0


def test_invalid_json_fails_all_structured_checks() -> None:
    results = _by_name('{"failure_rate":')

    assert all(result.score == 0.0 for result in results.values())


def test_missing_field_reports_partial_required_accuracy() -> None:
    results = _by_name('{"failure_rate": 0.025}')

    assert results["structured_parse_success"].score == 1.0
    assert results["structured_schema_success"].score == 0.0
    assert results["structured_required_field_accuracy"].score == 0.5
