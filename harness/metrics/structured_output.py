"""JSON parsing and JSON Schema validation metrics."""

import json
from typing import Any, cast

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict

from harness.metrics.base import Metric
from harness.schemas import JsonValue, MetricResult, MetricStatus


class StructuredOutputInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: JsonValue
    output_schema: dict[str, JsonValue]


class StructuredOutputMetric(Metric[StructuredOutputInput]):
    def evaluate(self, context: StructuredOutputInput) -> list[MetricResult]:
        parsed, parse_error = self._parse(context.output)
        parse_success = parse_error is None

        parse_result = MetricResult(
            name="structured_parse_success",
            status=MetricStatus.PASSED if parse_success else MetricStatus.FAILED,
            score=1.0 if parse_success else 0.0,
            explanation="Output is valid JSON." if parse_success else parse_error,
        )
        if not parse_success:
            return [
                parse_result,
                MetricResult(
                    name="structured_schema_success",
                    status=MetricStatus.FAILED,
                    score=0.0,
                    explanation="Schema validation requires parseable JSON.",
                ),
                MetricResult(
                    name="structured_required_field_accuracy",
                    status=MetricStatus.FAILED,
                    score=0.0,
                    explanation="Required fields cannot be inspected before parsing.",
                ),
            ]

        validator = Draft202012Validator(context.output_schema)
        errors = sorted(validator.iter_errors(parsed), key=lambda error: list(error.path))
        schema_success = not errors
        required = self._required_fields(context.output_schema)
        present = set(parsed) if isinstance(parsed, dict) else set()
        required_score = 1.0 if not required else len(required & present) / len(required)
        error_messages = [cast(JsonValue, error.message) for error in errors]
        required_fields = [cast(JsonValue, field) for field in sorted(required)]
        present_required_fields = [
            cast(JsonValue, field) for field in sorted(required & present)
        ]

        return [
            parse_result,
            MetricResult(
                name="structured_schema_success",
                status=MetricStatus.PASSED if schema_success else MetricStatus.FAILED,
                score=1.0 if schema_success else 0.0,
                explanation=(
                    "Output satisfies the JSON schema."
                    if schema_success
                    else "Output violates the JSON schema."
                ),
                details={"errors": error_messages},
            ),
            MetricResult(
                name="structured_required_field_accuracy",
                status=(
                    MetricStatus.PASSED if required_score == 1.0 else MetricStatus.FAILED
                ),
                score=required_score,
                details={
                    "required_fields": required_fields,
                    "present_required_fields": present_required_fields,
                },
            ),
        ]

    @staticmethod
    def _parse(output: JsonValue) -> tuple[Any, str | None]:
        if not isinstance(output, str):
            return output, None
        try:
            return json.loads(output), None
        except json.JSONDecodeError as error:
            return None, f"Output is not valid JSON: {error.msg}."

    @staticmethod
    def _required_fields(schema: dict[str, JsonValue]) -> set[str]:
        required = schema.get("required", [])
        if not isinstance(required, list):
            return set()
        return {field for field in required if isinstance(field, str)}
