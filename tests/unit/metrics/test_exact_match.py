"""Tests for exact-match evaluation."""

from harness.metrics.exact_match import ExactMatchInput, ExactMatchMetric
from harness.schemas import MetricStatus


def test_exact_match_supports_nested_json_values() -> None:
    result = ExactMatchMetric().evaluate(
        ExactMatchInput(expected={"answer": [1, "yes"]}, actual={"answer": [1, "yes"]})
    )[0]

    assert result.status is MetricStatus.PASSED
    assert result.score == 1.0


def test_exact_match_is_strict_by_default() -> None:
    result = ExactMatchMetric().evaluate(ExactMatchInput(expected="Answer", actual="answer"))[0]

    assert result.status is MetricStatus.FAILED
    assert result.score == 0.0


def test_exact_match_can_normalize_text() -> None:
    result = ExactMatchMetric().evaluate(
        ExactMatchInput(
            expected="The Answer",
            actual=" the   answer ",
            case_sensitive=False,
            normalize_whitespace=True,
        )
    )[0]

    assert result.status is MetricStatus.PASSED
