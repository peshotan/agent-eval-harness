import pytest

from harness.regression import RegressionComparator, RegressionPolicy
from harness.schemas import EvaluationRunResult, EvaluationType, JsonValue


def run(
    *,
    mean_score: float | None,
    pass_rate: float | None,
    evaluation_type: EvaluationType = EvaluationType.MODEL,
) -> EvaluationRunResult:
    aggregates: dict[str, JsonValue] = {}
    if mean_score is not None:
        aggregates["mean_score"] = mean_score
    if pass_rate is not None:
        aggregates["pass_rate"] = pass_rate
    return EvaluationRunResult(evaluation_type=evaluation_type, aggregates=aggregates)


def test_passes_when_candidate_satisfies_quality_policy() -> None:
    baseline = run(mean_score=0.90, pass_rate=0.90)
    candidate = run(mean_score=0.88, pass_rate=0.90)

    result = RegressionComparator().compare(
        baseline,
        candidate,
        RegressionPolicy(
            min_overall_score=0.85,
            min_pass_rate=0.85,
            max_score_regression=0.03,
        ),
    )

    assert result.passed is True
    assert result.metric_deltas["mean_score"] == pytest.approx(-0.02)
    assert result.regressions == []


def test_reports_each_violated_quality_threshold() -> None:
    result = RegressionComparator().compare(
        run(mean_score=0.90, pass_rate=1.0),
        run(mean_score=0.80, pass_rate=0.75),
        RegressionPolicy(
            min_overall_score=0.85,
            min_pass_rate=0.80,
            max_score_regression=0.05,
        ),
    )

    assert result.passed is False
    assert len(result.regressions) == 3
    assert result.metric_deltas["mean_score"] == pytest.approx(-0.10)


def test_missing_required_aggregate_fails_explicitly() -> None:
    result = RegressionComparator().compare(
        run(mean_score=0.90, pass_rate=1.0),
        run(mean_score=None, pass_rate=None),
        RegressionPolicy(min_overall_score=0.85, max_score_regression=0.03),
    )

    assert result.passed is False
    assert result.metric_deltas == {}
    assert any("unavailable" in message for message in result.regressions)
    assert any("score regression" in message for message in result.regressions)


def test_rejects_mismatched_evaluation_types() -> None:
    baseline = run(mean_score=0.9, pass_rate=1.0, evaluation_type=EvaluationType.MODEL)
    candidate = run(mean_score=0.9, pass_rate=1.0, evaluation_type=EvaluationType.AGENT)

    with pytest.raises(ValueError, match="baseline and candidate evaluation types must match"):
        RegressionComparator().compare(baseline, candidate, RegressionPolicy())
