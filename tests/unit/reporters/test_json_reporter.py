import json
from uuid import UUID

from harness.reporters import render_json
from harness.schemas import EvaluationRunResult, EvaluationType, RegressionResult


def test_renders_evaluation_run_as_valid_json() -> None:
    result = EvaluationRunResult(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        evaluation_type=EvaluationType.MODEL,
        aggregates={"mean_score": 0.9},
    )

    rendered = render_json(result)
    payload = json.loads(rendered)

    assert rendered.endswith("\n")
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == "00000000-0000-0000-0000-000000000001"
    assert payload["evaluation_type"] == "model"
    assert payload["aggregates"] == {"mean_score": 0.9}


def test_renders_regression_result_as_valid_json() -> None:
    result = RegressionResult(
        baseline_run_id=UUID("00000000-0000-0000-0000-000000000001"),
        candidate_run_id=UUID("00000000-0000-0000-0000-000000000002"),
        passed=False,
        metric_deltas={"mean_score": -0.1},
        regressions=["mean score decreased"],
    )

    payload = json.loads(render_json(result))

    assert payload["passed"] is False
    assert payload["metric_deltas"] == {"mean_score": -0.1}
    assert payload["regressions"] == ["mean score decreased"]
