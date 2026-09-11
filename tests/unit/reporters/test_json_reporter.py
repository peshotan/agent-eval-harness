import json
from pathlib import Path
from uuid import UUID

from harness.reporters import render_json, write_json
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


def test_atomically_writes_json_and_creates_parent_directory(tmp_path: Path) -> None:
    destination = tmp_path / "artifacts" / "run.json"
    result = EvaluationRunResult(evaluation_type=EvaluationType.AGENT)

    returned_path = write_json(result, destination)

    assert returned_path == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["evaluation_type"] == "agent"
    assert list(destination.parent.iterdir()) == [destination]
