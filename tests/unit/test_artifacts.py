from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.artifacts import load_evaluation_run
from harness.reporters import write_json
from harness.schemas import EvaluationRunResult, EvaluationType


def test_loads_written_evaluation_run(tmp_path: Path) -> None:
    source = tmp_path / "run.json"
    expected = EvaluationRunResult(
        evaluation_type=EvaluationType.MODEL,
        aggregates={"mean_score": 0.92},
    )
    write_json(expected, source)

    loaded = load_evaluation_run(source)

    assert loaded == expected


def test_rejects_unknown_artifact_fields(tmp_path: Path) -> None:
    source = tmp_path / "run.json"
    source.write_text(
        '{"evaluation_type":"model","unexpected":true}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unexpected"):
        load_evaluation_run(source)


def test_rejects_malformed_json(tmp_path: Path) -> None:
    source = tmp_path / "run.json"
    source.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValidationError, match="Invalid JSON"):
        load_evaluation_run(source)
