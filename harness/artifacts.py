"""Validated evaluation artifact loading."""

from pathlib import Path

from harness.schemas import EvaluationRunResult


def load_evaluation_run(source: Path) -> EvaluationRunResult:
    """Load and validate one evaluation-run JSON artifact."""
    return EvaluationRunResult.model_validate_json(source.read_text(encoding="utf-8"))
