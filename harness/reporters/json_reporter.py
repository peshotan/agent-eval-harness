"""Machine-readable result rendering."""

from harness.schemas import EvaluationRunResult, RegressionResult


def render_json(result: EvaluationRunResult | RegressionResult) -> str:
    """Render one validated result as indented JSON with a final newline."""
    return result.model_dump_json(indent=2) + "\n"
