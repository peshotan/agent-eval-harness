"""Machine-readable result rendering."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from harness.schemas import EvaluationRunResult, RegressionResult


def render_json(result: EvaluationRunResult | RegressionResult) -> str:
    """Render one validated result as indented JSON with a final newline."""
    return result.model_dump_json(indent=2) + "\n"


def write_json(
    result: EvaluationRunResult | RegressionResult,
    destination: Path,
) -> Path:
    """Atomically write a JSON result, creating parent directories as needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary.write(render_json(result))
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
