"""Smoke tests for the project skeleton."""

from typer.testing import CliRunner

from cli import app
from harness import __version__


def test_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_cli_reports_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
