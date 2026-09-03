"""Tests for validated runtime settings."""

import pytest
from pydantic import ValidationError

from harness.config import LogLevel, Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings()

    assert settings.log_level is LogLevel.INFO
    assert settings.max_concurrency == 5
    assert settings.test_timeout_seconds == 60.0


def test_settings_load_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_EVAL_LOG_LEVEL", "debug")
    monkeypatch.setenv("AGENT_EVAL_MAX_CONCURRENCY", "8")
    monkeypatch.setenv("AGENT_EVAL_TEST_TIMEOUT_SECONDS", "12.5")

    settings = Settings()

    assert settings.log_level is LogLevel.DEBUG
    assert settings.max_concurrency == 8
    assert settings.test_timeout_seconds == 12.5


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("AGENT_EVAL_MAX_CONCURRENCY", "0"),
        ("AGENT_EVAL_MAX_CONCURRENCY", "-1"),
        ("AGENT_EVAL_TEST_TIMEOUT_SECONDS", "0"),
        ("AGENT_EVAL_TEST_TIMEOUT_SECONDS", "-0.1"),
        ("AGENT_EVAL_LOG_LEVEL", "VERBOSE"),
    ],
)
def test_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings()
