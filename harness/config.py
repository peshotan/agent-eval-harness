"""Validated runtime configuration for evaluation runs."""

from enum import StrEnum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a local `.env` file."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: LogLevel = LogLevel.INFO
    max_concurrency: int = Field(default=5, gt=0)
    test_timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Accept conventional case-insensitive log-level values."""
        return value.upper() if isinstance(value, str) else value
