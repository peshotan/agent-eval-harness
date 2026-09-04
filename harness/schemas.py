"""Versioned data contracts shared by execution, metrics, and reporting."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import JsonValue as JsonValue


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields in persisted contracts."""

    model_config = ConfigDict(extra="forbid")


class EvaluationType(StrEnum):
    MODEL = "model"
    AGENT = "agent"


class TrajectoryStepType(StrEnum):
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRY = "retry"
    STATE_TRANSITION = "state_transition"
    FINAL_ANSWER = "final_answer"


class MetricStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "TokenUsage":
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class ToolExpectation(StrictModel):
    name: str = Field(min_length=1)
    required_arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolCall(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    success: bool = True
    result: JsonValue = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> "ToolResult":
        if self.success and self.error is not None:
            raise ValueError("successful tool results cannot contain an error")
        if not self.success and not self.error:
            raise ValueError("failed tool results must contain an error")
        return self


class TrajectoryStep(StrictModel):
    step: int = Field(ge=1)
    type: TrajectoryStepType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    content: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "TrajectoryStep":
        if self.type is TrajectoryStepType.TOOL_CALL and self.tool_call is None:
            raise ValueError("tool_call steps require a tool_call payload")
        if self.type is TrajectoryStepType.TOOL_RESULT and self.tool_result is None:
            raise ValueError("tool_result steps require a tool_result payload")
        if self.type is TrajectoryStepType.FINAL_ANSWER and self.content is None:
            raise ValueError("final_answer steps require content")
        if self.type is not TrajectoryStepType.TOOL_CALL and self.tool_call is not None:
            raise ValueError("tool_call payload is only valid for tool_call steps")
        if self.type is not TrajectoryStepType.TOOL_RESULT and self.tool_result is not None:
            raise ValueError("tool_result payload is only valid for tool_result steps")
        return self


class ModelTestCase(StrictModel):
    test_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_output: JsonValue = None
    output_schema: dict[str, JsonValue] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentTestCase(StrictModel):
    test_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    expected_tools: list[ToolExpectation] = Field(default_factory=list)
    ground_truth: JsonValue = None
    reference_answer: str | None = None
    optimal_steps: int | None = Field(default=None, ge=1)
    max_allowed_steps: int | None = Field(default=None, ge=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_step_bounds(self) -> "AgentTestCase":
        if (
            self.optimal_steps is not None
            and self.max_allowed_steps is not None
            and self.optimal_steps > self.max_allowed_steps
        ):
            raise ValueError("optimal_steps cannot exceed max_allowed_steps")
        return self


class ModelExecutionResult(StrictModel):
    test_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    output: JsonValue = None
    raw_output: str | None = None
    latency_ms: float = Field(ge=0)
    usage: TokenUsage | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None


class AgentExecutionResult(StrictModel):
    test_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    final_answer: str | None = None
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    usage: TokenUsage | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None


class MetricResult(StrictModel):
    name: str = Field(min_length=1)
    status: MetricStatus
    score: float | None = Field(default=None, ge=0, le=1)
    explanation: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TestEvaluationResult(StrictModel):
    test_id: str = Field(min_length=1)
    execution: ModelExecutionResult | AgentExecutionResult
    metrics: list[MetricResult] = Field(default_factory=list)
    overall_score: float | None = Field(default=None, ge=0, le=1)
    passed: bool


class EvaluationRunResult(StrictModel):
    schema_version: str = "1.0"
    run_id: UUID = Field(default_factory=uuid4)
    evaluation_type: EvaluationType
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    results: list[TestEvaluationResult] = Field(default_factory=list)
    aggregates: dict[str, JsonValue] = Field(default_factory=dict)


class RegressionResult(StrictModel):
    baseline_run_id: UUID
    candidate_run_id: UUID
    passed: bool
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)
