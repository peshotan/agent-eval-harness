"""Tests for shared evaluation contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.schemas import (
    AgentTestCase,
    ModelTestCase,
    TokenUsage,
    ToolCall,
    ToolResult,
    TrajectoryStep,
    TrajectoryStepType,
)

ROOT = Path(__file__).parents[2]


def test_example_datasets_validate() -> None:
    model_cases = json.loads((ROOT / "datasets/model_golden_dataset.json").read_text())
    agent_cases = json.loads((ROOT / "datasets/agent_golden_dataset.json").read_text())

    assert [ModelTestCase.model_validate(case).test_id for case in model_cases] == [
        "model_math_001"
    ]
    assert [AgentTestCase.model_validate(case).test_id for case in agent_cases] == [
        "agent_sales_001"
    ]


def test_token_usage_requires_consistent_total() -> None:
    with pytest.raises(ValidationError, match="total_tokens"):
        TokenUsage(input_tokens=10, output_tokens=5, total_tokens=12)


def test_failed_tool_result_requires_error() -> None:
    with pytest.raises(ValidationError, match="must contain an error"):
        ToolResult(call_id="call-1", name="lookup", success=False)


def test_trajectory_step_requires_matching_payload() -> None:
    with pytest.raises(ValidationError, match="tool_call payload"):
        TrajectoryStep(step=1, type=TrajectoryStepType.TOOL_CALL)

    step = TrajectoryStep(
        step=1,
        type=TrajectoryStepType.TOOL_CALL,
        tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"id": 7}),
    )
    assert step.tool_call is not None
    assert step.tool_call.arguments == {"id": 7}


def test_agent_case_rejects_inverted_step_bounds() -> None:
    with pytest.raises(ValidationError, match="optimal_steps"):
        AgentTestCase(
            test_id="case-1",
            category="test",
            user_input="Do the task",
            optimal_steps=5,
            max_allowed_steps=4,
        )
