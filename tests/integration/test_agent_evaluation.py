"""End-to-end deterministic agent evaluation tests."""

import json
from pathlib import Path

import pytest

from harness.evaluators import AgentEvaluator
from harness.schemas import AgentExecutionResult, AgentTestCase
from target_agent import BadMockAgent, GoodMockAgent

ROOT = Path(__file__).parents[2]


def _load_cases() -> list[AgentTestCase]:
    payload = json.loads((ROOT / "datasets/agent_golden_dataset.json").read_text())
    return [AgentTestCase.model_validate(item) for item in payload]


@pytest.mark.asyncio
async def test_good_agent_passes_and_bad_agent_is_measurably_worse() -> None:
    evaluator = AgentEvaluator(threshold=0.85, concurrency=2)
    good_run = await evaluator.evaluate(_load_cases(), GoodMockAgent())
    bad_run = await evaluator.evaluate(_load_cases(), BadMockAgent())

    good = good_run.results[0]
    bad = bad_run.results[0]

    assert good.passed
    assert good.overall_score == 1.0
    assert not bad.passed
    assert bad.overall_score is not None
    assert good.overall_score - bad.overall_score >= 0.5
    assert good_run.aggregates["pass_rate"] == 1.0
    assert bad_run.aggregates["pass_rate"] == 0.0


class FailingAgent:
    name = "failing"

    def __init__(self) -> None:
        self.known_tools: set[str] = set()

    async def execute(self, test_case: AgentTestCase) -> AgentExecutionResult:
        raise RuntimeError(f"cannot execute {test_case.test_id}")


@pytest.mark.asyncio
async def test_agent_error_becomes_failed_evaluation_result() -> None:
    evaluator = AgentEvaluator()
    run = await evaluator.evaluate(_load_cases(), FailingAgent())

    assert not run.results[0].passed
    assert run.results[0].overall_score == 0.0
    assert run.results[0].execution.error == "RuntimeError: cannot execute agent_sales_001"
