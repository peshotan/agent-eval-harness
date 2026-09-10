import pytest

from harness.evaluators import ModelEvaluator
from harness.metrics.llm_judge import LLMJudge
from harness.providers.base import ModelProvider
from harness.schemas import ModelRequest, ModelResponse, ModelTestCase, TokenUsage


class FakeProvider(ModelProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.model == "judge/model":
            return ModelResponse(
                model=request.model,
                content=(
                    '{"faithfulness":{"score":5,"explanation":"grounded"},'
                    '"goal_completion":{"score":5,"explanation":"complete"},'
                    '"relevance":{"score":5,"explanation":"focused"},'
                    '"hallucination_detected":false}'
                ),
            )
        return ModelResponse(
            model=request.model,
            content='{ "failure_rate": 0.025 }',
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=8,
                total_tokens=28,
                estimated_cost_usd=0.001,
            ),
            provider_metadata={"provider": "fake", "cost_status": "available"},
        )


@pytest.mark.asyncio
async def test_model_evaluation_with_repeats_judge_usage_and_cost() -> None:
    provider = FakeProvider()
    case = ModelTestCase(
        test_id="model-1",
        category="structured",
        prompt="Return failure rate",
        expected_output={"failure_rate": 0.025},
        output_schema={
            "type": "object",
            "properties": {"failure_rate": {"type": "number"}},
            "required": ["failure_rate"],
        },
    )
    evaluator = ModelEvaluator(
        provider,
        model="target/model",
        judge=LLMJudge(provider, model="judge/model", runs=2),
        repeats=2,
    )

    run = await evaluator.evaluate([case])

    assert len(run.results) == 2
    assert all(result.passed for result in run.results)
    assert run.aggregates["total_tokens"] == 56
    assert run.aggregates["total_cost_usd"] == 0.002
    assert run.aggregates["priced_execution_count"] == 2
    assert run.aggregates["mean_consistency"] == 1
    assert run.configuration["judge_model"] == "judge/model"
    assert run.results[0].metrics[-1].name == "repeat_consistency"


@pytest.mark.asyncio
async def test_provider_failure_is_isolated() -> None:
    class FailingProvider(ModelProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError(f"unavailable: {request.model}")

    case = ModelTestCase(test_id="failed", category="errors", prompt="prompt")
    run = await ModelEvaluator(FailingProvider(), model="target/model").evaluate([case])

    assert run.results[0].passed is False
    assert run.results[0].overall_score == 0
    assert "RuntimeError" in (run.results[0].execution.error or "")
    assert run.aggregates["mean_consistency"] == 0
