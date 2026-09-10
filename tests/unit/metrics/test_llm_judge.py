import json

import pytest

from harness.metrics.llm_judge import LLMJudge
from harness.providers.base import ModelProvider
from harness.schemas import MetricStatus, ModelRequest, ModelResponse


class SequenceProvider(ModelProvider):
    def __init__(self, contents: list[str | None]) -> None:
        self.contents = iter(contents)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        content = next(self.contents)
        return ModelResponse(
            model=request.model,
            content=content,
            error=None if content is not None else "judge failed",
        )


def verdict(score: int, hallucination: bool = False) -> str:
    return json.dumps(
        {
            "faithfulness": {"score": score, "explanation": "faith"},
            "goal_completion": {"score": score, "explanation": "goal"},
            "relevance": {"score": score, "explanation": "relevant"},
            "hallucination_detected": hallucination,
        }
    )


@pytest.mark.asyncio
async def test_averages_repeated_judge_runs() -> None:
    provider = SequenceProvider([verdict(5), verdict(3, hallucination=True)])
    results = await LLMJudge(provider, model="judge/model", runs=2).evaluate(
        task="Do the task", answer="done", ground_truth={"answer": "done"}
    )

    assert [result.score for result in results] == [0.8, 0.8, 0.8, 0.5]
    assert results[-1].status is MetricStatus.FAILED
    assert results[0].details["probabilistic"] is True
    assert all(request.temperature == 0 for request in provider.requests)


@pytest.mark.asyncio
async def test_retains_success_when_one_judge_run_fails() -> None:
    provider = SequenceProvider(["not json", verdict(5)])
    results = await LLMJudge(provider, model="judge/model", runs=2).evaluate(
        task="task", answer="answer", ground_truth="answer"
    )

    assert len(results) == 4
    assert results[0].score == 1
    assert results[0].details["successful_runs"] == 1
    errors = results[0].details["errors"]
    assert isinstance(errors, list)
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_reports_all_judge_failures_without_zero_score() -> None:
    provider = SequenceProvider([None, "invalid"])
    results = await LLMJudge(provider, model="judge/model", runs=2).evaluate(
        task="task", answer=None, ground_truth=None
    )

    assert len(results) == 1
    assert results[0].status is MetricStatus.ERROR
    assert results[0].score is None
    assert results[0].details["successful_runs"] == 0
