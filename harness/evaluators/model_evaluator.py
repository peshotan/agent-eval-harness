"""Evaluation pipeline for model generations."""

import json
from collections import Counter

from harness.metrics import (
    ExactMatchInput,
    ExactMatchMetric,
    StructuredOutputInput,
    StructuredOutputMetric,
)
from harness.metrics.llm_judge import LLMJudge
from harness.providers.base import ModelProvider
from harness.runner import AsyncRunner, RunOutcome
from harness.schemas import (
    EvaluationRunResult,
    EvaluationType,
    JsonValue,
    MetricResult,
    MetricStatus,
    ModelExecutionResult,
    ModelRequest,
    ModelResponse,
    ModelTestCase,
    TestEvaluationResult,
)


class ModelEvaluator:
    """Execute model cases and combine deterministic and optional judge metrics."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        judge: LLMJudge | None = None,
        repeats: int = 1,
        threshold: float = 0.85,
        concurrency: int = 5,
        timeout_seconds: float = 60.0,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.provider = provider
        self.model = model
        self.judge = judge
        self.repeats = repeats
        self.threshold = threshold
        self.runner = AsyncRunner(concurrency=concurrency, timeout_seconds=timeout_seconds)

    async def evaluate(self, test_cases: list[ModelTestCase]) -> EvaluationRunResult:
        repeated_cases = [(case, run) for case in test_cases for run in range(self.repeats)]
        outcomes = await self.runner.run(repeated_cases, self._generate)
        results = [
            await self._evaluate_outcome(case, run, outcome)
            for (case, run), outcome in zip(repeated_cases, outcomes, strict=True)
        ]
        consistency_scores = self._add_consistency_metrics(results)
        scores = [result.overall_score for result in results if result.overall_score is not None]
        usages = [
            result.execution.usage for result in results if result.execution.usage is not None
        ]
        costs = [
            usage.estimated_cost_usd for usage in usages if usage.estimated_cost_usd is not None
        ]
        return EvaluationRunResult(
            evaluation_type=EvaluationType.MODEL,
            configuration={
                "model": self.model,
                "repeats": self.repeats,
                "threshold": self.threshold,
                "judge_model": self.judge.model if self.judge else None,
            },
            results=results,
            aggregates={
                "test_count": len(results),
                "passed_count": sum(result.passed for result in results),
                "pass_rate": sum(result.passed for result in results) / len(results)
                if results
                else 0.0,
                "mean_score": sum(scores) / len(scores) if scores else None,
                "mean_consistency": (
                    sum(consistency_scores) / len(consistency_scores)
                    if consistency_scores
                    else None
                ),
                "input_tokens": sum(usage.input_tokens for usage in usages),
                "output_tokens": sum(usage.output_tokens for usage in usages),
                "total_tokens": sum(usage.total_tokens for usage in usages),
                "total_cost_usd": sum(costs) if costs else None,
                "average_cost_usd": sum(costs) / len(costs) if costs else None,
                "priced_execution_count": len(costs),
            },
        )

    async def _generate(self, item: tuple[ModelTestCase, int]) -> ModelResponse:
        case, _ = item
        return await self.provider.generate(ModelRequest(model=self.model, prompt=case.prompt))

    async def _evaluate_outcome(
        self,
        test_case: ModelTestCase,
        repeat: int,
        outcome: RunOutcome[ModelResponse],
    ) -> TestEvaluationResult:
        if not outcome.succeeded or outcome.value is None:
            return self._execution_error(test_case, repeat, outcome)
        response = outcome.value
        output = self._parse_output(response.content)
        execution = ModelExecutionResult(
            test_id=test_case.test_id,
            model=response.model,
            output=output,
            raw_output=response.content,
            latency_ms=outcome.latency_ms,
            usage=response.usage,
            provider_metadata={**response.provider_metadata, "repeat": repeat + 1},
            error=response.error,
        )
        metrics = ExactMatchMetric().evaluate(
            ExactMatchInput(expected=test_case.expected_output, actual=output)
        )
        if test_case.output_schema is not None:
            metrics.extend(
                StructuredOutputMetric().evaluate(
                    StructuredOutputInput(
                        output=response.content, output_schema=test_case.output_schema
                    )
                )
            )
        if self.judge is not None and response.error is None:
            metrics.extend(
                await self.judge.evaluate(
                    task=test_case.prompt,
                    answer=response.content,
                    ground_truth=test_case.expected_output,
                )
            )
        scores = [metric.score for metric in metrics if metric.score is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        return TestEvaluationResult(
            test_id=f"{test_case.test_id}#{repeat + 1}",
            execution=execution,
            metrics=metrics,
            overall_score=overall_score,
            passed=response.error is None and overall_score >= self.threshold,
        )

    def _execution_error(
        self,
        test_case: ModelTestCase,
        repeat: int,
        outcome: RunOutcome[ModelResponse],
    ) -> TestEvaluationResult:
        error = outcome.error or "provider returned no response"
        execution = ModelExecutionResult(
            test_id=test_case.test_id,
            model=self.model,
            latency_ms=outcome.latency_ms,
            provider_metadata={"repeat": repeat + 1},
            error=error,
        )
        return TestEvaluationResult(
            test_id=f"{test_case.test_id}#{repeat + 1}",
            execution=execution,
            metrics=[
                MetricResult(
                    name="execution_success", status=MetricStatus.ERROR, score=0, explanation=error
                )
            ],
            overall_score=0,
            passed=False,
        )

    def _add_consistency_metrics(self, results: list[TestEvaluationResult]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(results), self.repeats):
            group = results[start : start + self.repeats]
            successful_outputs = [
                self._canonical_output(result.execution.output)
                for result in group
                if isinstance(result.execution, ModelExecutionResult)
                and result.execution.error is None
            ]
            counts = Counter(successful_outputs)
            score = max(counts.values(), default=0) / self.repeats
            scores.append(score)
            metric = MetricResult(
                name="repeat_consistency",
                status=MetricStatus.PASSED if score == 1 else MetricStatus.FAILED,
                score=score,
                explanation="Share of repeated runs matching the most common successful output.",
                details={"repeats": self.repeats, "distinct_successful_outputs": len(counts)},
            )
            for result in group:
                result.metrics.append(metric.model_copy(deep=True))
                metric_scores = [item.score for item in result.metrics if item.score is not None]
                result.overall_score = sum(metric_scores) / len(metric_scores)
                result.passed = (
                    result.execution.error is None and result.overall_score >= self.threshold
                )
        return scores

    @staticmethod
    def _canonical_output(output: JsonValue) -> str:
        return json.dumps(output, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _parse_output(content: str | None) -> JsonValue:
        if content is None:
            return None
        try:
            parsed: JsonValue = json.loads(content)
        except json.JSONDecodeError:
            return content
        return parsed
