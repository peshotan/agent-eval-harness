"""Evaluation pipeline for observable agent executions."""

from typing import Protocol

from harness.metrics import (
    ExactMatchInput,
    ExactMatchMetric,
    ToolAccuracyInput,
    ToolAccuracyMetric,
    TrajectoryInput,
    TrajectoryMetric,
)
from harness.runner import AsyncRunner, RunOutcome
from harness.schemas import (
    AgentExecutionResult,
    AgentTestCase,
    EvaluationRunResult,
    EvaluationType,
    MetricResult,
    MetricStatus,
    TestEvaluationResult,
    ToolCall,
    TrajectoryStepType,
)


class EvaluatedAgent(Protocol):
    """Minimal adapter contract for an agent under evaluation."""

    name: str
    known_tools: set[str]

    async def execute(self, test_case: AgentTestCase) -> AgentExecutionResult:
        """Execute one test case and return its normalized observable result."""


class AgentEvaluator:
    """Execute agent cases and calculate deterministic evaluation results."""

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        concurrency: int = 5,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.runner = AsyncRunner(
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
        )

    async def evaluate(
        self,
        test_cases: list[AgentTestCase],
        agent: EvaluatedAgent,
    ) -> EvaluationRunResult:
        outcomes = await self.runner.run(test_cases, agent.execute)
        results = [
            self._evaluate_outcome(test_case, outcome, agent)
            for test_case, outcome in zip(test_cases, outcomes, strict=True)
        ]
        scores = [result.overall_score for result in results if result.overall_score is not None]
        mean_score = sum(scores) / len(scores) if scores else None

        return EvaluationRunResult(
            evaluation_type=EvaluationType.AGENT,
            configuration={
                "agent": agent.name,
                "threshold": self.threshold,
                "concurrency": self.runner.concurrency,
                "timeout_seconds": self.runner.timeout_seconds,
            },
            results=results,
            aggregates={
                "test_count": len(results),
                "passed_count": sum(result.passed for result in results),
                "pass_rate": (
                    sum(result.passed for result in results) / len(results) if results else 0.0
                ),
                "mean_score": mean_score,
            },
        )

    def _evaluate_outcome(
        self,
        test_case: AgentTestCase,
        outcome: RunOutcome[AgentExecutionResult],
        agent: EvaluatedAgent,
    ) -> TestEvaluationResult:
        if not outcome.succeeded or outcome.value is None:
            execution = AgentExecutionResult(
                test_id=test_case.test_id,
                agent=agent.name,
                latency_ms=outcome.latency_ms,
                error=outcome.error or "agent returned no execution result",
            )
            return TestEvaluationResult(
                test_id=test_case.test_id,
                execution=execution,
                metrics=[
                    MetricResult(
                        name="execution_success",
                        status=MetricStatus.ERROR,
                        score=0.0,
                        explanation=execution.error,
                        details={"timed_out": outcome.timed_out},
                    )
                ],
                overall_score=0.0,
                passed=False,
            )

        execution = outcome.value
        tool_calls = [
            step.tool_call
            for step in execution.trajectory
            if step.type is TrajectoryStepType.TOOL_CALL and step.tool_call is not None
        ]
        metrics: list[MetricResult] = []
        if test_case.reference_answer is not None:
            metrics.extend(
                ExactMatchMetric().evaluate(
                    ExactMatchInput(
                        expected=test_case.reference_answer,
                        actual=execution.final_answer,
                    )
                )
            )
        metrics.extend(
            ToolAccuracyMetric().evaluate(
                ToolAccuracyInput(
                    expected_tools=test_case.expected_tools,
                    actual_calls=tool_calls,
                    known_tools=agent.known_tools,
                )
            )
        )
        metrics.extend(
            TrajectoryMetric().evaluate(
                TrajectoryInput(
                    steps=execution.trajectory,
                    optimal_steps=test_case.optimal_steps
                    or self._default_optimal_steps(tool_calls),
                    max_allowed_steps=test_case.max_allowed_steps,
                )
            )
        )
        scores = [metric.score for metric in metrics if metric.score is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.0

        return TestEvaluationResult(
            test_id=test_case.test_id,
            execution=execution,
            metrics=metrics,
            overall_score=overall_score,
            passed=execution.error is None and overall_score >= self.threshold,
        )

    @staticmethod
    def _default_optimal_steps(tool_calls: list[ToolCall]) -> int:
        return max(1, len(tool_calls) * 2 + 1)
