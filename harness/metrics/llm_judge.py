"""Structured, repeatable LLM judge metric for open-ended outputs."""

import json
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.providers.base import ModelProvider
from harness.schemas import JsonValue, MetricResult, MetricStatus, ModelRequest


class JudgeDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    explanation: str = Field(min_length=1)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: JudgeDimension
    goal_completion: JudgeDimension
    relevance: JudgeDimension
    hallucination_detected: bool


class LLMJudge:
    """Average repeated structured judge calls and expose failures explicitly."""

    def __init__(self, provider: ModelProvider, *, model: str, runs: int = 1) -> None:
        if runs < 1:
            raise ValueError("runs must be positive")
        self.provider = provider
        self.model = model
        self.runs = runs

    async def evaluate(
        self,
        *,
        task: str,
        answer: str | None,
        ground_truth: JsonValue,
        evidence: JsonValue = None,
    ) -> list[MetricResult]:
        verdicts: list[JudgeVerdict] = []
        errors: list[str] = []
        for _ in range(self.runs):
            response = await self.provider.generate(
                ModelRequest(
                    model=self.model,
                    prompt=self._prompt(task, answer, ground_truth, evidence),
                    system_prompt=(
                        "You are an evaluation judge. Return only JSON matching the requested "
                        "schema. Scores are probabilistic assessments, not ground truth."
                    ),
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            )
            if response.error or response.content is None:
                errors.append(response.error or "judge returned no content")
                continue
            try:
                verdicts.append(JudgeVerdict.model_validate_json(response.content))
            except ValidationError as error:
                errors.append(f"invalid judge response: {error.errors()[0]['msg']}")

        if not verdicts:
            return [
                MetricResult(
                    name="llm_judge",
                    status=MetricStatus.ERROR,
                    score=None,
                    explanation="All judge runs failed.",
                    details={
                        "errors": cast(JsonValue, errors),
                        "successful_runs": 0,
                        "requested_runs": self.runs,
                    },
                )
            ]

        results = [
            self._dimension_result(name, verdicts, errors)
            for name in ("faithfulness", "goal_completion", "relevance")
        ]
        hallucination_rate = sum(v.hallucination_detected for v in verdicts) / len(verdicts)
        results.append(
            MetricResult(
                name="hallucination_free",
                status=MetricStatus.PASSED if hallucination_rate == 0 else MetricStatus.FAILED,
                score=1 - hallucination_rate,
                explanation="Fraction of successful judge runs that detected no hallucination.",
                details=self._details(errors, len(verdicts)),
            )
        )
        return results

    def _dimension_result(
        self,
        name: str,
        verdicts: list[JudgeVerdict],
        errors: list[str],
    ) -> MetricResult:
        dimensions = [getattr(verdict, name) for verdict in verdicts]
        score = sum(dimension.score for dimension in dimensions) / len(dimensions) / 5
        return MetricResult(
            name=f"judge_{name}",
            status=MetricStatus.PASSED if score >= 0.8 else MetricStatus.FAILED,
            score=score,
            explanation=" | ".join(dimension.explanation for dimension in dimensions),
            details=self._details(errors, len(verdicts)),
        )

    def _details(self, errors: list[str], successful_runs: int) -> dict[str, JsonValue]:
        return {
            "judge_model": self.model,
            "requested_runs": self.runs,
            "successful_runs": successful_runs,
            "errors": cast(JsonValue, errors),
            "probabilistic": True,
        }

    @staticmethod
    def _prompt(task: str, answer: str | None, ground_truth: JsonValue, evidence: JsonValue) -> str:
        return json.dumps(
            {
                "instruction": (
                    "Score faithfulness, goal_completion, and relevance from 1 to 5 with "
                    "explanations; also return hallucination_detected as a boolean."
                ),
                "task": task,
                "answer": answer,
                "ground_truth": ground_truth,
                "evidence": evidence,
            },
            sort_keys=True,
        )
