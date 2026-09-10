"""Aggregate quality comparison for baseline and candidate evaluation runs."""

from pydantic import BaseModel, ConfigDict, Field

from harness.schemas import EvaluationRunResult, JsonValue, RegressionResult


class RegressionPolicy(BaseModel):
    """Initial quality thresholds; operational thresholds are added separately."""

    model_config = ConfigDict(extra="forbid")

    min_overall_score: float | None = Field(default=None, ge=0, le=1)
    min_pass_rate: float | None = Field(default=None, ge=0, le=1)
    max_score_regression: float | None = Field(default=None, ge=0, le=1)


class RegressionComparator:
    """Compare aggregate quality while keeping absent values explicit."""

    def compare(
        self,
        baseline: EvaluationRunResult,
        candidate: EvaluationRunResult,
        policy: RegressionPolicy,
    ) -> RegressionResult:
        if baseline.evaluation_type is not candidate.evaluation_type:
            raise ValueError("baseline and candidate evaluation types must match")

        baseline_score = self._number(baseline, "mean_score")
        candidate_score = self._number(candidate, "mean_score")
        candidate_pass_rate = self._number(candidate, "pass_rate")
        regressions: list[str] = []
        deltas: dict[str, float] = {}

        if baseline_score is not None and candidate_score is not None:
            deltas["mean_score"] = candidate_score - baseline_score

        self._require_minimum(
            name="mean_score",
            actual=candidate_score,
            minimum=policy.min_overall_score,
            regressions=regressions,
        )
        self._require_minimum(
            name="pass_rate",
            actual=candidate_pass_rate,
            minimum=policy.min_pass_rate,
            regressions=regressions,
        )
        if policy.max_score_regression is not None:
            if baseline_score is None or candidate_score is None:
                regressions.append("mean_score is required to evaluate score regression")
            elif baseline_score - candidate_score > policy.max_score_regression:
                regressions.append(
                    "mean_score regression "
                    f"{baseline_score - candidate_score:.4f} exceeds "
                    f"{policy.max_score_regression:.4f}"
                )

        details: dict[str, JsonValue] = {
            "baseline_mean_score": baseline_score,
            "candidate_mean_score": candidate_score,
            "candidate_pass_rate": candidate_pass_rate,
        }
        return RegressionResult(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            passed=not regressions,
            metric_deltas=deltas,
            regressions=regressions,
            details=details,
        )

    @staticmethod
    def _number(run: EvaluationRunResult, name: str) -> float | None:
        value = run.aggregates.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _require_minimum(
        *,
        name: str,
        actual: float | None,
        minimum: float | None,
        regressions: list[str],
    ) -> None:
        if minimum is None:
            return
        if actual is None:
            regressions.append(f"{name} is required by the policy but unavailable")
        elif actual < minimum:
            regressions.append(f"{name} {actual:.4f} is below minimum {minimum:.4f}")
