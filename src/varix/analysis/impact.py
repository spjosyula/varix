"""Downstream impact estimator.

For a given source step, decide whether variance at that step reaches the
final pipeline output (`PROPAGATES`) or is absorbed by downstream steps
(`ABSORBED`). Reports are computed lazily and never persisted — the
underlying runs in the artifact are the source of truth.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from varix.analysis._helpers import outputs_differ
from varix.core import (
    Confidence,
    Evidence,
    ExactMatch,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


class ImpactBehavior(enum.Enum):
    """How a source step's variance manifests at the pipeline's final output."""

    ABSORBED = "absorbed"
    PROPAGATES = "propagates"


@dataclass(frozen=True, slots=True)
class ImpactReport:
    """Verdict on whether variance at `source_step_id` reaches the final step."""

    source_step_id: str
    final_step_id: str | None
    behavior: ImpactBehavior
    confidence: Confidence
    reason: str
    evidence: tuple[Evidence, ...] = ()


class ImpactEstimator:
    """Estimate downstream impact of variance at a single source step."""

    def __init__(self, metric: VarianceMetric | None = None) -> None:
        self._metric: VarianceMetric = metric if metric is not None else ExactMatch()

    def estimate(self, runs: Sequence[PipelineRun], source_step_id: str) -> ImpactReport:
        """Decide ABSORBED vs PROPAGATES for `source_step_id` under `runs`."""
        if len(runs) < 2:
            return ImpactReport(
                source_step_id=source_step_id,
                final_step_id=None,
                behavior=ImpactBehavior.ABSORBED,
                confidence=Confidence.UNAVAILABLE,
                reason="insufficient runs to compare (need at least 2)",
            )

        source_observations = _gather(runs, source_step_id)
        if len(source_observations) < 2:
            return ImpactReport(
                source_step_id=source_step_id,
                final_step_id=None,
                behavior=ImpactBehavior.ABSORBED,
                confidence=Confidence.UNAVAILABLE,
                reason=f"step {source_step_id!r} did not appear in two or more runs",
            )

        final_step_id = runs[0].step_runs[-1].step_id if runs[0].step_runs else None

        if not outputs_differ(source_observations, self._metric):
            return ImpactReport(
                source_step_id=source_step_id,
                final_step_id=final_step_id,
                behavior=ImpactBehavior.ABSORBED,
                confidence=Confidence.HIGH,
                reason=(
                    f"output of {source_step_id} is stable across runs; no variance to propagate"
                ),
            )

        if final_step_id is None or final_step_id == source_step_id:
            return ImpactReport(
                source_step_id=source_step_id,
                final_step_id=final_step_id,
                behavior=ImpactBehavior.ABSORBED,
                confidence=Confidence.HIGH,
                reason=f"{source_step_id} is the final step — no downstream to propagate to",
            )

        final_observations = _gather(runs, final_step_id)
        if len(final_observations) < 2:
            return ImpactReport(
                source_step_id=source_step_id,
                final_step_id=final_step_id,
                behavior=ImpactBehavior.ABSORBED,
                confidence=Confidence.UNAVAILABLE,
                reason=f"final step {final_step_id!r} did not appear in two or more runs",
            )

        source_unique = _count_unique(source_observations, self._metric)
        final_unique = _count_unique(final_observations, self._metric)

        evidence = (
            Evidence(
                kind="source_to_final_diff",
                description="source step output diversity vs. final step output diversity",
                data={
                    "source_step": source_step_id,
                    "final_step": final_step_id,
                    "source_unique_outputs": source_unique,
                    "final_unique_outputs": final_unique,
                },
            ),
        )

        if outputs_differ(final_observations, self._metric):
            return ImpactReport(
                source_step_id=source_step_id,
                final_step_id=final_step_id,
                behavior=ImpactBehavior.PROPAGATES,
                confidence=Confidence.HIGH,
                reason=f"variance at {source_step_id} reached final step {final_step_id}",
                evidence=evidence,
            )

        return ImpactReport(
            source_step_id=source_step_id,
            final_step_id=final_step_id,
            behavior=ImpactBehavior.ABSORBED,
            confidence=Confidence.HIGH,
            reason=(
                f"variance at {source_step_id} did not reach final step {final_step_id}; "
                "downstream absorbed it"
            ),
            evidence=evidence,
        )


def _gather(runs: Sequence[PipelineRun], step_id: str) -> list[StepRun]:
    out: list[StepRun] = []
    for run in runs:
        for sr in run.step_runs:
            if sr.step_id == step_id:
                out.append(sr)
                break
    return out


def _count_unique(observations: Sequence[StepRun], metric: VarianceMetric) -> int:
    unique: list[object] = []
    for sr in observations:
        if not any(metric.equivalent(sr.output, u) for u in unique):
            unique.append(sr.output)
    return len(unique)
