"""Prompt-side residual classifier — fires only when no other classifier fits.

The "by elimination" verdict: when output varies at a SOURCE step, capabilities
are sufficient to rule out provider/tool/ordering/time-state, AND none of those
signals are present, the most likely explanation is prompt or sampling
nondeterminism. MEDIUM confidence: it's reasoned-by-exclusion, not direct
evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from varix.analysis._helpers import (
    fingerprints_differ,
    gather_step_runs,
    outputs_differ,
    sequences_differ_with_same_multiset,
    time_or_state_markers,
    tool_results_differ_for_same_key,
)
from varix.core import (
    AdapterCapabilities,
    Classification,
    Confidence,
    Evidence,
    Finding,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


class PromptSideClassifier:
    """Residual: variance attributable to prompt/sampling once others rule out."""

    def name(self) -> str:
        return "prompt_side"

    def classify(
        self,
        step_id: str,
        localization: LocalizationOutcome,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        # Residual only applies to SOURCE steps. DOWNSTREAM variance is
        # upstream's responsibility; DETERMINISTIC has nothing to explain.
        if localization is not LocalizationOutcome.SOURCE:
            return []

        observations = gather_step_runs(step_id, runs, replays)
        if len(observations) < 2:
            return []
        if not outputs_differ(observations, metric):
            return []

        # Don't claim certainty when primary classifiers couldn't even check.
        if not (capabilities.exposes_fingerprint and capabilities.exposes_tool_calls):
            return []

        if (
            fingerprints_differ(observations)
            or tool_results_differ_for_same_key(observations, metric)
            or sequences_differ_with_same_multiset(observations)
            or time_or_state_markers(observations)
        ):
            return []

        return [
            Finding(
                step_id=step_id,
                localization=localization,
                confidence=Confidence.MEDIUM,
                metric_name=metric.name(),
                classification=Classification.PROMPT_SIDE,
                reason=(
                    "output varied across runs with no provider, tool, ordering, "
                    "or time/state signal — likely prompt or sampling"
                ),
                evidence=(
                    Evidence(
                        kind="residual_output_variance",
                        description="output varied with no upstream or tooling explanation",
                        data={"observation_count": len(observations)},
                    ),
                ),
            )
        ]
