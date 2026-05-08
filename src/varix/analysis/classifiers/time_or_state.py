"""Time/state heuristic classifier — flags obvious clock- or RNG-derived markers.

Conservative starter heuristics:
  - Tool names containing time/RNG-suggestive substrings (`time`, `now`,
    `clock`, `today`, `date`, `uuid`, `random`, `rand`).
  - More than one unique ISO-8601-shaped timestamp observed across runs.

Fires LOW because heuristics are noisy. The reason explicitly admits the
classifier is not certain.
"""

from __future__ import annotations

from collections.abc import Sequence

from varix.analysis._helpers import (
    gather_step_runs,
    outputs_differ,
    time_or_state_markers,
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


class TimeOrStateClassifier:
    """Heuristic for clock/RNG-driven variance (timestamps, named time tools)."""

    def name(self) -> str:
        return "time_or_state"

    def classify(
        self,
        step_id: str,
        localization: LocalizationOutcome,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        observations = gather_step_runs(step_id, runs, replays)
        if len(observations) < 2:
            return []
        if not outputs_differ(observations, metric):
            return []

        markers = time_or_state_markers(observations)
        if not markers:
            return []

        return [
            Finding(
                step_id=step_id,
                localization=localization,
                confidence=Confidence.LOW,
                metric_name=metric.name(),
                classification=Classification.TIME_OR_STATE,
                reason="heuristic match — could be clock or RNG; not definitive",
                evidence=(
                    Evidence(
                        kind="time_or_state_markers",
                        description=f"{len(markers)} heuristic marker(s) observed",
                        data={"markers": markers},
                    ),
                ),
            )
        ]
