"""Time/state heuristic — flags obvious time- or RNG-derived markers."""

from __future__ import annotations

from collections.abc import Sequence

from varix.core import (
    AdapterCapabilities,
    Finding,
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
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        return []
