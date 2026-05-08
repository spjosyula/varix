"""Prompt-side residual classifier — fires only when no other classifier fits."""

from __future__ import annotations

from collections.abc import Sequence

from varix.core import (
    AdapterCapabilities,
    Finding,
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
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        return []
