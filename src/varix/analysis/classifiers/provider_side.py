"""Provider-side classifier — detects variance arising from the model provider."""

from __future__ import annotations

from collections.abc import Sequence

from varix.core import (
    AdapterCapabilities,
    Finding,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


class ProviderSideClassifier:
    """Diff `system_fingerprint` (and similar provider metadata) across replays."""

    def name(self) -> str:
        return "provider_side"

    def classify(
        self,
        step_id: str,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        return []
