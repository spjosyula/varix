"""Tool-side classifier — same tool, same args, different result across replays."""

from __future__ import annotations

from collections.abc import Sequence

from varix.core import (
    AdapterCapabilities,
    Finding,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


class ToolSideClassifier:
    """Detect nondeterministic tool invocations."""

    def name(self) -> str:
        return "tool_side"

    def classify(
        self,
        step_id: str,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        return []
