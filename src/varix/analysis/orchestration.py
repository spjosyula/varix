"""Top-level analysis entry point.

Wires together the localizer and every shipped classifier. Returns both
per-step outcomes and the collected findings — callers (the CLI, the
reporter) typically need both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from varix.analysis.classifiers import (
    OrderingClassifier,
    PromptSideClassifier,
    ProviderSideClassifier,
    TimeOrStateClassifier,
    ToolSideClassifier,
)
from varix.analysis.localizer import Localizer
from varix.analysis.registry import ClassifierRegistry
from varix.core import (
    AdapterCapabilities,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Combined verdict from Localizer + classifier registry."""

    outcomes: dict[str, LocalizationOutcome]
    findings: tuple[Finding, ...]


def analyze(
    runs: Sequence[PipelineRun],
    capabilities: AdapterCapabilities,
    metric: VarianceMetric | None = None,
    *,
    replays_by_step: Mapping[str, Sequence[StepRun]] | None = None,
) -> AnalysisResult:
    """Localize each step and run every classifier; return the combined result."""
    actual_metric = metric if metric is not None else ExactMatch()
    localizer = Localizer(metric=actual_metric)
    outcomes = localizer.classify_steps(runs)

    registry = ClassifierRegistry(
        [
            ProviderSideClassifier(),
            ToolSideClassifier(),
            OrderingClassifier(),
            PromptSideClassifier(),
            TimeOrStateClassifier(),
        ]
    )

    replays = replays_by_step or {}
    findings: list[Finding] = []
    for step_id, outcome in outcomes.items():
        step_findings = registry.classify_step(
            step_id=step_id,
            localization=outcome,
            runs=runs,
            replays=list(replays.get(step_id, [])),
            capabilities=capabilities,
            metric=actual_metric,
        )
        findings.extend(step_findings)

    return AnalysisResult(outcomes=dict(outcomes), findings=tuple(findings))
