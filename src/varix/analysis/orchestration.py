"""Top-level analysis entry point.

Wires together the localizer and every shipped classifier. Returns both
per-step outcomes and the collected findings — callers (the CLI, the
reporter) typically need both.

Acts as the gatekeeper for structural validity: if the N runs disagree on
the step graph, it raises `StructuralMismatch` before any localization or
classification happens. There is no honest analysis to produce when the
runs aren't comparable.
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
    StructuralMismatch,
    VarianceMetric,
)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Combined verdict from Localizer + classifier registry."""

    outcomes: dict[str, LocalizationOutcome]
    findings: tuple[Finding, ...]


def detect_structural_mismatch(runs: Sequence[PipelineRun]) -> None:
    """Raise `StructuralMismatch` if `runs` disagree on the step-id sequence.

    With fewer than two runs there is nothing to compare; this is a no-op.
    """
    if len(runs) < 2:
        return
    expected = tuple(sr.step_id for sr in runs[0].step_runs)
    for index, run in enumerate(runs[1:], start=1):
        actual = tuple(sr.step_id for sr in run.step_runs)
        if actual != expected:
            raise StructuralMismatch(
                "pipeline structure varied across runs: "
                f"run 0 has steps {list(expected)}, "
                f"run {index} has steps {list(actual)}"
            )


def analyze(
    runs: Sequence[PipelineRun],
    capabilities: AdapterCapabilities,
    metric: VarianceMetric | None = None,
    *,
    replays_by_step: Mapping[str, Sequence[StepRun]] | None = None,
) -> AnalysisResult:
    """Localize each step and run every classifier; return the combined result.

    Raises `StructuralMismatch` when the N runs disagree on the step graph.
    """
    detect_structural_mismatch(runs)

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
