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

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    Confidence,
    Evidence,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    StepRun,
    StructuralMismatch,
    VarianceMetric,
)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Combined verdict from Localizer + classifier registry.

    `notes` carries classifier-derived warnings (e.g. partial-data exclusions)
    that callers should surface alongside their own notes (budget, n<2, etc.).
    """

    outcomes: dict[str, LocalizationOutcome]
    findings: tuple[Finding, ...]
    notes: tuple[str, ...] = ()


def infer_capabilities(analysis: PipelineAnalysis) -> AdapterCapabilities:
    """Best-effort reconstruction of `AdapterCapabilities` from a stored analysis.

    Used to make legacy schema-0.1 artifacts replayable: those don't carry
    `capabilities`, so we look at the runs themselves to decide what the
    original adapter must have exposed. Heuristic only — when the recorded
    field is present, prefer that over this.

    `supports_replay` is best-effort: an adapter that *could* replay but never
    did leaves no signal in the runs. The heuristic infers True only when
    `step_replays` is populated. This is safe because no current classifier
    branches on `supports_replay`; if that ever changes, replay correctness
    against legacy artifacts will need stronger evidence than a heuristic.
    """
    exposes_fingerprint = any(
        sr.provider_metadata is not None and "system_fingerprint" in sr.provider_metadata
        for run in analysis.runs
        for sr in run.step_runs
    )
    exposes_tool_calls = any(
        bool(sr.tool_calls) for run in analysis.runs for sr in run.step_runs
    )
    supports_replay = bool(analysis.step_replays)
    return AdapterCapabilities(
        exposes_fingerprint=exposes_fingerprint,
        exposes_tool_calls=exposes_tool_calls,
        supports_replay=supports_replay,
    )


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
        step_replays = list(replays.get(step_id, []))
        step_findings = registry.classify_step(
            step_id=step_id,
            localization=outcome,
            runs=runs,
            replays=step_replays,
            capabilities=capabilities,
            metric=actual_metric,
        )
        # If replays (held at fixed inputs) still vary, the step is its own
        # source. Surface this via evidence — localization stays DOWNSTREAM.
        if (
            outcome is LocalizationOutcome.DOWNSTREAM
            and len(step_replays) >= 2
            and _replays_show_independent_variance(step_replays, actual_metric)
        ):
            if step_findings:
                # Attach to the first finding only; otherwise the footer
                # would repeat in `varix explain`.
                step_findings = [
                    _attach_replay_disambiguation(
                        step_findings[0], step_replays, actual_metric
                    ),
                    *step_findings[1:],
                ]
            else:
                # Independent variance with no classifier match (e.g. LLM
                # drift at temp=0 with no tools). Synthesize a placeholder.
                step_findings = [
                    _synthesize_replay_promoted_finding(
                        step_id, step_replays, actual_metric
                    )
                ]
        findings.extend(step_findings)

    capped_findings, cap_notes = _cap_confidence_for_weak_evidence(findings, len(runs))

    return AnalysisResult(
        outcomes=dict(outcomes),
        findings=capped_findings,
        notes=(*_summarize_exclusions(capped_findings), *cap_notes),
    )


def _replays_show_independent_variance(
    replays: Sequence[StepRun], metric: VarianceMetric
) -> bool:
    """True if the replays (sharing fixed inputs) produced varying outputs."""
    first = replays[0].output
    return any(not metric.equivalent(first, r.output) for r in replays[1:])


def _attach_replay_disambiguation(
    finding: Finding, replays: Sequence[StepRun], metric: VarianceMetric
) -> Finding:
    """Append a `replay_disambiguation` evidence record to `finding`."""
    return dataclasses.replace(
        finding, evidence=(*finding.evidence, _replay_evidence(replays, metric))
    )


def _synthesize_replay_promoted_finding(
    step_id: str, replays: Sequence[StepRun], metric: VarianceMetric
) -> Finding:
    """Placeholder finding when replay proves independent variance but no
    classifier signal matched."""
    return Finding(
        step_id=step_id,
        localization=LocalizationOutcome.DOWNSTREAM,
        confidence=Confidence.MEDIUM,
        metric_name=metric.name(),
        classification=None,
        reason="varies under fixed inputs but no classifier signal matched",
        evidence=(_replay_evidence(replays, metric),),
    )


def _replay_evidence(replays: Sequence[StepRun], metric: VarianceMetric) -> Evidence:
    unique: list[Any] = []
    for r in replays:
        if not any(metric.equivalent(r.output, u) for u in unique):
            unique.append(r.output)
    return Evidence(
        kind="replay_disambiguation",
        description=(
            f"Replayed {len(replays)} times with fixed inputs; "
            f"observed {len(unique)} unique outputs."
        ),
        data={"replay_count": len(replays), "unique_output_count": len(unique)},
    )


def _cap_confidence_for_weak_evidence(
    findings: Sequence[Finding], n_runs: int
) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Downgrade HIGH to MEDIUM when N<3 runs back the finding.

    Findings carrying `replay_disambiguation` evidence are exempt — their
    replays add fixed-input observations beyond the runs themselves. Note
    is emitted only when at least one finding was actually capped.
    """
    if n_runs >= 3:
        return tuple(findings), ()

    out: list[Finding] = []
    any_capped = False
    for f in findings:
        has_replay_ev = any(ev.kind == "replay_disambiguation" for ev in f.evidence)
        if f.confidence is Confidence.HIGH and not has_replay_ev:
            out.append(dataclasses.replace(f, confidence=Confidence.MEDIUM))
            any_capped = True
        else:
            out.append(f)

    if not any_capped:
        return tuple(out), ()
    return tuple(out), (
        f"only {n_runs} run(s) available - HIGH-confidence findings reported as "
        "MEDIUM. Re-run with --n 3 or higher for stronger statistical signal.",
    )


def _summarize_exclusions(findings: Sequence[Finding]) -> tuple[str, ...]:
    """One-line analysis-level summary when any classifier excluded observations.

    The detail lives in per-finding `excluded_runs` evidence (surfaced by
    `varix explain`); this note is the discovery-level breadcrumb pointing
    users there.
    """
    total = sum(
        len(ev.data.get("excluded", []))
        for f in findings
        for ev in f.evidence
        if ev.kind == "excluded_runs"
    )
    if total == 0:
        return ()
    unit = "observation" if total == 1 else "observations"
    return (
        f"{total} {unit} excluded from classification due to missing data - "
        "see varix explain for details",
    )
