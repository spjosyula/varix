"""Terminal reporter — renders a `PipelineAnalysis` as human-readable text.

Internal taxonomy strings (`Classification.*.value`, `LocalizationOutcome.*.value`,
`Confidence.UNAVAILABLE.value`) are translated to plain English at this boundary
so the report is readable without learning varix's vocabulary. Internal names
stay inside the artifact JSON.
"""

from __future__ import annotations

from varix.analysis import ImpactEstimator, ImpactReport, Localizer
from varix.core import (
    Classification,
    Confidence,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    PipelineAnalysis,
)

__all__ = ["render_analysis", "render_explain", "render_impact"]


_LOCALIZATION_LABELS: dict[LocalizationOutcome, str] = {
    LocalizationOutcome.DETERMINISTIC: "deterministic",
    LocalizationOutcome.SOURCE: "source of variance",
    LocalizationOutcome.DOWNSTREAM: "inherited from upstream",
}

_CLASSIFICATION_LABELS: dict[Classification, str] = {
    Classification.PROVIDER_SIDE: "provider rolled the model",
    Classification.TOOL_SIDE: "tool returned different result",
    Classification.ORDERING: "tools called in different order",
    Classification.PROMPT_SIDE: "sampling / temperature",
    Classification.TIME_OR_STATE: "clock or random source",
}

_CONFIDENCE_LABELS: dict[Confidence, str] = {
    Confidence.HIGH: "high",
    Confidence.MEDIUM: "medium",
    Confidence.LOW: "low",
    Confidence.UNAVAILABLE: "cannot verify",
}


def _label_classification(c: Classification | None) -> str:
    return _CLASSIFICATION_LABELS[c] if c is not None else "unknown"


def _headline(analysis: PipelineAnalysis, outcomes: dict[str, LocalizationOutcome]) -> str | None:
    """One-sentence verdict in plain English, or None when inconclusive (n<2)."""
    if analysis.n < 2:
        return None  # WARNING banner already explains the inconclusive case

    n_sources = sum(1 for o in outcomes.values() if o is LocalizationOutcome.SOURCE)
    if n_sources == 0:
        return "every run produced the same output."

    final_step_id = analysis.runs[0].step_runs[-1].step_id if analysis.runs else None
    final_outcome = outcomes.get(final_step_id) if final_step_id is not None else None
    final_varies = (
        final_outcome is not None and final_outcome is not LocalizationOutcome.DETERMINISTIC
    )

    subj = "1 step varies" if n_sources == 1 else f"{n_sources} steps vary"
    if final_varies:
        return f"{subj}, and you get a different final output each run."
    return f"{subj}, but the final output is the same every run."


def render_analysis(analysis: PipelineAnalysis) -> str:
    """Return a plain-text report for `analysis`. ASCII-only, no trailing newline."""
    outcomes = Localizer(metric=ExactMatch()).classify_steps(analysis.runs)

    lines: list[str] = []
    lines.append("=== varix analysis ===")
    lines.append(f"pipeline:    {analysis.pipeline_name}")
    lines.append(f"analysis_id: {analysis.analysis_id}")
    lines.append(f"n:           {analysis.n}")
    lines.append(f"metric:      {analysis.metric_name}")
    lines.append(f"cost:        ${analysis.total_cost.dollars:.4f}")
    headline = _headline(analysis, outcomes)
    if headline is not None:
        lines.append(f"verdict:     {headline}")
    lines.append("")

    if analysis.notes:
        lines.append("WARNING:")
        for note in analysis.notes:
            lines.append(f"  {note}")
        lines.append("")

    findings_by_step: dict[str, list[Finding]] = {}
    for finding in analysis.findings:
        findings_by_step.setdefault(finding.step_id, []).append(finding)

    step_ids = [sr.step_id for sr in analysis.runs[0].step_runs] if analysis.runs else []
    estimator = ImpactEstimator()

    for sid in step_ids:
        outcome = outcomes.get(sid, LocalizationOutcome.DETERMINISTIC)
        line = f"step {sid}: {_LOCALIZATION_LABELS[outcome]}"
        if outcome is LocalizationOutcome.SOURCE:
            impact = estimator.estimate(analysis.runs, sid)
            line += f" [{impact.behavior.value}]"
        lines.append(line)
        for f in findings_by_step.get(sid, []):
            cat = _label_classification(f.classification)
            conf = _CONFIDENCE_LABELS[f.confidence]
            reason = f.reason or ""
            lines.append(f"  -> {cat} ({conf}): {reason}")

    return "\n".join(lines)


def render_explain(analysis: PipelineAnalysis, step_id: str) -> str:
    """Render the evidence trail for `step_id`'s findings from `analysis`.

    Uses only `analysis.findings` and their `Evidence` records — no Localizer,
    no classifier, no re-run. The artifact is the source of truth.
    """
    lines: list[str] = []
    lines.append(f"=== explain {step_id} ===")
    lines.append(f"analysis_id: {analysis.analysis_id}")
    lines.append(f"pipeline:    {analysis.pipeline_name}")
    lines.append("")

    step_findings = [f for f in analysis.findings if f.step_id == step_id]
    if not step_findings:
        lines.append(f"{step_id} has no findings.")
        return "\n".join(lines)

    lines.append(f"{step_id} has {len(step_findings)} finding(s):")
    lines.append("")

    for f in step_findings:
        cat = _label_classification(f.classification)
        lines.append(f"{cat} ({_CONFIDENCE_LABELS[f.confidence]})")
        if f.reason:
            lines.append(f"  reason: {f.reason}")
        if f.evidence:
            lines.append("  evidence:")
            for ev in f.evidence:
                lines.append(f"    [{ev.kind}] {ev.description}")
                for k, v in ev.data.items():
                    lines.append(f"      {k}: {v}")
        lines.append("")

    return "\n".join(lines).rstrip()


def render_impact(analysis: PipelineAnalysis, report: ImpactReport) -> str:
    """Render an `ImpactReport` for a single source step.

    ASCII-only, no trailing newline. Format mirrors `render_explain` for
    consistency across CLI commands.
    """
    lines: list[str] = []
    lines.append(f"=== impact {report.source_step_id} ===")
    lines.append(f"analysis_id: {analysis.analysis_id}")
    lines.append(f"pipeline:    {analysis.pipeline_name}")
    lines.append("")
    lines.append(f"behavior:    {report.behavior.value}")
    lines.append(f"confidence:  {_CONFIDENCE_LABELS[report.confidence]}")
    lines.append(f"reason:      {report.reason}")
    if report.evidence:
        lines.append("")
        lines.append("evidence:")
        for ev in report.evidence:
            lines.append(f"  [{ev.kind}] {ev.description}")
            for k, v in ev.data.items():
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)
