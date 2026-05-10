"""Terminal reporter — renders a `PipelineAnalysis` as plain-English text.

Internal taxonomy enums are translated to plain English at this boundary, not
in the JSON artifact, so users can read the report without learning varix's
vocabulary while machine consumers keep stable identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from varix.analysis import ImpactBehavior, ImpactEstimator, ImpactReport, Localizer
from varix.core import (
    Classification,
    Confidence,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
)

__all__ = ["render_analysis", "render_explain", "render_impact"]


# Pure, side-effect-free utilities used by the render functions below.
# Kept private; tests reach in via underscore imports.

_ID_DISPLAY_LEN = 8
_OUTPUT_TRUNCATE_LEN = 60


def _short_id(analysis_id: str) -> str:
    """First 8 characters of an analysis_id — enough to disambiguate recent runs."""
    return analysis_id[:_ID_DISPLAY_LEN]


def _truncate(text: str, max_chars: int = _OUTPUT_TRUNCATE_LEN) -> str:
    """Truncate `text` to `max_chars`, appending '...' when cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _format_duration(start: datetime, end: datetime) -> str:
    """Render a wall-clock delta as '0.0s', '14s', or '2m 14s'."""
    total = max((end - start).total_seconds(), 0.0)
    if total < 10:
        return f"{total:.1f}s"
    if total < 60:
        return f"{int(total)}s"
    minutes = int(total // 60)
    seconds = int(total - minutes * 60)
    return f"{minutes}m {seconds}s"


def _format_relative_time(when: datetime, now: datetime) -> str:
    """Render `when` relative to `now` ('just now', '5 minutes ago', '2 hours ago')."""
    seconds = int((now - when).total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit} ago"
    hours = minutes // 60
    if hours < 24:
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} ago"
    days = hours // 24
    unit = "day" if days == 1 else "days"
    return f"{days} {unit} ago"


def _rank_source_step_ids(
    runs: Sequence[PipelineRun],
    outcomes: Mapping[str, LocalizationOutcome],
    pipeline_step_ids: Sequence[str],
) -> list[str]:
    """Order SOURCE step ids by impact (PROPAGATES first), then pipeline order within tier."""
    estimator = ImpactEstimator()
    propagates: list[str] = []
    absorbed: list[str] = []
    for sid in pipeline_step_ids:
        if outcomes.get(sid) is not LocalizationOutcome.SOURCE:
            continue
        report = estimator.estimate(runs, sid)
        if report.behavior is ImpactBehavior.PROPAGATES:
            propagates.append(sid)
        else:
            absorbed.append(sid)
    return propagates + absorbed


def _format_receipt(analysis: PipelineAnalysis) -> str:
    """Single-line receipt: 'n=3 | $0.0007 | 14s | analysis abc12345'."""
    duration = _format_duration(analysis.started_at, analysis.finished_at)
    return (
        f"n={analysis.n} | ${analysis.total_cost.dollars:.4f} | "
        f"{duration} | analysis {_short_id(analysis.analysis_id)}"
    )


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

_IMPACT_SUFFIX_LABELS: dict[ImpactBehavior, str] = {
    ImpactBehavior.PROPAGATES: "changes the final output",
    ImpactBehavior.ABSORBED: "absorbed before final output",
}


def _label_classification(c: Classification | None) -> str:
    return _CLASSIFICATION_LABELS[c] if c is not None else "unknown"


def _paren_confidence(c: Confidence) -> str:
    # "cannot verify" is already a verb phrase; the others need "confidence" appended.
    if c is Confidence.UNAVAILABLE:
        return _CONFIDENCE_LABELS[c]
    return f"{_CONFIDENCE_LABELS[c]} confidence"


def _impact_verdict(report: ImpactReport) -> str:
    sid = report.source_step_id
    conf = _paren_confidence(report.confidence)
    if report.confidence is Confidence.UNAVAILABLE:
        return f"impact of {sid} could not be determined ({conf})."
    if report.behavior is ImpactBehavior.PROPAGATES:
        return f"{sid} changes the final output of every run ({conf})."
    return f"{sid}'s variance is absorbed before the final output ({conf})."


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
            line += f" ({_IMPACT_SUFFIX_LABELS[impact.behavior]})"
        lines.append(line)
        for f in findings_by_step.get(sid, []):
            cat = _label_classification(f.classification)
            reason = f.reason or ""
            lines.append(f"  - {cat} ({_paren_confidence(f.confidence)}): {reason}")

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
        lines.append(f"{cat} ({_paren_confidence(f.confidence)})")
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
    """Render an `ImpactReport` for a single source step. ASCII-only, no trailing newline."""
    lines: list[str] = []
    lines.append(f"=== impact {report.source_step_id} ===")
    lines.append(f"analysis_id: {analysis.analysis_id}")
    lines.append(f"pipeline:    {analysis.pipeline_name}")
    lines.append("")
    lines.append(f"verdict:     {_impact_verdict(report)}")
    lines.append(f"reason:      {report.reason}")
    if report.evidence:
        lines.append("")
        lines.append("evidence:")
        for ev in report.evidence:
            lines.append(f"  [{ev.kind}] {ev.description}")
            for k, v in ev.data.items():
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)
