"""Terminal reporter — renders a `PipelineAnalysis` as human-readable text."""

from __future__ import annotations

from varix.analysis import ImpactEstimator, ImpactReport, Localizer
from varix.core import ExactMatch, Finding, LocalizationOutcome, PipelineAnalysis

__all__ = ["render_analysis", "render_explain", "render_impact"]


def render_analysis(analysis: PipelineAnalysis) -> str:
    """Return a plain-text report for `analysis`. ASCII-only, no trailing newline."""
    lines: list[str] = []
    lines.append("=== varix analysis ===")
    lines.append(f"pipeline:    {analysis.pipeline_name}")
    lines.append(f"analysis_id: {analysis.analysis_id}")
    lines.append(f"n:           {analysis.n}")
    lines.append(f"metric:      {analysis.metric_name}")
    lines.append(f"cost:        ${analysis.total_cost.dollars:.4f}")
    lines.append("")

    outcomes = Localizer(metric=ExactMatch()).classify_steps(analysis.runs)

    findings_by_step: dict[str, list[Finding]] = {}
    for finding in analysis.findings:
        findings_by_step.setdefault(finding.step_id, []).append(finding)

    step_ids = [sr.step_id for sr in analysis.runs[0].step_runs] if analysis.runs else []
    estimator = ImpactEstimator()

    for sid in step_ids:
        outcome = outcomes.get(sid, LocalizationOutcome.DETERMINISTIC)
        line = f"step {sid}: {outcome.value}"
        if outcome is LocalizationOutcome.SOURCE:
            impact = estimator.estimate(analysis.runs, sid)
            line += f" [{impact.behavior.value}]"
        lines.append(line)
        for f in findings_by_step.get(sid, []):
            cat = f.classification.value if f.classification is not None else "unknown"
            conf = f.confidence.value
            reason = f.reason or ""
            lines.append(f"  -> {cat} ({conf}): {reason}")

    n_findings = len(analysis.findings)
    n_sources = sum(1 for o in outcomes.values() if o is LocalizationOutcome.SOURCE)
    lines.append("")
    lines.append(f"{n_findings} finding(s), {n_sources} source step(s)")

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
        cat = f.classification.value if f.classification is not None else "unknown"
        lines.append(f"{cat} ({f.confidence.value})")
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
    lines.append(f"confidence:  {report.confidence.value}")
    lines.append(f"reason:      {report.reason}")
    if report.evidence:
        lines.append("")
        lines.append("evidence:")
        for ev in report.evidence:
            lines.append(f"  [{ev.kind}] {ev.description}")
            for k, v in ev.data.items():
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)
