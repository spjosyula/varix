"""Terminal reporter — renders a `PipelineAnalysis` as human-readable text."""

from __future__ import annotations

from varix.analysis import Localizer
from varix.core import ExactMatch, Finding, LocalizationOutcome, PipelineAnalysis


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

    for sid in step_ids:
        outcome = outcomes.get(sid, LocalizationOutcome.DETERMINISTIC)
        lines.append(f"step {sid}: {outcome.value}")
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
