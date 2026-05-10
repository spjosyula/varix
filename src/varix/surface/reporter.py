"""Terminal reporter — renders a `PipelineAnalysis` as plain-English text.

Internal taxonomy enums are translated to plain English at this boundary, not
in the JSON artifact, so users can read the report without learning varix's
vocabulary while machine consumers keep stable identifiers.
"""

from __future__ import annotations

import os
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


def _display_pipeline_name(name: str) -> str:
    """Strip the directory prefix when name is a file path; pass import strings through.

    `agent.py` reads better than `C:\\Users\\...\\tmp\\agent.py` in the headline.
    Import strings like `pkg.mod:object` have no path separators and are unchanged.
    """
    if "/" in name or "\\" in name:
        return os.path.basename(name)
    return name


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


# Long-form labels — used by render_explain / render_impact.
_CLASSIFICATION_LABELS: dict[Classification, str] = {
    Classification.PROVIDER_SIDE: "provider rolled the model",
    Classification.TOOL_SIDE: "tool returned different result",
    Classification.ORDERING: "tools called in different order",
    Classification.PROMPT_SIDE: "sampling / temperature",
    Classification.TIME_OR_STATE: "clock or random source",
}

# Short-form labels — used inline by render_analysis source lines.
_CLASSIFICATION_SHORT: dict[Classification, str] = {
    Classification.PROVIDER_SIDE: "provider-side",
    Classification.TOOL_SIDE: "tool-side",
    Classification.ORDERING: "ordering",
    Classification.PROMPT_SIDE: "prompt-side",
    Classification.TIME_OR_STATE: "time/state",
}

_CONFIDENCE_LABELS: dict[Confidence, str] = {
    Confidence.HIGH: "high",
    Confidence.MEDIUM: "medium",
    Confidence.LOW: "low",
    Confidence.UNAVAILABLE: "cannot verify",
}

# Inline impact suffix used in source-step lines.
_IMPACT_SUFFIX_SHORT: dict[ImpactBehavior, str] = {
    ImpactBehavior.PROPAGATES: "propagates downstream",
    ImpactBehavior.ABSORBED: "absorbed downstream",
}


def _label_classification(c: Classification | None) -> str:
    return _CLASSIFICATION_LABELS[c] if c is not None else "unknown"


def _paren_confidence(c: Confidence) -> str:
    # "cannot verify" is already a verb phrase; the others need "confidence" appended.
    if c is Confidence.UNAVAILABLE:
        return _CONFIDENCE_LABELS[c]
    return f"{_CONFIDENCE_LABELS[c]} confidence"


def _count_modal_final_output(
    runs: Sequence[PipelineRun],
    final_step_id: str,
    metric: ExactMatch,
) -> int:
    """Largest equivalence-class size among final-step outputs across runs.

    With finals (c1, c1, c2, c3, c4): modal class is c1 (size 2) → 2 runs were
    'absorbed' to the modal answer, the other 3 reached different answers.
    """
    finals: list[object] = []
    for r in runs:
        for sr in r.step_runs:
            if sr.step_id == final_step_id:
                finals.append(sr.output)
                break
    if not finals:
        return 0
    classes: list[list[object]] = []
    for v in finals:
        for c in classes:
            if metric.equivalent(v, c[0]):
                c.append(v)
                break
        else:
            classes.append([v])
    return max(len(c) for c in classes)


def _impact_source_unique(report: ImpactReport) -> int:
    """Pull source_unique_outputs from the impact evidence; 0 when absent."""
    for ev in report.evidence:
        if ev.kind == "source_to_final_diff":
            return int(ev.data.get("source_unique_outputs", 0))
    return 0


# Confidence ranking — used to pick the primary finding when a step has several.
_CONFIDENCE_ORDER: dict[Confidence, int] = {
    Confidence.HIGH: 0,
    Confidence.MEDIUM: 1,
    Confidence.LOW: 2,
    Confidence.UNAVAILABLE: 3,
}


def _classification_short(c: Classification | None) -> str:
    """Short hyphenated label for a classification ('prompt-side', 'provider-side', ...)."""
    return _CLASSIFICATION_SHORT[c] if c is not None else "unclassified"


def _primary_finding(findings: list[Finding]) -> Finding | None:
    """Pick the highest-confidence finding for a step; tiebreak by classifier order."""
    if not findings:
        return None
    return min(findings, key=lambda f: _CONFIDENCE_ORDER[f.confidence])


def _environmental_findings(
    outcomes: Mapping[str, LocalizationOutcome],
    findings: Sequence[Finding],
) -> list[Finding]:
    """HIGH-confidence findings on DETERMINISTIC steps — variance varix saw but outputs absorbed."""
    return [
        f
        for f in findings
        if f.confidence is Confidence.HIGH
        and outcomes.get(f.step_id) is LocalizationOutcome.DETERMINISTIC
    ]


def _env_finding_detail(finding: Finding) -> str:
    """One-line inline detail for an environmental finding (currently provider-side only)."""
    if finding.classification is Classification.PROVIDER_SIDE:
        for ev in finding.evidence:
            if ev.kind == "fingerprint_diff":
                unique = [str(u) for u in ev.data.get("unique", [])]
                if len(unique) == 2:
                    return f"fingerprint changed ({unique[0]} -> {unique[1]})"
                if unique:
                    return f"fingerprint changed ({', '.join(unique)})"
        return "fingerprint changed"
    return _classification_short(finding.classification)


def _render_warning_block(notes: Sequence[str]) -> list[str]:
    """WARNING banner shown above the headline. Empty when no notes."""
    if not notes:
        return []
    lines = ["WARNING:"]
    for note in notes:
        lines.append(f"  {note}")
    lines.append("")
    return lines


def _render_source_lines(
    sources: list[str],
    impacts: dict[str, ImpactBehavior],
    findings_by_step: dict[str, list[Finding]],
) -> list[str]:
    """Indented per-source lines: '  step `<id>`  ->  <classification>, <impact>'."""
    if not sources:
        return []
    pad = max(len(sid) for sid in sources)
    out: list[str] = []
    for sid in sources:
        primary = _primary_finding(findings_by_step.get(sid, []))
        if primary is None:
            class_label = "unclassified"
        elif primary.confidence is Confidence.UNAVAILABLE:
            class_label = "unclassified (cannot verify)"
        else:
            class_label = _classification_short(primary.classification)
        impact_label = _IMPACT_SUFFIX_SHORT[impacts[sid]]
        padded_id = f"`{sid}`".ljust(pad + 2)
        out.append(f"  step {padded_id}  ->  {class_label}, {impact_label}")
    return out


def _render_next_block(
    sources: list[str],
    impacts: dict[str, ImpactBehavior],
) -> list[str]:
    """`Next:` block: educational descriptions for single source, bare commands for many."""
    if not sources:
        return []
    out = ["Next:"]
    if len(sources) == 1:
        sid = sources[0]
        impact_cmd = f"varix impact {sid}"
        explain_cmd = f"varix explain {sid}"
        pad = max(len(impact_cmd), len(explain_cmd)) + 4
        out.append(f"  {impact_cmd.ljust(pad)}see how much this changes your output")
        out.append(f"  {explain_cmd.ljust(pad)}see the evidence varix used")
        return out
    # Multi-source: one line per source, choose verb by impact.
    for sid in sources:
        verb = "impact" if impacts[sid] is ImpactBehavior.PROPAGATES else "explain"
        out.append(f"  varix {verb} {sid}")
    return out


def render_analysis(analysis: PipelineAnalysis) -> str:
    """Return a plain-text report for `analysis`. ASCII-only, no trailing newline.

    Four cases:
      1. n<2: inconclusive — WARNING banner + receipt.
      2. n>=2, no sources, no env findings: 'No nondeterminism found' + receipt.
      3. n>=2, no sources but HIGH-confidence env findings: 'outputs stable but
         routing varied' + finding detail + receipt + Next.
      4. n>=2, has sources: 'Found N sources' + ranked source lines + receipt + Next.
    """
    outcomes = Localizer(metric=ExactMatch()).classify_steps(analysis.runs)
    step_ids = [sr.step_id for sr in analysis.runs[0].step_runs] if analysis.runs else []
    pipeline_label = _display_pipeline_name(analysis.pipeline_name)

    findings_by_step: dict[str, list[Finding]] = {}
    for f in analysis.findings:
        findings_by_step.setdefault(f.step_id, []).append(f)

    lines: list[str] = []
    lines.extend(_render_warning_block(analysis.notes))
    if analysis.n < 2:
        lines.append(_format_receipt(analysis))
        return "\n".join(lines)

    sources = _rank_source_step_ids(analysis.runs, outcomes, step_ids)
    estimator = ImpactEstimator()
    impacts = {sid: estimator.estimate(analysis.runs, sid).behavior for sid in sources}

    if sources:
        plural = "s" if len(sources) != 1 else ""
        lines.append(
            f"Found {len(sources)} source{plural} of nondeterminism in {pipeline_label}."
        )
        lines.append("")
        lines.extend(_render_source_lines(sources, impacts, findings_by_step))
        lines.append("")
        lines.append(_format_receipt(analysis))
        lines.append("")
        lines.extend(_render_next_block(sources, impacts))
        return "\n".join(lines)

    env_findings = _environmental_findings(outcomes, analysis.findings)
    if env_findings:
        lines.append(f"Your pipeline's outputs were stable across {analysis.n} runs.")
        lines.append("")
        lines.append("  varix did detect provider routing changes during the runs:")
        for f in env_findings:
            lines.append(f"    step `{f.step_id}`  ->  {_env_finding_detail(f)}")
        lines.append("")
        lines.append("  This didn't affect your output - but it means the provider routed")
        lines.append("  your requests to different model infrastructure. Future runs may")
        lines.append("  behave differently.")
        lines.append("")
        lines.append(_format_receipt(analysis))
        lines.append("")
        lines.append("Next:")
        for f in env_findings:
            cmd = f"varix explain {f.step_id}"
            lines.append(f"  {cmd.ljust(len(cmd) + 6)}see the fingerprint evidence")
        return "\n".join(lines)
    lines.append(f"No nondeterminism found in {pipeline_label}.")
    lines.append("")
    lines.append(_format_receipt(analysis))
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
    """Render an `ImpactReport` for a single source step. ASCII-only, no trailing newline.

    Four cases:
      - UNAVAILABLE: 'Impact of <step> could not be determined' + reason.
      - <step> IS the final step: definitional sentence (variance IS the output's variance).
      - PROPAGATES: '<step>'s variance changes the final output' + ratio prose.
      - ABSORBED downstream: '<step>'s variance is absorbed' + diversity prose.
    """
    sid = report.source_step_id
    short_id = _short_id(analysis.analysis_id)
    lines: list[str] = []

    if report.confidence is Confidence.UNAVAILABLE:
        lines.append(f"Impact of {sid} could not be determined.")
        lines.append("")
        lines.append(f"  {report.reason}")
        lines.append("")
        lines.append(f"confidence: {_CONFIDENCE_LABELS[Confidence.UNAVAILABLE]}")
        lines.append(f"analysis: {short_id}")
        return "\n".join(lines)

    is_final_step = (
        report.behavior is ImpactBehavior.ABSORBED
        and (report.final_step_id is None or report.final_step_id == sid)
    )

    if is_final_step:
        lines.append(
            f"{sid} is the final step in the pipeline; "
            f"its variance IS the final output's variance."
        )
    elif report.behavior is ImpactBehavior.PROPAGATES:
        n = analysis.n
        modal = _count_modal_final_output(analysis.runs, report.final_step_id or sid, ExactMatch())
        lines.append(f"{sid}'s variance changes the final output.")
        lines.append("")
        if modal <= 1:
            lines.append(f"  {n} of {n} runs reached a different final answer.")
        else:
            different = n - modal
            lines.append(
                f"  {different} of {n} runs reached a different final answer; "
                f"{modal} were absorbed."
            )
    else:
        # ABSORBED, downstream pipeline normalized the variance.
        n = analysis.n
        source_unique = _impact_source_unique(report)
        lines.append(f"{sid}'s variance is absorbed before the final output.")
        lines.append("")
        lines.append(
            f"  {source_unique} different {sid} outputs produced only 1 final answer "
            f"across {n} runs."
        )
        lines.append("  The downstream pipeline normalized the differences.")

    lines.append("")
    lines.append(f"confidence: {_CONFIDENCE_LABELS[report.confidence]}")
    lines.append(f"analysis: {short_id}")
    lines.append("")
    lines.append("Next:")
    cmd = f"varix explain {sid}"
    lines.append(f"  {cmd.ljust(len(cmd) + 6)}see the evidence varix used")
    return "\n".join(lines)
