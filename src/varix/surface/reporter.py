"""Terminal reporter — renders a `PipelineAnalysis` as plain-English text.

Internal taxonomy enums are translated to plain English at this boundary, not
in the JSON artifact, so users can read the report without learning varix's
vocabulary while machine consumers keep stable identifiers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from varix.analysis import ImpactBehavior, ImpactEstimator, ImpactReport, Localizer
from varix.core import (
    Classification,
    Confidence,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    StepRun,
)

__all__ = ["render_analysis", "render_explain", "render_impact", "render_list", "render_replay"]


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

    Hand-rolled instead of `os.path.basename`: that one only honors the host
    platform's separator, so Windows paths leak through unchanged on Linux CI.
    Treating both `/` and `\\` as separators is correct for our two input shapes
    (paths from either platform; import strings have neither).
    """
    last_sep = max(name.rfind("/"), name.rfind("\\"))
    if last_sep >= 0:
        return name[last_sep + 1 :]
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


def _has_replay_disambiguation(findings: Sequence[Finding]) -> bool:
    """True if any finding for this step carries `replay_disambiguation` evidence."""
    return any(
        ev.kind == "replay_disambiguation"
        for f in findings
        for ev in f.evidence
    )


def _rank_source_step_ids(
    runs: Sequence[PipelineRun],
    outcomes: Mapping[str, LocalizationOutcome],
    pipeline_step_ids: Sequence[str],
    findings_by_step: Mapping[str, Sequence[Finding]],
) -> list[str]:
    """Order effective-source step ids by impact (PROPAGATES first), then by
    pipeline order. Effective-source = localizer-SOURCE OR DOWNSTREAM with
    `replay_disambiguation` evidence."""
    estimator = ImpactEstimator()
    propagates: list[str] = []
    absorbed: list[str] = []
    for sid in pipeline_step_ids:
        outcome = outcomes.get(sid)
        is_promoted = (
            outcome is LocalizationOutcome.DOWNSTREAM
            and _has_replay_disambiguation(findings_by_step.get(sid, ()))
        )
        if outcome is not LocalizationOutcome.SOURCE and not is_promoted:
            continue
        report = estimator.estimate(runs, sid)
        if report.behavior is ImpactBehavior.PROPAGATES:
            propagates.append(sid)
        else:
            absorbed.append(sid)
    return propagates + absorbed


def _format_receipt(
    analysis: PipelineAnalysis,
    now: datetime | None = None,
    replayed: bool = False,
) -> str:
    """Single-line receipt: 'n=3 | $0.0007 | 14s | analysis abc12345'.

    `now` appends ' | ran <relative-time>' (used by `varix show`).
    `replayed` appends ' | replayed' (used by `varix replay`).
    The two are mutually exclusive in practice — replay's preamble already
    carries timing context.
    """
    duration = _format_duration(analysis.started_at, analysis.finished_at)
    line = (
        f"n={analysis.n} | ${analysis.total_cost.dollars:.4f} | "
        f"{duration} | analysis {_short_id(analysis.analysis_id)}"
    )
    if now is not None:
        line += f" | ran {_format_relative_time(analysis.finished_at, now)}"
    if replayed:
        line += " | replayed"
    return line


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


def render_analysis(
    analysis: PipelineAnalysis,
    now: datetime | None = None,
    replayed: bool = False,
) -> str:
    """Return a plain-text report for `analysis`. ASCII-only, no trailing newline.

    Four cases:
      1. n<2: inconclusive — WARNING banner + receipt.
      2. n>=2, no sources, no env findings: 'No nondeterminism found' + receipt.
      3. n>=2, no sources but HIGH-confidence env findings: 'outputs stable but
         routing varied' + finding detail + receipt + Next.
      4. n>=2, has sources: 'Found N sources' + ranked source lines + receipt + Next.

    Receipt suffix:
      - `now` adds ' | ran <relative-time>' (used by `varix show`)
      - `replayed=True` adds ' | replayed' (used by `varix replay`)
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
        lines.append(_format_receipt(analysis, now=now, replayed=replayed))
        return "\n".join(lines)

    sources = _rank_source_step_ids(analysis.runs, outcomes, step_ids, findings_by_step)
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
        lines.append(_format_receipt(analysis, now=now, replayed=replayed))
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
        lines.append(_format_receipt(analysis, now=now, replayed=replayed))
        lines.append("")
        lines.append("Next:")
        for f in env_findings:
            cmd = f"varix explain {f.step_id}"
            lines.append(f"  {cmd.ljust(len(cmd) + 6)}see the fingerprint evidence")
        return "\n".join(lines)
    lines.append(f"No nondeterminism found in {pipeline_label}.")
    lines.append("")
    lines.append(_format_receipt(analysis, now=now, replayed=replayed))
    return "\n".join(lines)


def render_replay(analysis: PipelineAnalysis, now: datetime) -> str:
    """Return a plain-text replay report: preamble + render_analysis body.

    The preamble identifies the artifact and how long ago it was run. The
    body is byte-for-byte what `render_analysis` would produce for the same
    artifact under the current varix version, with `| replayed` appended to
    the receipt so the reader knows nothing was newly billed.
    """
    preamble = (
        f"replay of analysis {_short_id(analysis.analysis_id)} from "
        f"{_format_relative_time(analysis.finished_at, now)}."
    )
    return f"{preamble}\n\n{render_analysis(analysis, replayed=True)}"


def _classification_long(c: Classification | None) -> str:
    """Long-form name used in explain headlines ('provider-side variance', ...)."""
    if c is None:
        return "unclassified variance"
    return f"{_CLASSIFICATION_SHORT[c]} variance"


def _confidence_phrase(c: Confidence) -> str:
    """'high confidence' / 'medium confidence' / 'cannot verify'."""
    if c is Confidence.UNAVAILABLE:
        return _CONFIDENCE_LABELS[c]
    return f"{_CONFIDENCE_LABELS[c]} confidence"


def _runs_word(n: int) -> str:
    return "run" if n == 1 else "runs"


def _explain_headline(finding: Finding, step_id: str) -> str:
    return (
        f"step `{step_id}` was classified as "
        f"{_classification_long(finding.classification)}, "
        f"{_confidence_phrase(finding.confidence)}."
    )


def _gather_step_observations(runs: Sequence[PipelineRun], step_id: str) -> list[StepRun]:
    """Collect each run's StepRun matching step_id (first occurrence per run)."""
    obs: list[StepRun] = []
    for r in runs:
        for sr in r.step_runs:
            if sr.step_id == step_id:
                obs.append(sr)
                break
    return obs


def _explain_provider_side(finding: Finding, analysis: PipelineAnalysis, step_id: str) -> list[str]:
    fingerprints: list[str] = []
    excluded: list[dict[str, Any]] = []
    for ev in finding.evidence:
        if ev.kind == "fingerprint_diff":
            fingerprints = [str(fp) for fp in ev.data.get("fingerprints", [])]
        elif ev.kind == "excluded_runs":
            excluded = list(ev.data.get("excluded", []))
    # Compared count is the fingerprints we actually analyzed; total spans those
    # plus any we had to drop due to missing data.
    n_compared = len(fingerprints) if fingerprints else analysis.n
    n_total = n_compared + len(excluded)
    runs_phrase = f"{n_compared} {_runs_word(n_compared)}"
    lines = [
        _explain_headline(finding, step_id),
        "",
        "Why this classification:",
        f"  Across {runs_phrase} of `{step_id}`, system_fingerprint changed:",
        "",
    ]
    if fingerprints:
        counts = Counter(fingerprints)
        # Stable ordering: most-frequent first, name asc as tiebreak.
        for fp, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"    {fp}   used in {count} {_runs_word(count)}")
    if excluded:
        lines.extend([
            "",
            f"  Note: {len(excluded)} of {n_total} {_runs_word(n_total)} excluded from this "
            "classifier - missing system_fingerprint.",
        ])
    lines.extend([
        "",
        "  Different fingerprints mean the provider routed your requests to",
        "  different model infrastructure. This is variance from the provider",
        "  side - there is nothing in your pipeline causing it.",
    ])
    return lines


def _explain_prompt_side(finding: Finding, analysis: PipelineAnalysis, step_id: str) -> list[str]:
    obs = _gather_step_observations(analysis.runs, step_id)
    n = analysis.n

    fps = [(sr.provider_metadata or {}).get("system_fingerprint") for sr in obs]
    fp_present = [str(fp) for fp in fps if fp is not None]
    if fp_present and len(set(fp_present)) == 1:
        fp_line = (
            f"    - provider fingerprints were stable ({fp_present[0]} "
            f"in all {len(fp_present)})"
        )
    else:
        fp_line = "    - provider fingerprints were stable across runs"

    total_tools = sum(len(sr.tool_calls) for sr in obs)
    if total_tools == 0:
        tool_line = "    - no tool calls were made"
    else:
        tool_line = f"    - {total_tools} tool call(s) observed, all matched across runs"

    lines = [
        _explain_headline(finding, step_id),
        "",
        "Why this classification:",
        f"  Across {n} {_runs_word(n)} of `{step_id}` with identical inputs:",
        fp_line,
        tool_line,
        "    - no time/state markers detected in outputs",
        "    - the outputs themselves differed",
        "",
        "  When provider, tool, ordering, and time/state are ruled out, varix",
        "  labels the residual variance as prompt-side. Medium confidence",
        "  because it's a residual category - if a non-obvious variance source",
        "  leaked through, varix would attribute it here.",
    ]

    if obs:
        lines.append("")
        lines.append(f"The {len(obs)} {_runs_word(len(obs))} varix observed:")
        for i, sr in enumerate(obs, 1):
            lines.append(f'  run {i}: "{_truncate(str(sr.output))}"')
    return lines


def _explain_tool_side(finding: Finding, analysis: PipelineAnalysis, step_id: str) -> list[str]:
    diffs: list[dict[str, Any]] = []
    for ev in finding.evidence:
        if ev.kind == "tool_result_diff":
            diffs = list(ev.data.get("diffs", []))
            break
    n = analysis.n
    lines = [
        _explain_headline(finding, step_id),
        "",
        "Why this classification:",
        f"  Across {n} {_runs_word(n)} of `{step_id}`, the same tool returned different results:",
        "",
    ]
    for d in diffs:
        tool = d.get("tool", "?")
        unique_count = int(d.get("unique_count", 0))
        plural = "s" if unique_count != 1 else ""
        lines.append(f"    tool `{tool}`: {unique_count} distinct result{plural}")
    lines.extend([
        "",
        "  varix paired tool calls by (name, arguments). Same input, different",
        "  output - that's tool-side variance, originating outside your pipeline.",
    ])
    return lines


def _explain_ordering(finding: Finding, analysis: PipelineAnalysis, step_id: str) -> list[str]:
    sequences: list[str] = []
    unique_count = 0
    for ev in finding.evidence:
        if ev.kind == "ordering_diff":
            sequences = [str(s) for s in ev.data.get("sequences", [])]
            unique_count = int(ev.data.get("unique_sequence_count", 0))
            break
    n = analysis.n
    lines = [
        _explain_headline(finding, step_id),
        "",
        "Why this classification:",
        f"  Across {n} {_runs_word(n)} of `{step_id}`, the same set of tool calls",
        f"  appeared in {unique_count} different sequences:",
        "",
    ]
    for i, seq in enumerate(sequences, 1):
        lines.append(f"    seq {i}: {seq}")
    lines.extend([
        "",
        "  Same calls, same results, different order - the variance is in",
        "  scheduling, not in the tools themselves.",
    ])
    return lines


def _format_time_marker(marker: object) -> str:
    """Turn a time_or_state_markers dict into a readable line.

    Two known shapes from `analysis._helpers.time_or_state_markers`:
      - {'kind': 'time_tool_name', 'tools': [...]}
      - {'kind': 'varying_timestamp_in_output', 'timestamps': [...]}
    Plain strings pass through; unknown shapes fall back to repr.
    """
    if isinstance(marker, str):
        return marker
    if isinstance(marker, dict):
        kind = marker.get("kind")
        if kind == "time_tool_name":
            tools = ", ".join(str(t) for t in marker.get("tools", []))
            return f"tool name(s) suggest clock or RNG: {tools}"
        if kind == "varying_timestamp_in_output":
            stamps = ", ".join(str(s) for s in marker.get("timestamps", []))
            return f"timestamps in output varied: {stamps}"
    return str(marker)


def _explain_time_or_state(finding: Finding, analysis: PipelineAnalysis, step_id: str) -> list[str]:
    markers: list[object] = []
    for ev in finding.evidence:
        if ev.kind == "time_or_state_markers":
            markers = list(ev.data.get("markers", []))
            break
    n = analysis.n
    lines = [
        _explain_headline(finding, step_id),
        "",
        "Why this classification:",
        f"  Across {n} {_runs_word(n)} of `{step_id}`, varix detected heuristic markers",
        "  suggesting clock or RNG:",
        "",
    ]
    for m in markers:
        lines.append(f"    - {_format_time_marker(m)}")
    lines.extend([
        "",
        "  This is a low-confidence heuristic - markers like tool names containing",
        "  'time'/'rand' or ISO timestamps. Verify by inspecting the step's logic.",
    ])
    return lines


def _explain_unavailable(finding: Finding, step_id: str) -> list[str]:
    lines = [
        _explain_headline(finding, step_id),
        "",
        "Why varix could not classify:",
    ]
    if finding.reason:
        lines.append(f"  {finding.reason}")
    if finding.classification is not None:
        attempted = _CLASSIFICATION_SHORT[finding.classification]
        lines.extend([
            "",
            f"  varix tried to verify {attempted} variance but the adapter",
            "  doesn't expose the data needed. Expose the relevant capability",
            "  on your adapter to enable this classification.",
        ])
    return lines


def _explain_block(finding: Finding, analysis: PipelineAnalysis, step_id: str) -> list[str]:
    """Per-finding prose body. UNAVAILABLE first; classification dispatch otherwise."""
    if finding.confidence is Confidence.UNAVAILABLE:
        lines = _explain_unavailable(finding, step_id)
    elif finding.classification is Classification.PROVIDER_SIDE:
        lines = _explain_provider_side(finding, analysis, step_id)
    elif finding.classification is Classification.PROMPT_SIDE:
        lines = _explain_prompt_side(finding, analysis, step_id)
    elif finding.classification is Classification.TOOL_SIDE:
        lines = _explain_tool_side(finding, analysis, step_id)
    elif finding.classification is Classification.ORDERING:
        lines = _explain_ordering(finding, analysis, step_id)
    elif finding.classification is Classification.TIME_OR_STATE:
        lines = _explain_time_or_state(finding, analysis, step_id)
    else:
        # Fallback for findings without a classification — should not normally occur.
        lines = [_explain_headline(finding, step_id)]
        if finding.reason:
            lines.extend(["", f"Reason: {finding.reason}"])
    lines.extend(_explain_replay_disambiguation(finding))
    return lines


def _explain_replay_disambiguation(finding: Finding) -> list[str]:
    """Footer block surfacing replay-disambiguation evidence when present."""
    for ev in finding.evidence:
        if ev.kind != "replay_disambiguation":
            continue
        k = int(ev.data.get("replay_count", 0))
        u = int(ev.data.get("unique_output_count", 0))
        return [
            "",
            "Replay disambiguation:",
            f"  Replayed {k} times with fixed inputs; observed {u} unique outputs.",
            "  The step has its own variance source — not just cascade from upstream.",
        ]
    return []


def render_explain(analysis: PipelineAnalysis, step_id: str) -> str:
    """Return a plain-text evidence trail for `step_id`. ASCII-only, no trailing newline.

    Cases:
      - No findings: short 'deterministic, nothing to explain' sentence.
      - One finding: a single classification block followed by analysis + Next.
      - Multiple findings: each block separated by '---', sorted HIGH→LOW confidence.
    """
    short_id = _short_id(analysis.analysis_id)
    step_findings = [f for f in analysis.findings if f.step_id == step_id]

    if not step_findings:
        return (
            f"step `{step_id}` has no findings - it was deterministic across "
            f"the runs, so there is nothing to explain.\n"
            f"\n"
            f"analysis: {short_id}"
        )

    sorted_findings = sorted(step_findings, key=lambda f: _CONFIDENCE_ORDER[f.confidence])

    lines: list[str] = []
    for i, f in enumerate(sorted_findings):
        if i > 0:
            lines.extend(["", "---", ""])
        lines.extend(_explain_block(f, analysis, step_id))

    lines.extend(["", f"analysis: {short_id}", "", "Next:"])
    cmd = f"varix impact {step_id}"
    lines.append(f"  {cmd.ljust(len(cmd) + 6)}see how this changes your final output")
    return "\n".join(lines)


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


_LIST_NAME_MAX = 40
_LIST_NAME_TRUNCATE_AT = _LIST_NAME_MAX - 3  # leave room for the trailing "..."


def _list_summary(analysis: PipelineAnalysis) -> str:
    """One-line summary for the `varix list` table.

    Derives the verdict from stored findings + ImpactEstimator (re-walks runs).
    Localizer outcomes aren't stored per-step, so we read each finding's
    own `localization` field to identify source steps.
    """
    if analysis.n < 2:
        return f"inconclusive (n={analysis.n})"

    # Effective sources: localizer-SOURCE plus DOWNSTREAM steps that replay
    # disambiguated as independently varying.
    findings_by_step: dict[str, list[Finding]] = {}
    for f in analysis.findings:
        findings_by_step.setdefault(f.step_id, []).append(f)
    source_steps = sorted(
        sid
        for sid, fs in findings_by_step.items()
        if any(f.localization is LocalizationOutcome.SOURCE for f in fs)
        or _has_replay_disambiguation(fs)
    )
    if source_steps:
        estimator = ImpactEstimator()
        behaviors = [estimator.estimate(analysis.runs, sid).behavior for sid in source_steps]
        propagates = sum(1 for b in behaviors if b is ImpactBehavior.PROPAGATES)
        absorbed = sum(1 for b in behaviors if b is ImpactBehavior.ABSORBED)
        if propagates == len(behaviors):
            qualifier = "propagates"
        elif absorbed == len(behaviors):
            qualifier = "absorbed"
        else:
            qualifier = "mixed"
        n = len(source_steps)
        return f"{n} source{'s' if n != 1 else ''} ({qualifier})"

    env = [
        f
        for f in analysis.findings
        if f.confidence is Confidence.HIGH
        and f.localization is LocalizationOutcome.DETERMINISTIC
    ]
    if env:
        return "stable, routing varied"

    return "clean"


def render_list(analyses: list[PipelineAnalysis], now: datetime) -> str:
    """Render a 'recent analyses' table. ASCII-only, no trailing newline."""
    if not analyses:
        return "no analyses found."

    rows: list[tuple[str, str, str, str]] = []
    for a in analyses:
        rows.append(
            (
                _short_id(a.analysis_id),
                _format_relative_time(a.finished_at, now),
                _display_pipeline_name(a.pipeline_name),
                _list_summary(a),
            )
        )

    when_w = max(len(r[1]) for r in rows)
    name_w = min(_LIST_NAME_MAX, max(len(r[2]) for r in rows))

    lines = ["recent analyses:", ""]
    for short, when, name, summary in rows:
        displayed_name = (
            name if len(name) <= _LIST_NAME_MAX else name[:_LIST_NAME_TRUNCATE_AT] + "..."
        )
        lines.append(
            f"  {short}  {when.ljust(when_w)}  {displayed_name.ljust(name_w)}  {summary}"
        )
    lines.append("")
    suffix = "analysis" if len(analyses) == 1 else "analyses"
    lines.append(f"showing {len(analyses)} {suffix}.")
    return "\n".join(lines)
