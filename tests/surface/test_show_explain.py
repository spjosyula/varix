"""Tests for execute_show / execute_explain dispatch and render_explain output."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from varix.core import (
    SCHEMA_VERSION,
    Classification,
    Confidence,
    CostSnapshot,
    Evidence,
    Finding,
    FrozenClock,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    RefusalRequired,
    SequenceRng,
    StepRun,
    ToolCall,
)
from varix.surface.dispatch import execute_explain, execute_run, execute_show
from varix.surface.reporter import render_explain
from varix.surface.storage import latest_analysis, save

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def _make_analysis(analysis_id: str, *, with_finding: bool = False) -> PipelineAnalysis:
    step_run = StepRun(
        step_id="s1",
        inputs="hello",
        output="out",
        tool_calls=(ToolCall(name="lookup", arguments={"q": "hello"}, result="hit"),),
        provider_metadata={"system_fingerprint": "fp_a"},
    )
    pipeline_run = PipelineRun(run_id="r1", step_runs=(step_run,), started_at=_T, finished_at=_T)
    findings: tuple[Finding, ...] = ()
    if with_finding:
        findings = (
            Finding(
                step_id="s1",
                localization=LocalizationOutcome.SOURCE,
                confidence=Confidence.HIGH,
                metric_name="exact",
                classification=Classification.PROVIDER_SIDE,
                evidence=(
                    Evidence(
                        kind="fingerprint_diff",
                        description="system_fingerprint values observed across runs",
                        data={"fingerprints": ["fp_a", "fp_b"], "unique": ["fp_a", "fp_b"]},
                    ),
                ),
                reason="system_fingerprint varied across runs: ['fp_a', 'fp_b']",
            ),
        )
    return PipelineAnalysis(
        analysis_id=analysis_id,
        pipeline_name="varix.adapters:FakeAdapter",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pipeline_run,),
        findings=findings,
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )


# --- latest_analysis -----------------------------------------------------------


def test_latest_analysis_returns_none_when_empty(tmp_path: Path) -> None:
    assert latest_analysis(base_dir=tmp_path) is None


def test_latest_analysis_returns_most_recent_mtime(tmp_path: Path) -> None:
    save(_make_analysis("first"), base_dir=tmp_path)
    time.sleep(0.02)  # ensure mtime resolution differs
    save(_make_analysis("second"), base_dir=tmp_path)
    latest = latest_analysis(base_dir=tmp_path)
    assert latest is not None
    assert latest.stem == "second"


# --- execute_show --------------------------------------------------------------


def test_execute_show_by_id_returns_rendered_text(tmp_path: Path) -> None:
    save(_make_analysis("abc-123"), base_dir=tmp_path)
    rendered = execute_show("abc-123", base_dir=tmp_path)
    assert "analysis abc-123" in rendered


def test_execute_show_by_path_returns_rendered_text(tmp_path: Path) -> None:
    path = save(_make_analysis("abc-123"), base_dir=tmp_path)
    rendered = execute_show(str(path))
    assert "analysis abc-123" in rendered


def test_execute_show_unknown_id_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute_show("nonexistent", base_dir=tmp_path)


def test_execute_show_newer_schema_raises_refusal(tmp_path: Path) -> None:
    artifact = tmp_path / "future.json"
    data = _make_analysis("future").to_dict()
    data["schema_version"] = "9.99"
    artifact.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RefusalRequired):
        execute_show("future", base_dir=tmp_path)


def test_execute_show_round_trips_run_output(tmp_path: Path) -> None:
    """varix run + varix show should render the same text."""
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=3,
        base_dir=tmp_path,
        clock=FrozenClock(_T),
        rng=SequenceRng(["round-trip-id"]),
    )
    from varix.surface.reporter import render_analysis

    fresh = render_analysis(analysis)
    reloaded = execute_show("round-trip-id", base_dir=tmp_path)
    assert fresh == reloaded


# --- execute_explain -----------------------------------------------------------


def test_execute_explain_with_explicit_analysis_id(tmp_path: Path) -> None:
    save(_make_analysis("abc-123", with_finding=True), base_dir=tmp_path)
    rendered = execute_explain("s1", "abc-123", base_dir=tmp_path)
    assert "step `s1` was classified as provider-side variance, high confidence." in rendered
    assert "system_fingerprint changed" in rendered


def test_execute_explain_falls_back_to_latest_when_no_target(tmp_path: Path) -> None:
    save(_make_analysis("older", with_finding=False), base_dir=tmp_path)
    time.sleep(0.02)
    save(_make_analysis("newer", with_finding=True), base_dir=tmp_path)
    rendered = execute_explain("s1", base_dir=tmp_path)
    assert "analysis: newer" in rendered
    assert "step `s1` was classified as provider-side variance, high confidence." in rendered


def test_execute_explain_no_saved_analyses_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute_explain("s1", base_dir=tmp_path)


def test_execute_explain_unknown_step_raises_value_error(tmp_path: Path) -> None:
    save(_make_analysis("abc"), base_dir=tmp_path)
    with pytest.raises(ValueError, match="not_a_step"):
        execute_explain("not_a_step", "abc", base_dir=tmp_path)


# --- render_explain ------------------------------------------------------------


def test_render_explain_step_with_no_findings_says_so() -> None:
    analysis = _make_analysis("abc")
    rendered = render_explain(analysis, "s1")
    assert "step `s1` has no findings" in rendered
    assert "deterministic across the runs" in rendered


def test_render_explain_provider_side_renders_fingerprint_table() -> None:
    analysis = _make_analysis("abc", with_finding=True)
    rendered = render_explain(analysis, "s1")
    assert "step `s1` was classified as provider-side variance, high confidence." in rendered
    assert "system_fingerprint changed" in rendered
    assert "fp_a   used in 1 run" in rendered
    assert "fp_b   used in 1 run" in rendered
    assert "different model infrastructure" in rendered


def test_render_explain_prompt_side_renders_ruled_out_facts_and_outputs() -> None:
    def _run(rid: str, output: str) -> PipelineRun:
        return PipelineRun(
            run_id=rid,
            step_runs=(
                StepRun(
                    step_id="s1",
                    inputs="i_const",
                    output=output,
                    provider_metadata={"system_fingerprint": "fp_a"},
                ),
            ),
            started_at=_T,
            finished_at=_T,
        )

    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.MEDIUM,
        metric_name="exact",
        classification=Classification.PROMPT_SIDE,
        reason="residual",
    )
    analysis = PipelineAnalysis(
        analysis_id="prompt-id",
        pipeline_name="fake",
        n=3,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(_run("r1", "alpha output"), _run("r2", "beta output"), _run("r3", "gamma output")),
        findings=(finding,),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )
    rendered = render_explain(analysis, "s1")
    assert "step `s1` was classified as prompt-side variance, medium confidence." in rendered
    assert "provider fingerprints were stable (fp_a in all 3)" in rendered
    assert "no tool calls were made" in rendered
    assert "no time/state markers detected in outputs" in rendered
    assert "the outputs themselves differed" in rendered
    assert "The 3 runs varix observed:" in rendered
    assert 'run 1: "alpha output"' in rendered
    assert 'run 3: "gamma output"' in rendered


def test_format_time_marker_handles_known_dict_shapes() -> None:
    from varix.surface.reporter import _format_time_marker

    assert (
        _format_time_marker({"kind": "time_tool_name", "tools": ["get_current_time"]})
        == "tool name(s) suggest clock or RNG: get_current_time"
    )
    assert (
        _format_time_marker(
            {"kind": "varying_timestamp_in_output", "timestamps": ["2026-05-10T12:00:00Z"]}
        )
        == "timestamps in output varied: 2026-05-10T12:00:00Z"
    )
    assert _format_time_marker("plain string") == "plain string"
    assert _format_time_marker({"kind": "unknown"}) == "{'kind': 'unknown'}"


def test_render_explain_unavailable_finding_says_could_not_classify() -> None:
    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.UNAVAILABLE,
        metric_name="exact",
        classification=Classification.PROVIDER_SIDE,
        reason="adapter does not expose system_fingerprint",
    )
    analysis = PipelineAnalysis(
        analysis_id="unavail-id",
        pipeline_name="fake",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr, pr),
        findings=(finding,),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )
    rendered = render_explain(analysis, "s1")
    assert "step `s1` was classified as provider-side variance, cannot verify." in rendered
    assert "Why varix could not classify:" in rendered
    assert "adapter does not expose system_fingerprint" in rendered


def test_render_explain_with_multiple_findings_renders_each_block() -> None:
    base = _make_analysis("abc", with_finding=True)
    extra = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.MEDIUM,
        metric_name="exact",
        classification=Classification.PROMPT_SIDE,
        reason="residual",
    )
    analysis = PipelineAnalysis(
        analysis_id=base.analysis_id,
        pipeline_name=base.pipeline_name,
        n=base.n,
        metric_name=base.metric_name,
        schema_version=base.schema_version,
        runs=base.runs,
        findings=(*base.findings, extra),
        started_at=base.started_at,
        finished_at=base.finished_at,
        total_cost=base.total_cost,
    )
    rendered = render_explain(analysis, "s1")
    assert "provider-side variance, high confidence." in rendered
    assert "prompt-side variance, medium confidence." in rendered
    # HIGH-confidence block must come first under the sort.
    assert rendered.index("provider-side variance") < rendered.index("prompt-side variance")
    assert "---" in rendered  # divider between blocks
