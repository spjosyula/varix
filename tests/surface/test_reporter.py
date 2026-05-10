"""Tests for the terminal reporter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from varix.core import (
    FrozenClock,
    SequenceRng,
)
from varix.surface.dispatch import execute_run
from varix.surface.reporter import render_analysis

_FROZEN = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def _execute(pipeline_target: str, tmp_path: Path) -> str:
    analysis, _ = execute_run(
        pipeline=pipeline_target,
        input_text="hello",
        n=3,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["golden-test-id"]),
    )
    return render_analysis(analysis)


def test_report_header_lists_required_fields(tmp_path: Path) -> None:
    report = _execute("varix.adapters:FakeAdapter", tmp_path)
    assert "=== varix analysis ===" in report
    assert "pipeline:" in report
    assert "analysis_id: golden-test-id" in report
    assert "n:           3" in report
    assert "metric:      exact" in report
    assert "cost:" in report


def test_deterministic_report_shows_no_findings(tmp_path: Path) -> None:
    report = _execute("varix.adapters:FakeAdapter", tmp_path)
    assert "verdict:     every run produced the same output." in report
    assert "step s1: deterministic" in report
    assert "step s2: deterministic" in report
    assert "step s5: deterministic" in report


def test_step_order_matches_pipeline_order(tmp_path: Path) -> None:
    report = _execute("varix.adapters:FakeAdapter", tmp_path)
    s1 = report.index("step s1:")
    s2 = report.index("step s2:")
    s5 = report.index("step s5:")
    assert s1 < s2 < s5


def test_deterministic_report_byte_for_byte_golden(tmp_path: Path) -> None:
    report = _execute("varix.adapters:FakeAdapter", tmp_path)
    expected = (
        "=== varix analysis ===\n"
        "pipeline:    varix.adapters:FakeAdapter\n"
        "analysis_id: golden-test-id\n"
        "n:           3\n"
        "metric:      exact\n"
        "cost:        $0.0000\n"
        "verdict:     every run produced the same output.\n"
        "\n"
        "step s1: deterministic\n"
        "step s2: deterministic\n"
        "step s3: deterministic\n"
        "step s4: deterministic\n"
        "step s5: deterministic"
    )
    assert report == expected


def _adapter_file(tmp_path: Path, variance_kw: str) -> Path:
    """Write a small Python file that exposes a FakeAdapter with the given variance."""
    path = tmp_path / "agent.py"
    path.write_text(
        "from varix.adapters import FakeAdapter\n"
        "from varix.core import Classification\n"
        f"adapter = FakeAdapter(variance={variance_kw})\n",
        encoding="utf-8",
    )
    return path


def test_provider_side_scenario_renders_high_finding(tmp_path: Path) -> None:
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROVIDER_SIDE}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["provider-test-id"]),
    )
    report = render_analysis(analysis)
    # PROVIDER_SIDE in FakeAdapter keeps output stable but flips fingerprints.
    # Localizer reads everything as DETERMINISTIC; classifier still emits HIGH.
    assert "step s2: deterministic" in report
    assert "- provider rolled the model (high confidence):" in report
    assert "system_fingerprint" in report
    # Output is stable (only fingerprint flips), so the headline says so.
    assert "verdict:     every run produced the same output." in report


def test_prompt_side_scenario_renders_medium_residual(tmp_path: Path) -> None:
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["prompt-test-id"]),
    )
    report = render_analysis(analysis)
    assert "step s2: source of variance" in report
    assert "- sampling / temperature (medium confidence):" in report
    assert "step s3: inherited from upstream" in report
    assert "step s5: inherited from upstream" in report
    assert "verdict:     1 step varies, and you get a different final output each run." in report


def test_time_or_state_scenario_renders_low_finding(tmp_path: Path) -> None:
    agent = _adapter_file(tmp_path, "{'s5': Classification.TIME_OR_STATE}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["time-test-id"]),
    )
    report = render_analysis(analysis)
    assert "step s5: source of variance" in report
    assert "- clock or random source (low confidence):" in report
    assert "verdict:     1 step varies, and you get a different final output each run." in report


@pytest.mark.parametrize("scenario", ["PROVIDER_SIDE", "TOOL_SIDE", "ORDERING"])
def test_high_confidence_scenarios_each_emit_exactly_one_finding(
    tmp_path: Path, scenario: str
) -> None:
    agent = _adapter_file(tmp_path, f"{{'s2': Classification.{scenario}}}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng([f"{scenario}-id"]),
    )
    assert len(analysis.findings) == 1
    assert analysis.findings[0].confidence.value == "high"


def test_headline_pluralizes_for_multiple_source_steps() -> None:
    """Two source steps with stable inputs but varying outputs; headline says `2 steps vary`."""
    from varix.core import (
        SCHEMA_VERSION,
        CostSnapshot,
        PipelineAnalysis,
        PipelineRun,
        StepRun,
    )
    from varix.surface.reporter import render_analysis

    def _run(run_id: str, s1_out: str, s2_out: str) -> PipelineRun:
        return PipelineRun(
            run_id=run_id,
            step_runs=(
                StepRun(step_id="s1", inputs="constant_in_1", output=s1_out),
                StepRun(step_id="s2", inputs="constant_in_2", output=s2_out),
            ),
            started_at=_FROZEN,
            finished_at=_FROZEN,
        )

    analysis = PipelineAnalysis(
        analysis_id="multi-id",
        pipeline_name="manual",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(_run("r1", "a", "c"), _run("r2", "A", "C")),
        findings=(),
        started_at=_FROZEN,
        finished_at=_FROZEN,
        total_cost=CostSnapshot(),
    )
    report = render_analysis(analysis)
    assert "verdict:     2 steps vary, and you get a different final output each run." in report


def test_headline_omitted_when_n_is_one(tmp_path: Path) -> None:
    """At n<2 the WARNING banner explains inconclusiveness; no verdict line."""
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=1,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["solo-id"]),
    )
    report = render_analysis(analysis)
    assert "verdict:" not in report
    assert "WARNING:" in report


def test_render_emits_warning_banner_when_notes_present() -> None:
    """Truncation notes must be loud at the top of the report, not buried."""
    from varix.core import (
        SCHEMA_VERSION,
        CostSnapshot,
        PipelineAnalysis,
        PipelineRun,
        StepRun,
    )
    from varix.surface.reporter import render_analysis

    step_run = StepRun(step_id="s1", inputs="i", output="o")
    pipeline_run = PipelineRun(
        run_id="r1", step_runs=(step_run,), started_at=_FROZEN, finished_at=_FROZEN
    )
    analysis = PipelineAnalysis(
        analysis_id="notes-test",
        pipeline_name="fake",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pipeline_run,),
        findings=(),
        started_at=_FROZEN,
        finished_at=_FROZEN,
        total_cost=CostSnapshot(),
        notes=("run 3 of 5 failed: RuntimeError: provider stalled",),
    )
    report = render_analysis(analysis)
    assert "WARNING:" in report
    assert "provider stalled" in report
    # Banner sits between the header block and the per-step lines.
    warn_at = report.index("WARNING:")
    step_at = report.index("step s1:")
    assert warn_at < step_at


def test_render_clean_analysis_has_no_warning_section() -> None:
    """Empty notes must not introduce a phantom WARNING banner."""
    from varix.core import (
        SCHEMA_VERSION,
        CostSnapshot,
        PipelineAnalysis,
        PipelineRun,
        StepRun,
    )
    from varix.surface.reporter import render_analysis

    step_run = StepRun(step_id="s1", inputs="i", output="o")
    pipeline_run = PipelineRun(
        run_id="r1", step_runs=(step_run,), started_at=_FROZEN, finished_at=_FROZEN
    )
    analysis = PipelineAnalysis(
        analysis_id="clean-test",
        pipeline_name="fake",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pipeline_run,),
        findings=(),
        started_at=_FROZEN,
        finished_at=_FROZEN,
        total_cost=CostSnapshot(),
    )
    assert "WARNING" not in render_analysis(analysis)


def test_short_id_takes_first_eight_chars() -> None:
    from varix.surface.reporter import _short_id

    assert _short_id("c13cfc73-8f25-49c5-a8a2-6a513f740598") == "c13cfc73"


def test_short_id_handles_input_shorter_than_eight() -> None:
    from varix.surface.reporter import _short_id

    assert _short_id("abc") == "abc"


def test_truncate_returns_input_when_under_limit() -> None:
    from varix.surface.reporter import _truncate

    assert _truncate("hello", max_chars=10) == "hello"


def test_truncate_appends_ellipsis_when_over_limit() -> None:
    from varix.surface.reporter import _truncate

    assert _truncate("0123456789abcdef", max_chars=10) == "0123456789..."


def test_format_duration_subsecond_keeps_decimal() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_duration

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_duration(t, t) == "0.0s"
    assert _format_duration(t, t + timedelta(seconds=2.5)) == "2.5s"


def test_format_duration_seconds_int_above_ten() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_duration

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_duration(t, t + timedelta(seconds=14)) == "14s"


def test_format_duration_minutes_and_seconds() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_duration

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_duration(t, t + timedelta(minutes=2, seconds=14)) == "2m 14s"


def test_format_duration_clamps_negative_to_zero() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_duration

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_duration(t + timedelta(seconds=5), t) == "0.0s"


def test_format_relative_time_just_now() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_relative_time

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_relative_time(t, t + timedelta(seconds=30)) == "just now"


def test_format_relative_time_minutes_pluralization() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_relative_time

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_relative_time(t, t + timedelta(minutes=1)) == "1 minute ago"
    assert _format_relative_time(t, t + timedelta(minutes=5)) == "5 minutes ago"


def test_format_relative_time_hours_and_days() -> None:
    from datetime import timedelta

    from varix.surface.reporter import _format_relative_time

    t = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert _format_relative_time(t, t + timedelta(hours=2)) == "2 hours ago"
    assert _format_relative_time(t, t + timedelta(days=3)) == "3 days ago"
    assert _format_relative_time(t, t + timedelta(hours=1)) == "1 hour ago"


def test_rank_source_step_ids_propagates_first_then_absorbed() -> None:
    """A source step that's also the final step is ABSORBED; other sources PROPAGATE."""
    from varix.core import (
        SCHEMA_VERSION,
        CostSnapshot,
        LocalizationOutcome,
        PipelineAnalysis,
        PipelineRun,
        StepRun,
    )
    from varix.surface.reporter import _rank_source_step_ids

    def _run(run_id: str, a_out: str, c_out: str) -> PipelineRun:
        return PipelineRun(
            run_id=run_id,
            step_runs=(
                StepRun(step_id="a", inputs="in_a", output=a_out),
                StepRun(step_id="b", inputs="in_b", output="b_const"),
                StepRun(step_id="c", inputs="in_c", output=c_out),
            ),
            started_at=_FROZEN,
            finished_at=_FROZEN,
        )

    runs = (_run("r1", "a1", "c1"), _run("r2", "a2", "c2"))
    outcomes = {
        "a": LocalizationOutcome.SOURCE,
        "b": LocalizationOutcome.DETERMINISTIC,
        "c": LocalizationOutcome.SOURCE,
    }
    # Reference unused PipelineAnalysis import for mypy-strict.
    _ = PipelineAnalysis(
        analysis_id="x",
        pipeline_name="x",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=runs,
        findings=(),
        started_at=_FROZEN,
        finished_at=_FROZEN,
        total_cost=CostSnapshot(),
    )
    ranked = _rank_source_step_ids(runs, outcomes, ["a", "b", "c"])
    # `a` propagates (final c varies); `c` is the final step → ABSORBED special-case.
    assert ranked == ["a", "c"]


def test_rank_source_step_ids_skips_non_sources() -> None:
    from varix.core import LocalizationOutcome, PipelineRun, StepRun
    from varix.surface.reporter import _rank_source_step_ids

    runs = (
        PipelineRun(
            run_id="r1",
            step_runs=(StepRun(step_id="a", inputs="i", output="o"),),
            started_at=_FROZEN,
            finished_at=_FROZEN,
        ),
    )
    outcomes = {"a": LocalizationOutcome.DETERMINISTIC}
    assert _rank_source_step_ids(runs, outcomes, ["a"]) == []


def test_format_receipt_assembles_n_cost_duration_short_id() -> None:
    from datetime import timedelta

    from varix.core import (
        SCHEMA_VERSION,
        CostSnapshot,
        PipelineAnalysis,
        PipelineRun,
        StepRun,
    )
    from varix.surface.reporter import _format_receipt

    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_FROZEN,
        finished_at=_FROZEN,
    )
    analysis = PipelineAnalysis(
        analysis_id="abc12345-rest-of-the-uuid",
        pipeline_name="fake",
        n=3,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr,),
        findings=(),
        started_at=_FROZEN,
        finished_at=_FROZEN + timedelta(seconds=14),
        total_cost=CostSnapshot(input_tokens=0, output_tokens=0, dollars=0.0007),
    )
    assert _format_receipt(analysis) == "n=3 | $0.0007 | 14s | analysis abc12345"


def test_render_includes_unavailable_findings() -> None:
    """Regression: UNAVAILABLE findings (from missing capability) survive into the report."""
    from varix.core import (
        SCHEMA_VERSION,
        Classification,
        Confidence,
        CostSnapshot,
        Finding,
        LocalizationOutcome,
        PipelineAnalysis,
        PipelineRun,
        StepRun,
    )
    from varix.surface.reporter import render_analysis

    step_run = StepRun(step_id="s1", inputs="i", output="o")
    pipeline_run = PipelineRun(
        run_id="r1", step_runs=(step_run,), started_at=_FROZEN, finished_at=_FROZEN
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
        analysis_id="unavailable-test",
        pipeline_name="fake",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pipeline_run,),
        findings=(finding,),
        started_at=_FROZEN,
        finished_at=_FROZEN,
        total_cost=CostSnapshot(),
    )
    report = render_analysis(analysis)
    assert "provider rolled the model (cannot verify):" in report
    assert "adapter does not expose system_fingerprint" in report
