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
    assert "0 finding(s), 0 source step(s)" in report
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
        "\n"
        "step s1: deterministic\n"
        "step s2: deterministic\n"
        "step s3: deterministic\n"
        "step s4: deterministic\n"
        "step s5: deterministic\n"
        "\n"
        "0 finding(s), 0 source step(s)"
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
    assert "-> provider_side (high):" in report
    assert "system_fingerprint" in report
    assert "1 finding(s), 0 source step(s)" in report


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
    assert "step s2: source" in report
    assert "-> prompt_side (medium):" in report
    assert "step s3: downstream" in report
    assert "step s5: downstream" in report
    assert "1 finding(s), 1 source step(s)" in report


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
    assert "step s5: source" in report
    assert "-> time_or_state (low):" in report
    assert "1 finding(s), 1 source step(s)" in report


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
    assert "provider_side (unavailable):" in report
    assert "adapter does not expose system_fingerprint" in report
