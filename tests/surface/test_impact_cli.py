"""Tests for execute_impact, render_impact, and the impact suffix in run reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from varix.adapters import FakeAdapter
from varix.analysis import ImpactBehavior, ImpactEstimator, ImpactReport
from varix.core import (
    Classification,
    Confidence,
    Evidence,
    FrozenClock,
    PipelineRun,
    SequenceRng,
    StepRun,
)
from varix.execution import run_n
from varix.surface.cli import app
from varix.surface.dispatch import execute_impact, execute_run
from varix.surface.reporter import render_analysis, render_impact
from varix.surface.storage import save

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
_FROZEN = _T

_PROPAGATES_VERDICT = "verdict:     s2 changes the final output of every run (high confidence)."


def _adapter_file(tmp_path: Path, variance_kw: str) -> Path:
    path = tmp_path / "agent.py"
    path.write_text(
        "from varix.adapters import FakeAdapter\n"
        "from varix.core import Classification\n"
        f"adapter = FakeAdapter(variance={variance_kw})\n",
        encoding="utf-8",
    )
    return path


# --- render_impact ----------------------------------------------------------


def test_render_impact_includes_header_and_verdict(tmp_path: Path) -> None:
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["impact-test-id"]),
    )
    report = ImpactEstimator().estimate(analysis.runs, "s2")
    rendered = render_impact(analysis, report)
    assert "=== impact s2 ===" in rendered
    assert "analysis_id: impact-test-id" in rendered
    assert _PROPAGATES_VERDICT in rendered
    assert "evidence:" in rendered


def test_render_impact_without_evidence_omits_evidence_section() -> None:
    from varix.core import SCHEMA_VERSION, CostSnapshot, PipelineAnalysis

    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    analysis = PipelineAnalysis(
        analysis_id="x",
        pipeline_name="fake",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr,),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )
    bare = ImpactReport(
        source_step_id="s1",
        final_step_id="s1",
        behavior=ImpactBehavior.ABSORBED,
        confidence=Confidence.UNAVAILABLE,
        reason="insufficient runs",
    )
    rendered = render_impact(analysis, bare)
    assert "evidence:" not in rendered


# --- execute_impact ---------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_impact_with_explicit_analysis_id(tmp_path: Path) -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    from varix.core import SCHEMA_VERSION, CostSnapshot, PipelineAnalysis

    analysis = PipelineAnalysis(
        analysis_id="impact-id",
        pipeline_name="fake",
        n=3,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=tuple(runs),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )
    save(analysis, base_dir=tmp_path)
    rendered = execute_impact("s2", "impact-id", base_dir=tmp_path)
    assert _PROPAGATES_VERDICT in rendered
    assert "=== impact s2 ===" in rendered


def test_execute_impact_unknown_step_raises_value_error(tmp_path: Path) -> None:
    from varix.core import SCHEMA_VERSION, CostSnapshot, PipelineAnalysis

    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    analysis = PipelineAnalysis(
        analysis_id="abc",
        pipeline_name="fake",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr,),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )
    save(analysis, base_dir=tmp_path)
    with pytest.raises(ValueError, match="not_a_step"):
        execute_impact("not_a_step", "abc", base_dir=tmp_path)


def test_execute_impact_no_artifacts_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute_impact("s1", base_dir=tmp_path)


# --- run report integration -------------------------------------------------


def test_run_report_appends_impact_suffix_for_source_step(tmp_path: Path) -> None:
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    rendered = render_analysis(analysis)
    # FakeAdapter's input-cascade makes s2's variance reach the final step.
    assert "step `s2`  ->  prompt-side, propagates downstream" in rendered


def test_run_report_omits_source_lines_for_clean_pipelines(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    rendered = render_analysis(analysis)
    assert "propagates downstream" not in rendered
    assert "absorbed downstream" not in rendered
    assert "step `s" not in rendered


# --- CLI command ------------------------------------------------------------


cli_runner = CliRunner()


def test_cli_impact_command_renders_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    cli_runner.invoke(app, ["run", str(agent), "--input", "hello", "-n", "3"])
    result = cli_runner.invoke(app, ["impact", "s2"])
    assert result.exit_code == 0
    assert "=== impact s2 ===" in result.output
    assert _PROPAGATES_VERDICT in result.output


def test_cli_impact_unknown_step_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    cli_runner.invoke(app, ["run", "varix.adapters:FakeAdapter", "--input", "hello"])
    result = cli_runner.invoke(app, ["impact", "no_such_step"])
    assert result.exit_code == 1
    assert "varix impact:" in result.output
    assert "no_such_step" in result.output


def test_cli_impact_no_artifacts_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = cli_runner.invoke(app, ["impact", "s1"])
    assert result.exit_code == 1
    assert "varix impact:" in result.output


def _evidence_unused() -> Evidence:
    """Reference Evidence so the import is exercised by mypy strict."""
    return Evidence(kind="x", description="y")
