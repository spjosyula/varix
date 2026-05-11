"""Tests for `varix list` — at-a-glance directory of recent analyses."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from varix.core import (
    SCHEMA_VERSION,
    AdapterCapabilities,
    Classification,
    Confidence,
    CostSnapshot,
    Evidence,
    Finding,
    FrozenClock,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    SequenceRng,
    StepRun,
)
from varix.surface.cli import app
from varix.surface.dispatch import execute_list, execute_run
from varix.surface.reporter import render_list
from varix.surface.storage import recent_analyses, save

_T = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
runner = CliRunner()


def _clean_analysis(aid: str) -> PipelineAnalysis:
    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    return PipelineAnalysis(
        analysis_id=aid,
        pipeline_name="agent.py",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr, pr),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
        capabilities=AdapterCapabilities(),
    )


def _source_analysis(aid: str, *, propagating: bool = True) -> PipelineAnalysis:
    pr1 = PipelineRun(
        run_id="r1",
        step_runs=(
            StepRun(step_id="s1", inputs="i_const", output="o1"),
            StepRun(
                step_id="s2",
                inputs="i_const",
                output="x" if propagating else "shared",
            ),
        ),
        started_at=_T,
        finished_at=_T,
    )
    pr2 = PipelineRun(
        run_id="r2",
        step_runs=(
            StepRun(step_id="s1", inputs="i_const", output="o2"),
            StepRun(
                step_id="s2",
                inputs="i_const",
                output="y" if propagating else "shared",
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
    )
    return PipelineAnalysis(
        analysis_id=aid,
        pipeline_name="agent.py",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr1, pr2),
        findings=(finding,),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
        capabilities=AdapterCapabilities(),
    )


def _env_finding_analysis(aid: str) -> PipelineAnalysis:
    pr1 = PipelineRun(
        run_id="r1",
        step_runs=(
            StepRun(
                step_id="s1",
                inputs="i",
                output="o",
                provider_metadata={"system_fingerprint": "fp_a"},
            ),
        ),
        started_at=_T,
        finished_at=_T,
    )
    pr2 = PipelineRun(
        run_id="r2",
        step_runs=(
            StepRun(
                step_id="s1",
                inputs="i",
                output="o",
                provider_metadata={"system_fingerprint": "fp_b"},
            ),
        ),
        started_at=_T,
        finished_at=_T,
    )
    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        confidence=Confidence.HIGH,
        metric_name="exact",
        classification=Classification.PROVIDER_SIDE,
        evidence=(
            Evidence(
                kind="fingerprint_diff",
                description="x",
                data={"fingerprints": ["fp_a", "fp_b"], "unique": ["fp_a", "fp_b"]},
            ),
        ),
    )
    return PipelineAnalysis(
        analysis_id=aid,
        pipeline_name="agent.py",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr1, pr2),
        findings=(finding,),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
    )


def _inconclusive_analysis(aid: str) -> PipelineAnalysis:
    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    return PipelineAnalysis(
        analysis_id=aid,
        pipeline_name="agent.py",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr,),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
        capabilities=AdapterCapabilities(),
    )


# --- recent_analyses walker ---------------------------------------------------


def test_recent_analyses_returns_empty_when_no_artifacts(tmp_path: Path) -> None:
    assert recent_analyses(tmp_path) == []


def test_recent_analyses_returns_most_recent_first(tmp_path: Path) -> None:
    save(_clean_analysis("first"), base_dir=tmp_path)
    time.sleep(0.02)
    save(_clean_analysis("second"), base_dir=tmp_path)
    time.sleep(0.02)
    save(_clean_analysis("third"), base_dir=tmp_path)
    out = recent_analyses(tmp_path)
    assert [a.analysis_id for a in out] == ["third", "second", "first"]


def test_recent_analyses_respects_limit(tmp_path: Path) -> None:
    for i in range(5):
        save(_clean_analysis(f"a{i}"), base_dir=tmp_path)
        time.sleep(0.01)
    out = recent_analyses(tmp_path, limit=2)
    assert len(out) == 2


def test_recent_analyses_applies_predicate(tmp_path: Path) -> None:
    save(_clean_analysis("clean-one"), base_dir=tmp_path)
    time.sleep(0.01)
    save(_source_analysis("has-source"), base_dir=tmp_path)
    out = recent_analyses(tmp_path, predicate=lambda a: bool(a.findings))
    assert [a.analysis_id for a in out] == ["has-source"]


def test_recent_analyses_skips_unreadable_artifacts(tmp_path: Path) -> None:
    save(_clean_analysis("good"), base_dir=tmp_path)
    # Future-schema artifact: refused by storage, should be skipped silently.
    future = tmp_path / "future.json"
    data = _clean_analysis("future").to_dict()
    data["schema_version"] = "9.99"
    future.write_text(json.dumps(data), encoding="utf-8")
    # Corrupt artifact: invalid JSON.
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")

    out = recent_analyses(tmp_path)
    assert [a.analysis_id for a in out] == ["good"]


# --- render_list --------------------------------------------------------------


def test_render_list_empty_says_no_analyses() -> None:
    assert render_list([], now=_T) == "no analyses found."


def test_render_list_clean_analysis_summary_is_clean() -> None:
    rendered = render_list([_clean_analysis("aaa11111")], now=_T)
    assert "clean" in rendered
    assert "aaa11111" in rendered


def test_render_list_source_analysis_propagates() -> None:
    rendered = render_list([_source_analysis("src1", propagating=True)], now=_T)
    assert "1 source (propagates)" in rendered


def test_render_list_source_analysis_absorbed() -> None:
    rendered = render_list([_source_analysis("abs1", propagating=False)], now=_T)
    assert "1 source (absorbed)" in rendered


def test_render_list_env_finding_summary() -> None:
    rendered = render_list([_env_finding_analysis("env1")], now=_T)
    assert "stable, routing varied" in rendered


def test_render_list_inconclusive_summary() -> None:
    rendered = render_list([_inconclusive_analysis("inc1")], now=_T)
    assert "inconclusive (n=1)" in rendered


def test_render_list_relative_time_uses_finished_at() -> None:
    rendered = render_list([_clean_analysis("aged")], now=_T + timedelta(hours=2))
    assert "2 hours ago" in rendered


def test_render_list_truncates_long_pipeline_names() -> None:
    long_name = "x" * 60
    analysis = _clean_analysis("longname")
    analysis = PipelineAnalysis(
        analysis_id=analysis.analysis_id,
        pipeline_name=long_name,
        n=analysis.n,
        metric_name=analysis.metric_name,
        schema_version=analysis.schema_version,
        runs=analysis.runs,
        findings=analysis.findings,
        started_at=analysis.started_at,
        finished_at=analysis.finished_at,
        total_cost=analysis.total_cost,
        capabilities=analysis.capabilities,
    )
    rendered = render_list([analysis], now=_T)
    assert "..." in rendered
    # Truncated name shouldn't bleed into the summary column.
    assert "clean" in rendered


def test_render_list_singular_count_in_footer() -> None:
    rendered = render_list([_clean_analysis("only")], now=_T)
    assert "showing 1 analysis." in rendered


def test_render_list_plural_count_in_footer() -> None:
    rendered = render_list([_clean_analysis("a"), _clean_analysis("b")], now=_T)
    assert "showing 2 analyses." in rendered


# --- execute_list -------------------------------------------------------------


def test_execute_list_against_real_run(tmp_path: Path) -> None:
    execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=3,
        base_dir=tmp_path,
        clock=FrozenClock(_T),
        rng=SequenceRng(["real-id"]),
    )
    rendered = execute_list(base_dir=tmp_path, clock=FrozenClock(_T))
    assert "real-id" in rendered
    assert "clean" in rendered
    assert "varix.adapters:FakeAdapter" in rendered


def test_execute_list_empty_dir(tmp_path: Path) -> None:
    rendered = execute_list(base_dir=tmp_path, clock=FrozenClock(_T))
    assert rendered == "no analyses found."


# --- CLI ----------------------------------------------------------------------


def test_cli_list_command_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    runner.invoke(app, ["run", "varix.adapters:FakeAdapter", "--input", "hi", "-n", "3"])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "recent analyses:" in result.output
    assert "clean" in result.output


def test_cli_list_empty_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "no analyses found." in result.output


def test_cli_list_appears_in_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    import re

    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "list" in clean
