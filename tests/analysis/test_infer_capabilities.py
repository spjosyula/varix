"""Tests for `infer_capabilities` — the heuristic fallback for legacy artifacts."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from varix.adapters import FakeAdapter
from varix.analysis import analyze, infer_capabilities
from varix.core import (
    SCHEMA_VERSION,
    AdapterCapabilities,
    Classification,
    CostSnapshot,
    ExactMatch,
    PipelineAnalysis,
    PipelineRun,
    StepRun,
    ToolCall,
)
from varix.execution import run_n

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def _analysis(runs: tuple[PipelineRun, ...]) -> PipelineAnalysis:
    return PipelineAnalysis(
        analysis_id="x",
        pipeline_name="x",
        n=len(runs),
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=runs,
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )


def test_infer_returns_all_false_when_runs_have_no_signal() -> None:
    sr = StepRun(step_id="s1", inputs="i", output="o")
    runs = (PipelineRun(run_id="r1", step_runs=(sr,), started_at=_T, finished_at=_T),)
    inferred = infer_capabilities(_analysis(runs))
    assert inferred == AdapterCapabilities(
        exposes_fingerprint=False, exposes_tool_calls=False, supports_replay=False
    )


def test_infer_detects_fingerprint_when_any_run_has_it() -> None:
    sr_with_fp = StepRun(
        step_id="s1", inputs="i", output="o", provider_metadata={"system_fingerprint": "fp_a"}
    )
    sr_without = StepRun(step_id="s1", inputs="i", output="o")
    runs = (
        PipelineRun(run_id="r1", step_runs=(sr_with_fp,), started_at=_T, finished_at=_T),
        PipelineRun(run_id="r2", step_runs=(sr_without,), started_at=_T, finished_at=_T),
    )
    inferred = infer_capabilities(_analysis(runs))
    assert inferred.exposes_fingerprint is True


def test_infer_detects_tool_calls_when_any_run_has_them() -> None:
    sr_with_tools = StepRun(
        step_id="s1",
        inputs="i",
        output="o",
        tool_calls=(ToolCall(name="search", arguments={"q": "x"}, result="r"),),
    )
    runs = (PipelineRun(run_id="r1", step_runs=(sr_with_tools,), started_at=_T, finished_at=_T),)
    inferred = infer_capabilities(_analysis(runs))
    assert inferred.exposes_tool_calls is True


def test_infer_detects_supports_replay_when_step_replays_present() -> None:
    sr = StepRun(step_id="s1", inputs="i", output="o")
    runs = (PipelineRun(run_id="r1", step_runs=(sr,), started_at=_T, finished_at=_T),)
    analysis = dataclasses.replace(_analysis(runs), step_replays={"s1": (sr,)})
    inferred = infer_capabilities(analysis)
    assert inferred.supports_replay is True


@pytest.mark.asyncio
async def test_heuristic_findings_match_recorded_findings() -> None:
    """The strategic guarantee, stated as findings-equivalence rather than
    capabilities-equality: re-analyzing a legacy 0.1 artifact with heuristic
    capabilities produces the same findings as the 0.2 path with recorded
    capabilities. Drift here means replay would silently produce different
    findings — exactly what the schema commitment forbids."""
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    runs_tuple = tuple(runs)

    recorded_caps = adapter.capabilities()
    legacy_analysis = PipelineAnalysis(
        analysis_id="legacy",
        pipeline_name="fake",
        n=len(runs_tuple),
        metric_name="exact",
        schema_version="0.1",
        runs=runs_tuple,
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
        capabilities=None,
    )
    inferred_caps = infer_capabilities(legacy_analysis)

    # The two flags classifiers actually consume must agree.
    assert inferred_caps.exposes_fingerprint == recorded_caps.exposes_fingerprint
    assert inferred_caps.exposes_tool_calls == recorded_caps.exposes_tool_calls

    metric = ExactMatch()
    legacy_findings = analyze(runs_tuple, inferred_caps, metric=metric).findings
    recorded_findings = analyze(runs_tuple, recorded_caps, metric=metric).findings
    assert legacy_findings == recorded_findings
