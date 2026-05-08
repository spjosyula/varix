"""Tests for the top-level analyze() entry point."""

from __future__ import annotations

import pytest

from varix.adapters import FakeAdapter
from varix.analysis import AnalysisResult, analyze
from varix.core import Classification, Confidence, LocalizationOutcome
from varix.execution import run_n


@pytest.mark.asyncio
async def test_deterministic_pipeline_produces_no_findings() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    assert isinstance(result, AnalysisResult)
    assert result.findings == ()
    assert all(o is LocalizationOutcome.DETERMINISTIC for o in result.outcomes.values())


@pytest.mark.asyncio
async def test_provider_side_variance_produces_provider_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    s2_findings = [f for f in result.findings if f.step_id == "s2"]
    assert len(s2_findings) == 1
    assert s2_findings[0].classification is Classification.PROVIDER_SIDE
    assert s2_findings[0].confidence is Confidence.HIGH


@pytest.mark.asyncio
async def test_prompt_side_variance_produces_residual_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    s2_findings = [f for f in result.findings if f.step_id == "s2"]
    assert len(s2_findings) == 1
    assert s2_findings[0].classification is Classification.PROMPT_SIDE
    assert s2_findings[0].confidence is Confidence.MEDIUM
    assert result.outcomes["s2"] is LocalizationOutcome.SOURCE
    for sid in ("s3", "s4", "s5"):
        assert result.outcomes[sid] is LocalizationOutcome.DOWNSTREAM


@pytest.mark.asyncio
async def test_time_or_state_variance_produces_low_finding_only() -> None:
    adapter = FakeAdapter(variance={"s5": Classification.TIME_OR_STATE})
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    s5_findings = [f for f in result.findings if f.step_id == "s5"]
    assert len(s5_findings) == 1
    assert s5_findings[0].classification is Classification.TIME_OR_STATE
    assert s5_findings[0].confidence is Confidence.LOW


@pytest.mark.asyncio
async def test_empty_runs_returns_empty_result() -> None:
    adapter = FakeAdapter()
    result = analyze([], adapter.capabilities())
    assert result.findings == ()
    assert result.outcomes == {}
