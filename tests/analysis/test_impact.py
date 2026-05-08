"""Tests for ImpactEstimator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from varix.adapters import FakeAdapter
from varix.analysis import ImpactBehavior, ImpactEstimator
from varix.core import (
    Classification,
    Confidence,
    ExactMatch,
    PipelineRun,
    StepRun,
)
from varix.execution import run_n

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def _run(*pairs: tuple[str, object]) -> PipelineRun:
    """Build a PipelineRun from (step_id, output) pairs."""
    return PipelineRun(
        run_id="r",
        step_runs=tuple(StepRun(step_id=sid, inputs="i", output=out) for sid, out in pairs),
        started_at=_T,
        finished_at=_T,
    )


def test_source_output_stable_returns_absorbed_high() -> None:
    runs = [
        _run(("s1", "x"), ("s2", "y"), ("s3", "z")),
        _run(("s1", "x"), ("s2", "y"), ("s3", "z")),
    ]
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.ABSORBED
    assert report.confidence is Confidence.HIGH
    assert report.source_step_id == "s2"


def test_source_varies_final_stable_returns_absorbed() -> None:
    # Source (s2) varies, but final (s3) is the same across runs → absorbed.
    runs = [
        _run(("s1", "in"), ("s2", "a"), ("s3", "FINAL")),
        _run(("s1", "in"), ("s2", "b"), ("s3", "FINAL")),
    ]
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.ABSORBED
    assert report.confidence is Confidence.HIGH
    assert report.final_step_id == "s3"
    assert report.evidence[0].kind == "source_to_final_diff"


def test_source_varies_final_varies_returns_propagates() -> None:
    runs = [
        _run(("s1", "in"), ("s2", "a"), ("s3", "FINAL_a")),
        _run(("s1", "in"), ("s2", "b"), ("s3", "FINAL_b")),
    ]
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.PROPAGATES
    assert report.confidence is Confidence.HIGH
    assert report.evidence[0].data["source_unique_outputs"] == 2
    assert report.evidence[0].data["final_unique_outputs"] == 2


def test_source_is_final_step_returns_absorbed() -> None:
    runs = [_run(("s1", "in"), ("s2", "a")), _run(("s1", "in"), ("s2", "b"))]
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.ABSORBED
    assert report.confidence is Confidence.HIGH
    assert "final step" in report.reason


def test_fewer_than_two_runs_returns_unavailable() -> None:
    runs = [_run(("s1", "in"), ("s2", "a"))]
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.ABSORBED  # default safe verdict
    assert report.confidence is Confidence.UNAVAILABLE


def test_zero_runs_returns_unavailable() -> None:
    report = ImpactEstimator().estimate([], "s2")
    assert report.confidence is Confidence.UNAVAILABLE


def test_step_not_in_runs_returns_unavailable() -> None:
    runs = [_run(("s1", "x")), _run(("s1", "x"))]
    report = ImpactEstimator().estimate(runs, "no_such_step")
    assert report.confidence is Confidence.UNAVAILABLE


@pytest.mark.asyncio
async def test_against_fake_adapter_prompt_side_propagates() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    # FakeAdapter propagates inputs through outputs, so PROMPT_SIDE on s2
    # makes s5's output vary too → PROPAGATES.
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.PROPAGATES
    assert report.confidence is Confidence.HIGH


@pytest.mark.asyncio
async def test_against_fake_adapter_deterministic_step_absorbed() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    report = ImpactEstimator().estimate(runs, "s2")
    assert report.behavior is ImpactBehavior.ABSORBED


def test_estimator_accepts_custom_metric() -> None:
    metric = ExactMatch()
    estimator = ImpactEstimator(metric=metric)
    runs = [_run(("s1", "x")), _run(("s1", "x"))]
    report = estimator.estimate(runs, "s1")
    assert report.behavior is ImpactBehavior.ABSORBED
