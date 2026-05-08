"""Tests for the pipeline runner and cost accumulator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.adapters import FakeAdapter
from varix.core import (
    Adapter,
    AdapterCapabilities,
    AdapterError,
    BudgetExceeded,
    Classification,
    CostSnapshot,
    PipelineRun,
    RunFailed,
    StepGraph,
    StepRun,
)
from varix.execution import CostAccumulator, run_n


@pytest.mark.asyncio
async def test_run_n_returns_n_pipeline_runs() -> None:
    runs = await run_n(FakeAdapter(), "hello", n=5)
    assert len(runs) == 5


@pytest.mark.asyncio
async def test_run_n_with_zero_returns_empty_list() -> None:
    runs = await run_n(FakeAdapter(), "hello", n=0)
    assert runs == []


@pytest.mark.asyncio
async def test_deterministic_mode_yields_identical_step_runs() -> None:
    runs = await run_n(FakeAdapter(), "hello", n=3)
    first = runs[0].step_runs
    for r in runs[1:]:
        assert r.step_runs == first


@pytest.mark.asyncio
async def test_provider_side_variance_yields_varying_metadata() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    fingerprints = {str(r.step_runs[1].provider_metadata) for r in runs}
    assert len(fingerprints) >= 2


@pytest.mark.asyncio
async def test_tool_side_variance_yields_varying_tool_results() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    results = {r.step_runs[1].tool_calls[0].result for r in runs}
    assert len(results) == 3


def test_cost_accumulator_starts_at_zero() -> None:
    cost = CostAccumulator()
    assert cost.snapshot() == CostSnapshot()


def test_cost_accumulator_sums_snapshots() -> None:
    cost = CostAccumulator()
    cost.add(CostSnapshot(input_tokens=10, output_tokens=5, dollars=0.01))
    cost.add(CostSnapshot(input_tokens=20, output_tokens=10, dollars=0.02))
    total = cost.snapshot()
    assert total.input_tokens == 30
    assert total.output_tokens == 15
    assert abs(total.dollars - 0.03) < 1e-9


@pytest.mark.asyncio
async def test_run_n_accumulates_costs_when_supplied() -> None:
    cost = CostAccumulator()
    await run_n(FakeAdapter(), "hello", n=3, cost=cost)
    # FakeAdapter reports zero cost; accumulator stays at zero but the
    # call path is exercised.
    assert cost.snapshot() == CostSnapshot()


@pytest.mark.asyncio
async def test_run_n_without_accumulator_does_not_crash() -> None:
    runs = await run_n(FakeAdapter(), "hello", n=2)
    assert len(runs) == 2


_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


class _CostlyAdapter:
    """Test adapter that reports a fixed dollar cost per run."""

    def __init__(self, dollars_per_run: float) -> None:
        self._dollars = dollars_per_run

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(steps=())

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        return PipelineRun(
            run_id="r",
            step_runs=(),
            started_at=_T,
            finished_at=_T,
            cost=CostSnapshot(dollars=self._dollars),
        )

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_run_n_under_budget_completes() -> None:
    adapter: Adapter = _CostlyAdapter(dollars_per_run=0.001)
    runs = await run_n(adapter, "x", n=5, max_cost=1.0)
    assert len(runs) == 5


@pytest.mark.asyncio
async def test_run_n_without_max_cost_does_not_enforce_budget() -> None:
    adapter: Adapter = _CostlyAdapter(dollars_per_run=10.0)
    runs = await run_n(adapter, "x", n=4)
    assert len(runs) == 4


@pytest.mark.asyncio
async def test_run_n_raises_budget_exceeded_after_limit() -> None:
    # 1st run: $0.6 (under $1.0)
    # 2nd run: $1.2 (over)  → halts after 2 runs
    adapter: Adapter = _CostlyAdapter(dollars_per_run=0.6)
    with pytest.raises(BudgetExceeded) as ei:
        await run_n(adapter, "x", n=10, max_cost=1.0)
    assert len(ei.value.partial_runs) == 2


@pytest.mark.asyncio
async def test_budget_exact_match_is_not_exceeded() -> None:
    # 1st: $0.5, 2nd: $1.0 (== budget, not over). 3rd: $1.5 → halts after 3.
    adapter: Adapter = _CostlyAdapter(dollars_per_run=0.5)
    with pytest.raises(BudgetExceeded) as ei:
        await run_n(adapter, "x", n=10, max_cost=1.0)
    assert len(ei.value.partial_runs) == 3


@pytest.mark.asyncio
async def test_budget_uses_caller_accumulator_when_provided() -> None:
    adapter: Adapter = _CostlyAdapter(dollars_per_run=0.4)
    cost = CostAccumulator()
    with pytest.raises(BudgetExceeded):
        await run_n(adapter, "x", n=10, max_cost=1.0, cost=cost)
    # Caller's accumulator reflects spend up to (and including) the run that tripped.
    assert cost.snapshot().dollars > 1.0


class _FailingAdapter:
    """Test adapter that succeeds for `succeed_for` runs then raises."""

    def __init__(self, succeed_for: int, exc: BaseException) -> None:
        self._succeed_for = succeed_for
        self._exc = exc
        self._calls = 0

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(steps=())

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        self._calls += 1
        if self._calls > self._succeed_for:
            raise self._exc
        return PipelineRun(run_id=f"r{self._calls}", step_runs=(), started_at=_T, finished_at=_T)

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_run_n_wraps_adapter_exception_in_run_failed() -> None:
    adapter: Adapter = _FailingAdapter(succeed_for=2, exc=RuntimeError("provider stalled"))
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "x", n=5)
    assert "run 3 of 5" in str(ei.value)
    assert "RuntimeError" in str(ei.value)
    assert "provider stalled" in str(ei.value)


@pytest.mark.asyncio
async def test_run_failed_carries_partial_runs_completed_before_failure() -> None:
    adapter: Adapter = _FailingAdapter(succeed_for=3, exc=RuntimeError("network blip"))
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "x", n=10)
    assert len(ei.value.partial_runs) == 3


@pytest.mark.asyncio
async def test_run_failed_chains_original_cause() -> None:
    original = RuntimeError("network blip")
    adapter: Adapter = _FailingAdapter(succeed_for=0, exc=original)
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "x", n=2)
    assert ei.value.__cause__ is original


@pytest.mark.asyncio
async def test_run_n_passes_through_varix_errors_unchanged() -> None:
    """`AdapterError` from the adapter is varix-typed; it should not be re-wrapped."""
    adapter: Adapter = _FailingAdapter(succeed_for=1, exc=AdapterError("malformed result"))
    with pytest.raises(AdapterError):
        await run_n(adapter, "x", n=3)


class _UnsavableAdapter:
    """Test adapter: `succeed_for` clean runs, then a run with non-JSON output."""

    def __init__(self, succeed_for: int) -> None:
        self._succeed_for = succeed_for
        self._calls = 0

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(steps=())

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        self._calls += 1
        if self._calls > self._succeed_for:
            bad = StepRun(step_id="s", inputs="i", output=object())
            return PipelineRun(
                run_id=f"r{self._calls}",
                step_runs=(bad,),
                started_at=_T,
                finished_at=_T,
            )
        return PipelineRun(run_id=f"r{self._calls}", step_runs=(), started_at=_T, finished_at=_T)

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_run_n_aborts_when_run_is_not_json_serializable() -> None:
    adapter: Adapter = _UnsavableAdapter(succeed_for=2)
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "x", n=5)
    assert "run 3 of 5" in str(ei.value)
    assert "non-JSON-serializable" in str(ei.value)


@pytest.mark.asyncio
async def test_unsavable_failure_carries_only_clean_partial_runs() -> None:
    """The bad run must NOT appear in partial_runs; only the runs that serialized."""
    adapter: Adapter = _UnsavableAdapter(succeed_for=3)
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "x", n=10)
    assert len(ei.value.partial_runs) == 3
    # And every preserved run round-trips through JSON cleanly.
    import json as _json

    for r in ei.value.partial_runs:
        _json.dumps(r.to_dict())


@pytest.mark.asyncio
async def test_unsavable_failure_chains_underlying_typeerror() -> None:
    adapter: Adapter = _UnsavableAdapter(succeed_for=0)
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "x", n=2)
    assert isinstance(ei.value.__cause__, TypeError)
