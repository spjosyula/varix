"""Tests for the pipeline runner and cost accumulator."""

from __future__ import annotations

import pytest

from varix.adapters import FakeAdapter
from varix.core import Classification, CostSnapshot
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
