"""Run a pipeline N times via an adapter and accumulate per-run costs."""

from __future__ import annotations

from typing import Any

from varix.core import Adapter, BudgetExceeded, CostSnapshot, PipelineRun


class CostAccumulator:
    """Mutable accumulator for `CostSnapshot` values across multiple runs."""

    def __init__(self) -> None:
        self._total = CostSnapshot()

    def add(self, snapshot: CostSnapshot) -> None:
        self._total = self._total + snapshot

    def snapshot(self) -> CostSnapshot:
        return self._total


async def run_n(
    adapter: Adapter,
    pipeline_input: Any,
    n: int,
    *,
    cost: CostAccumulator | None = None,
    max_cost: float | None = None,
) -> list[PipelineRun]:
    """Execute the pipeline `n` times.

    If `max_cost` is given, raises `BudgetExceeded` after the first run that
    pushes accumulated dollars over the budget; the exception's `partial_runs`
    contains every run that completed first.
    """
    if max_cost is not None and cost is None:
        cost = CostAccumulator()
    runs: list[PipelineRun] = []
    for _ in range(n):
        run = await adapter.run_pipeline(pipeline_input)
        runs.append(run)
        if cost is not None:
            cost.add(run.cost)
            if max_cost is not None and cost.snapshot().dollars > max_cost:
                raise BudgetExceeded(
                    f"max_cost ${max_cost} exceeded after {len(runs)} run(s) "
                    f"(actual: ${cost.snapshot().dollars:.4f})",
                    partial_runs=tuple(runs),
                )
    return runs
