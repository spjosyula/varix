"""Run a pipeline N times via an adapter and accumulate per-run costs."""

from __future__ import annotations

from typing import Any

from varix.core import Adapter, CostSnapshot, PipelineRun


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
) -> list[PipelineRun]:
    """Execute the pipeline `n` times. Optionally accumulate per-run costs."""
    runs: list[PipelineRun] = []
    for _ in range(n):
        run = await adapter.run_pipeline(pipeline_input)
        runs.append(run)
        if cost is not None:
            cost.add(run.cost)
    return runs
