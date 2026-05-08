"""Run a pipeline N times via an adapter and accumulate per-run costs."""

from __future__ import annotations

import json
from typing import Any

from varix.core import Adapter, BudgetExceeded, CostSnapshot, PipelineRun, RunFailed, VarixError


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

    If the adapter raises any non-`VarixError` exception during a run, the
    failure is wrapped in `RunFailed` carrying every run that completed
    cleanly before the failure. The original exception is chained via
    `__cause__`. Varix-typed errors (e.g. `BudgetExceeded`, `AdapterError`)
    pass through unchanged.

    Each completed run is validated for JSON-serializability before it is
    accepted; an unsavable run is also surfaced as `RunFailed` so we abort
    on the first offender rather than discovering the problem at save time
    (after N runs of cost). The bad run is *not* included in `partial_runs`.
    """
    if max_cost is not None and cost is None:
        cost = CostAccumulator()
    runs: list[PipelineRun] = []
    for i in range(n):
        try:
            run = await adapter.run_pipeline(pipeline_input)
        except VarixError:
            raise
        except Exception as exc:
            raise RunFailed(
                f"run {i + 1} of {n} failed: {type(exc).__name__}: {exc}",
                partial_runs=tuple(runs),
            ) from exc
        try:
            json.dumps(run.to_dict())
        except (TypeError, ValueError, RecursionError) as exc:
            raise RunFailed(
                f"run {i + 1} of {n} produced non-JSON-serializable output: {exc}",
                partial_runs=tuple(runs),
            ) from exc
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
