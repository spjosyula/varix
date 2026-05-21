"""Replay a single pipeline step N times with fixed inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from varix.core import (
    Adapter,
    CapabilityMissing,
    ExactMatch,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    VarianceMetric,
)
from varix.execution.runner import CostAccumulator


async def replay_n(
    adapter: Adapter,
    step_id: str,
    fixed_inputs: Any,
    n: int,
) -> list[StepRun]:
    """Replay `step_id` `n` times against `adapter` with `fixed_inputs` held constant.

    Raises `CapabilityMissing` if the adapter does not declare `supports_replay`.
    """
    if not adapter.capabilities().supports_replay:
        raise CapabilityMissing(
            f"adapter {type(adapter).__name__} does not declare supports_replay; "
            "cannot replay steps"
        )
    runs: list[StepRun] = []
    for _ in range(n):
        runs.append(await adapter.replay_step(step_id, fixed_inputs))
    return runs


async def gather_disambiguation_replays(
    adapter: Adapter,
    runs: Sequence[PipelineRun],
    outcomes: Mapping[str, LocalizationOutcome],
    *,
    k: int = 3,
    metric: VarianceMetric | None = None,
    cost: CostAccumulator | None = None,
    max_cost: float | None = None,
) -> dict[str, list[StepRun]]:
    """Replay each DOWNSTREAM step with output variance, holding its inputs
    fixed at the first run's values. Reveals whether the step has its own
    variance source vs. just cascading from upstream.

    No-ops when the adapter doesn't support replay. Honors `max_cost`. If
    `replay_step` raises for a step, that step's gathering stops; others
    continue.
    """
    if not adapter.capabilities().supports_replay:
        return {}
    actual_metric = metric if metric is not None else ExactMatch()
    replays: dict[str, list[StepRun]] = {}

    for step_id, outcome in outcomes.items():
        if outcome is not LocalizationOutcome.DOWNSTREAM:
            continue
        observations = [
            sr for run in runs for sr in run.step_runs if sr.step_id == step_id
        ]
        if len(observations) < 2 or not _outputs_vary(observations, actual_metric):
            continue

        fixed_inputs = observations[0].inputs
        step_replays: list[StepRun] = []
        for _ in range(k):
            if (
                cost is not None
                and max_cost is not None
                and cost.snapshot().dollars >= max_cost
            ):
                break
            try:
                sr = await adapter.replay_step(step_id, fixed_inputs)
            except Exception:
                break
            if cost is not None:
                cost.add(sr.cost)
            step_replays.append(sr)
        if step_replays:
            replays[step_id] = step_replays
    return replays


def _outputs_vary(step_runs: Sequence[StepRun], metric: VarianceMetric) -> bool:
    if len(step_runs) < 2:
        return False
    first = step_runs[0].output
    return any(not metric.equivalent(first, sr.output) for sr in step_runs[1:])
