"""Replay a single pipeline step N times with fixed inputs."""

from __future__ import annotations

from typing import Any

from varix.core import Adapter, CapabilityMissing, StepRun


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
