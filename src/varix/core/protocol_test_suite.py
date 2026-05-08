"""Conformance checks every Adapter must pass.

Adapter authors run these against their adapter to validate the contract.
A check raises `AdapterError` on violation; the message names the rule.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

from varix.core.adapter import Adapter
from varix.core.errors import AdapterError

_ASYNC_METHODS: tuple[str, ...] = ("pipeline_structure", "run_pipeline", "replay_step")


def check_methods_are_async(adapter: Adapter) -> None:
    """Each Protocol-declared `async def` method must actually be a coroutine function."""
    for method_name in _ASYNC_METHODS:
        if not inspect.iscoroutinefunction(getattr(adapter, method_name)):
            raise AdapterError(
                f"adapter method {method_name!r} must be `async def`; "
                "Protocol declares it async but found a regular function"
            )


def check_capabilities_idempotent(adapter: Adapter) -> None:
    """`capabilities()` must return equal results across repeated calls."""
    first = adapter.capabilities()
    second = adapter.capabilities()
    if first != second:
        raise AdapterError(f"capabilities() is not idempotent: {first!r} then {second!r}")


async def check_pipeline_structure_stable(adapter: Adapter, sample_input: Any) -> None:
    """`pipeline_structure(input)` must return equal graphs across calls."""
    a = await adapter.pipeline_structure(sample_input)
    b = await adapter.pipeline_structure(sample_input)
    if a != b:
        raise AdapterError(
            f"pipeline_structure is unstable across calls for the same input: {a!r} vs {b!r}"
        )


async def check_run_pipeline_aligns_with_structure(adapter: Adapter, sample_input: Any) -> None:
    """`run_pipeline` must produce one StepRun per declared step, in order."""
    graph = await adapter.pipeline_structure(sample_input)
    run = await adapter.run_pipeline(sample_input)
    declared = [s.id for s in graph.steps]
    actual = [sr.step_id for sr in run.step_runs]
    if declared != actual:
        raise AdapterError(
            f"run_pipeline step IDs do not match pipeline_structure: "
            f"declared={declared!r}, actual={actual!r}"
        )


async def check_replay_step_does_not_mutate_inputs(adapter: Adapter, sample_input: Any) -> None:
    """`replay_step` must not mutate `fixed_inputs`. Skipped when supports_replay is False."""
    if not adapter.capabilities().supports_replay:
        return

    graph = await adapter.pipeline_structure(sample_input)
    if not graph.steps:
        return

    run = await adapter.run_pipeline(sample_input)
    if not run.step_runs:
        return

    target = run.step_runs[0]
    fixed_inputs = target.inputs
    snapshot = copy.deepcopy(fixed_inputs)
    await adapter.replay_step(target.step_id, fixed_inputs)
    if fixed_inputs != snapshot:
        raise AdapterError("replay_step mutated fixed_inputs; inputs must be treated as immutable")


async def validate_adapter(adapter: Adapter, sample_input: Any) -> None:
    """Run every conformance check. Raises `AdapterError` on first violation."""
    check_methods_are_async(adapter)
    check_capabilities_idempotent(adapter)
    await check_pipeline_structure_stable(adapter, sample_input)
    await check_run_pipeline_aligns_with_structure(adapter, sample_input)
    await check_replay_step_does_not_mutate_inputs(adapter, sample_input)
