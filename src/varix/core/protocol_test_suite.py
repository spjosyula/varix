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
from varix.core.variance import ExactMatch

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


async def check_step_inputs_track_upstream_state(
    adapter: Adapter, sample_input: Any
) -> None:
    """Two runs with the same input: structure must match, and any step
    whose upstream is stable must have stable `inputs`. Catches LLM prompts
    or other run-local state being leaked into the inputs field — the
    footgun that silently masks SOURCE steps as DOWNSTREAM. Cascading
    variance from a legitimately-varying upstream step is not flagged.
    """
    run_a = await adapter.run_pipeline(sample_input)
    run_b = await adapter.run_pipeline(sample_input)

    ids_a = [sr.step_id for sr in run_a.step_runs]
    ids_b = [sr.step_id for sr in run_b.step_runs]
    if ids_a != ids_b:
        raise AdapterError(
            "adapter produced different step sequences across two runs with "
            f"the same input: {ids_a!r} vs {ids_b!r}. Pipeline structure must "
            "be deterministic for varix to analyze it."
        )

    metric = ExactMatch()
    for i, (sr_a, sr_b) in enumerate(zip(run_a.step_runs, run_b.step_runs, strict=False)):
        upstream_stable = all(
            metric.equivalent(run_a.step_runs[j].output, run_b.step_runs[j].output)
            for j in range(i)
        )
        if upstream_stable and not metric.equivalent(sr_a.inputs, sr_b.inputs):
            raise AdapterError(
                f"step {i} ({sr_a.step_id!r}): inputs varied across two runs "
                "even though every upstream step produced equivalent output. "
                "StepRun.inputs must be the logical pipeline input that flowed "
                "in from upstream, not the LLM prompt or other run-local "
                "state. See StepRun docstring for the contract."
            )


async def validate_adapter(adapter: Adapter, sample_input: Any) -> None:
    """Run every conformance check. Raises `AdapterError` on first violation."""
    check_methods_are_async(adapter)
    check_capabilities_idempotent(adapter)
    await check_pipeline_structure_stable(adapter, sample_input)
    await check_run_pipeline_aligns_with_structure(adapter, sample_input)
    await check_replay_step_does_not_mutate_inputs(adapter, sample_input)
    await check_step_inputs_track_upstream_state(adapter, sample_input)
