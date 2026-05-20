"""Tests for the protocol_test_suite checks.

Each check is exercised against a deliberately broken adapter that violates
exactly one rule, plus a clean adapter that should pass everything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.core import (
    AdapterCapabilities,
    PipelineRun,
    Step,
    StepGraph,
    StepRun,
)
from varix.core.errors import AdapterError
from varix.core.protocol_test_suite import (
    check_capabilities_idempotent,
    check_methods_are_async,
    check_pipeline_structure_stable,
    check_replay_step_does_not_mutate_inputs,
    check_run_pipeline_aligns_with_structure,
    check_step_inputs_track_upstream_state,
    validate_adapter,
)


def _now() -> datetime:
    return datetime.now(UTC)


class _CleanAdapter:
    """Adapter that satisfies every check."""

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(supports_replay=True)

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(
            steps=(
                Step(id="s1", name="planner", index=0),
                Step(id="s2", name="responder", index=1),
            )
        )

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        runs = (
            StepRun(step_id="s1", inputs=pipeline_input, output="planned"),
            StepRun(step_id="s2", inputs="planned", output="answered"),
        )
        now = _now()
        return PipelineRun(run_id="r1", step_runs=runs, started_at=now, finished_at=now)

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        return StepRun(step_id=step_id, inputs=fixed_inputs, output="replayed")


class _FlakyCapabilities(_CleanAdapter):
    """capabilities() returns different results across calls."""

    _calls = 0

    def capabilities(self) -> AdapterCapabilities:
        self._calls += 1
        return AdapterCapabilities(supports_replay=True, exposes_fingerprint=self._calls % 2 == 0)


class _UnstableStructure(_CleanAdapter):
    """pipeline_structure returns different graphs on each call."""

    _calls = 0

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        type(self)._calls += 1
        if type(self)._calls % 2 == 0:
            return StepGraph(steps=(Step(id="different", name="other", index=0),))
        return await super().pipeline_structure(pipeline_input)


class _MisalignedRun(_CleanAdapter):
    """run_pipeline returns step IDs that don't match pipeline_structure."""

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        runs = (
            StepRun(step_id="s1", inputs=pipeline_input, output="planned"),
            StepRun(step_id="WRONG", inputs="planned", output="answered"),
        )
        now = _now()
        return PipelineRun(run_id="r1", step_runs=runs, started_at=now, finished_at=now)


class _MutatingReplay(_CleanAdapter):
    """replay_step mutates fixed_inputs."""

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        if isinstance(fixed_inputs, dict):
            fixed_inputs["__poisoned__"] = True
        return StepRun(step_id=step_id, inputs=fixed_inputs, output="replayed")


@pytest.mark.asyncio
async def test_clean_adapter_passes_full_suite() -> None:
    await validate_adapter(_CleanAdapter(), {"q": "hello"})


def test_flaky_capabilities_is_caught() -> None:
    with pytest.raises(AdapterError, match="idempotent"):
        check_capabilities_idempotent(_FlakyCapabilities())


@pytest.mark.asyncio
async def test_unstable_structure_is_caught() -> None:
    with pytest.raises(AdapterError, match="unstable"):
        await check_pipeline_structure_stable(_UnstableStructure(), {"q": "hi"})


@pytest.mark.asyncio
async def test_misaligned_run_is_caught() -> None:
    with pytest.raises(AdapterError, match="do not match"):
        await check_run_pipeline_aligns_with_structure(_MisalignedRun(), {"q": "hi"})


@pytest.mark.asyncio
async def test_mutating_replay_is_caught() -> None:
    with pytest.raises(AdapterError, match="mutated"):
        await check_replay_step_does_not_mutate_inputs(_MutatingReplay(), {"q": "hi"})


def test_clean_adapter_passes_async_methods_check() -> None:
    check_methods_are_async(_CleanAdapter())


def test_sync_run_pipeline_method_is_caught() -> None:
    class _SyncRun(_CleanAdapter):
        def run_pipeline(  # type: ignore[override]
            self, pipeline_input: Any, seed: int | None = None
        ) -> PipelineRun:
            now = _now()
            return PipelineRun(run_id="r", step_runs=(), started_at=now, finished_at=now)

    with pytest.raises(AdapterError, match=r"run_pipeline.*async def"):
        check_methods_are_async(_SyncRun())


@pytest.mark.asyncio
async def test_replay_check_skipped_when_capability_false() -> None:
    class _NoReplay(_CleanAdapter):
        def capabilities(self) -> AdapterCapabilities:
            return AdapterCapabilities(supports_replay=False)

        async def replay_step(
            self, step_id: str, fixed_inputs: Any, seed: int | None = None
        ) -> StepRun:
            raise AssertionError("should not be called")

    await check_replay_step_does_not_mutate_inputs(_NoReplay(), {"q": "hi"})


# --- StepRun.inputs contract -------------------------------------------------


@pytest.mark.asyncio
async def test_clean_adapter_satisfies_inputs_contract() -> None:
    """The base _CleanAdapter passes upstream state through to inputs cleanly."""
    await check_step_inputs_track_upstream_state(_CleanAdapter(), {"q": "hi"})


@pytest.mark.asyncio
async def test_structural_variance_across_runs_is_caught() -> None:
    """run_pipeline produces a different step sequence on the second call —
    structure must be deterministic for the same input."""

    class _AlternatingStructure(_CleanAdapter):
        _calls = 0

        async def run_pipeline(
            self, pipeline_input: Any, seed: int | None = None
        ) -> PipelineRun:
            type(self)._calls += 1
            now = _now()
            if type(self)._calls % 2 == 0:
                # Truncated second run.
                step_runs: tuple[StepRun, ...] = (
                    StepRun(step_id="s1", inputs=pipeline_input, output="planned"),
                )
            else:
                step_runs = (
                    StepRun(step_id="s1", inputs=pipeline_input, output="planned"),
                    StepRun(step_id="s2", inputs="planned", output="answered"),
                )
            return PipelineRun(
                run_id="r", step_runs=step_runs, started_at=now, finished_at=now
            )

    with pytest.raises(AdapterError, match="different step sequences"):
        await check_step_inputs_track_upstream_state(
            _AlternatingStructure(), {"q": "hi"}
        )


@pytest.mark.asyncio
async def test_inputs_leaking_run_local_state_is_caught() -> None:
    """Adapter that bakes a per-call counter into StepRun.inputs is leaking
    run-local state — the exact footgun the dogfood tripped earlier."""

    class _LeakyInputs(_CleanAdapter):
        _calls = 0

        async def run_pipeline(
            self, pipeline_input: Any, seed: int | None = None
        ) -> PipelineRun:
            type(self)._calls += 1
            n = type(self)._calls
            now = _now()
            return PipelineRun(
                run_id=f"r{n}",
                step_runs=(
                    StepRun(
                        step_id="s1",
                        # Bug: per-call value bleeds into the logical input.
                        inputs=f"prompt-with-counter-{n}",
                        output="planned",
                    ),
                ),
                started_at=now,
                finished_at=now,
            )

    with pytest.raises(AdapterError, match="inputs varied"):
        await check_step_inputs_track_upstream_state(_LeakyInputs(), {"q": "hi"})


@pytest.mark.asyncio
async def test_downstream_cascade_does_not_trip_inputs_contract() -> None:
    """When step 0 legitimately varies across runs, step 1's inputs cascade
    differently — that's correct and must not be flagged."""

    class _CascadingVariance(_CleanAdapter):
        _calls = 0

        async def run_pipeline(
            self, pipeline_input: Any, seed: int | None = None
        ) -> PipelineRun:
            type(self)._calls += 1
            n = type(self)._calls
            step0_output = f"planned-v{n}"  # legitimate upstream variance
            now = _now()
            return PipelineRun(
                run_id=f"r{n}",
                step_runs=(
                    StepRun(step_id="s1", inputs=pipeline_input, output=step0_output),
                    StepRun(step_id="s2", inputs=step0_output, output="answered"),
                ),
                started_at=now,
                finished_at=now,
            )

    # s2.inputs differs across runs, but only because s1.output legitimately
    # varied — the conditional invariant must allow this.
    await check_step_inputs_track_upstream_state(_CascadingVariance(), {"q": "hi"})
