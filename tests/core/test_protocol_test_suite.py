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
    check_pipeline_structure_stable,
    check_replay_step_does_not_mutate_inputs,
    check_run_pipeline_aligns_with_structure,
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
