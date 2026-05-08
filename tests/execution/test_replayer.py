"""Tests for the step replayer."""

from __future__ import annotations

from typing import Any

import pytest

from varix.adapters import FakeAdapter
from varix.core import (
    Adapter,
    AdapterCapabilities,
    CapabilityMissing,
    Classification,
    PipelineRun,
    StepGraph,
    StepRun,
)
from varix.execution import replay_n


@pytest.mark.asyncio
async def test_replay_n_returns_n_step_runs() -> None:
    runs = await replay_n(FakeAdapter(), "s2", "hello", n=4)
    assert len(runs) == 4
    assert all(r.step_id == "s2" for r in runs)


@pytest.mark.asyncio
async def test_replay_n_with_zero_returns_empty_list() -> None:
    runs = await replay_n(FakeAdapter(), "s2", "hello", n=0)
    assert runs == []


@pytest.mark.asyncio
async def test_deterministic_replay_yields_identical_step_runs() -> None:
    runs = await replay_n(FakeAdapter(), "s3", "hello", n=3)
    assert all(r == runs[0] for r in runs)


@pytest.mark.asyncio
async def test_provider_side_variance_yields_varying_metadata_under_replay() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await replay_n(adapter, "s2", "hello", n=3)
    fingerprints = {str(r.provider_metadata) for r in runs}
    assert len(fingerprints) >= 2


@pytest.mark.asyncio
async def test_tool_side_variance_yields_varying_results_under_replay() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs = await replay_n(adapter, "s2", "hello", n=3)
    results = {r.tool_calls[0].result for r in runs}
    assert len(results) == 3


class _NoReplayAdapter:
    """Adapter that explicitly does not support replay."""

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            exposes_fingerprint=True,
            exposes_tool_calls=True,
            supports_replay=False,
        )

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        raise NotImplementedError

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        raise NotImplementedError

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_replay_n_refuses_when_capability_missing() -> None:
    adapter: Adapter = _NoReplayAdapter()
    with pytest.raises(CapabilityMissing, match="supports_replay"):
        await replay_n(adapter, "s2", "hello", n=3)


@pytest.mark.asyncio
async def test_replay_n_refuses_before_any_call() -> None:
    # _NoReplayAdapter.replay_step raises NotImplementedError; if we ever
    # reached it, the test would fail with that instead of CapabilityMissing.
    adapter: Adapter = _NoReplayAdapter()
    with pytest.raises(CapabilityMissing):
        await replay_n(adapter, "s2", "hello", n=1)
