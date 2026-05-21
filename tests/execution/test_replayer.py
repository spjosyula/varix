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


# --- gather_disambiguation_replays --------------------------------------------


from varix.core import LocalizationOutcome  # noqa: E402
from varix.execution import gather_disambiguation_replays  # noqa: E402
from varix.execution.runner import CostAccumulator, run_n  # noqa: E402


@pytest.mark.asyncio
async def test_gather_returns_empty_when_replay_unsupported() -> None:
    """Adapter without replay capability: no replays attempted, no notes."""
    adapter: Adapter = _NoReplayAdapter()
    out, notes = await gather_disambiguation_replays(
        adapter, runs=[], outcomes={"s1": LocalizationOutcome.DOWNSTREAM}
    )
    assert out == {}
    assert notes == []


@pytest.mark.asyncio
async def test_gather_skips_deterministic_and_source_steps() -> None:
    """Only DOWNSTREAM steps with output variance trigger replays."""
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hi", n=3)
    outcomes = {
        "s1": LocalizationOutcome.DETERMINISTIC,
        "s2": LocalizationOutcome.SOURCE,
        # No DOWNSTREAM step → nothing to replay.
    }
    out, notes = await gather_disambiguation_replays(adapter, runs, outcomes, k=3)
    assert out == {}
    assert notes == []


@pytest.mark.asyncio
async def test_gather_replays_downstream_with_variance() -> None:
    """A DOWNSTREAM step whose outputs vary gets replayed k times."""
    # FakeAdapter with PROMPT_SIDE on s2 makes s2 a SOURCE and s3+ cascade as
    # DOWNSTREAM with varying outputs (the prompt-side suffix propagates).
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    outcomes = {"s3": LocalizationOutcome.DOWNSTREAM}
    out, notes = await gather_disambiguation_replays(adapter, runs, outcomes, k=3)
    assert "s3" in out
    assert len(out["s3"]) == 3
    assert all(sr.step_id == "s3" for sr in out["s3"])
    assert notes == []


@pytest.mark.asyncio
async def test_gather_honors_max_cost_and_stops_partway() -> None:
    """Budget exhaustion bails gracefully without raising."""

    class _CostlyReplay(FakeAdapter):
        async def replay_step(
            self, step_id: str, fixed_inputs: Any, seed: int | None = None
        ) -> StepRun:
            sr = await super().replay_step(step_id, fixed_inputs, seed)
            # Each replay claims $0.01 — pushing past tight budgets quickly.
            from varix.core import CostSnapshot

            return StepRun(
                step_id=sr.step_id,
                inputs=sr.inputs,
                output=sr.output,
                tool_calls=sr.tool_calls,
                provider_metadata=sr.provider_metadata,
                cost=CostSnapshot(dollars=0.01),
                seed=sr.seed,
            )

    adapter = _CostlyReplay(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    cost = CostAccumulator()
    out, notes = await gather_disambiguation_replays(
        adapter,
        runs,
        {"s3": LocalizationOutcome.DOWNSTREAM},
        k=5,
        cost=cost,
        max_cost=0.025,  # allows ~2 replays before exhaustion
    )
    assert "s3" in out
    assert len(out["s3"]) < 5  # bailed early
    assert notes == []  # budget exhaustion is not a failure note


@pytest.mark.asyncio
async def test_gather_continues_other_steps_when_one_step_raises() -> None:
    """If replay_step raises mid-gather, that step stops; others proceed."""

    class _RaisingOnS3(FakeAdapter):
        async def replay_step(
            self, step_id: str, fixed_inputs: Any, seed: int | None = None
        ) -> StepRun:
            if step_id == "s3":
                raise RuntimeError("transient failure")
            return await super().replay_step(step_id, fixed_inputs, seed)

    adapter = _RaisingOnS3(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    outcomes = {
        "s3": LocalizationOutcome.DOWNSTREAM,
        "s4": LocalizationOutcome.DOWNSTREAM,
    }
    out, notes = await gather_disambiguation_replays(adapter, runs, outcomes, k=3)
    # s3 failed → either absent or empty; s4 succeeded with K replays.
    assert "s4" in out and len(out["s4"]) == 3
    assert out.get("s3", []) == []
    # The failure is surfaced as a note rather than silently dropped.
    assert any("s3" in n and "transient failure" in n for n in notes)


@pytest.mark.asyncio
async def test_gather_emits_note_on_first_replay_failure() -> None:
    """When replay_step raises on the very first call, the step gets no
    entry in `replays` but the failure is named in `notes` — otherwise the
    user has no way to know the gather ran at all."""

    class _AlwaysRaises(FakeAdapter):
        async def replay_step(
            self, step_id: str, fixed_inputs: Any, seed: int | None = None
        ) -> StepRun:
            raise RuntimeError("API down")

    adapter = _AlwaysRaises(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    out, notes = await gather_disambiguation_replays(
        adapter, runs, {"s3": LocalizationOutcome.DOWNSTREAM}, k=3
    )
    assert "s3" not in out
    assert len(notes) == 1
    assert "s3" in notes[0]
    assert "0 successful replay" in notes[0]
    assert "API down" in notes[0]
