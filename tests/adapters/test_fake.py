"""Tests for FakeAdapter.

Verifies the deterministic baseline, each variance category, and full
conformance to the protocol_test_suite.
"""

from __future__ import annotations

import pytest

from varix.adapters import FakeAdapter
from varix.core import Adapter, Classification
from varix.core.protocol_test_suite import validate_adapter


def test_fake_adapter_satisfies_protocol() -> None:
    assert isinstance(FakeAdapter(), Adapter)


@pytest.mark.asyncio
async def test_fake_adapter_passes_protocol_suite() -> None:
    await validate_adapter(FakeAdapter(), {"q": "hello"})


@pytest.mark.asyncio
async def test_deterministic_run_yields_identical_step_runs() -> None:
    adapter = FakeAdapter()
    a = await adapter.run_pipeline("hello")
    b = await adapter.run_pipeline("hello")
    assert a.step_runs == b.step_runs


@pytest.mark.asyncio
async def test_deterministic_replay_yields_identical_step_runs() -> None:
    adapter = FakeAdapter()
    a = await adapter.replay_step("s2", "hello")
    b = await adapter.replay_step("s2", "hello")
    assert a == b


@pytest.mark.asyncio
async def test_provider_side_variance_changes_fingerprint() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    a = await adapter.run_pipeline("hello")
    b = await adapter.run_pipeline("hello")
    assert a.step_runs[1].provider_metadata != b.step_runs[1].provider_metadata
    # Other steps remain stable
    assert a.step_runs[0] == b.step_runs[0]


@pytest.mark.asyncio
async def test_tool_side_variance_changes_tool_result() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    a = await adapter.run_pipeline("hello")
    b = await adapter.run_pipeline("hello")
    assert a.step_runs[1].tool_calls[0].result != b.step_runs[1].tool_calls[0].result


@pytest.mark.asyncio
async def test_ordering_variance_reorders_tool_calls() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.ORDERING})
    a = await adapter.run_pipeline("hello")
    b = await adapter.run_pipeline("hello")
    a_names = [tc.result for tc in a.step_runs[1].tool_calls]
    b_names = [tc.result for tc in b.step_runs[1].tool_calls]
    assert a_names != b_names
    assert sorted(a_names) == sorted(b_names)


@pytest.mark.asyncio
async def test_prompt_side_variance_changes_output() -> None:
    adapter = FakeAdapter(variance={"s4": Classification.PROMPT_SIDE})
    a = await adapter.run_pipeline("hello")
    b = await adapter.run_pipeline("hello")
    assert a.step_runs[3].output != b.step_runs[3].output


@pytest.mark.asyncio
async def test_time_or_state_variance_embeds_timestamp() -> None:
    adapter = FakeAdapter(variance={"s5": Classification.TIME_OR_STATE})
    a = await adapter.run_pipeline("hello")
    b = await adapter.run_pipeline("hello")
    assert "2026-05-08T12:00:" in str(a.step_runs[4].output)
    assert a.step_runs[4].output != b.step_runs[4].output


@pytest.mark.asyncio
async def test_replay_variance_uses_replay_counter() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    a = await adapter.replay_step("s2", "hello")
    b = await adapter.replay_step("s2", "hello")
    assert a.output != b.output


@pytest.mark.asyncio
async def test_run_and_replay_counters_are_independent() -> None:
    adapter = FakeAdapter(variance={"s1": Classification.PROMPT_SIDE})
    run_a = await adapter.run_pipeline("hello")
    replay_a = await adapter.replay_step("s1", "hello")
    run_b = await adapter.run_pipeline("hello")
    # First run uses run_counter=1, first replay uses replay_counter=1.
    # They share the same suffix in PROMPT_SIDE, so outputs match.
    assert run_a.step_runs[0].output == replay_a.output
    # Second run uses run_counter=2 → different from first run's output.
    assert run_b.step_runs[0].output != run_a.step_runs[0].output


@pytest.mark.asyncio
async def test_pipeline_structure_returns_five_steps() -> None:
    graph = await FakeAdapter().pipeline_structure("anything")
    assert [s.id for s in graph.steps] == ["s1", "s2", "s3", "s4", "s5"]
