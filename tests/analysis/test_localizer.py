"""Tests for the per-step Localizer."""

from __future__ import annotations

import pytest

from varix.adapters import FakeAdapter
from varix.analysis import Localizer
from varix.core import Classification, LocalizationOutcome
from varix.execution import run_n


@pytest.mark.asyncio
async def test_deterministic_pipeline_marks_every_step_deterministic() -> None:
    runs = await run_n(FakeAdapter(), "hello", n=3)
    outcomes = Localizer().classify_steps(runs)
    assert all(o is LocalizationOutcome.DETERMINISTIC for o in outcomes.values())
    assert set(outcomes) == {"s1", "s2", "s3", "s4", "s5"}


@pytest.mark.asyncio
async def test_metadata_only_variance_reads_as_deterministic_to_localizer() -> None:
    # PROVIDER_SIDE flips provider_metadata only; outputs stay stable.
    # The output-based Localizer correctly says "no variance observed" —
    # the PROVIDER_SIDE classifier (separate path) is what surfaces this case.
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    outcomes = Localizer().classify_steps(runs)
    assert all(o is LocalizationOutcome.DETERMINISTIC for o in outcomes.values())


@pytest.mark.asyncio
async def test_prompt_side_variance_at_s2_marks_s2_source_and_descendants_downstream() -> None:
    # PROMPT_SIDE varies s2's output while s1 stays stable → s2 inputs identical
    # → s2 SOURCE. Variance propagates through to s3-s5 inputs → DOWNSTREAM.
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    outcomes = Localizer().classify_steps(runs)
    assert outcomes["s1"] is LocalizationOutcome.DETERMINISTIC
    assert outcomes["s2"] is LocalizationOutcome.SOURCE
    for sid in ("s3", "s4", "s5"):
        assert outcomes[sid] is LocalizationOutcome.DOWNSTREAM


def test_empty_runs_returns_empty_dict() -> None:
    assert Localizer().classify_steps([]) == {}


@pytest.mark.asyncio
async def test_single_run_marks_every_step_deterministic() -> None:
    runs = await run_n(FakeAdapter(), "hello", n=1)
    outcomes = Localizer().classify_steps(runs)
    assert all(o is LocalizationOutcome.DETERMINISTIC for o in outcomes.values())
