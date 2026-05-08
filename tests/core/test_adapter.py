"""Tests for the Adapter Protocol and the unavailable_finding helper."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.core import (
    Adapter,
    AdapterCapabilities,
    Confidence,
    LocalizationOutcome,
    PipelineRun,
    Step,
    StepGraph,
    StepRun,
    unavailable_finding,
)


class _NoOpAdapter:
    """Minimal Adapter implementation used to verify Protocol structural conformance."""

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(supports_replay=True)

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(steps=(Step(id="s1", name="step", index=0),))

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        sr = StepRun(step_id="s1", inputs=pipeline_input, output="ok")
        now = datetime.now(UTC)
        return PipelineRun(run_id="r1", step_runs=(sr,), started_at=now, finished_at=now)

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        return StepRun(step_id=step_id, inputs=fixed_inputs, output="ok")


def test_no_op_adapter_satisfies_protocol_at_runtime() -> None:
    assert isinstance(_NoOpAdapter(), Adapter)


def test_unavailable_finding_defaults_to_source() -> None:
    f = unavailable_finding(
        step_id="s1",
        metric_name="exact",
        reason="adapter does not expose system_fingerprint",
    )
    assert f.confidence is Confidence.UNAVAILABLE
    assert f.localization is LocalizationOutcome.SOURCE
    assert f.classification is None
    assert f.reason and "system_fingerprint" in f.reason


def test_unavailable_finding_localization_overridable() -> None:
    f = unavailable_finding(
        step_id="s1",
        metric_name="exact",
        reason="replay not supported",
        localization=LocalizationOutcome.DOWNSTREAM,
    )
    assert f.localization is LocalizationOutcome.DOWNSTREAM


@pytest.mark.asyncio
async def test_no_op_adapter_run_pipeline_returns_one_step_run() -> None:
    adapter = _NoOpAdapter()
    run = await adapter.run_pipeline("hello")
    assert len(run.step_runs) == 1
    assert run.step_runs[0].step_id == "s1"
