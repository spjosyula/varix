"""Tests for the prompt-side residual classifier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.adapters import FakeAdapter
from varix.analysis.classifiers import PromptSideClassifier
from varix.core import (
    AdapterCapabilities,
    Classification,
    Confidence,
    ExactMatch,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    ToolCall,
)
from varix.execution import run_n

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
_METRIC = ExactMatch()
_FULL_CAPS = AdapterCapabilities(
    exposes_fingerprint=True, exposes_tool_calls=True, supports_replay=True
)


def _wrap(*step_runs: StepRun) -> list[PipelineRun]:
    return [
        PipelineRun(run_id=f"r{i}", step_runs=(sr,), started_at=_T, finished_at=_T)
        for i, sr in enumerate(step_runs)
    ]


def _step(
    output: object,
    *,
    calls: tuple[ToolCall, ...] = (),
    fingerprint: str = "fp_stable",
) -> StepRun:
    return StepRun(
        step_id="s1",
        inputs="in",
        output=output,
        tool_calls=calls,
        provider_metadata={"system_fingerprint": fingerprint},
    )


def test_emits_medium_when_only_output_varies() -> None:
    runs = _wrap(_step("a"), _step("b"))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].classification is Classification.PROMPT_SIDE
    assert findings[0].confidence is Confidence.MEDIUM
    assert findings[0].evidence[0].kind == "residual_output_variance"


def test_abstains_when_outputs_match() -> None:
    runs = _wrap(_step("x"), _step("x"))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_localization_is_downstream() -> None:
    runs = _wrap(_step("a"), _step("b"))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.DOWNSTREAM,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_localization_is_deterministic() -> None:
    runs = _wrap(_step("a"), _step("b"))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_fingerprints_differ() -> None:
    runs = _wrap(_step("a", fingerprint="fp_a"), _step("b", fingerprint="fp_b"))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_tool_results_differ() -> None:
    a: dict[str, Any] = {"q": "x"}
    runs = _wrap(
        _step("a", calls=(ToolCall(name="lookup", arguments=a, result="r1"),)),
        _step("b", calls=(ToolCall(name="lookup", arguments=a, result="r2"),)),
    )
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_only_ordering_changes() -> None:
    a = ToolCall(name="lookup", arguments={"q": "1"}, result="alpha")
    b = ToolCall(name="lookup", arguments={"q": "2"}, result="beta")
    runs = _wrap(_step("a", calls=(a, b)), _step("b", calls=(b, a)))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_time_or_state_marker_present() -> None:
    runs = _wrap(
        _step("r1 at 2026-05-08T12:00:01Z"),
        _step("r2 at 2026-05-08T12:00:02Z"),
    )
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_capability_missing_fingerprint() -> None:
    runs = _wrap(_step("a"), _step("b"))
    caps = AdapterCapabilities(exposes_fingerprint=False, exposes_tool_calls=True)
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=caps,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_capability_missing_tool_calls() -> None:
    runs = _wrap(_step("a"), _step("b"))
    caps = AdapterCapabilities(exposes_fingerprint=True, exposes_tool_calls=False)
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=caps,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_with_fewer_than_two_observations() -> None:
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=_wrap(_step("a")),
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_propagates_localization_onto_finding() -> None:
    runs = _wrap(_step("a"), _step("b"))
    findings = PromptSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings[0].localization is LocalizationOutcome.SOURCE


@pytest.mark.asyncio
async def test_against_fake_adapter_prompt_side_variance() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    findings = PromptSideClassifier().classify(
        step_id="s2",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.MEDIUM


@pytest.mark.asyncio
async def test_against_fake_adapter_provider_side_does_not_fire_residual() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    findings = PromptSideClassifier().classify(
        step_id="s2",
        localization=LocalizationOutcome.DETERMINISTIC,  # output stable in this scenario
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert findings == []


@pytest.mark.asyncio
async def test_against_fake_adapter_tool_side_does_not_fire_residual() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    findings = PromptSideClassifier().classify(
        step_id="s2",
        localization=LocalizationOutcome.DETERMINISTIC,  # output stable
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert findings == []


@pytest.mark.asyncio
async def test_against_fake_adapter_time_or_state_does_not_fire_residual() -> None:
    adapter = FakeAdapter(variance={"s5": Classification.TIME_OR_STATE})
    runs = await run_n(adapter, "hello", n=3)
    # FakeAdapter TIME_OR_STATE makes output vary (timestamp suffix),
    # localization=SOURCE for s5. Residual should still abstain because the
    # time/state marker is present in the output.
    findings = PromptSideClassifier().classify(
        step_id="s5",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert findings == []
