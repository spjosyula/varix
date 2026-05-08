"""Tests for the tool-side classifier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.adapters import FakeAdapter
from varix.analysis.classifiers import ToolSideClassifier
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


def _step(*calls: ToolCall, output: object = "out", step_id: str = "s1") -> StepRun:
    return StepRun(step_id=step_id, inputs="in", output=output, tool_calls=calls)


def _call(
    name: str = "lookup",
    args: dict[str, Any] | None = None,
    result: object = "hit",
) -> ToolCall:
    return ToolCall(name=name, arguments=args or {"q": "x"}, result=result)


def test_emits_high_when_same_tool_args_returns_different_result() -> None:
    runs = _wrap(_step(_call(result="hit_1")), _step(_call(result="hit_2")))
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].classification is Classification.TOOL_SIDE
    assert findings[0].confidence is Confidence.HIGH
    assert findings[0].evidence[0].kind == "tool_result_diff"
    assert findings[0].evidence[0].data["diffs"][0]["tool"] == "lookup"


def test_abstains_when_tool_results_match() -> None:
    runs = _wrap(_step(_call(result="hit")), _step(_call(result="hit")))
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_only_ordering_changes() -> None:
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    runs = _wrap(_step(a, b), _step(b, a))
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_no_tool_calls() -> None:
    runs = _wrap(_step(output="a"), _step(output="b"))
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_emits_unavailable_when_capability_missing_and_outputs_differ() -> None:
    runs = _wrap(_step(_call(result="hit"), output="a"), _step(_call(result="hit"), output="b"))
    caps = AdapterCapabilities(exposes_fingerprint=True, exposes_tool_calls=False)
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=caps,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.UNAVAILABLE
    assert findings[0].classification is Classification.TOOL_SIDE
    assert findings[0].reason and "tool_calls" in findings[0].reason


def test_abstains_when_capability_missing_and_outputs_match() -> None:
    runs = _wrap(_step(_call(), output="x"), _step(_call(), output="x"))
    caps = AdapterCapabilities(exposes_tool_calls=False)
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=caps,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_with_fewer_than_two_observations() -> None:
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=_wrap(_step(_call(result="hit"))),
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_replays_contribute_to_diff_pool() -> None:
    runs = _wrap(_step(_call(result="hit_1")))
    replays = [_step(_call(result="hit_2"))]
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=replays,
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.HIGH


def test_propagates_localization_onto_finding() -> None:
    runs = _wrap(_step(_call(result="hit_1")), _step(_call(result="hit_2")))
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.DOWNSTREAM,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings[0].localization is LocalizationOutcome.DOWNSTREAM


def test_args_canonicalization_handles_key_order() -> None:
    # Same args, different key insertion order — should still pair up.
    runs = _wrap(
        _step(_call(args={"a": 1, "b": 2}, result="r1")),
        _step(_call(args={"b": 2, "a": 1}, result="r2")),
    )
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1


def test_only_flags_keys_seen_in_two_or_more_observations() -> None:
    # tool A appears in run 1 only; tool B appears in both with same result.
    # Neither qualifies as a diff.
    runs = _wrap(
        _step(_call(name="alpha", result="x"), _call(name="beta", result="b")),
        _step(_call(name="beta", result="b")),
    )
    findings = ToolSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


@pytest.mark.asyncio
async def test_against_fake_adapter_tool_side_variance() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    findings = ToolSideClassifier().classify(
        step_id="s2",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.HIGH


@pytest.mark.asyncio
async def test_against_fake_adapter_ordering_does_not_fire_tool_side() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.ORDERING})
    runs = await run_n(adapter, "hello", n=3)
    findings = ToolSideClassifier().classify(
        step_id="s2",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert findings == []


@pytest.mark.asyncio
async def test_against_fake_adapter_no_variance_abstains() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    for sid in ("s1", "s2", "s3", "s4", "s5"):
        findings = ToolSideClassifier().classify(
            step_id=sid,
            localization=LocalizationOutcome.DETERMINISTIC,
            runs=runs,
            replays=[],
            capabilities=adapter.capabilities(),
            metric=_METRIC,
        )
        assert findings == []
