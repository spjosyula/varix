"""Tests for the time/state heuristic classifier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.adapters import FakeAdapter
from varix.analysis.classifiers import TimeOrStateClassifier
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
    output: object = "out",
    *,
    calls: tuple[ToolCall, ...] = (),
) -> StepRun:
    return StepRun(step_id="s1", inputs="in", output=output, tool_calls=calls)


def test_emits_low_when_output_contains_varying_iso8601_timestamps() -> None:
    runs = _wrap(
        _step(output="result at 2026-05-08T12:00:01Z"),
        _step(output="result at 2026-05-08T12:00:02Z"),
    )
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].classification is Classification.TIME_OR_STATE
    assert findings[0].confidence is Confidence.LOW
    kinds = [m["kind"] for m in findings[0].evidence[0].data["markers"]]
    assert "varying_timestamp_in_output" in kinds


def test_emits_low_when_time_named_tool_was_called() -> None:
    a: dict[str, Any] = {}
    runs = _wrap(
        _step(
            output="r1",
            calls=(ToolCall(name="get_current_time", arguments=a, result="t1"),),
        ),
        _step(
            output="r2",
            calls=(ToolCall(name="get_current_time", arguments=a, result="t2"),),
        ),
    )
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1
    kinds = [m["kind"] for m in findings[0].evidence[0].data["markers"]]
    assert "time_tool_name" in kinds


def test_reason_admits_uncertainty() -> None:
    runs = _wrap(
        _step(output="r1 2026-05-08T12:00:01Z"),
        _step(output="r2 2026-05-08T12:00:02Z"),
    )
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings[0].reason and "not definitive" in findings[0].reason


def test_abstains_when_outputs_match() -> None:
    a: dict[str, Any] = {}
    runs = _wrap(
        _step(output="r", calls=(ToolCall(name="get_current_time", arguments=a, result="t"),)),
        _step(output="r", calls=(ToolCall(name="get_current_time", arguments=a, result="t"),)),
    )
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_no_markers_present() -> None:
    runs = _wrap(_step(output="hello world"), _step(output="hello there"))
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_timestamps_match_across_observations() -> None:
    runs = _wrap(
        _step(output="r1 at 2026-05-08T12:00:00Z"),
        _step(output="r2 at 2026-05-08T12:00:00Z"),
    )
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    # Outputs differ ("r1" vs "r2") but timestamp is constant — no marker fires.
    assert findings == []


def test_abstains_with_fewer_than_two_observations() -> None:
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=_wrap(_step(output="2026-05-08T12:00:01Z")),
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_propagates_localization_onto_finding() -> None:
    runs = _wrap(
        _step(output="r 2026-05-08T12:00:01Z"),
        _step(output="r 2026-05-08T12:00:02Z"),
    )
    findings = TimeOrStateClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.DOWNSTREAM,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings[0].localization is LocalizationOutcome.DOWNSTREAM


def test_tool_marker_recognizes_various_substrings() -> None:
    # uuid, random, now, clock — each should trigger.
    a: dict[str, Any] = {}
    for tool_name in ("uuid4", "random_choice", "now", "wall_clock"):
        runs = _wrap(
            _step(output="x", calls=(ToolCall(name=tool_name, arguments=a, result="t"),)),
            _step(output="y", calls=(ToolCall(name=tool_name, arguments=a, result="t"),)),
        )
        findings = TimeOrStateClassifier().classify(
            step_id="s1",
            localization=LocalizationOutcome.SOURCE,
            runs=runs,
            replays=[],
            capabilities=_FULL_CAPS,
            metric=_METRIC,
        )
        assert findings, f"expected marker for tool name {tool_name!r}"


@pytest.mark.asyncio
async def test_against_fake_adapter_time_or_state_variance() -> None:
    adapter = FakeAdapter(variance={"s5": Classification.TIME_OR_STATE})
    runs = await run_n(adapter, "hello", n=3)
    findings = TimeOrStateClassifier().classify(
        step_id="s5",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.LOW


@pytest.mark.asyncio
async def test_against_fake_adapter_no_variance_abstains() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    for sid in ("s1", "s2", "s3", "s4", "s5"):
        findings = TimeOrStateClassifier().classify(
            step_id=sid,
            localization=LocalizationOutcome.DETERMINISTIC,
            runs=runs,
            replays=[],
            capabilities=adapter.capabilities(),
            metric=_METRIC,
        )
        assert findings == []
