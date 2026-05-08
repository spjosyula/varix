"""Tests for the ordering classifier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from varix.adapters import FakeAdapter
from varix.analysis.classifiers import OrderingClassifier, ToolSideClassifier
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


def _step(*calls: ToolCall, output: object = "out") -> StepRun:
    return StepRun(step_id="s1", inputs="in", output=output, tool_calls=calls)


def _call(
    name: str = "lookup",
    args: dict[str, Any] | None = None,
    result: object = "hit",
) -> ToolCall:
    return ToolCall(name=name, arguments=args or {"q": "x"}, result=result)


def test_emits_high_when_same_multiset_different_sequence() -> None:
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    runs = _wrap(_step(a, b), _step(b, a))
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].classification is Classification.ORDERING
    assert findings[0].confidence is Confidence.HIGH
    assert findings[0].evidence[0].kind == "ordering_diff"
    assert findings[0].evidence[0].data["unique_sequence_count"] == 2


def test_abstains_when_sequences_identical() -> None:
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    runs = _wrap(_step(a, b), _step(a, b))
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_multisets_differ_pure_tool_side_scenario() -> None:
    # Same (name, args) but different result across runs → tool-side territory.
    runs = _wrap(_step(_call(result="hit_1")), _step(_call(result="hit_2")))
    findings = OrderingClassifier().classify(
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
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_when_only_one_tool_call() -> None:
    runs = _wrap(_step(_call()), _step(_call()))
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_emits_unavailable_when_capability_missing_and_outputs_differ() -> None:
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    runs = _wrap(_step(a, b, output="x"), _step(b, a, output="y"))
    caps = AdapterCapabilities(exposes_fingerprint=True, exposes_tool_calls=False)
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=caps,
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.UNAVAILABLE
    assert findings[0].classification is Classification.ORDERING


def test_abstains_when_capability_missing_and_outputs_match() -> None:
    runs = _wrap(_step(_call(), output="x"), _step(_call(), output="x"))
    caps = AdapterCapabilities(exposes_tool_calls=False)
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=caps,
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_with_fewer_than_two_observations() -> None:
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=_wrap(_step(a, b)),
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings == []


def test_replays_contribute_to_sequence_pool() -> None:
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    runs = _wrap(_step(a, b))  # one run
    replays = [_step(b, a)]  # one replay with reversed order
    findings = OrderingClassifier().classify(
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
    a = _call(args={"q": "first"}, result="alpha")
    b = _call(args={"q": "second"}, result="beta")
    runs = _wrap(_step(a, b), _step(b, a))
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.DOWNSTREAM,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert findings[0].localization is LocalizationOutcome.DOWNSTREAM


def test_args_canonicalization_handles_key_order() -> None:
    # Same args, different key insertion order — multisets must align.
    a1 = _call(args={"a": 1, "b": 2}, result="r1")
    a2 = _call(args={"b": 2, "a": 1}, result="r1")
    b = _call(args={"x": 0}, result="r2")
    runs = _wrap(_step(a1, b), _step(b, a2))
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1


def test_handles_repeated_calls_within_observation() -> None:
    # Multiset preserves multiplicity: [A, A, B] vs [A, B, A] same multiset.
    a = _call(args={"q": "x"}, result="r")
    b = _call(args={"q": "y"}, result="r")
    runs = _wrap(_step(a, a, b), _step(a, b, a))
    findings = OrderingClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=_FULL_CAPS,
        metric=_METRIC,
    )
    assert len(findings) == 1


@pytest.mark.asyncio
async def test_against_fake_adapter_ordering_variance() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.ORDERING})
    runs = await run_n(adapter, "hello", n=3)
    findings = OrderingClassifier().classify(
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
async def test_against_fake_adapter_tool_side_does_not_fire_ordering() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    findings = OrderingClassifier().classify(
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
        findings = OrderingClassifier().classify(
            step_id=sid,
            localization=LocalizationOutcome.DETERMINISTIC,
            runs=runs,
            replays=[],
            capabilities=adapter.capabilities(),
            metric=_METRIC,
        )
        assert findings == []


@pytest.mark.asyncio
async def test_ordering_and_tool_side_are_mutually_exclusive() -> None:
    """When both classifiers run on the same step, only the appropriate one fires."""
    ordering_classifier = OrderingClassifier()
    tool_side_classifier = ToolSideClassifier()

    # ORDERING scenario: ordering fires, tool-side does not.
    adapter_ord = FakeAdapter(variance={"s2": Classification.ORDERING})
    runs_ord = await run_n(adapter_ord, "hello", n=3)
    args_kwargs: dict[str, Any] = {
        "step_id": "s2",
        "localization": LocalizationOutcome.SOURCE,
        "replays": [],
        "capabilities": adapter_ord.capabilities(),
        "metric": _METRIC,
    }
    assert len(ordering_classifier.classify(runs=runs_ord, **args_kwargs)) == 1
    assert tool_side_classifier.classify(runs=runs_ord, **args_kwargs) == []

    # TOOL_SIDE scenario: tool-side fires, ordering does not.
    adapter_ts = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs_ts = await run_n(adapter_ts, "hello", n=3)
    args_kwargs["capabilities"] = adapter_ts.capabilities()
    assert ordering_classifier.classify(runs=runs_ts, **args_kwargs) == []
    assert len(tool_side_classifier.classify(runs=runs_ts, **args_kwargs)) == 1
