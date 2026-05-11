"""Tests for the provider-side classifier."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from varix.adapters import FakeAdapter
from varix.analysis.classifiers import ProviderSideClassifier
from varix.core import (
    AdapterCapabilities,
    Classification,
    Confidence,
    ExactMatch,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
)
from varix.execution import run_n

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
_METRIC = ExactMatch()


def _wrap(*step_runs: StepRun) -> list[PipelineRun]:
    """Wrap StepRuns into single-step PipelineRuns for ergonomic testing."""
    return [
        PipelineRun(run_id=f"r{i}", step_runs=(sr,), started_at=_T, finished_at=_T)
        for i, sr in enumerate(step_runs)
    ]


def _step(
    step_id: str = "s1",
    inputs: object = "in",
    output: object = "out",
    fingerprint: str | None = "fp_stable",
) -> StepRun:
    metadata: dict[str, object] | None = (
        {"system_fingerprint": fingerprint} if fingerprint is not None else None
    )
    return StepRun(step_id=step_id, inputs=inputs, output=output, provider_metadata=metadata)


def test_emits_high_when_fingerprints_differ() -> None:
    runs = _wrap(_step(fingerprint="fp_a"), _step(fingerprint="fp_b"), _step(fingerprint="fp_a"))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.HIGH
    assert findings[0].classification is Classification.PROVIDER_SIDE
    assert findings[0].evidence[0].kind == "fingerprint_diff"
    assert findings[0].evidence[0].data["unique"] == ["fp_a", "fp_b"]


def test_emits_excluded_runs_evidence_when_some_observations_lack_fingerprint() -> None:
    runs = _wrap(
        _step(fingerprint="fp_a"),
        _step(fingerprint=None),  # missing despite capability
        _step(fingerprint="fp_b"),
    )
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert len(findings) == 1
    f = findings[0]
    kinds = [ev.kind for ev in f.evidence]
    assert "fingerprint_diff" in kinds
    assert "excluded_runs" in kinds
    excluded_ev = next(ev for ev in f.evidence if ev.kind == "excluded_runs")
    excluded = excluded_ev.data["excluded"]
    assert len(excluded) == 1
    assert excluded[0]["observation_index"] == 1
    assert excluded[0]["reason"] == "no system_fingerprint"


def test_no_excluded_runs_evidence_when_all_observations_have_fingerprint() -> None:
    runs = _wrap(_step(fingerprint="fp_a"), _step(fingerprint="fp_b"))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert all(ev.kind != "excluded_runs" for ev in findings[0].evidence)


def test_abstains_when_fingerprints_match() -> None:
    runs = _wrap(_step(fingerprint="fp_a"), _step(fingerprint="fp_a"))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert findings == []


def test_emits_unavailable_when_capability_missing_and_outputs_differ() -> None:
    runs = _wrap(_step(output="a"), _step(output="b"))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=False),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.UNAVAILABLE
    assert findings[0].classification is Classification.PROVIDER_SIDE
    assert findings[0].reason and "system_fingerprint" in findings[0].reason


def test_abstains_when_capability_missing_and_outputs_match() -> None:
    runs = _wrap(_step(output="x"), _step(output="x"))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=False),
        metric=_METRIC,
    )
    assert findings == []


def test_abstains_with_fewer_than_two_observations() -> None:
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=_wrap(_step(fingerprint="fp_a")),
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert findings == []


def test_replays_contribute_to_fingerprint_pool() -> None:
    runs = _wrap(_step(fingerprint="fp_a"))  # one run only
    replays = [_step(fingerprint="fp_b")]  # plus one replay
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=replays,
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.HIGH


def test_propagates_localization_onto_finding() -> None:
    runs = _wrap(_step(fingerprint="fp_a"), _step(fingerprint="fp_b"))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.DOWNSTREAM,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert findings[0].localization is LocalizationOutcome.DOWNSTREAM


def test_abstains_when_only_one_run_has_fingerprint_data() -> None:
    runs = _wrap(_step(fingerprint="fp_a"), _step(fingerprint=None))
    findings = ProviderSideClassifier().classify(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
        metric=_METRIC,
    )
    assert findings == []


@pytest.mark.asyncio
async def test_against_fake_adapter_provider_side_variance() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    findings = ProviderSideClassifier().classify(
        step_id="s2",
        localization=LocalizationOutcome.SOURCE,
        runs=runs,
        replays=[],
        capabilities=adapter.capabilities(),
        metric=_METRIC,
    )
    assert len(findings) == 1
    assert findings[0].confidence is Confidence.HIGH
    assert sorted(findings[0].evidence[0].data["unique"]) == ["fp_a", "fp_b"]


@pytest.mark.asyncio
async def test_against_fake_adapter_no_variance_abstains() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    for sid in ("s1", "s2", "s3", "s4", "s5"):
        findings = ProviderSideClassifier().classify(
            step_id=sid,
            localization=LocalizationOutcome.DETERMINISTIC,
            runs=runs,
            replays=[],
            capabilities=adapter.capabilities(),
            metric=_METRIC,
        )
        assert findings == []
