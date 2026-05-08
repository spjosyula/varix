"""Integration test for the analysis pipeline.

Wires FakeAdapter → run_n → Localizer → ClassifierRegistry. Per-classifier
assertions accumulate as each scaffold is filled in.
"""

from __future__ import annotations

import pytest

from varix.adapters import FakeAdapter
from varix.analysis import ClassifierRegistry, Localizer
from varix.analysis.classifiers import (
    OrderingClassifier,
    PromptSideClassifier,
    ProviderSideClassifier,
    TimeOrStateClassifier,
    ToolSideClassifier,
)
from varix.core import Classification, Confidence, ExactMatch
from varix.execution import run_n


def _all_classifiers() -> ClassifierRegistry:
    return ClassifierRegistry(
        [
            ProviderSideClassifier(),
            ToolSideClassifier(),
            OrderingClassifier(),
            PromptSideClassifier(),
            TimeOrStateClassifier(),
        ]
    )


@pytest.mark.asyncio
async def test_deterministic_pipeline_emits_no_findings() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    assert set(outcomes) == {"s1", "s2", "s3", "s4", "s5"}

    registry = _all_classifiers()
    capabilities = adapter.capabilities()
    for step_id, localization in outcomes.items():
        findings = registry.classify_step(
            step_id=step_id,
            localization=localization,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        assert findings == []


@pytest.mark.asyncio
async def test_provider_side_variance_at_s2_produces_high_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    registry = _all_classifiers()
    capabilities = adapter.capabilities()

    findings_by_step = {
        step_id: registry.classify_step(
            step_id=step_id,
            localization=localization,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        for step_id, localization in outcomes.items()
    }

    # Only s2 produces a finding; everywhere else, classifiers abstain.
    assert findings_by_step["s2"], "expected a provider-side finding for s2"
    for sid in ("s1", "s3", "s4", "s5"):
        assert findings_by_step[sid] == [], f"unexpected finding on {sid}"

    s2_findings = findings_by_step["s2"]
    assert len(s2_findings) == 1
    finding = s2_findings[0]
    assert finding.classification is Classification.PROVIDER_SIDE
    assert finding.confidence is Confidence.HIGH
    assert finding.evidence and finding.evidence[0].kind == "fingerprint_diff"


@pytest.mark.asyncio
async def test_tool_side_variance_at_s2_produces_high_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.TOOL_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    registry = _all_classifiers()
    capabilities = adapter.capabilities()

    findings_by_step = {
        step_id: registry.classify_step(
            step_id=step_id,
            localization=localization,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        for step_id, localization in outcomes.items()
    }

    # s2 fires (tool result varies); other steps stay quiet.
    assert len(findings_by_step["s2"]) == 1
    for sid in ("s1", "s3", "s4", "s5"):
        assert findings_by_step[sid] == [], f"unexpected finding on {sid}"

    finding = findings_by_step["s2"][0]
    assert finding.classification is Classification.TOOL_SIDE
    assert finding.confidence is Confidence.HIGH
    assert finding.evidence[0].kind == "tool_result_diff"


@pytest.mark.asyncio
async def test_ordering_variance_at_s2_produces_high_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.ORDERING})
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    registry = _all_classifiers()
    capabilities = adapter.capabilities()

    findings_by_step = {
        step_id: registry.classify_step(
            step_id=step_id,
            localization=localization,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        for step_id, localization in outcomes.items()
    }

    # Exactly one ordering finding on s2; nothing on the other steps.
    assert len(findings_by_step["s2"]) == 1
    for sid in ("s1", "s3", "s4", "s5"):
        assert findings_by_step[sid] == [], f"unexpected finding on {sid}"

    finding = findings_by_step["s2"][0]
    assert finding.classification is Classification.ORDERING
    assert finding.confidence is Confidence.HIGH
    assert finding.evidence[0].kind == "ordering_diff"


@pytest.mark.asyncio
async def test_prompt_side_variance_at_s2_produces_medium_residual_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    registry = _all_classifiers()
    capabilities = adapter.capabilities()

    findings_by_step = {
        step_id: registry.classify_step(
            step_id=step_id,
            localization=localization,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        for step_id, localization in outcomes.items()
    }

    # s2 is the SOURCE; only the residual prompt-side classifier fires there.
    assert len(findings_by_step["s2"]) == 1
    finding = findings_by_step["s2"][0]
    assert finding.classification is Classification.PROMPT_SIDE
    assert finding.confidence is Confidence.MEDIUM

    # s3-s5 are DOWNSTREAM (variance propagates) — residual abstains, no
    # other classifier sees a signal there either.
    for sid in ("s1", "s3", "s4", "s5"):
        assert findings_by_step[sid] == [], f"unexpected finding on {sid}"


@pytest.mark.asyncio
async def test_time_or_state_variance_at_s5_produces_low_finding_only() -> None:
    adapter = FakeAdapter(variance={"s5": Classification.TIME_OR_STATE})
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    registry = _all_classifiers()
    capabilities = adapter.capabilities()

    findings_by_step = {
        step_id: registry.classify_step(
            step_id=step_id,
            localization=localization,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        for step_id, localization in outcomes.items()
    }

    # Time/state heuristic fires LOW; residual stays quiet because the marker
    # is present (residual doesn't fire alongside another classifier).
    s5_findings = findings_by_step["s5"]
    assert len(s5_findings) == 1
    finding = s5_findings[0]
    assert finding.classification is Classification.TIME_OR_STATE
    assert finding.confidence is Confidence.LOW

    for sid in ("s1", "s2", "s3", "s4"):
        assert findings_by_step[sid] == [], f"unexpected finding on {sid}"
