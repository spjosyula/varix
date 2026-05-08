"""Tests for ClassifierRegistry."""

from __future__ import annotations

from collections.abc import Sequence

from varix.analysis import Classifier, ClassifierRegistry
from varix.analysis.classifiers import (
    OrderingClassifier,
    PromptSideClassifier,
    ProviderSideClassifier,
    TimeOrStateClassifier,
    ToolSideClassifier,
)
from varix.core import (
    AdapterCapabilities,
    Confidence,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


class _StubFixedFinding:
    """Stub classifier that always emits one finding for any step."""

    def name(self) -> str:
        return "stub"

    def classify(
        self,
        step_id: str,
        localization: LocalizationOutcome,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        return [
            Finding(
                step_id=step_id,
                localization=localization,
                confidence=Confidence.LOW,
                metric_name=metric.name(),
                reason="stub",
            )
        ]


def test_empty_registry_returns_no_findings() -> None:
    registry = ClassifierRegistry()
    findings = registry.classify_step(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=(),
        replays=(),
        capabilities=AdapterCapabilities(),
        metric=ExactMatch(),
    )
    assert findings == []


def test_registry_with_only_scaffolds_returns_no_findings() -> None:
    registry = ClassifierRegistry(
        [
            ProviderSideClassifier(),
            ToolSideClassifier(),
            OrderingClassifier(),
            PromptSideClassifier(),
            TimeOrStateClassifier(),
        ]
    )
    findings = registry.classify_step(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        runs=(),
        replays=(),
        capabilities=AdapterCapabilities(),
        metric=ExactMatch(),
    )
    assert findings == []


def test_registry_collects_findings_from_all_classifiers() -> None:
    registry = ClassifierRegistry([_StubFixedFinding(), _StubFixedFinding()])
    findings = registry.classify_step(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        runs=(),
        replays=(),
        capabilities=AdapterCapabilities(),
        metric=ExactMatch(),
    )
    assert len(findings) == 2
    assert all(f.step_id == "s1" for f in findings)
    assert all(f.localization is LocalizationOutcome.SOURCE for f in findings)


def test_registry_add_appends_classifier() -> None:
    registry = ClassifierRegistry()
    assert registry.classifiers == ()
    registry.add(ProviderSideClassifier())
    assert len(registry.classifiers) == 1


def test_scaffolds_satisfy_classifier_protocol() -> None:
    for c in (
        ProviderSideClassifier(),
        ToolSideClassifier(),
        OrderingClassifier(),
        PromptSideClassifier(),
        TimeOrStateClassifier(),
    ):
        assert isinstance(c, Classifier)
