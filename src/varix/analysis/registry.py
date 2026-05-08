"""Classifier Protocol + registry.

Each classifier receives the per-step view (N runs + any replays for that step)
and emits zero, one, or more findings. The registry runs every classifier and
collects their findings; the orchestration loop (which steps to classify, how
many replays to gather) lives in the surface layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from varix.core import (
    AdapterCapabilities,
    Finding,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    VarianceMetric,
)


@runtime_checkable
class Classifier(Protocol):
    """A single-category classifier.

    Receives the localizer's verdict so emitted findings carry the correct
    `localization`. The classifier itself decides — based on its own domain
    evidence — whether to fire; localization is not a gate.
    """

    def name(self) -> str: ...

    def classify(
        self,
        step_id: str,
        localization: LocalizationOutcome,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]: ...


class ClassifierRegistry:
    """Holds the active list of classifiers and runs them against one step."""

    def __init__(self, classifiers: Iterable[Classifier] = ()) -> None:
        self._classifiers: list[Classifier] = list(classifiers)

    def add(self, classifier: Classifier) -> None:
        self._classifiers.append(classifier)

    @property
    def classifiers(self) -> tuple[Classifier, ...]:
        return tuple(self._classifiers)

    def classify_step(
        self,
        step_id: str,
        localization: LocalizationOutcome,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for classifier in self._classifiers:
            findings.extend(
                classifier.classify(step_id, localization, runs, replays, capabilities, metric)
            )
        return findings
