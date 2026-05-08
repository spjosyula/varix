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
    PipelineRun,
    StepRun,
    VarianceMetric,
)


@runtime_checkable
class Classifier(Protocol):
    """A single-category classifier.

    Returns a list of findings (often zero or one). Multiple findings allow
    a classifier to emit both a residual verdict and a heuristic note.
    """

    def name(self) -> str: ...

    def classify(
        self,
        step_id: str,
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
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for classifier in self._classifiers:
            findings.extend(classifier.classify(step_id, runs, replays, capabilities, metric))
        return findings
