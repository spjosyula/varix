"""Integration test for the analysis pipeline.

Wires FakeAdapter → run_n → Localizer → ClassifierRegistry. Initially asserts
only that the wiring runs without crashing. Subsequent commits add per-classifier
assertions as each scaffold is filled in.
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
from varix.core import ExactMatch
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
async def test_full_analysis_pipeline_runs_against_fake_adapter() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    metric = ExactMatch()

    outcomes = Localizer(metric=metric).classify_steps(runs)
    assert set(outcomes) == {"s1", "s2", "s3", "s4", "s5"}

    registry = _all_classifiers()
    capabilities = adapter.capabilities()
    for step_id in outcomes:
        findings = registry.classify_step(
            step_id=step_id,
            runs=runs,
            replays=[],
            capabilities=capabilities,
            metric=metric,
        )
        # Scaffolds abstain.
        assert findings == []
