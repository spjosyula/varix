"""Shared helpers used by classifiers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from varix.core import PipelineRun, StepRun, VarianceMetric


def gather_step_runs(
    step_id: str,
    runs: Sequence[PipelineRun],
    replays: Sequence[StepRun],
) -> list[StepRun]:
    """Collect every observed StepRun for `step_id` from runs and replays.

    Each pipeline run contributes its first matching step (one per run);
    replays are appended in order.
    """
    out: list[StepRun] = []
    for run in runs:
        for sr in run.step_runs:
            if sr.step_id == step_id:
                out.append(sr)
                break
    out.extend(sr for sr in replays if sr.step_id == step_id)
    return out


def outputs_differ(observations: Sequence[StepRun], metric: VarianceMetric) -> bool:
    """Return True if any observation's output differs from the first under the metric."""
    if not observations:
        return False
    first = observations[0].output
    return any(not metric.equivalent(first, sr.output) for sr in observations[1:])


def args_key(arguments: dict[str, Any]) -> str:
    """Stable, hashable canonical form of a tool's argument dict (JSON, sorted keys)."""
    return json.dumps(arguments, sort_keys=True, default=str)
