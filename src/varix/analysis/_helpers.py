"""Shared helpers used by classifiers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from varix.core import PipelineRun, StepRun, VarianceMetric

_FINGERPRINT_KEY = "system_fingerprint"

_ISO8601_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)

_TIME_TOOL_MARKERS: tuple[str, ...] = (
    "time",
    "now",
    "clock",
    "today",
    "date",
    "uuid",
    "random",
    "rand",
)


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


def fingerprints_differ(observations: Sequence[StepRun]) -> bool:
    """True if `system_fingerprint` varies across observations that report it."""
    seen: set[str] = set()
    for sr in observations:
        meta = sr.provider_metadata or {}
        fp = meta.get(_FINGERPRINT_KEY)
        if fp is not None:
            seen.add(str(fp))
            if len(seen) > 1:
                return True
    return False


def tool_results_differ_for_same_key(
    observations: Sequence[StepRun], metric: VarianceMetric
) -> bool:
    """True if any (tool_name, args) key has divergent results across observations."""
    per_obs: list[dict[tuple[str, str], Any]] = []
    for obs in observations:
        first: dict[tuple[str, str], Any] = {}
        for tc in obs.tool_calls:
            key = (tc.name, args_key(tc.arguments))
            if key not in first:
                first[key] = tc.result
        per_obs.append(first)

    all_keys: set[tuple[str, str]] = set().union(*per_obs)
    for key in all_keys:
        results = [obs[key] for obs in per_obs if key in obs]
        if len(results) < 2:
            continue
        unique: list[Any] = []
        for r in results:
            if not any(metric.equivalent(r, u) for u in unique):
                unique.append(r)
        if len(unique) > 1:
            return True
    return False


def sequences_differ_with_same_multiset(observations: Sequence[StepRun]) -> bool:
    """True if observations share a tool-call multiset but their sequences differ."""
    sequences = [
        tuple((tc.name, args_key(tc.arguments), str(tc.result)) for tc in obs.tool_calls)
        for obs in observations
    ]
    multisets = {tuple(sorted(seq)) for seq in sequences}
    if len(multisets) > 1:
        return False
    return len(set(sequences)) > 1


def time_or_state_markers(observations: Sequence[StepRun]) -> list[dict[str, Any]]:
    """Return heuristic markers suggesting clock/RNG-driven variance.

    Each marker is a dict describing what was observed. An empty list means
    no heuristic signal fired.
    """
    markers: list[dict[str, Any]] = []

    suggestive_tools: set[str] = set()
    for obs in observations:
        for tc in obs.tool_calls:
            lowered = tc.name.lower()
            if any(needle in lowered for needle in _TIME_TOOL_MARKERS):
                suggestive_tools.add(tc.name)
    if suggestive_tools:
        markers.append({"kind": "time_tool_name", "tools": sorted(suggestive_tools)})

    all_timestamps: set[str] = set()
    for obs in observations:
        all_timestamps.update(_ISO8601_PATTERN.findall(str(obs.output)))
    if len(all_timestamps) > 1:
        markers.append(
            {"kind": "varying_timestamp_in_output", "timestamps": sorted(all_timestamps)}
        )

    return markers
