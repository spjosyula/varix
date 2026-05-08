"""Tool-side classifier — same tool, same args, different result across observations.

Pairs tool calls across observations by `(tool_name, canonical_args)` and flags
keys whose results vary under the metric. Ordering variance alone (same
multiset, different sequence) does not trigger this classifier, since matched
keys still see matching results.

Limitation: when a single observation invokes the same `(tool, args)` more
than once, only the first occurrence is compared. Internal stateful tools
that diverge within a single run are out of scope here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from varix.analysis._helpers import args_key, gather_step_runs, outputs_differ
from varix.core import (
    AdapterCapabilities,
    Classification,
    Confidence,
    Evidence,
    Finding,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    VarianceMetric,
    unavailable_finding,
)


class ToolSideClassifier:
    """Detect nondeterministic tool invocations."""

    def name(self) -> str:
        return "tool_side"

    def classify(
        self,
        step_id: str,
        localization: LocalizationOutcome,
        runs: Sequence[PipelineRun],
        replays: Sequence[StepRun],
        capabilities: AdapterCapabilities,
        metric: VarianceMetric,
    ) -> list[Finding]:
        observations = gather_step_runs(step_id, runs, replays)
        if len(observations) < 2:
            return []

        if not capabilities.exposes_tool_calls:
            if not outputs_differ(observations, metric):
                return []
            return [
                unavailable_finding(
                    step_id=step_id,
                    metric_name=metric.name(),
                    reason="adapter does not expose tool_calls",
                    classification=Classification.TOOL_SIDE,
                    localization=localization,
                )
            ]

        per_obs_first_results = [_first_results_by_key(obs) for obs in observations]
        all_keys: set[tuple[str, str]] = set().union(*per_obs_first_results)

        diffs: list[dict[str, Any]] = []
        for key in sorted(all_keys):
            results_present = [obs[key] for obs in per_obs_first_results if key in obs]
            if len(results_present) < 2:
                continue
            unique: list[Any] = []
            for r in results_present:
                if not any(metric.equivalent(r, u) for u in unique):
                    unique.append(r)
            if len(unique) > 1:
                tool_name, _ = key
                diffs.append(
                    {
                        "tool": tool_name,
                        "results": [str(r) for r in results_present],
                        "unique_count": len(unique),
                    }
                )

        if not diffs:
            return []

        return [
            Finding(
                step_id=step_id,
                localization=localization,
                confidence=Confidence.HIGH,
                metric_name=metric.name(),
                classification=Classification.TOOL_SIDE,
                reason=f"tool result varied across runs for {len(diffs)} (tool, args) pair(s)",
                evidence=(
                    Evidence(
                        kind="tool_result_diff",
                        description=f"{len(diffs)} (tool, args) pair(s) returned different results",
                        data={"diffs": diffs},
                    ),
                ),
            )
        ]


def _first_results_by_key(obs: StepRun) -> dict[tuple[str, str], Any]:
    """Map (tool_name, canonical_args) → first observed result in this StepRun."""
    out: dict[tuple[str, str], Any] = {}
    for tc in obs.tool_calls:
        key = (tc.name, args_key(tc.arguments))
        if key not in out:
            out[key] = tc.result
    return out
