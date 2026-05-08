"""Ordering classifier — same multiset of tool calls observed in different sequences.

Compares each observation's tool-call sequence. If every observation has the
same multiset of (name, args, result) triples but the *sequence* differs,
ordering variance fires HIGH. Differing multisets are someone else's territory
(typically tool-side), so this classifier abstains there.
"""

from __future__ import annotations

from collections.abc import Sequence

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

_Triple = tuple[str, str, str]
_Sequence = tuple[_Triple, ...]


class OrderingClassifier:
    """Detect tool-call sequencing variance: same calls, different order."""

    def name(self) -> str:
        return "ordering"

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
                    classification=Classification.ORDERING,
                    localization=localization,
                )
            ]

        sequences: list[_Sequence] = [_sequence_sig(obs) for obs in observations]
        multisets = {tuple(sorted(seq)) for seq in sequences}

        if len(multisets) > 1:
            return []  # multisets differ → not pure ordering variance
        if len(set(sequences)) <= 1:
            return []  # sequences identical → no variance at all

        unique_sequences = sorted(set(sequences))
        return [
            Finding(
                step_id=step_id,
                localization=localization,
                confidence=Confidence.HIGH,
                metric_name=metric.name(),
                classification=Classification.ORDERING,
                reason=(
                    f"{len(unique_sequences)} distinct tool-call sequences observed "
                    f"with the same {len(sequences[0])}-call multiset"
                ),
                evidence=(
                    Evidence(
                        kind="ordering_diff",
                        description="same multiset of tool calls observed in different sequences",
                        data={
                            "unique_sequence_count": len(unique_sequences),
                            "sequences": [_format_sequence(seq) for seq in unique_sequences],
                        },
                    ),
                ),
            )
        ]


def _sequence_sig(obs: StepRun) -> _Sequence:
    return tuple((tc.name, args_key(tc.arguments), str(tc.result)) for tc in obs.tool_calls)


def _format_sequence(seq: _Sequence) -> str:
    return " -> ".join(f"{name}({args})={result}" for name, args, result in seq)
