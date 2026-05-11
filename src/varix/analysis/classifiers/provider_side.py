"""Provider-side classifier — detects variance arising from the model provider.

Diffs `system_fingerprint` across the runs (and replays, when available) for
the step under analysis. Emits HIGH when fingerprints differ, UNAVAILABLE when
the adapter does not expose fingerprint data and there is observable variance
to explain, and abstains otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

from varix.analysis._helpers import excluded_runs_evidence, gather_step_runs, outputs_differ
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

_FINGERPRINT_KEY = "system_fingerprint"


class ProviderSideClassifier:
    """Diff `system_fingerprint` (and similar provider metadata) across runs."""

    def name(self) -> str:
        return "provider_side"

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

        if not capabilities.exposes_fingerprint:
            if not outputs_differ(observations, metric):
                return []
            return [
                unavailable_finding(
                    step_id=step_id,
                    metric_name=metric.name(),
                    reason="adapter does not expose system_fingerprint",
                    classification=Classification.PROVIDER_SIDE,
                    localization=localization,
                )
            ]

        fingerprints = [(sr.provider_metadata or {}).get(_FINGERPRINT_KEY) for sr in observations]
        # Capability claimed fingerprints would be exposed; track any obs that lacked one
        # so the finding's evidence is honest about partial-data exclusions.
        excluded = [
            (i, "no system_fingerprint")
            for i, fp in enumerate(fingerprints)
            if fp is None
        ]
        present = [fp for fp in fingerprints if fp is not None]
        if len(present) < 2:
            return []

        unique = sorted({str(fp) for fp in present})
        if len(unique) <= 1:
            return []

        evidence_items: list[Evidence] = [
            Evidence(
                kind="fingerprint_diff",
                description="system_fingerprint values observed across runs",
                data={"fingerprints": [str(fp) for fp in present], "unique": unique},
            ),
        ]
        ex_ev = excluded_runs_evidence(excluded)
        if ex_ev is not None:
            evidence_items.append(ex_ev)

        return [
            Finding(
                step_id=step_id,
                localization=localization,
                confidence=Confidence.HIGH,
                metric_name=metric.name(),
                classification=Classification.PROVIDER_SIDE,
                reason=f"system_fingerprint varied across runs: {unique}",
                evidence=tuple(evidence_items),
            )
        ]
