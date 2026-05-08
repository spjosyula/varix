"""Per-step localization across N runs."""

from __future__ import annotations

from collections.abc import Sequence

from varix.core import ExactMatch, LocalizationOutcome, PipelineRun, VarianceMetric


class Localizer:
    """Decide each step's localization across N runs.

    Outcomes:
      - DETERMINISTIC: every run's step output is equivalent under the metric.
      - SOURCE: outputs differ but inputs are identical → variance originates here.
      - DOWNSTREAM: outputs differ AND inputs differ → variance arrived from above.

    Assumes all runs share the same step ordering. Structural mismatch is
    handled separately (refusal path).
    """

    def __init__(self, metric: VarianceMetric | None = None) -> None:
        self._metric: VarianceMetric = metric if metric is not None else ExactMatch()

    def classify_steps(self, runs: Sequence[PipelineRun]) -> dict[str, LocalizationOutcome]:
        if not runs:
            return {}
        if len(runs) < 2:
            return {sr.step_id: LocalizationOutcome.DETERMINISTIC for sr in runs[0].step_runs}

        n_steps = len(runs[0].step_runs)
        outcomes: dict[str, LocalizationOutcome] = {}
        for i in range(n_steps):
            step_runs = [run.step_runs[i] for run in runs]
            step_id = step_runs[0].step_id
            outputs_match = all(
                self._metric.equivalent(step_runs[0].output, sr.output) for sr in step_runs[1:]
            )
            if outputs_match:
                outcomes[step_id] = LocalizationOutcome.DETERMINISTIC
                continue
            inputs_match = all(
                self._metric.equivalent(step_runs[0].inputs, sr.inputs) for sr in step_runs[1:]
            )
            outcomes[step_id] = (
                LocalizationOutcome.SOURCE if inputs_match else LocalizationOutcome.DOWNSTREAM
            )
        return outcomes
