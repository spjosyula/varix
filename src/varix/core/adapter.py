"""The Adapter Protocol — varix's public API for plugging into agent frameworks and SDKs.

Every built-in and community adapter satisfies this Protocol. `capabilities()`
declares which methods are meaningful; classifiers emit `Confidence.UNAVAILABLE`
when a required capability is absent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from varix.core.types import (
    AdapterCapabilities,
    Classification,
    Confidence,
    Finding,
    LocalizationOutcome,
    PipelineRun,
    StepGraph,
    StepRun,
)


@runtime_checkable
class Adapter(Protocol):
    """The contract every varix adapter satisfies."""

    def capabilities(self) -> AdapterCapabilities:
        """Declare what this adapter can honestly provide. Must be idempotent."""
        ...

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        """Return the ordered set of steps that will execute for `pipeline_input`.

        Must be stable across repeated calls with the same input.
        """
        ...

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        """Execute the pipeline end-to-end.

        The returned `PipelineRun` must contain one `StepRun` per step declared
        by `pipeline_structure(pipeline_input)`, in the same order. Pass `seed`
        through to any RNG-controllable component when given.
        """
        ...

    async def replay_step(
        self,
        step_id: str,
        fixed_inputs: Any,
        seed: int | None = None,
    ) -> StepRun:
        """Re-execute a single step with `fixed_inputs` held constant.

        Required when `capabilities().supports_replay` is True. Must not
        mutate `fixed_inputs`.
        """
        ...


def unavailable_finding(
    step_id: str,
    metric_name: str,
    reason: str,
    *,
    classification: Classification | None = None,
    localization: LocalizationOutcome = LocalizationOutcome.SOURCE,
) -> Finding:
    """Construct a `Finding` carrying `Confidence.UNAVAILABLE` with a stated reason.

    `classification` is optional: classifiers that know which category they
    *would* have detected can pass it through so the report can name the gap.
    """
    return Finding(
        step_id=step_id,
        localization=localization,
        confidence=Confidence.UNAVAILABLE,
        metric_name=metric_name,
        classification=classification,
        reason=reason,
    )
