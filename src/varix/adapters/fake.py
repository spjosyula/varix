"""FakeAdapter — a controlled five-step pipeline for tests.

Variance can be injected per step per category to validate localizer and
classifier behavior without touching real model providers. With no variance
configured, every call to `run_pipeline` and `replay_step` produces
identical step output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from varix.core import (
    AdapterCapabilities,
    Classification,
    PipelineRun,
    Step,
    StepGraph,
    StepRun,
    ToolCall,
)

_STEPS: tuple[Step, ...] = (
    Step(id="s1", name="planner", index=0),
    Step(id="s2", name="researcher", index=1),
    Step(id="s3", name="ranker", index=2),
    Step(id="s4", name="summarizer", index=3),
    Step(id="s5", name="responder", index=4),
)

_STABLE_TIME = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


class FakeAdapter:
    """Five-step test pipeline. Pass `variance={step_id: Classification(...)}` to inject.

    Setting `structure_variance=True` makes `run_pipeline` return the full
    five-step sequence on odd-numbered calls and a truncated three-step
    sequence on even-numbered calls — used to exercise varix's
    structural-mismatch refusal path.
    """

    def __init__(
        self,
        *,
        variance: dict[str, Classification] | None = None,
        structure_variance: bool = False,
    ) -> None:
        self._variance: dict[str, Classification] = dict(variance) if variance else {}
        self._structure_variance = structure_variance
        self._run_counter = 0
        self._replay_counter = 0

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            exposes_fingerprint=True,
            exposes_tool_calls=True,
            supports_replay=True,
        )

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(steps=_STEPS)

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        self._run_counter += 1
        n = self._run_counter
        steps = self._steps_for_run(n)
        prev_output: Any = pipeline_input
        step_runs: list[StepRun] = []
        for step in steps:
            sr = _build_step_run(step, prev_output, n, self._variance.get(step.id))
            step_runs.append(sr)
            prev_output = sr.output
        return PipelineRun(
            run_id=f"r{n}",
            step_runs=tuple(step_runs),
            started_at=_STABLE_TIME,
            finished_at=_STABLE_TIME,
        )

    def _steps_for_run(self, run_index: int) -> tuple[Step, ...]:
        if not self._structure_variance:
            return _STEPS
        # Alternate between the full five-step graph and a truncated three-step
        # graph so analyze() sees inconsistent step sequences across runs.
        return _STEPS if run_index % 2 == 1 else _STEPS[:3]

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        self._replay_counter += 1
        step = next(s for s in _STEPS if s.id == step_id)
        return _build_step_run(
            step, fixed_inputs, self._replay_counter, self._variance.get(step_id)
        )


def _build_step_run(
    step: Step, inputs: Any, run_index: int, category: Classification | None
) -> StepRun:
    # Include inputs so variance propagates through downstream steps under the
    # baseline (no-category) path — gives the Localizer real DOWNSTREAM cases.
    base_output = f"{step.id}_output:{inputs}"
    base_tool_calls: tuple[ToolCall, ...] = (
        ToolCall(name="lookup", arguments={"q": str(inputs)}, result="hit"),
    )
    base_metadata = {"system_fingerprint": "fp_stable"}

    if category is None:
        return StepRun(
            step_id=step.id,
            inputs=inputs,
            output=base_output,
            tool_calls=base_tool_calls,
            provider_metadata=base_metadata,
        )

    if category is Classification.PROVIDER_SIDE:
        fp = "fp_a" if run_index % 2 == 1 else "fp_b"
        return StepRun(
            step_id=step.id,
            inputs=inputs,
            output=base_output,
            tool_calls=base_tool_calls,
            provider_metadata={"system_fingerprint": fp},
        )

    if category is Classification.TOOL_SIDE:
        result = f"hit_{run_index}"
        return StepRun(
            step_id=step.id,
            inputs=inputs,
            output=base_output,
            tool_calls=(ToolCall(name="lookup", arguments={"q": str(inputs)}, result=result),),
            provider_metadata=base_metadata,
        )

    if category is Classification.ORDERING:
        a = ToolCall(name="lookup", arguments={"q": "first"}, result="alpha")
        b = ToolCall(name="lookup", arguments={"q": "second"}, result="beta")
        ordered = (a, b) if run_index % 2 == 1 else (b, a)
        return StepRun(
            step_id=step.id,
            inputs=inputs,
            output=base_output,
            tool_calls=ordered,
            provider_metadata=base_metadata,
        )

    if category is Classification.PROMPT_SIDE:
        return StepRun(
            step_id=step.id,
            inputs=inputs,
            output=f"{base_output}_v{run_index}",
            tool_calls=base_tool_calls,
            provider_metadata=base_metadata,
        )

    if category is Classification.TIME_OR_STATE:
        return StepRun(
            step_id=step.id,
            inputs=inputs,
            output=f"{base_output} at 2026-05-08T12:00:{run_index:02d}Z",
            tool_calls=base_tool_calls,
            provider_metadata=base_metadata,
        )

    raise ValueError(f"unhandled category: {category}")
