"""Single-call Gemini adapter for varix.

Sends one prompt to a Gemini model and returns the response as a one-step
pipeline. For multi-step agents, compose this adapter (or its `_call` logic)
into your own `Adapter` class.

Capabilities:
    exposes_fingerprint=True   `model_version` from Gemini's response is mapped
                               to `system_fingerprint` for the provider classifier.
    exposes_tool_calls=True    Faithful zero — this adapter does not wire tool
                               calls; reporting zero each run is honest, not silent.
    supports_replay=False      Single-shot; `replay_step` raises CapabilityMissing.

Caveats:
    - Default cost rates apply to `gemini-2.5-flash-lite`. Pass
      `input_dollars_per_token` and `output_dollars_per_token` when using a
      different model, or `--max-cost` enforcement will be wrong. A warning
      is logged on construction if the model differs from default but rates
      do not.
    - `seed` is accepted for protocol conformance but not forwarded; Gemini's
      `generate_content` does not currently take a seed. A warning is logged
      the first time a non-None seed is passed.

Requires `google-genai`. Install with `pip install varix[gemini]`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

try:
    from google import genai
except ImportError as exc:
    raise ImportError() from exc

from varix.core import (
    AdapterCapabilities,
    AdapterError,
    CapabilityMissing,
    CostSnapshot,
    PipelineRun,
    Step,
    StepGraph,
    StepRun,
)

__all__ = ["GeminiSingleCallAdapter"]

_LOGGER: Final = logging.getLogger(__name__)

# Cheapest GA Gemini Flash variant
_DEFAULT_MODEL: Final = "gemini-2.5-flash-lite"
_DEFAULT_INPUT_DOLLARS_PER_TOKEN: Final = 0.10 / 1_000_000
_DEFAULT_OUTPUT_DOLLARS_PER_TOKEN: Final = 0.40 / 1_000_000

_RESPONSE_STEP_ID: Final = "response"
_STEPS: Final[tuple[Step, ...]] = (Step(id=_RESPONSE_STEP_ID, name="response", index=0),)


@dataclass(frozen=True, slots=True)
class _CallResult:
    output: str
    metadata: dict[str, Any]
    cost: CostSnapshot


def _resolve_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiSingleCallAdapter:
    """Send one prompt to a Gemini model and return the response as a one-step pipeline."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        client: Any = None,
        input_dollars_per_token: float = _DEFAULT_INPUT_DOLLARS_PER_TOKEN,
        output_dollars_per_token: float = _DEFAULT_OUTPUT_DOLLARS_PER_TOKEN,
    ) -> None:
        if client is None:
            api_key = _resolve_api_key()
            if not api_key:
                raise AdapterError(
                    "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                    "in the environment, or pass an explicit `client` argument."
                )
            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model
        self._input_rate = input_dollars_per_token
        self._output_rate = output_dollars_per_token
        self._warned_seed = False

        # Default rates apply to _DEFAULT_MODEL only. Warn if the user picks a
        # different model without overriding rates, so --max-cost stays honest.
        if (
            model != _DEFAULT_MODEL
            and input_dollars_per_token == _DEFAULT_INPUT_DOLLARS_PER_TOKEN
            and output_dollars_per_token == _DEFAULT_OUTPUT_DOLLARS_PER_TOKEN
        ):
            _LOGGER.warning(
                "model=%r uses default rates for %r; cost figures (and --max-cost) "
                "may be wrong. Pass input_dollars_per_token / output_dollars_per_token "
                "to override.",
                model,
                _DEFAULT_MODEL,
            )

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            exposes_fingerprint=True,
            exposes_tool_calls=True,
            supports_replay=False,
        )

    async def pipeline_structure(self, pipeline_input: Any) -> StepGraph:
        return StepGraph(steps=_STEPS)

    async def run_pipeline(self, pipeline_input: Any, seed: int | None = None) -> PipelineRun:
        if seed is not None and not self._warned_seed:
            # Accepted for protocol conformance but not forwarded — Gemini's
            # generate_content does not currently take a seed.
            _LOGGER.warning("seed=%r passed but ignored; Gemini API does not honor it.", seed)
            self._warned_seed = True
        prompt = str(pipeline_input)
        started = datetime.now(tz=UTC)
        result = await self._call(prompt)
        finished = datetime.now(tz=UTC)
        return PipelineRun(
            run_id=f"r-{started.timestamp()}",
            step_runs=(
                StepRun(
                    step_id=_RESPONSE_STEP_ID,
                    inputs=prompt,
                    output=result.output,
                    provider_metadata=result.metadata or None,
                    cost=result.cost,
                ),
            ),
            started_at=started,
            finished_at=finished,
            cost=result.cost,
        )

    async def replay_step(
        self, step_id: str, fixed_inputs: Any, seed: int | None = None
    ) -> StepRun:
        raise CapabilityMissing(
            "GeminiSingleCallAdapter does not support replay (supports_replay=False)"
        )

    async def _call(self, prompt: str) -> _CallResult:
        # SDK errors (genai_errors.APIError, network errors) propagate unwrapped:
        # the runner catches them as non-VarixError exceptions and surfaces them
        # via RunFailed with the partial_runs that succeeded before the failure.
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
        )

        usage = getattr(resp, "usage_metadata", None)
        if usage is None:
            _LOGGER.warning("Gemini response missing usage_metadata; cost recorded as zero.")
            input_tokens = 0
            output_tokens = 0
        else:
            input_tokens = getattr(usage, "prompt_token_count", None) or 0
            output_tokens = getattr(usage, "candidates_token_count", None) or 0

        metadata: dict[str, Any] = {}
        model_version = getattr(resp, "model_version", None)
        if model_version:
            metadata["system_fingerprint"] = str(model_version)

        return _CallResult(
            output=getattr(resp, "text", "") or "",
            metadata=metadata,
            cost=CostSnapshot(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                dollars=(input_tokens * self._input_rate + output_tokens * self._output_rate),
            ),
        )
