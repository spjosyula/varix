"""Tests for GeminiSingleCallAdapter.

Skipped entirely when google-genai is not installed (`pytest.importorskip`),
so the rest of the suite passes without the optional dependency.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.genai")

from google.genai import types as genai_types

from varix.adapters.gemini import GeminiSingleCallAdapter
from varix.core import Adapter, AdapterError, RunFailed
from varix.core.protocol_test_suite import validate_adapter
from varix.execution import run_n

# --- Test stubs --------------------------------------------------------------


@dataclass
class _FakeUsage:
    prompt_token_count: int
    candidates_token_count: int


@dataclass
class _FakeResponse:
    text: str
    model_version: str | None
    usage_metadata: _FakeUsage | None


class _FakeAioModels:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: str, config: Any = None) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class _FakeAio:
    def __init__(self, response: Any) -> None:
        self.models = _FakeAioModels(response)


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.aio = _FakeAio(response)


def _ok_response() -> _FakeResponse:
    return _FakeResponse(
        text="The sky is blue because of Rayleigh scattering.",
        model_version="gemini-2.5-flash-lite-001",
        usage_metadata=_FakeUsage(prompt_token_count=10, candidates_token_count=20),
    )


# --- Construction ------------------------------------------------------------


def test_init_without_api_key_raises_adapter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="GEMINI_API_KEY"):
        GeminiSingleCallAdapter()


def test_init_with_explicit_client_skips_env_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))


def test_init_warns_when_non_default_model_keeps_default_rates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silent-budget-breakage guard: model changed without rate override should warn."""
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        GeminiSingleCallAdapter(model="gemini-2.5-pro", client=_FakeClient(_ok_response()))
    assert "default rates" in caplog.text
    assert "gemini-2.5-pro" in caplog.text


def test_init_no_warning_when_non_default_model_overrides_rates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit rate override is the user opting in; no warning."""
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        GeminiSingleCallAdapter(
            model="gemini-2.5-pro",
            client=_FakeClient(_ok_response()),
            input_dollars_per_token=1.25 / 1_000_000,
            output_dollars_per_token=5.0 / 1_000_000,
        )
    assert "default rates" not in caplog.text


def test_init_no_warning_when_default_model_default_rates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The default config is the happy path; never warn."""
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    assert "default rates" not in caplog.text


# --- Capabilities & protocol conformance -------------------------------------


def test_capabilities_match_design() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    caps = adapter.capabilities()
    assert caps.exposes_fingerprint is True
    assert caps.exposes_tool_calls is True
    assert caps.supports_replay is True


def test_satisfies_runtime_checkable_protocol() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    assert isinstance(adapter, Adapter)


@pytest.mark.asyncio
async def test_passes_protocol_test_suite() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    await validate_adapter(adapter, "any input")


# --- run_pipeline behaviour --------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_records_output_and_fingerprint() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    run = await adapter.run_pipeline("Why is the sky blue?")
    assert len(run.step_runs) == 1
    sr = run.step_runs[0]
    assert sr.step_id == "response"
    assert sr.inputs == "Why is the sky blue?"
    assert "Rayleigh" in str(sr.output)
    assert sr.provider_metadata == {
        "system_fingerprint": "gemini-2.5-flash-lite-001",
        "temperature": 0.0,
    }


@pytest.mark.asyncio
async def test_run_pipeline_computes_cost_from_usage_metadata() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    run = await adapter.run_pipeline("hi")
    sr = run.step_runs[0]
    expected = (10 * 0.10 / 1_000_000) + (20 * 0.40 / 1_000_000)
    assert sr.cost.input_tokens == 10
    assert sr.cost.output_tokens == 20
    assert sr.cost.dollars == pytest.approx(expected)


@pytest.mark.asyncio
async def test_run_pipeline_coerces_input_to_string() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    await adapter.run_pipeline(42)
    assert fake.aio.models.calls[0]["contents"] == "42"


@pytest.mark.asyncio
async def test_run_pipeline_passes_configured_model_to_client() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(model="gemini-2.5-pro", client=fake)
    await adapter.run_pipeline("hi")
    assert fake.aio.models.calls[0]["model"] == "gemini-2.5-pro"


# --- Provider metadata edge cases --------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_records_temperature_when_no_fingerprint() -> None:
    response = _FakeResponse(text="ok", model_version=None, usage_metadata=_FakeUsage(0, 0))
    adapter = GeminiSingleCallAdapter(client=_FakeClient(response))
    run = await adapter.run_pipeline("hi")
    # Even without a fingerprint, the resolved temperature is recorded so
    # future replays know what conditions produced the artifact.
    assert run.step_runs[0].provider_metadata == {"temperature": 0.0}


@pytest.mark.asyncio
async def test_run_pipeline_warns_when_usage_metadata_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(text="ok", model_version="x", usage_metadata=None)
    adapter = GeminiSingleCallAdapter(client=_FakeClient(response))
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        run = await adapter.run_pipeline("hi")
    assert "usage_metadata" in caplog.text
    assert run.step_runs[0].cost.dollars == 0.0


# --- SDK errors propagate so the runner can preserve partial_runs ------------


class _IntermittentModels:
    """First `succeed_for` calls return _ok_response; subsequent calls raise `exc`."""

    def __init__(self, succeed_for: int, exc: BaseException) -> None:
        self._calls = 0
        self._succeed_for = succeed_for
        self._exc = exc

    async def generate_content(
        self, *, model: str, contents: str, config: Any = None
    ) -> _FakeResponse:
        self._calls += 1
        if self._calls > self._succeed_for:
            raise self._exc
        return _ok_response()


def _intermittent_client(succeed_for: int, exc: BaseException) -> SimpleNamespace:
    return SimpleNamespace(aio=SimpleNamespace(models=_IntermittentModels(succeed_for, exc)))


@pytest.mark.asyncio
async def test_mid_loop_sdk_error_surfaces_as_run_failed_with_partial_runs() -> None:
    """Run 4 of 5 hits a network error; runs 1-3 must survive in partial_runs."""
    client = _intermittent_client(succeed_for=3, exc=ConnectionError("dns failed"))
    adapter = GeminiSingleCallAdapter(client=client)
    with pytest.raises(RunFailed) as ei:
        await run_n(adapter, "hi", n=5)
    assert len(ei.value.partial_runs) == 3
    assert "ConnectionError" in str(ei.value)
    assert "dns failed" in str(ei.value)


# --- temperature default & resolution ---------------------------------------


@pytest.mark.asyncio
async def test_temperature_defaults_to_zero_in_sdk_config() -> None:
    """varix's diagnostic identity depends on temperature=0 being the default."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    await adapter.run_pipeline("hi")
    cfg = fake.aio.models.calls[0]["config"]
    assert cfg.temperature == 0.0


@pytest.mark.asyncio
async def test_temperature_kwarg_overrides_default() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake, temperature=0.7)
    await adapter.run_pipeline("hi")
    assert fake.aio.models.calls[0]["config"].temperature == 0.7


@pytest.mark.asyncio
async def test_temperature_kwarg_wins_over_generation_config() -> None:
    """The kwarg is varix's opinionated lane; the dict is pass-through."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(
        client=fake,
        temperature=0.7,
        generation_config={"temperature": 0.3},
    )
    await adapter.run_pipeline("hi")
    assert fake.aio.models.calls[0]["config"].temperature == 0.7


@pytest.mark.asyncio
async def test_generation_config_without_temperature_still_gets_default() -> None:
    """A user passing only top_p still gets temperature=0; default isn't conditional."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(
        client=fake,
        generation_config={"top_p": 0.9},
    )
    await adapter.run_pipeline("hi")
    cfg = fake.aio.models.calls[0]["config"]
    assert cfg.temperature == 0.0
    assert cfg.top_p == 0.9


@pytest.mark.asyncio
async def test_resolved_temperature_recorded_in_provider_metadata() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake, temperature=0.5)
    run = await adapter.run_pipeline("hi")
    assert run.step_runs[0].provider_metadata == {
        "system_fingerprint": "gemini-2.5-flash-lite-001",
        "temperature": 0.5,
    }


# --- generation_config pass-through ------------------------------------------


@pytest.mark.asyncio
async def test_generation_config_passes_through_to_sdk() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(
        client=fake,
        generation_config={"top_p": 0.9, "max_output_tokens": 100},
    )
    await adapter.run_pipeline("hi")
    cfg = fake.aio.models.calls[0]["config"]
    assert cfg.top_p == 0.9
    assert cfg.max_output_tokens == 100


# --- timeout -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_defaults_to_60_seconds() -> None:
    """SDK takes milliseconds; user-facing API is seconds."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    await adapter.run_pipeline("hi")
    http_opts = fake.aio.models.calls[0]["config"].http_options
    assert http_opts.timeout == 60_000


@pytest.mark.asyncio
async def test_timeout_kwarg_honored() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake, timeout=10.0)
    await adapter.run_pipeline("hi")
    assert fake.aio.models.calls[0]["config"].http_options.timeout == 10_000


# --- seed forwarding ---------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_forwarded_to_sdk_when_provided() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    run = await adapter.run_pipeline("hi", seed=42)
    assert fake.aio.models.calls[0]["config"].seed == 42
    # StepRun.seed records what was used so replays know the conditions.
    assert run.step_runs[0].seed == 42


@pytest.mark.asyncio
async def test_seed_none_means_no_seed_in_config() -> None:
    """The runner passes seed=None today; that path must not inject a seed."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    run = await adapter.run_pipeline("hi")
    assert fake.aio.models.calls[0]["config"].seed is None
    assert run.step_runs[0].seed is None


@pytest.mark.asyncio
async def test_per_call_seed_overrides_generation_config_seed() -> None:
    """Per-call seed is the more specific signal; it wins."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake, generation_config={"seed": 1})
    await adapter.run_pipeline("hi", seed=999)
    assert fake.aio.models.calls[0]["config"].seed == 999


@pytest.mark.asyncio
async def test_seed_no_longer_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Forwarding is the new behavior; the old ignored-seed warning is gone."""
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        await adapter.run_pipeline("hi", seed=42)
    assert not any("ignored" in r.getMessage() for r in caplog.records)


# --- run_id is a UUID, not a timestamp --------------------------------------


@pytest.mark.asyncio
async def test_run_id_is_uuid_and_unique_per_call() -> None:
    """Timestamp-based IDs collided when two runs started in the same second.
    UUID4s never do."""
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    run1 = await adapter.run_pipeline("hi")
    run2 = await adapter.run_pipeline("hi")
    # Both must parse as valid UUIDs and must differ.
    uuid.UUID(run1.run_id)
    uuid.UUID(run2.run_id)
    assert run1.run_id != run2.run_id


# --- http_options passed in generation_config is preserved -------------------


@pytest.mark.asyncio
async def test_http_options_in_generation_config_is_preserved() -> None:
    """When the user provides http_options, the adapter's timeout default
    must not silently overwrite it. Power users set base_url / headers here."""
    fake = _FakeClient(_ok_response())
    user_http = genai_types.HttpOptions(timeout=5000)  # 5s, user's choice
    adapter = GeminiSingleCallAdapter(
        client=fake,
        generation_config={"http_options": user_http},
        timeout=99.0,  # ignored when user supplies http_options
    )
    await adapter.run_pipeline("hi")
    sent_http = fake.aio.models.calls[0]["config"].http_options
    assert sent_http.timeout == 5000  # user's setting wins; 99_000 not applied


# --- Replay --------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_step_returns_step_run_with_fixed_inputs() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    sr = await adapter.replay_step("response", "fixed prompt")
    assert sr.step_id == "response"
    assert sr.inputs == "fixed prompt"
    assert "Rayleigh" in str(sr.output)
    assert sr.cost.dollars > 0
    # Construction-time temperature default still applies; recorded in metadata.
    assert sr.provider_metadata == {
        "system_fingerprint": "gemini-2.5-flash-lite-001",
        "temperature": 0.0,
    }


@pytest.mark.asyncio
async def test_replay_step_forwards_seed_to_sdk() -> None:
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake)
    sr = await adapter.replay_step("response", "fixed prompt", seed=7)
    assert fake.aio.models.calls[0]["config"].seed == 7
    assert sr.seed == 7


@pytest.mark.asyncio
async def test_replay_step_uses_construction_temperature() -> None:
    """Replay must reproduce — temperature is locked to construction time."""
    fake = _FakeClient(_ok_response())
    adapter = GeminiSingleCallAdapter(client=fake, temperature=0.3)
    await adapter.replay_step("response", "fixed prompt")
    cfg = fake.aio.models.calls[0]["config"]
    assert cfg.temperature == 0.3


@pytest.mark.asyncio
async def test_replay_step_raises_on_unknown_step_id() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    with pytest.raises(AdapterError, match="unknown step_id"):
        await adapter.replay_step("not_a_step", "anything")


# --- Opt-in integration test -------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="requires GEMINI_API_KEY or GOOGLE_API_KEY",
)
@pytest.mark.asyncio
async def test_integration_real_call_returns_artifact_shape() -> None:
    """Hits real Gemini. Skipped without an API key."""
    adapter = GeminiSingleCallAdapter()
    run = await adapter.run_pipeline("Say the word 'hello' and nothing else.")
    assert len(run.step_runs) == 1
    sr = run.step_runs[0]
    assert sr.output
    assert sr.cost.dollars > 0
