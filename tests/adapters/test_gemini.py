"""Tests for GeminiSingleCallAdapter.

Skipped entirely when google-genai is not installed (`pytest.importorskip`),
so the rest of the suite passes without the optional dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.genai")

from varix.adapters.gemini import GeminiSingleCallAdapter
from varix.core import Adapter, AdapterError, CapabilityMissing, RunFailed
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

    async def generate_content(self, *, model: str, contents: str) -> Any:
        self.calls.append({"model": model, "contents": contents})
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
    assert caps.supports_replay is False


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
    assert sr.provider_metadata == {"system_fingerprint": "gemini-2.5-flash-lite-001"}


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
async def test_run_pipeline_omits_metadata_when_no_fingerprint() -> None:
    response = _FakeResponse(text="ok", model_version=None, usage_metadata=_FakeUsage(0, 0))
    adapter = GeminiSingleCallAdapter(client=_FakeClient(response))
    run = await adapter.run_pipeline("hi")
    # Empty metadata dict collapses to None so the JSON artifact stays clean.
    assert run.step_runs[0].provider_metadata is None


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

    async def generate_content(self, *, model: str, contents: str) -> _FakeResponse:
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


# --- seed parameter is accepted but ignored (with one-shot warning) ---------


@pytest.mark.asyncio
async def test_run_pipeline_warns_when_seed_passed(caplog: pytest.LogCaptureFixture) -> None:
    """Protocol allows passing a seed; we don't honor it. Surface that fact."""
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        await adapter.run_pipeline("hi", seed=42)
    assert "seed=42" in caplog.text
    assert "ignored" in caplog.text


@pytest.mark.asyncio
async def test_run_pipeline_seed_warning_only_fires_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Don't spam the user with one warning per run in an n=10 loop."""
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        await adapter.run_pipeline("hi", seed=42)
        await adapter.run_pipeline("hi", seed=42)
        await adapter.run_pipeline("hi", seed=99)
    seed_warnings = [r for r in caplog.records if "ignored" in r.getMessage()]
    assert len(seed_warnings) == 1


@pytest.mark.asyncio
async def test_run_pipeline_no_warning_when_seed_is_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The varix runner currently passes seed=None; that path must stay quiet."""
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    with caplog.at_level(logging.WARNING, logger="varix.adapters.gemini"):
        await adapter.run_pipeline("hi")
    assert not any("seed" in r.getMessage() for r in caplog.records)


# --- Replay refusal ----------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_step_raises_capability_missing() -> None:
    adapter = GeminiSingleCallAdapter(client=_FakeClient(_ok_response()))
    with pytest.raises(CapabilityMissing, match="does not support replay"):
        await adapter.replay_step("response", "any input")


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
