"""Tests for surface.dispatch (execute_run orchestration)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from varix.core import (
    SCHEMA_VERSION,
    FrozenClock,
    SequenceRng,
)
from varix.surface import load
from varix.surface.dispatch import execute_run, resolve_runs_dir

_FROZEN = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def test_execute_run_writes_loadable_artifact(tmp_path: Path) -> None:
    analysis, path = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=3,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["fixed-id-1"]),
    )
    assert path.exists()
    loaded = load("fixed-id-1", base_dir=tmp_path)
    assert loaded == analysis


def test_execute_run_uses_injected_clock_and_rng(tmp_path: Path) -> None:
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=2,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["abc-123"]),
    )
    assert analysis.analysis_id == "abc-123"
    assert analysis.started_at == _FROZEN
    assert analysis.finished_at == _FROZEN


def test_execute_run_produces_n_runs(tmp_path: Path) -> None:
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=4,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    assert analysis.n == 4
    assert len(analysis.runs) == 4


def test_execute_run_writes_empty_findings_stub(tmp_path: Path) -> None:
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=2,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    assert analysis.findings == ()
    assert analysis.schema_version == SCHEMA_VERSION
    assert analysis.metric_name == "exact"


def test_execute_run_records_pipeline_target_as_name(tmp_path: Path) -> None:
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=1,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    assert analysis.pipeline_name == "varix.adapters:FakeAdapter"


def test_execute_run_with_max_cost_under_budget_completes(tmp_path: Path) -> None:
    # FakeAdapter reports zero cost, so any budget is satisfied.
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=3,
        max_cost=1.0,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    assert analysis.n == 3


def test_execute_run_resolution_failure_raises_before_save(tmp_path: Path) -> None:
    with pytest.raises(ImportError):
        execute_run(
            pipeline="not_a_real_module_xyz:adapter",
            input_text="hello",
            n=1,
            base_dir=tmp_path,
            clock=FrozenClock(_FROZEN),
            rng=SequenceRng(["id"]),
        )
    # No artifact written.
    assert list(tmp_path.glob("*.json")) == []


def test_resolve_runs_dir_prefers_explicit_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"
    assert resolve_runs_dir(explicit) == explicit


def test_resolve_runs_dir_falls_back_to_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    assert resolve_runs_dir(None) == tmp_path


def test_resolve_runs_dir_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VARIX_RUNS_DIR", raising=False)
    assert resolve_runs_dir(None) is None
