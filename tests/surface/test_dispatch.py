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


def test_execute_run_structural_mismatch_writes_no_artifact(tmp_path: Path) -> None:
    """An adapter that returns inconsistent step graphs produces a refusal."""
    from varix.core import StructuralMismatch

    agent = tmp_path / "agent.py"
    agent.write_text(
        "from varix.adapters import FakeAdapter\nadapter = FakeAdapter(structure_variance=True)\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"
    with pytest.raises(StructuralMismatch):
        execute_run(
            pipeline=str(agent),
            input_text="hello",
            n=3,
            base_dir=runs_dir,
            clock=FrozenClock(_FROZEN),
            rng=SequenceRng(["id"]),
        )
    # Refusal is loud; nothing is persisted.
    assert not runs_dir.exists() or list(runs_dir.glob("*.json")) == []


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


def test_execute_run_saves_partial_artifact_when_adapter_fails(tmp_path: Path) -> None:
    """A run-loop interrupted by an adapter exception still produces a saved artifact.

    The completed runs become the analysis input; the truncation reason is
    recorded in `analysis.notes`. This is the explicit anti-silent-failure
    contract: cost is never silently discarded.
    """
    agent = tmp_path / "agent.py"
    agent.write_text(
        "from typing import Any\n"
        "from datetime import UTC, datetime\n"
        "from varix.core import (\n"
        "    AdapterCapabilities, PipelineRun, StepGraph, StepRun,\n"
        ")\n"
        "_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)\n"
        "\n"
        "class _Flaky:\n"
        "    def __init__(self) -> None: self._calls = 0\n"
        "    def capabilities(self) -> AdapterCapabilities:\n"
        "        return AdapterCapabilities()\n"
        "    async def pipeline_structure(self, pi: Any) -> StepGraph:\n"
        "        return StepGraph(steps=())\n"
        "    async def run_pipeline(self, pi: Any, seed: int | None = None) -> PipelineRun:\n"
        "        self._calls += 1\n"
        "        if self._calls > 2: raise RuntimeError('provider stalled')\n"
        "        return PipelineRun(run_id=f'r{self._calls}', step_runs=(),\n"
        "            started_at=_T, finished_at=_T)\n"
        "    async def replay_step(self, sid: str, fi: Any,\n"
        "        seed: int | None = None) -> StepRun:\n"
        "        raise NotImplementedError\n"
        "\n"
        "adapter = _Flaky()\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"
    analysis, path = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=5,
        base_dir=runs_dir,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["partial-id"]),
    )
    assert path.exists()
    assert analysis.n == 2  # only the runs that completed
    assert len(analysis.notes) == 1
    note = analysis.notes[0]
    assert "RuntimeError" in note
    assert "provider stalled" in note
    assert "run 3 of 5" in note


def test_execute_run_clean_run_records_no_notes(tmp_path: Path) -> None:
    analysis, _ = execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=2,
        base_dir=tmp_path,
        clock=FrozenClock(_FROZEN),
        rng=SequenceRng(["id"]),
    )
    assert analysis.notes == ()
