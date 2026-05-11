"""Tests for `varix replay` — reproducible re-classification without an adapter import."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from varix.core import (
    SCHEMA_VERSION,
    AdapterCapabilities,
    Classification,
    Confidence,
    CostSnapshot,
    Evidence,
    Finding,
    FrozenClock,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    RefusalRequired,
    SequenceRng,
    StepRun,
)
from varix.surface.cli import app
from varix.surface.dispatch import execute_replay, execute_run
from varix.surface.reporter import render_analysis, render_replay
from varix.surface.storage import save

_T = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
runner = CliRunner()


def _adapter_file(tmp_path: Path, variance_kw: str) -> Path:
    path = tmp_path / "agent.py"
    path.write_text(
        "from varix.adapters import FakeAdapter\n"
        "from varix.core import Classification\n"
        f"adapter = FakeAdapter(variance={variance_kw})\n",
        encoding="utf-8",
    )
    return path


def test_replay_body_byte_equals_render_analysis_body(tmp_path: Path) -> None:
    """The reproducibility property: replay's body is exactly what
    render_analysis(..., replayed=True) produces. The preamble is the only
    addition; nothing else in the body differs."""
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    runs_dir = tmp_path / "runs"
    analysis, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_T),
        rng=SequenceRng(["replay-id"]),
    )
    expected_body = render_analysis(analysis, replayed=True)
    replayed = execute_replay("replay-id", base_dir=runs_dir, clock=FrozenClock(_T))

    assert expected_body in replayed
    assert replayed.startswith("replay of analysis replay-i from just now.")
    assert "| replayed" in replayed


def test_replay_works_when_adapter_source_no_longer_exists(tmp_path: Path) -> None:
    """Replay must not import the adapter — the artifact alone is enough."""
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    runs_dir = tmp_path / "runs"
    execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_T),
        rng=SequenceRng(["no-adapter-id"]),
    )
    agent.unlink()  # delete the adapter source

    rendered = execute_replay("no-adapter-id", base_dir=runs_dir, clock=FrozenClock(_T))
    assert "Found 1 source of nondeterminism" in rendered
    assert "| replayed" in rendered


def test_replay_of_legacy_v0_1_artifact_uses_heuristic_capabilities(tmp_path: Path) -> None:
    """A schema-0.1 artifact (no capabilities recorded) must replay via
    infer_capabilities and produce the same findings as the 0.2 path."""
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROVIDER_SIDE}")
    runs_dir = tmp_path / "runs"
    analysis_v02, _ = execute_run(
        pipeline=str(agent),
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_T),
        rng=SequenceRng(["legacy-id"]),
    )

    legacy_path = runs_dir / "legacy-id.json"
    data = json.loads(legacy_path.read_text(encoding="utf-8"))
    data["schema_version"] = "0.1"
    del data["capabilities"]
    legacy_path.write_text(json.dumps(data), encoding="utf-8")

    rendered = execute_replay("legacy-id", base_dir=runs_dir, clock=FrozenClock(_T))
    expected_body = render_analysis(analysis_v02, replayed=True)
    assert expected_body in rendered


def test_replay_refuses_unknown_id(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute_replay("does-not-exist", base_dir=tmp_path)


def test_replay_refuses_newer_schema(tmp_path: Path) -> None:
    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    future_data = PipelineAnalysis(
        analysis_id="future",
        pipeline_name="x",
        n=1,
        metric_name="exact",
        schema_version="9.99",
        runs=(pr,),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    ).to_dict()
    future_data["schema_version"] = "9.99"
    artifact = tmp_path / "future.json"
    artifact.write_text(json.dumps(future_data), encoding="utf-8")

    with pytest.raises(RefusalRequired):
        execute_replay("future", base_dir=tmp_path)


def test_replay_preamble_relative_time(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    execute_run(
        pipeline="varix.adapters:FakeAdapter",
        input_text="hello",
        n=3,
        base_dir=runs_dir,
        clock=FrozenClock(_T),
        rng=SequenceRng(["aged-id"]),
    )
    later = FrozenClock(_T + timedelta(hours=2))
    rendered = execute_replay("aged-id", base_dir=runs_dir, clock=later)
    assert rendered.startswith("replay of analysis aged-id from 2 hours ago.")


def test_replay_preserves_runtime_notes_from_artifact() -> None:
    """Notes like 'budget exceeded' belong to the run, not the analysis. Replay
    must preserve them even though they're not re-derivable from the runs."""
    pr = PipelineRun(
        run_id="r1",
        step_runs=(
            StepRun(
                step_id="s1",
                inputs="i_const",
                output="o1",
                provider_metadata={"system_fingerprint": "fp_a"},
            ),
        ),
        started_at=_T,
        finished_at=_T,
    )
    pr2 = dataclasses.replace(
        pr,
        run_id="r2",
        step_runs=(
            StepRun(
                step_id="s1",
                inputs="i_const",
                output="o2",
                provider_metadata={"system_fingerprint": "fp_b"},
            ),
        ),
    )
    analysis = PipelineAnalysis(
        analysis_id="budget-id",
        pipeline_name="x",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr, pr2),
        findings=(),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
        notes=("budget exceeded after 2 of 5 runs",),
        capabilities=AdapterCapabilities(exposes_fingerprint=True),
    )

    # Need a tmp dir for save + replay
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save(analysis, base_dir=tmp_path)
        rendered = execute_replay("budget-id", base_dir=tmp_path, clock=FrozenClock(_T))
        assert "budget exceeded after 2 of 5 runs" in rendered


def test_cli_replay_command_renders_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    agent = _adapter_file(tmp_path, "{'s2': Classification.PROMPT_SIDE}")
    runner.invoke(app, ["run", str(agent), "--input", "hello", "-n", "3"])
    artifact = next(tmp_path.glob("*.json"))
    result = runner.invoke(app, ["replay", artifact.stem])
    assert result.exit_code == 0
    assert result.output.startswith("replay of analysis")
    assert "Found 1 source of nondeterminism" in result.output
    assert "| replayed" in result.output


def test_cli_replay_unknown_id_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["replay", "does-not-exist"])
    assert result.exit_code == 1
    assert "varix replay:" in result.output


def test_render_replay_directly_with_synthetic_analysis() -> None:
    """Unit-level render_replay test — preamble shape, body composition, no
    dependency on dispatch or storage."""
    pr = PipelineRun(
        run_id="r1",
        step_runs=(StepRun(step_id="s1", inputs="i", output="o"),),
        started_at=_T,
        finished_at=_T,
    )
    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        confidence=Confidence.HIGH,
        metric_name="exact",
        classification=Classification.PROVIDER_SIDE,
        evidence=(
            Evidence(
                kind="fingerprint_diff",
                description="x",
                data={"fingerprints": ["fp_a", "fp_b"], "unique": ["fp_a", "fp_b"]},
            ),
        ),
    )
    analysis = PipelineAnalysis(
        analysis_id="synth-id",
        pipeline_name="fake",
        n=2,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pr, pr),
        findings=(finding,),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(),
    )
    rendered = render_replay(analysis, now=_T + timedelta(days=3))
    assert rendered.startswith("replay of analysis synth-id from 3 days ago.")
    assert "| replayed" in rendered
