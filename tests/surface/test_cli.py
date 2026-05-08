"""Tests for the Typer CLI."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from varix import __version__
from varix.surface import load
from varix.surface.cli import app

runner = CliRunner()

# Typer renders help via Rich, which emits ANSI escape sequences and wraps to
# the detected terminal width. Strip the escapes so substring assertions don't
# trip over color codes; widen via the COLUMNS env var to prevent wrap-induced
# line splits inside option columns.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_WIDE_HELP_ENV = {"COLUMNS": "200"}


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def test_version_flag_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_works_at_top_level() -> None:
    result = runner.invoke(app, ["--help"], env=_WIDE_HELP_ENV)
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "run" in output
    assert "show" in output
    assert "explain" in output
    assert "impact" in output


def test_no_args_shows_help() -> None:
    # no_args_is_help=True → bare invocation yields help and a non-zero exit.
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.output


def test_run_writes_loadable_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = runner.invoke(
        app,
        ["run", "varix.adapters:FakeAdapter", "--input", "hello", "-n", "3"],
    )
    assert result.exit_code == 0, result.output
    artifacts = list(tmp_path.glob("*.json"))
    assert len(artifacts) == 1
    loaded = load(artifacts[0].stem, base_dir=tmp_path)
    assert loaded.n == 3
    assert loaded.findings == ()  # stub: real analysis lands later
    assert loaded.pipeline_name == "varix.adapters:FakeAdapter"


def test_run_prints_artifact_path_and_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["run", "varix.adapters:FakeAdapter", "--input", "hello"])
    assert result.exit_code == 0
    assert "=== varix analysis ===" in result.output  # rendered report
    assert "wrote" in result.output
    assert "analysis_id:" in result.output
    assert "n:           3" in result.output  # default n=3


def test_run_resolution_failure_exits_one_with_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["run", "not_a_real_module_xyz:adapter"])
    assert result.exit_code == 1
    assert "varix run:" in result.output
    assert list(tmp_path.glob("*.json")) == []


def test_run_structure_variance_exits_two_with_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path / "runs"))
    agent = tmp_path / "agent.py"
    agent.write_text(
        "from varix.adapters import FakeAdapter\nadapter = FakeAdapter(structure_variance=True)\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", str(agent), "--input", "hello", "-n", "3"])
    assert result.exit_code == 2
    assert "refusing to produce an analysis" in result.output
    assert "structure varied" in result.output
    # No artifact written.
    runs_dir = tmp_path / "runs"
    if runs_dir.exists():
        assert list(runs_dir.glob("*.json")) == []


def test_show_renders_saved_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    # Save an artifact via run, then show it.
    runner.invoke(app, ["run", "varix.adapters:FakeAdapter", "--input", "hello", "-n", "2"])
    artifact = next(tmp_path.glob("*.json"))
    result = runner.invoke(app, ["show", artifact.stem])
    assert result.exit_code == 0
    assert "=== varix analysis ===" in result.output
    assert "step s1: deterministic" in result.output


def test_show_unknown_id_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["show", "does-not-exist"])
    assert result.exit_code == 1
    assert "varix show:" in result.output


def test_explain_renders_evidence_for_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    agent = tmp_path / "agent.py"
    agent.write_text(
        "from varix.adapters import FakeAdapter\n"
        "from varix.core import Classification\n"
        "adapter = FakeAdapter(variance={'s2': Classification.PROVIDER_SIDE})\n",
        encoding="utf-8",
    )
    runner.invoke(app, ["run", str(agent), "--input", "hello", "-n", "3"])
    result = runner.invoke(app, ["explain", "s2"])
    assert result.exit_code == 0
    assert "=== explain s2 ===" in result.output
    assert "provider_side (high)" in result.output
    assert "evidence:" in result.output
    assert "fingerprint_diff" in result.output


def test_explain_unknown_step_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    runner.invoke(app, ["run", "varix.adapters:FakeAdapter", "--input", "hello"])
    result = runner.invoke(app, ["explain", "no_such_step"])
    assert result.exit_code == 1
    assert "varix explain:" in result.output
    assert "no_such_step" in result.output


def test_explain_with_no_saved_analyses_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))  # empty dir
    result = runner.invoke(app, ["explain", "s1"])
    assert result.exit_code == 1
    assert "varix explain:" in result.output


def test_run_help_lists_planned_options() -> None:
    result = runner.invoke(app, ["run", "--help"], env=_WIDE_HELP_ENV)
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--n" in output
    assert "--input" in output
    assert "--max-cost" in output


def test_verbose_flag_sets_debug_level_on_varix_logger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VARIX_RUNS_DIR", str(tmp_path))
    logger = logging.getLogger("varix")
    original_level = logger.level
    try:
        runner.invoke(
            app,
            ["--verbose", "run", "varix.adapters:FakeAdapter", "--input", "x", "-n", "1"],
        )
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(original_level)
