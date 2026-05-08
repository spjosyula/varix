"""Tests for the Typer CLI scaffold."""

from __future__ import annotations

import logging

from typer.testing import CliRunner

from varix import __version__
from varix.surface.cli import app

runner = CliRunner()


def test_version_flag_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_works_at_top_level() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "show" in result.output
    assert "explain" in result.output
    assert "impact" in result.output


def test_no_args_shows_help() -> None:
    # no_args_is_help=True → bare invocation yields help and a non-zero exit.
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.output


def test_run_stub_exits_non_zero_with_message() -> None:
    result = runner.invoke(app, ["run", "agent.py"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_show_stub_exits_non_zero_with_message() -> None:
    result = runner.invoke(app, ["show", "abc-123"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_explain_stub_exits_non_zero_with_message() -> None:
    result = runner.invoke(app, ["explain", "s1"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_impact_stub_exits_non_zero_with_message() -> None:
    result = runner.invoke(app, ["impact", "s1"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_run_help_lists_planned_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--n" in result.output
    assert "--input" in result.output
    assert "--max-cost" in result.output


def test_verbose_flag_sets_debug_level_on_varix_logger() -> None:
    logger = logging.getLogger("varix")
    original_level = logger.level
    try:
        runner.invoke(app, ["--verbose", "run", "agent.py"])
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(original_level)
