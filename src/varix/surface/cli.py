"""Typer-based CLI for varix."""

from __future__ import annotations

import logging

import typer

from varix import __version__
from varix.core import RefusalRequired, StructuralMismatch
from varix.surface.dispatch import (
    execute_explain,
    execute_impact,
    execute_run,
    execute_show,
)
from varix.surface.reporter import render_analysis

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A nondeterminism classifier for agent pipelines.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"varix {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show varix version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """varix — a nondeterminism classifier for agent pipelines."""
    if verbose:
        logging.getLogger("varix").setLevel(logging.DEBUG)


def _not_yet(command: str) -> None:
    typer.echo(f"varix {command}: not yet implemented", err=True)
    raise typer.Exit(code=1)


@app.command("run")
def run_cmd(
    pipeline: str = typer.Argument(
        ...,
        help="Pipeline target: a file path (`agent.py`) or import string (`pkg.mod:object`).",
    ),
    input_text: str = typer.Option(
        "",
        "--input",
        help="Input string passed to the pipeline.",
    ),
    n: int = typer.Option(
        3,
        "--n",
        "-n",
        min=1,
        help="Number of runs to execute.",
    ),
    max_cost: float | None = typer.Option(
        None,
        "--max-cost",
        help="Halt if accumulated cost in dollars exceeds this budget.",
    ),
) -> None:
    """Run the pipeline N times and write an analysis artifact."""
    try:
        analysis, path = execute_run(
            pipeline=pipeline,
            input_text=input_text,
            n=n,
            max_cost=max_cost,
        )
    except (StructuralMismatch, RefusalRequired) as exc:
        typer.echo("varix run: refusing to produce an analysis", err=True)
        typer.echo(f"  {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, ImportError, AttributeError, TypeError) as exc:
        typer.echo(f"varix run: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_analysis(analysis))
    typer.echo("")
    typer.echo(f"wrote {path}")


@app.command("show")
def show_cmd(
    analysis_id: str = typer.Argument(
        ...,
        help="Analysis ID, or a path to a saved JSON artifact.",
    ),
) -> None:
    """Render a saved analysis to the terminal."""
    try:
        rendered = execute_show(analysis_id)
    except RefusalRequired as exc:
        typer.echo(f"varix show: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, OSError) as exc:
        typer.echo(f"varix show: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(rendered)


@app.command("explain")
def explain_cmd(
    step_id: str = typer.Argument(
        ...,
        help="Step ID whose finding evidence trail should be printed.",
    ),
    analysis_id: str | None = typer.Option(
        None,
        "--analysis",
        help="Analysis ID or path. Defaults to the most recent saved artifact.",
    ),
) -> None:
    """Print the evidence trail behind a finding for a single step."""
    try:
        rendered = execute_explain(step_id, analysis_id)
    except RefusalRequired as exc:
        typer.echo(f"varix explain: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        typer.echo(f"varix explain: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(rendered)


@app.command("impact")
def impact_cmd(
    step_id: str = typer.Argument(
        ...,
        help="Source step ID to estimate downstream impact for.",
    ),
    analysis_id: str | None = typer.Option(
        None,
        "--analysis",
        help="Analysis ID or path. Defaults to the most recent saved artifact.",
    ),
) -> None:
    """Estimate downstream impact of variance at the given source step."""
    try:
        rendered = execute_impact(step_id, analysis_id)
    except RefusalRequired as exc:
        typer.echo(f"varix impact: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        typer.echo(f"varix impact: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(rendered)


if __name__ == "__main__":
    app()
