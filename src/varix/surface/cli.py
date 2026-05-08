"""Typer-based CLI scaffold for varix.

Defines the four user-facing commands (`run`, `show`, `explain`, `impact`)
with their planned argument signatures. Each command currently exits with
code 1 and a "not yet implemented" message; subsequent commits replace the
bodies one at a time.
"""

from __future__ import annotations

import logging

import typer

from varix import __version__

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
    _ = pipeline, input_text, n, max_cost
    _not_yet("run")


@app.command("show")
def show_cmd(
    analysis_id: str = typer.Argument(
        ...,
        help="Analysis ID, or a path to a saved JSON artifact.",
    ),
) -> None:
    """Render a saved analysis to the terminal."""
    _ = analysis_id
    _not_yet("show")


@app.command("explain")
def explain_cmd(
    step_id: str = typer.Argument(
        ...,
        help="Step ID whose finding evidence trail should be printed.",
    ),
    analysis_id: str | None = typer.Option(
        None,
        "--analysis",
        help="Analysis ID. Defaults to the most recent.",
    ),
) -> None:
    """Print the evidence trail behind a finding for a single step."""
    _ = step_id, analysis_id
    _not_yet("explain")


@app.command("impact")
def impact_cmd(
    step_id: str = typer.Argument(
        ...,
        help="Source step ID to estimate downstream impact for.",
    ),
    analysis_id: str | None = typer.Option(
        None,
        "--analysis",
        help="Analysis ID. Defaults to the most recent.",
    ),
) -> None:
    """Estimate downstream impact of variance at the given source step."""
    _ = step_id, analysis_id
    _not_yet("impact")


if __name__ == "__main__":
    app()
