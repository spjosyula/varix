"""Command dispatch — orchestrates the work behind each CLI command.

Lives between the Typer command handlers (which deal with argument parsing
and user output) and the lower layers (execution, analysis, storage). Tests
target these functions directly to avoid going through the CLI runner.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from varix.analysis import analyze
from varix.core import (
    SCHEMA_VERSION,
    BudgetExceeded,
    Clock,
    ExactMatch,
    PipelineAnalysis,
    Rng,
    SystemClock,
    SystemRng,
)
from varix.execution import CostAccumulator, run_n
from varix.surface.loader import load_adapter
from varix.surface.reporter import render_analysis, render_explain
from varix.surface.storage import latest_analysis, load, load_path, save

_RUNS_DIR_ENV = "VARIX_RUNS_DIR"


def execute_run(
    pipeline: str,
    input_text: str,
    n: int,
    *,
    max_cost: float | None = None,
    base_dir: Path | None = None,
    clock: Clock | None = None,
    rng: Rng | None = None,
) -> tuple[PipelineAnalysis, Path]:
    """Run `pipeline` `n` times, persist a stub analysis, and return both.

    `findings` on the returned analysis is empty for now — real analysis is
    wired in a later commit. The artifact is fully schema-valid and round-trips
    through `varix.surface.load`.

    Raises `FileNotFoundError`, `ImportError`, `AttributeError`, or `TypeError`
    when the pipeline target cannot be resolved. `BudgetExceeded` is caught:
    the partial runs that completed are written to disk and the function
    returns normally.
    """
    actual_clock = clock if clock is not None else SystemClock()
    actual_rng = rng if rng is not None else SystemRng()

    adapter = load_adapter(pipeline)
    cost = CostAccumulator()
    started = actual_clock.now()

    try:
        runs = asyncio.run(run_n(adapter, input_text, n, cost=cost, max_cost=max_cost))
    except BudgetExceeded as exc:
        runs = list(exc.partial_runs)

    finished = actual_clock.now()

    metric = ExactMatch()
    result = analyze(runs, adapter.capabilities(), metric=metric)

    analysis = PipelineAnalysis(
        analysis_id=actual_rng.new_id(),
        pipeline_name=pipeline,
        n=len(runs),
        metric_name=metric.name(),
        schema_version=SCHEMA_VERSION,
        runs=tuple(runs),
        findings=result.findings,
        started_at=started,
        finished_at=finished,
        total_cost=cost.snapshot(),
        step_replays={},
    )

    resolved_dir = resolve_runs_dir(base_dir)
    path = save(analysis, base_dir=resolved_dir)
    return analysis, path


def resolve_runs_dir(explicit: Path | None) -> Path | None:
    """Resolve the artifact directory: explicit > env var > storage default."""
    if explicit is not None:
        return explicit
    env = os.environ.get(_RUNS_DIR_ENV)
    return Path(env).expanduser() if env else None


def execute_show(target: str, *, base_dir: Path | None = None) -> str:
    """Load an artifact (by ID or path) and return the rendered analysis text."""
    analysis = _load_target(target, base_dir)
    return render_analysis(analysis)


def execute_explain(
    step_id: str,
    analysis_target: str | None = None,
    *,
    base_dir: Path | None = None,
) -> str:
    """Load an artifact and render the evidence trail for `step_id`.

    When `analysis_target` is None, the most recently modified artifact is used.
    Raises `FileNotFoundError` if no artifact can be located, or `ValueError`
    if `step_id` is not present in the chosen artifact.
    """
    if analysis_target is not None:
        analysis = _load_target(analysis_target, base_dir)
    else:
        resolved_dir = resolve_runs_dir(base_dir)
        latest = latest_analysis(resolved_dir)
        if latest is None:
            raise FileNotFoundError("no saved analyses found")
        analysis = load_path(latest)

    step_ids = {sr.step_id for run in analysis.runs for sr in run.step_runs}
    if step_id not in step_ids:
        raise ValueError(f"step {step_id!r} not found in analysis {analysis.analysis_id!r}")
    return render_explain(analysis, step_id)


def _load_target(target: str, base_dir: Path | None) -> PipelineAnalysis:
    """Load an artifact by ID or by path, depending on the target's shape."""
    if target.endswith(".json") or "/" in target or os.sep in target:
        return load_path(Path(target).expanduser())
    resolved_dir = resolve_runs_dir(base_dir)
    return load(target, base_dir=resolved_dir)
