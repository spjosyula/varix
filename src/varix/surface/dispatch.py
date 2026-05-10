"""Command dispatch — orchestrates the work behind each CLI command.

Lives between the Typer command handlers (which deal with argument parsing
and user output) and the lower layers (execution, analysis, storage). Tests
target these functions directly to avoid going through the CLI runner.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from varix.analysis import ImpactEstimator, analyze
from varix.core import (
    SCHEMA_VERSION,
    BudgetExceeded,
    Clock,
    ExactMatch,
    PipelineAnalysis,
    Rng,
    RunFailed,
    SystemClock,
    SystemRng,
)
from varix.execution import CostAccumulator, run_n
from varix.surface.loader import load_adapter
from varix.surface.reporter import render_analysis, render_explain, render_impact
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
    """Run `pipeline` `n` times, analyze, persist, and return both.

    Raises `FileNotFoundError`, `ImportError`, `AttributeError`, or `TypeError`
    when the pipeline target cannot be resolved. `BudgetExceeded` and
    `RunFailed` are caught: the partial runs that completed are saved with
    a note describing the truncation, and the function returns normally.
    """
    actual_clock = clock if clock is not None else SystemClock()
    actual_rng = rng if rng is not None else SystemRng()

    adapter = load_adapter(pipeline)
    cost = CostAccumulator()
    started = actual_clock.now()

    notes: list[str] = []
    try:
        runs = asyncio.run(run_n(adapter, input_text, n, cost=cost, max_cost=max_cost))
    except BudgetExceeded as exc:
        runs = list(exc.partial_runs)
        notes.append(str(exc))
    except RunFailed as exc:
        runs = list(exc.partial_runs)
        notes.append(str(exc))

    if len(runs) < 2:
        notes.append(
            f"analysis is inconclusive: {len(runs)} run(s) completed; "
            "variance analysis requires at least 2 runs to compare"
        )

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
        notes=tuple(notes),
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


def execute_show(
    target: str,
    *,
    base_dir: Path | None = None,
    clock: Clock | None = None,
) -> str:
    """Load an artifact (by ID or path) and return the rendered analysis text.

    The receipt grows a 'ran <relative-time>' suffix derived from `clock.now()`
    versus the artifact's `finished_at`, so the engineer has temporal context
    when re-reading a past run.
    """
    analysis = _load_target(target, base_dir)
    actual_clock = clock if clock is not None else SystemClock()
    return render_analysis(analysis, now=actual_clock.now())


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


def execute_impact(
    step_id: str,
    analysis_target: str | None = None,
    *,
    base_dir: Path | None = None,
) -> str:
    """Load an artifact and render the impact report for `step_id`.

    Mirrors `execute_explain`: defaults to the latest artifact, raises
    `FileNotFoundError` when none exist, and `ValueError` when `step_id`
    is not present in the chosen artifact.
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

    report = ImpactEstimator().estimate(analysis.runs, step_id)
    return render_impact(analysis, report)


def _load_target(target: str, base_dir: Path | None) -> PipelineAnalysis:
    """Load an artifact by ID or by path, depending on the target's shape."""
    if target.endswith(".json") or "/" in target or os.sep in target:
        return load_path(Path(target).expanduser())
    resolved_dir = resolve_runs_dir(base_dir)
    return load(target, base_dir=resolved_dir)
