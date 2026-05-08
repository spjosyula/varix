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
from varix.surface.storage import save

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
