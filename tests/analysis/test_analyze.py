"""Tests for the top-level analyze() entry point."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from varix.adapters import FakeAdapter
from varix.analysis import AnalysisResult, analyze, detect_structural_mismatch
from varix.core import (
    AdapterCapabilities,
    Classification,
    Confidence,
    LocalizationOutcome,
    PipelineRun,
    StepRun,
    StructuralMismatch,
)
from varix.execution import run_n

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def _run(*step_ids: str) -> PipelineRun:
    return PipelineRun(
        run_id="r",
        step_runs=tuple(StepRun(step_id=sid, inputs="i", output="o") for sid in step_ids),
        started_at=_T,
        finished_at=_T,
    )


def test_analyze_emits_exclusion_note_when_classifiers_drop_observations() -> None:
    """When provider_side excludes a run for missing fingerprint, the analysis-level
    `notes` carries a one-line summary so it surfaces in run/show output."""
    from varix.analysis import analyze

    def _r(rid: str, fp: str | None) -> PipelineRun:
        metadata = {"system_fingerprint": fp} if fp is not None else None
        return PipelineRun(
            run_id=rid,
            step_runs=(StepRun(step_id="s1", inputs="i", output="o", provider_metadata=metadata),),
            started_at=_T,
            finished_at=_T,
        )

    runs = (_r("r1", "fp_a"), _r("r2", None), _r("r3", "fp_b"))
    result = analyze(runs, AdapterCapabilities(exposes_fingerprint=True))
    assert any("excluded" in note for note in result.notes)
    assert any("varix explain" in note for note in result.notes)
    # ASCII glyphs only — em-dashes belong in docstrings, not user-facing strings.
    assert not any("—" in note for note in result.notes)


def test_analyze_has_empty_notes_when_no_exclusions() -> None:
    from varix.analysis import analyze

    def _r(rid: str, fp: str) -> PipelineRun:
        return PipelineRun(
            run_id=rid,
            step_runs=(
                StepRun(
                    step_id="s1", inputs="i", output="o",
                    provider_metadata={"system_fingerprint": fp},
                ),
            ),
            started_at=_T,
            finished_at=_T,
        )

    runs = (_r("r1", "fp_a"), _r("r2", "fp_b"))
    result = analyze(runs, AdapterCapabilities(exposes_fingerprint=True))
    assert result.notes == ()


@pytest.mark.asyncio
async def test_deterministic_pipeline_produces_no_findings() -> None:
    adapter = FakeAdapter()
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    assert isinstance(result, AnalysisResult)
    assert result.findings == ()
    assert all(o is LocalizationOutcome.DETERMINISTIC for o in result.outcomes.values())


@pytest.mark.asyncio
async def test_provider_side_variance_produces_provider_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROVIDER_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    s2_findings = [f for f in result.findings if f.step_id == "s2"]
    assert len(s2_findings) == 1
    assert s2_findings[0].classification is Classification.PROVIDER_SIDE
    assert s2_findings[0].confidence is Confidence.HIGH


@pytest.mark.asyncio
async def test_prompt_side_variance_produces_residual_finding() -> None:
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    s2_findings = [f for f in result.findings if f.step_id == "s2"]
    assert len(s2_findings) == 1
    assert s2_findings[0].classification is Classification.PROMPT_SIDE
    assert s2_findings[0].confidence is Confidence.MEDIUM
    assert result.outcomes["s2"] is LocalizationOutcome.SOURCE
    for sid in ("s3", "s4", "s5"):
        assert result.outcomes[sid] is LocalizationOutcome.DOWNSTREAM


@pytest.mark.asyncio
async def test_time_or_state_variance_produces_low_finding_only() -> None:
    adapter = FakeAdapter(variance={"s5": Classification.TIME_OR_STATE})
    runs = await run_n(adapter, "hello", n=3)
    result = analyze(runs, adapter.capabilities())
    s5_findings = [f for f in result.findings if f.step_id == "s5"]
    assert len(s5_findings) == 1
    assert s5_findings[0].classification is Classification.TIME_OR_STATE
    assert s5_findings[0].confidence is Confidence.LOW


@pytest.mark.asyncio
async def test_empty_runs_returns_empty_result() -> None:
    adapter = FakeAdapter()
    result = analyze([], adapter.capabilities())
    assert result.findings == ()
    assert result.outcomes == {}


# --- structural mismatch detection ----------------------------------------


def test_detect_structural_mismatch_no_op_for_empty_or_single_run() -> None:
    detect_structural_mismatch([])
    detect_structural_mismatch([_run("s1", "s2")])  # no AssertionError


def test_detect_structural_mismatch_passes_consistent_runs() -> None:
    detect_structural_mismatch([_run("s1", "s2"), _run("s1", "s2"), _run("s1", "s2")])


def test_detect_structural_mismatch_raises_on_different_lengths() -> None:
    with pytest.raises(StructuralMismatch, match="varied across runs"):
        detect_structural_mismatch([_run("s1", "s2", "s3"), _run("s1", "s2")])


def test_detect_structural_mismatch_raises_on_different_step_ids_same_position() -> None:
    with pytest.raises(StructuralMismatch):
        detect_structural_mismatch([_run("s1", "s2"), _run("s1", "sX")])


def test_detect_structural_mismatch_finds_third_run_inconsistent() -> None:
    with pytest.raises(StructuralMismatch, match="run 2"):
        detect_structural_mismatch([_run("s1", "s2"), _run("s1", "s2"), _run("s1")])


def test_analyze_propagates_structural_mismatch() -> None:
    runs = [_run("s1", "s2", "s3"), _run("s1", "s2")]
    with pytest.raises(StructuralMismatch):
        analyze(runs, AdapterCapabilities())


@pytest.mark.asyncio
async def test_fake_adapter_structure_variance_triggers_refusal() -> None:
    adapter = FakeAdapter(structure_variance=True)
    runs = await run_n(adapter, "hello", n=3)
    # Run 1: 5 steps, run 2: 3 steps, run 3: 5 steps → analyze refuses.
    with pytest.raises(StructuralMismatch):
        analyze(runs, adapter.capabilities())
