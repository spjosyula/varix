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

    # N=3 so the n<3 confidence cap doesn't add a note of its own.
    runs = (_r("r1", "fp_a"), _r("r2", "fp_b"), _r("r3", "fp_c"))
    result = analyze(runs, AdapterCapabilities(exposes_fingerprint=True))
    assert result.notes == ()


def test_analyze_caps_high_confidence_when_only_two_runs() -> None:
    """N=2 has weak statistical signal; HIGH findings get downgraded to MEDIUM
    and a single note explains the cap."""

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
    assert len(result.findings) == 1
    assert result.findings[0].confidence is Confidence.MEDIUM
    assert any("only 2 run(s)" in note for note in result.notes)


def test_cap_exempts_findings_with_replay_disambiguation_evidence() -> None:
    """HIGH stays HIGH when replay-disambiguation evidence is present, even at N=2."""
    from varix.analysis.orchestration import _cap_confidence_for_weak_evidence
    from varix.core import Evidence, Finding

    plain_high = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.HIGH,
        metric_name="exact",
    )
    with_replay_ev = Finding(
        step_id="s2",
        localization=LocalizationOutcome.DOWNSTREAM,
        confidence=Confidence.HIGH,
        metric_name="exact",
        evidence=(Evidence(kind="replay_disambiguation", description="3 replays varied"),),
    )

    capped, notes = _cap_confidence_for_weak_evidence((plain_high, with_replay_ev), n_runs=2)
    assert capped[0].confidence is Confidence.MEDIUM
    assert capped[1].confidence is Confidence.HIGH
    assert len(notes) == 1


def test_cap_emits_no_note_when_nothing_was_capped() -> None:
    """N<3 with no HIGH findings to cap stays quiet."""
    from varix.analysis.orchestration import _cap_confidence_for_weak_evidence
    from varix.core import Finding

    medium = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.MEDIUM,
        metric_name="exact",
    )
    capped, notes = _cap_confidence_for_weak_evidence((medium,), n_runs=2)
    assert capped[0].confidence is Confidence.MEDIUM
    assert notes == ()


def test_analyze_does_not_cap_when_three_or_more_runs() -> None:
    """N>=3 keeps HIGH findings untouched and emits no cap note."""

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

    runs = (_r("r1", "fp_a"), _r("r2", "fp_b"), _r("r3", "fp_c"))
    result = analyze(runs, AdapterCapabilities(exposes_fingerprint=True))
    assert result.findings[0].confidence is Confidence.HIGH
    assert not any("statistical signal" in note for note in result.notes)


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


# --- replay disambiguation evidence -------------------------------------------


@pytest.mark.asyncio
async def test_analyze_attaches_replay_disambiguation_evidence() -> None:
    """When DOWNSTREAM step's replays show independent variance, findings on
    that step get a replay_disambiguation evidence record."""
    # PROMPT_SIDE on s2 → s2 is SOURCE, s3+ are DOWNSTREAM with varying output.
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    # Synthesize replays for s3 with deliberately varying outputs to simulate
    # what gather_disambiguation_replays would produce on a step with its own
    # independent variance.
    s3_replays = (
        StepRun(step_id="s3", inputs="fixed", output="r1"),
        StepRun(step_id="s3", inputs="fixed", output="r2"),
        StepRun(step_id="s3", inputs="fixed", output="r3"),
    )
    result = analyze(runs, adapter.capabilities(), replays_by_step={"s3": s3_replays})

    s3_findings = [f for f in result.findings if f.step_id == "s3"]
    assert s3_findings, "expected at least one finding on s3"
    for f in s3_findings:
        kinds = [ev.kind for ev in f.evidence]
        assert "replay_disambiguation" in kinds


@pytest.mark.asyncio
async def test_analyze_does_not_attach_evidence_when_replays_are_stable() -> None:
    """If replays produce identical outputs, the step is NOT independently
    a source — no disambiguation evidence."""
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    # All replays identical — no independent variance.
    s3_replays = (
        StepRun(step_id="s3", inputs="fixed", output="same"),
        StepRun(step_id="s3", inputs="fixed", output="same"),
        StepRun(step_id="s3", inputs="fixed", output="same"),
    )
    result = analyze(runs, adapter.capabilities(), replays_by_step={"s3": s3_replays})
    for f in result.findings:
        if f.step_id == "s3":
            for ev in f.evidence:
                assert ev.kind != "replay_disambiguation"


@pytest.mark.asyncio
async def test_analyze_does_not_attach_evidence_on_source_step() -> None:
    """SOURCE steps are already correctly classified — they don't need
    disambiguation even if replays were provided (which shouldn't happen
    in practice, but the analyze logic must not over-apply)."""
    adapter = FakeAdapter(variance={"s2": Classification.PROMPT_SIDE})
    runs = await run_n(adapter, "hi", n=3)
    # Pretend replays were gathered for s2 (the SOURCE) anyway.
    s2_replays = (
        StepRun(step_id="s2", inputs="fixed", output="v1"),
        StepRun(step_id="s2", inputs="fixed", output="v2"),
    )
    result = analyze(runs, adapter.capabilities(), replays_by_step={"s2": s2_replays})
    for f in result.findings:
        if f.step_id == "s2":
            kinds = [ev.kind for ev in f.evidence]
            assert "replay_disambiguation" not in kinds
