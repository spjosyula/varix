"""Tests for varix.surface.storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from varix.core import (
    SCHEMA_VERSION,
    Classification,
    Confidence,
    CostSnapshot,
    Evidence,
    Finding,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    RefusalRequired,
    StepRun,
    ToolCall,
)
from varix.surface import (
    default_runs_dir,
    list_analyses,
    load,
    load_path,
    save,
)

_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)


def _make_analysis(analysis_id: str = "abc-123") -> PipelineAnalysis:
    step_run = StepRun(
        step_id="s1",
        inputs="hello",
        output="hello_out",
        tool_calls=(ToolCall(name="lookup", arguments={"q": "hello"}, result="hit"),),
        provider_metadata={"system_fingerprint": "fp_a"},
        cost=CostSnapshot(input_tokens=10, output_tokens=5, dollars=0.001),
    )
    pipeline_run = PipelineRun(
        run_id="r1",
        step_runs=(step_run,),
        started_at=_T,
        finished_at=_T,
        cost=CostSnapshot(input_tokens=10, output_tokens=5, dollars=0.001),
    )
    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.HIGH,
        metric_name="exact",
        classification=Classification.PROVIDER_SIDE,
        evidence=(Evidence(kind="fingerprint_diff", description="fp varied"),),
        reason="fingerprints differ",
    )
    return PipelineAnalysis(
        analysis_id=analysis_id,
        pipeline_name="fake_pipeline",
        n=3,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(pipeline_run,),
        findings=(finding,),
        started_at=_T,
        finished_at=_T,
        total_cost=CostSnapshot(input_tokens=30, output_tokens=15, dollars=0.003),
        step_replays={"s1": (step_run,)},
    )


def test_save_writes_json_file_named_by_analysis_id(tmp_path: Path) -> None:
    analysis = _make_analysis("abc-123")
    written = save(analysis, base_dir=tmp_path)
    assert written == tmp_path / "abc-123.json"
    assert written.exists()


def test_save_creates_base_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "runs"
    save(_make_analysis(), base_dir=nested)
    assert nested.is_dir()


def test_save_overwrites_existing_file(tmp_path: Path) -> None:
    save(_make_analysis(), base_dir=tmp_path)
    save(_make_analysis(), base_dir=tmp_path)  # second save with same ID
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_save_then_load_round_trip_preserves_every_field(tmp_path: Path) -> None:
    original = _make_analysis()
    save(original, base_dir=tmp_path)
    loaded = load("abc-123", base_dir=tmp_path)
    assert loaded == original


def test_load_path_with_explicit_path(tmp_path: Path) -> None:
    original = _make_analysis()
    written = save(original, base_dir=tmp_path)
    loaded = load_path(written)
    assert loaded == original


def test_round_trip_with_unavailable_finding_preserves_classification_none(
    tmp_path: Path,
) -> None:
    base = _make_analysis()
    findings = (
        Finding(
            step_id="s1",
            localization=LocalizationOutcome.SOURCE,
            confidence=Confidence.UNAVAILABLE,
            metric_name="exact",
            classification=None,  # the UNAVAILABLE-no-category code path
            reason="adapter does not expose system_fingerprint",
        ),
    )
    analysis = PipelineAnalysis(
        analysis_id=base.analysis_id,
        pipeline_name=base.pipeline_name,
        n=base.n,
        metric_name=base.metric_name,
        schema_version=base.schema_version,
        runs=base.runs,
        findings=findings,
        started_at=base.started_at,
        finished_at=base.finished_at,
        total_cost=base.total_cost,
        step_replays=base.step_replays,
    )
    save(analysis, base_dir=tmp_path)
    loaded = load(analysis.analysis_id, base_dir=tmp_path)
    assert loaded.findings[0].classification is None
    assert loaded.findings[0].confidence is Confidence.UNAVAILABLE


def test_save_uses_atomic_rename_no_lingering_tmp(tmp_path: Path) -> None:
    save(_make_analysis(), base_dir=tmp_path)
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert tmp_files == []


def test_save_writes_pretty_sorted_json(tmp_path: Path) -> None:
    analysis = _make_analysis()
    written = save(analysis, base_dir=tmp_path)
    text = written.read_text(encoding="utf-8")
    # Pretty-printed (indent=2) → at least one newline + leading spaces.
    assert "\n  " in text
    # sort_keys=True → analysis_id comes alphabetically before schema_version.
    assert text.find('"analysis_id"') < text.find('"schema_version"')


def test_list_analyses_returns_sorted_json_files_only(tmp_path: Path) -> None:
    save(_make_analysis("zeta"), base_dir=tmp_path)
    save(_make_analysis("alpha"), base_dir=tmp_path)
    # Drop a stray non-json and a fake .tmp file in the dir.
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / ".half.json.tmp").write_text("partial", encoding="utf-8")

    paths = list_analyses(base_dir=tmp_path)
    names = [p.name for p in paths]
    assert names == ["alpha.json", "zeta.json"]


def test_list_analyses_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    assert list_analyses(base_dir=tmp_path / "does-not-exist") == []


def test_load_refuses_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    data = _make_analysis().to_dict()
    data["schema_version"] = "9.99"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RefusalRequired, match=r"9\.99"):
        load_path(path)


def test_load_refuses_missing_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    data = _make_analysis().to_dict()
    del data["schema_version"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RefusalRequired, match="schema_version"):
        load_path(path)


def test_default_runs_dir_resolves_under_user_home() -> None:
    path = default_runs_dir()
    assert path.is_absolute()
    assert path.parts[-2:] == (".varix", "runs")


def test_round_trip_preserves_notes(tmp_path: Path) -> None:
    base = _make_analysis()
    analysis = PipelineAnalysis(
        analysis_id=base.analysis_id,
        pipeline_name=base.pipeline_name,
        n=base.n,
        metric_name=base.metric_name,
        schema_version=base.schema_version,
        runs=base.runs,
        findings=base.findings,
        started_at=base.started_at,
        finished_at=base.finished_at,
        total_cost=base.total_cost,
        step_replays=base.step_replays,
        notes=("run 3 of 5 failed: RuntimeError: timeout",),
    )
    save(analysis, base_dir=tmp_path)
    loaded = load(analysis.analysis_id, base_dir=tmp_path)
    assert loaded.notes == ("run 3 of 5 failed: RuntimeError: timeout",)


def test_load_old_artifact_without_notes_field(tmp_path: Path) -> None:
    """Artifacts written before the notes field existed must still load."""
    path = tmp_path / "old.json"
    data = _make_analysis().to_dict()
    del data["notes"]  # simulate a pre-notes artifact
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_path(path)
    assert loaded.notes == ()
