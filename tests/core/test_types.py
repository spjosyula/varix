"""Round-trip and basic-shape tests for varix.core types.

Property-based tests use hypothesis to ensure every domain type survives
to_dict / from_dict round-trip across a wide range of valid inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from varix.core import (
    SCHEMA_VERSION,
    AdapterCapabilities,
    Classification,
    Confidence,
    CostSnapshot,
    Evidence,
    Finding,
    LocalizationOutcome,
    PipelineAnalysis,
    PipelineRun,
    Step,
    StepGraph,
    StepRun,
    ToolCall,
)

# JSON-native value strategy. Bounded depth so generation is fast and
# deterministic; no NaN/Infinity since they don't round-trip through JSON.
_json_primitive: st.SearchStrategy[Any] = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)
_json_value: st.SearchStrategy[Any] = st.recursive(
    _json_primitive,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=4),
    ),
    max_leaves=8,
)

_short_text = st.text(min_size=1, max_size=20)


@st.composite
def _step(draw: st.DrawFn) -> Step:
    return Step(
        id=draw(_short_text),
        name=draw(_short_text),
        index=draw(st.integers(min_value=0, max_value=100)),
    )


@st.composite
def _cost(draw: st.DrawFn) -> CostSnapshot:
    return CostSnapshot(
        input_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        output_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        dollars=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
    )


@st.composite
def _capabilities(draw: st.DrawFn) -> AdapterCapabilities:
    return AdapterCapabilities(
        exposes_fingerprint=draw(st.booleans()),
        exposes_tool_calls=draw(st.booleans()),
        supports_replay=draw(st.booleans()),
    )


@st.composite
def _tool_call(draw: st.DrawFn) -> ToolCall:
    return ToolCall(
        name=draw(_short_text),
        arguments=draw(st.dictionaries(_short_text, _json_value, max_size=4)),
        result=draw(_json_value),
    )


@st.composite
def _step_run(draw: st.DrawFn) -> StepRun:
    return StepRun(
        step_id=draw(_short_text),
        inputs=draw(_json_value),
        output=draw(_json_value),
        tool_calls=tuple(draw(st.lists(_tool_call(), max_size=3))),
        provider_metadata=draw(
            st.one_of(st.none(), st.dictionaries(_short_text, _json_value, max_size=4))
        ),
        cost=draw(_cost()),
        seed=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10_000))),
    )


@st.composite
def _evidence(draw: st.DrawFn) -> Evidence:
    return Evidence(
        kind=draw(_short_text),
        description=draw(st.text(max_size=80)),
        data=draw(st.dictionaries(_short_text, _json_value, max_size=4)),
    )


@st.composite
def _finding(draw: st.DrawFn) -> Finding:
    return Finding(
        step_id=draw(_short_text),
        localization=draw(st.sampled_from(list(LocalizationOutcome))),
        confidence=draw(st.sampled_from(list(Confidence))),
        metric_name=draw(_short_text),
        classification=draw(st.one_of(st.none(), st.sampled_from(list(Classification)))),
        evidence=tuple(draw(st.lists(_evidence(), max_size=3))),
        reason=draw(st.one_of(st.none(), st.text(max_size=80))),
    )


def test_schema_version_is_string() -> None:
    assert isinstance(SCHEMA_VERSION, str)
    assert SCHEMA_VERSION


def test_confidence_unavailable_is_first_class() -> None:
    assert Confidence.UNAVAILABLE.value == "unavailable"
    assert Confidence("unavailable") is Confidence.UNAVAILABLE


def test_capabilities_default_is_safe() -> None:
    caps = AdapterCapabilities()
    assert not caps.exposes_fingerprint
    assert not caps.exposes_tool_calls
    assert not caps.supports_replay


def test_cost_snapshot_addition() -> None:
    a = CostSnapshot(input_tokens=10, output_tokens=20, dollars=0.5)
    b = CostSnapshot(input_tokens=5, output_tokens=15, dollars=0.25)
    total = a + b
    assert total.input_tokens == 15
    assert total.output_tokens == 35
    assert total.dollars == 0.75


@given(_cost())
def test_cost_snapshot_roundtrip(cost: CostSnapshot) -> None:
    assert CostSnapshot.from_dict(cost.to_dict()) == cost


@given(_capabilities())
def test_capabilities_roundtrip(caps: AdapterCapabilities) -> None:
    assert AdapterCapabilities.from_dict(caps.to_dict()) == caps


@given(_step())
def test_step_roundtrip(step: Step) -> None:
    assert Step.from_dict(step.to_dict()) == step


@given(st.lists(_step(), max_size=5))
def test_step_graph_roundtrip(steps: list[Step]) -> None:
    graph = StepGraph(steps=tuple(steps))
    assert StepGraph.from_dict(graph.to_dict()) == graph


@given(_tool_call())
def test_tool_call_roundtrip(tc: ToolCall) -> None:
    assert ToolCall.from_dict(tc.to_dict()) == tc


@given(_step_run())
def test_step_run_roundtrip(sr: StepRun) -> None:
    assert StepRun.from_dict(sr.to_dict()) == sr


@given(_evidence())
def test_evidence_roundtrip(e: Evidence) -> None:
    assert Evidence.from_dict(e.to_dict()) == e


@given(_finding())
def test_finding_roundtrip(f: Finding) -> None:
    assert Finding.from_dict(f.to_dict()) == f


def test_pipeline_run_roundtrip_concrete() -> None:
    sr = StepRun(step_id="s1", inputs={"q": "hi"}, output="hello")
    run = PipelineRun(
        run_id="run-1",
        step_runs=(sr,),
        started_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 8, 12, 1, 0, tzinfo=UTC),
    )
    assert PipelineRun.from_dict(run.to_dict()) == run


def test_pipeline_analysis_roundtrip_concrete() -> None:
    sr = StepRun(step_id="s1", inputs={"q": "hi"}, output="hello")
    run = PipelineRun(
        run_id="run-1",
        step_runs=(sr,),
        started_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 8, 12, 1, 0, tzinfo=UTC),
    )
    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        confidence=Confidence.HIGH,
        metric_name="exact",
    )
    analysis = PipelineAnalysis(
        analysis_id="a-1",
        pipeline_name="example.py",
        n=1,
        metric_name="exact",
        schema_version=SCHEMA_VERSION,
        runs=(run,),
        findings=(finding,),
        started_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 8, 12, 1, 0, tzinfo=UTC),
        step_replays={"s1": (sr,)},
    )
    assert PipelineAnalysis.from_dict(analysis.to_dict()) == analysis


def test_finding_classification_optional() -> None:
    f = Finding(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        confidence=Confidence.HIGH,
        metric_name="exact",
    )
    assert f.classification is None
    assert Finding.from_dict(f.to_dict()) == f


def test_finding_unavailable_carries_reason() -> None:
    f = Finding(
        step_id="s1",
        localization=LocalizationOutcome.SOURCE,
        confidence=Confidence.UNAVAILABLE,
        metric_name="exact",
        reason="adapter does not expose system_fingerprint",
    )
    round_tripped = Finding.from_dict(f.to_dict())
    assert round_tripped.confidence is Confidence.UNAVAILABLE
    assert round_tripped.reason == f.reason
