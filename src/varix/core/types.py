"""Core domain types for varix.

All types are frozen dataclasses with JSON round-trip support via
`to_dict()` / `from_dict()`. They have no I/O, no framework imports, and
define the data shape every other layer of varix consumes.

The schema version is bumped on breaking changes only; additive changes
keep the version stable so old artifacts remain readable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SCHEMA_VERSION = "1.0"


class Confidence(enum.Enum):
    """How much weight to give a finding.

    `UNAVAILABLE` is a real outcome, not an error: it means the analysis
    could not run honestly (missing capability, missing metadata, etc.) and
    the report says so explicitly rather than guessing.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class LocalizationOutcome(enum.Enum):
    """The localizer's structural verdict for a step.

    Distinct from `Classification` (the cause). A step that varies only
    because its inputs varied is `DOWNSTREAM`, not a `SOURCE`.
    """

    DETERMINISTIC = "deterministic"
    SOURCE = "source"
    DOWNSTREAM = "downstream"


class Classification(enum.Enum):
    """The category of nondeterminism for a source step."""

    PROVIDER_SIDE = "provider_side"
    TOOL_SIDE = "tool_side"
    ORDERING = "ordering"
    PROMPT_SIDE = "prompt_side"
    TIME_OR_STATE = "time_or_state"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Token and dollar accounting at a point in time.

    Snapshots are immutable values in core. The mutable accumulator that
    produces them lives in the execution layer.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    dollars: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "dollars": self.dollars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostSnapshot:
        return cls(
            input_tokens=int(data["input_tokens"]),
            output_tokens=int(data["output_tokens"]),
            dollars=float(data["dollars"]),
        )

    def __add__(self, other: CostSnapshot) -> CostSnapshot:
        return CostSnapshot(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            dollars=self.dollars + other.dollars,
        )


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """What an adapter can honestly provide.

    Each flag drives a specific analysis path. When a flag is False, the
    matching classifier emits `Confidence.UNAVAILABLE` rather than guessing.
    Resist adding new flags until a classifier needs one.
    """

    exposes_fingerprint: bool = False
    exposes_tool_calls: bool = False
    supports_replay: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposes_fingerprint": self.exposes_fingerprint,
            "exposes_tool_calls": self.exposes_tool_calls,
            "supports_replay": self.supports_replay,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterCapabilities:
        return cls(
            exposes_fingerprint=bool(data["exposes_fingerprint"]),
            exposes_tool_calls=bool(data["exposes_tool_calls"]),
            supports_replay=bool(data["supports_replay"]),
        )


@dataclass(frozen=True, slots=True)
class Step:
    """A named position in the pipeline structure."""

    id: str
    name: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "index": self.index}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        return cls(id=str(data["id"]), name=str(data["name"]), index=int(data["index"]))


@dataclass(frozen=True, slots=True)
class StepGraph:
    """The ordered set of steps an adapter declares for a given input.

    v1 requires this to be stable across N runs; the runner uses it for
    structural-mismatch refusal.
    """

    steps: tuple[Step, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepGraph:
        return cls(steps=tuple(Step.from_dict(s) for s in data["steps"]))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation observed inside a step run."""

    name: str
    arguments: dict[str, Any]
    result: Any

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "result": self.result}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=str(data["name"]),
            arguments=dict(data["arguments"]),
            result=data["result"],
        )


@dataclass(frozen=True, slots=True)
class StepRun:
    """One execution of a single step, captured for analysis.

    `inputs` and `output` are required to be JSON-serializable. varix does
    not validate this; serialization will fail at storage time if violated.
    """

    step_id: str
    inputs: Any
    output: Any
    tool_calls: tuple[ToolCall, ...] = ()
    provider_metadata: dict[str, Any] | None = None
    cost: CostSnapshot = field(default_factory=CostSnapshot)
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "inputs": self.inputs,
            "output": self.output,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "provider_metadata": self.provider_metadata,
            "cost": self.cost.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRun:
        return cls(
            step_id=str(data["step_id"]),
            inputs=data["inputs"],
            output=data["output"],
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in data["tool_calls"]),
            provider_metadata=data.get("provider_metadata"),
            cost=CostSnapshot.from_dict(data["cost"]),
            seed=data.get("seed"),
        )


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """One end-to-end execution of the pipeline (one of N)."""

    run_id: str
    step_runs: tuple[StepRun, ...]
    started_at: datetime
    finished_at: datetime
    cost: CostSnapshot = field(default_factory=CostSnapshot)
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_runs": [sr.to_dict() for sr in self.step_runs],
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "cost": self.cost.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineRun:
        return cls(
            run_id=str(data["run_id"]),
            step_runs=tuple(StepRun.from_dict(sr) for sr in data["step_runs"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            cost=CostSnapshot.from_dict(data["cost"]),
            seed=data.get("seed"),
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    """A piece of evidence backing a finding.

    `kind` is a stable machine-readable tag (e.g. "fingerprint_diff").
    `description` is human-readable for the report. `data` is the raw
    payload reviewers can inspect via `varix explain`.
    """

    kind: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "description": self.description, "data": self.data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            kind=str(data["kind"]),
            description=str(data["description"]),
            data=dict(data.get("data", {})),
        )


@dataclass(frozen=True, slots=True)
class Finding:
    """The analysis verdict for a single step.

    A `Finding` carries enough provenance (`evidence`) for the reporter to
    render `varix explain` without re-querying analysis internals.
    """

    step_id: str
    localization: LocalizationOutcome
    confidence: Confidence
    metric_name: str
    classification: Classification | None = None
    evidence: tuple[Evidence, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "localization": self.localization.value,
            "confidence": self.confidence.value,
            "metric_name": self.metric_name,
            "classification": self.classification.value if self.classification else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        cls_value = data.get("classification")
        return cls(
            step_id=str(data["step_id"]),
            localization=LocalizationOutcome(data["localization"]),
            confidence=Confidence(data["confidence"]),
            metric_name=str(data["metric_name"]),
            classification=Classification(cls_value) if cls_value else None,
            evidence=tuple(Evidence.from_dict(e) for e in data["evidence"]),
            reason=data.get("reason"),
        )


@dataclass(frozen=True, slots=True)
class PipelineAnalysis:
    """The top-level artifact varix writes to disk.

    One `PipelineAnalysis` corresponds to one invocation of `varix run`.
    The `schema_version` field gates compatibility: readers may refuse a
    `PipelineAnalysis` whose schema is newer than they understand.
    """

    analysis_id: str
    pipeline_name: str
    n: int
    metric_name: str
    schema_version: str
    runs: tuple[PipelineRun, ...]
    findings: tuple[Finding, ...]
    started_at: datetime
    finished_at: datetime
    total_cost: CostSnapshot = field(default_factory=CostSnapshot)
    step_replays: dict[str, tuple[StepRun, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "pipeline_name": self.pipeline_name,
            "n": self.n,
            "metric_name": self.metric_name,
            "schema_version": self.schema_version,
            "runs": [r.to_dict() for r in self.runs],
            "findings": [f.to_dict() for f in self.findings],
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "total_cost": self.total_cost.to_dict(),
            "step_replays": {
                step_id: [sr.to_dict() for sr in replays]
                for step_id, replays in self.step_replays.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineAnalysis:
        return cls(
            analysis_id=str(data["analysis_id"]),
            pipeline_name=str(data["pipeline_name"]),
            n=int(data["n"]),
            metric_name=str(data["metric_name"]),
            schema_version=str(data["schema_version"]),
            runs=tuple(PipelineRun.from_dict(r) for r in data["runs"]),
            findings=tuple(Finding.from_dict(f) for f in data["findings"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            total_cost=CostSnapshot.from_dict(data["total_cost"]),
            step_replays={
                step_id: tuple(StepRun.from_dict(sr) for sr in replays)
                for step_id, replays in data.get("step_replays", {}).items()
            },
        )
