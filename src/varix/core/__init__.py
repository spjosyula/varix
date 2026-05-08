"""varix.core — domain types, errors, and contracts.

This package has no I/O, no network, and no framework imports. Everything
varix depends on internally is built on top of these types.
"""

from varix.core.adapter import Adapter, unavailable_finding
from varix.core.clock import Clock, FrozenClock, SystemClock
from varix.core.errors import (
    AdapterError,
    BudgetExceeded,
    CapabilityMissing,
    RefusalRequired,
    StructuralMismatch,
    VarixError,
)
from varix.core.rng import Rng, SequenceRng, SystemRng
from varix.core.types import (
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
from varix.core.variance import ExactMatch, VarianceMetric

__all__ = [
    "SCHEMA_VERSION",
    "Adapter",
    "AdapterCapabilities",
    "AdapterError",
    "BudgetExceeded",
    "CapabilityMissing",
    "Classification",
    "Clock",
    "Confidence",
    "CostSnapshot",
    "Evidence",
    "ExactMatch",
    "Finding",
    "FrozenClock",
    "LocalizationOutcome",
    "PipelineAnalysis",
    "PipelineRun",
    "RefusalRequired",
    "Rng",
    "SequenceRng",
    "Step",
    "StepGraph",
    "StepRun",
    "StructuralMismatch",
    "SystemClock",
    "SystemRng",
    "ToolCall",
    "VarianceMetric",
    "VarixError",
    "unavailable_finding",
]
