"""varix.core — domain types, errors, and contracts.

This package has no I/O, no network, and no framework imports. Everything
varix depends on internally is built on top of these types.
"""

from varix.core.errors import (
    AdapterError,
    BudgetExceeded,
    CapabilityMissing,
    RefusalRequired,
    StructuralMismatch,
    VarixError,
)
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

__all__ = [
    "SCHEMA_VERSION",
    "AdapterCapabilities",
    "AdapterError",
    "BudgetExceeded",
    "CapabilityMissing",
    "Classification",
    "Confidence",
    "CostSnapshot",
    "Evidence",
    "Finding",
    "LocalizationOutcome",
    "PipelineAnalysis",
    "PipelineRun",
    "RefusalRequired",
    "Step",
    "StepGraph",
    "StepRun",
    "StructuralMismatch",
    "ToolCall",
    "VarixError",
]
