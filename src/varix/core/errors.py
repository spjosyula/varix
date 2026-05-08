"""varix error hierarchy.

Every error varix raises is a subclass of `VarixError`.
"""

from __future__ import annotations


class VarixError(Exception):
    """Base class for every error varix raises."""


class AdapterError(VarixError):
    """The adapter raised an error or returned a malformed result."""


class CapabilityMissing(VarixError):
    """An operation required an adapter capability that was not declared."""


class BudgetExceeded(VarixError):
    """The cost budget (`--max-cost`) was reached."""


class StructuralMismatch(VarixError):
    """Pipeline runs produced different step graphs and cannot be aligned."""


class RefusalRequired(VarixError):
    """varix declined to produce a finding rather than guess."""
