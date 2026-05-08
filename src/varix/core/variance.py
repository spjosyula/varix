"""The VarianceMetric Protocol and the ExactMatch implementation.

Classifiers and the localizer use a `VarianceMetric` to decide whether two
outputs are equivalent. The metric's `name()` is recorded on each `Finding`
so reports can show which equivalence rule produced the verdict.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VarianceMetric(Protocol):
    """Decides whether two outputs are equivalent for variance analysis."""

    def name(self) -> str:
        """Stable identifier; recorded on `Finding.metric_name`."""
        ...

    def equivalent(self, a: object, b: object) -> bool:
        """Return True if `a` and `b` are equivalent under this metric."""
        ...


class ExactMatch:
    """Equivalence by Python `==`. Strict: whitespace, casing, and ordering matter."""

    def name(self) -> str:
        return "exact"

    def equivalent(self, a: object, b: object) -> bool:
        return a == b
