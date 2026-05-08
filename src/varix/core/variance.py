"""The VarianceMetric Protocol and the ExactMatch implementation.

Classifiers and the localizer use a `VarianceMetric` to decide whether two
outputs are equivalent. The metric's `name()` is recorded on each `Finding`
so reports can show which equivalence rule produced the verdict.
"""

from __future__ import annotations

import math
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
    """Equivalence by Python `==`, with NaN floats treated as equivalent.

    NaN-equality is the only deliberate deviation: IEEE 754 says `nan != nan`,
    which would mark every step containing a NaN as a variance source.
    """

    def name(self) -> str:
        return "exact"

    def equivalent(self, a: object, b: object) -> bool:
        return _equiv(a, b)


def _equiv(a: object, b: object) -> bool:
    if a == b:
        return True
    if isinstance(a, float) and isinstance(b, float):
        return math.isnan(a) and math.isnan(b)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_equiv(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_equiv(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(_equiv(x, y) for x, y in zip(a, b, strict=False))
    return False
