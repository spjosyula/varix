"""Error hierarchy tests."""

from __future__ import annotations

import pytest

from varix.core import (
    AdapterError,
    BudgetExceeded,
    CapabilityMissing,
    RefusalRequired,
    StructuralMismatch,
    VarixError,
)


@pytest.mark.parametrize(
    "exc_class",
    [AdapterError, CapabilityMissing, BudgetExceeded, StructuralMismatch, RefusalRequired],
)
def test_subclass_of_varix_error(exc_class: type[VarixError]) -> None:
    assert issubclass(exc_class, VarixError)


def test_varix_error_caught_uniformly() -> None:
    for exc in (AdapterError, CapabilityMissing, BudgetExceeded, StructuralMismatch):
        with pytest.raises(VarixError):
            raise exc("boom")


def test_errors_carry_message() -> None:
    err = CapabilityMissing("supports_replay is False")
    assert "supports_replay" in str(err)
