"""Tests for the Rng seam."""

from __future__ import annotations

import pytest

from varix.core import Rng, SequenceRng, SystemRng


def test_system_rng_satisfies_rng_protocol() -> None:
    assert isinstance(SystemRng(), Rng)


def test_system_rng_returns_distinct_strings() -> None:
    rng = SystemRng()
    values = {rng.new_id() for _ in range(50)}
    assert len(values) == 50


def test_system_rng_returns_uuid4_string() -> None:
    value = SystemRng().new_id()
    # UUID4 string is 36 chars with 4 hyphens, e.g. xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    assert len(value) == 36
    assert value.count("-") == 4
    assert value[14] == "4"


def test_sequence_rng_emits_in_order() -> None:
    rng = SequenceRng(["a", "b", "c"])
    assert rng.new_id() == "a"
    assert rng.new_id() == "b"
    assert rng.new_id() == "c"


def test_sequence_rng_raises_when_exhausted() -> None:
    rng = SequenceRng(["only"])
    rng.new_id()
    with pytest.raises(RuntimeError, match="exhausted"):
        rng.new_id()


def test_sequence_rng_satisfies_rng_protocol() -> None:
    assert isinstance(SequenceRng(["x"]), Rng)
