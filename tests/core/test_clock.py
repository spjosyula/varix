"""Tests for the Clock seam."""

from __future__ import annotations

from datetime import UTC, datetime

from varix.core import Clock, FrozenClock, SystemClock


def test_system_clock_returns_timezone_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is UTC


def test_system_clock_satisfies_clock_protocol() -> None:
    assert isinstance(SystemClock(), Clock)


def test_frozen_clock_returns_fixed_time() -> None:
    fixed = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert FrozenClock(fixed).now() == fixed


def test_frozen_clock_returns_same_value_across_calls() -> None:
    fixed = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    clock = FrozenClock(fixed)
    assert clock.now() == clock.now() == fixed


def test_frozen_clock_satisfies_clock_protocol() -> None:
    assert isinstance(FrozenClock(datetime(2026, 1, 1, tzinfo=UTC)), Clock)
