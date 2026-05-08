"""Clock Protocol — an injectable time seam so varix's own behavior is testable."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Returns the current time. Inject in tests with `FrozenClock`."""

    def now(self) -> datetime: ...


class SystemClock:
    """Real wall-clock time in UTC. The default."""

    def now(self) -> datetime:
        return datetime.now(tz=UTC)


class FrozenClock:
    """Clock that always returns a fixed time. For deterministic tests."""

    def __init__(self, time: datetime) -> None:
        self._time = time

    def now(self) -> datetime:
        return self._time
