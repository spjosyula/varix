"""Rng Protocol — an injectable identifier source for run IDs and seeds."""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable


@runtime_checkable
class Rng(Protocol):
    """Source of fresh unique identifiers. Inject in tests with `SequenceRng`."""

    def new_id(self) -> str: ...


class SystemRng:
    """UUID4-backed identifier source. The default."""

    def new_id(self) -> str:
        return str(uuid.uuid4())


class SequenceRng:
    """Rng that emits a pre-recorded sequence of IDs. For deterministic tests."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self._index = 0

    def new_id(self) -> str:
        if self._index >= len(self._ids):
            raise RuntimeError("SequenceRng exhausted")
        out = self._ids[self._index]
        self._index += 1
        return out
