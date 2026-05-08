"""varix error hierarchy.

Every error varix raises is a subclass of `VarixError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from varix.core.types import PipelineRun


class VarixError(Exception):
    """Base class for every error varix raises."""


class AdapterError(VarixError):
    """The adapter raised an error or returned a malformed result."""


class CapabilityMissing(VarixError):
    """An operation required an adapter capability that was not declared."""


class BudgetExceeded(VarixError):
    """The cost budget (`--max-cost`) was reached.

    `partial_runs` carries every run that completed before the limit, so the
    caller can still write a (partial) artifact.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_runs: tuple[PipelineRun, ...] = (),
    ) -> None:
        super().__init__(message)
        self.partial_runs = partial_runs


class RunFailed(VarixError):
    """The run loop was interrupted by an adapter exception.

    `partial_runs` carries every run that completed cleanly before the
    failure. The original cause is chained via `raise ... from exc` and
    accessible through `__cause__`.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_runs: tuple[PipelineRun, ...] = (),
    ) -> None:
        super().__init__(message)
        self.partial_runs = partial_runs


class StructuralMismatch(VarixError):
    """Pipeline runs produced different step graphs and cannot be aligned."""


class RefusalRequired(VarixError):
    """varix declined to produce a finding rather than guess."""
