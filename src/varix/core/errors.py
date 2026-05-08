"""varix error hierarchy.

Every error varix raises is a subclass of `VarixError`. This lets callers
catch all varix-originating errors with a single except clause while still
distinguishing causes when they need to.
"""

from __future__ import annotations


class VarixError(Exception):
    """Base class for every error varix raises."""


class AdapterError(VarixError):
    """The adapter raised an error or returned a malformed result."""


class CapabilityMissing(VarixError):
    """An operation required an adapter capability that was not declared.

    Raised, for example, when the step replayer is asked to replay a step
    on an adapter whose `capabilities().supports_replay` is False. The
    correct caller response is usually to emit an `UNAVAILABLE` finding,
    not to retry.
    """


class BudgetExceeded(VarixError):
    """The cost budget (`--max-cost`) was reached.

    Analysis halts with the partial state it has accumulated so far; the
    surface layer is responsible for writing whatever artifact is possible.
    """


class StructuralMismatch(VarixError):
    """N runs of the pipeline produced different step graphs.

    v1 cannot align findings across runs whose structure varies, and refuses
    to produce a misleading analysis. The user-facing report explains what
    differed and exits with a refusal code.
    """


class RefusalRequired(VarixError):
    """varix has decided not to produce a finding rather than guess.

    Distinct from `StructuralMismatch` — this is the generic refusal raised
    when an analysis path concludes it cannot honestly answer.
    """
