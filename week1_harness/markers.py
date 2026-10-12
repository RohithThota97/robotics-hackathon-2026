"""Core value-marker model (Requirement 4).

This is the heart of the harness's honesty guarantee. Every value the harness
emits is wrapped in a :class:`MarkedValue` tagged ``REAL``, ``PLACEHOLDER``, or
``ESTIMATE``. Non-real values must carry a reason, and the marker-reason
invariant is enforced at construction time -- so no step can emit an unlabeled
non-real value, and nobody can mistake harness output for a real measured
Week 1 turnaround.

The marker-reason invariant (R4.2, R4.4, R4.5, R4.6):

    marker == REAL                      <=>  reason is None
    marker in {PLACEHOLDER, ESTIMATE}   <=>  reason is a non-empty string
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Marker(str, Enum):
    """How a value was produced during this run."""

    REAL = "REAL"               # measured on the local CPU during THIS run (R4.1, R4.6)
    PLACEHOLDER = "PLACEHOLDER"  # stubbed computation we chose not to run for real (R4.4)
    ESTIMATE = "ESTIMATE"        # needs absent hardware / gated asset (R4.5)


@dataclass(frozen=True)
class MarkedValue:
    """A value wrapped with its marker and (for non-real values) a reason.

    Fields:
        value:  the underlying value (any type -- a number, string, path, ...).
        marker: one of :class:`Marker`.
        reason: ``None`` iff ``marker is REAL``; otherwise a non-empty string
                recording why the value is not a real local-CPU measurement
                (the missing hardware / gated asset for an estimate, or what was
                stubbed for a placeholder).
        unit:   optional display unit appended to the value, e.g. ``"s"`` or
                ``" fps"``.
    """

    value: Any
    marker: Marker
    reason: str | None = None     # None iff REAL; required otherwise
    unit: str | None = None       # e.g. "s", "fps"

    def __post_init__(self) -> None:
        # Enforce the marker-reason invariant at construction (R4.2, R4.4-R4.6).
        if self.marker is Marker.REAL:
            assert self.reason is None, "REAL values must carry no reason (R4.6)"
        else:
            assert self.reason, "PLACEHOLDER/ESTIMATE must carry a non-empty reason (R4.4, R4.5)"

    def render(self) -> str:
        """Render for display in the Timing_Table / Dataset_Card.

        REAL values render untagged (R4.6/R13.3); placeholder and estimate
        values append ``[MARKER: reason]`` so the marker is visible inline
        (R4.4/R4.5/R13.4). Examples::

            3.42s
            n/a [PLACEHOLDER: training stubbed]
            ~25 min [ESTIMATE: NVIDIA GPU absent]
        """
        shown = f"{self.value}{self.unit or ''}"
        if self.marker is Marker.REAL:
            return shown
        return f"{shown} [{self.marker.value}: {self.reason}]"


# Factory helpers used everywhere, so the invariant is impossible to violate.
def real(value: Any, unit: str | None = None) -> MarkedValue:
    """Build a REAL value measured on the local CPU this run (carries no reason)."""
    return MarkedValue(value, Marker.REAL, None, unit)


def placeholder(value: Any, reason: str, unit: str | None = None) -> MarkedValue:
    """Build a PLACEHOLDER value: a stubbed result, with the reason it is not real (R4.4)."""
    return MarkedValue(value, Marker.PLACEHOLDER, reason, unit)


def estimate(value: Any, reason: str, unit: str | None = None) -> MarkedValue:
    """Build an ESTIMATE value: a figure requiring absent hardware/gated asset (R4.5)."""
    return MarkedValue(value, Marker.ESTIMATE, reason, unit)
