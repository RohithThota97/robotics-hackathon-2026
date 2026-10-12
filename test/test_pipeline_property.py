"""Property-based test for the pipeline runner's per-step progress line (Task 5.2).

Feature: week1-data-loop-harness, Property 14: Progress line per step -- for any
executed step, the run emits exactly one progress line that contains that step's
number and its canonical name. The runner also drives the steps in list order,
collects one ``StepResult`` per step in that order, and stores each step's
payload in ``ctx.artifacts`` under the step's name. A step that raises halts the
run and surfaces a ``StepFailure`` naming that step (R1.5).

Exercises :func:`week1_harness.pipeline.run`, the
:class:`week1_harness.pipeline.Step` protocol, and
:class:`week1_harness.pipeline.StepFailure` against a seeded
:class:`week1_harness.context.RunContext` and fake steps.

**Validates: Requirements 1.4**
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness import pipeline
from week1_harness.context import RunContext
from week1_harness.datamodel import StepResult
from week1_harness.markers import Marker, real
from week1_harness.pipeline import StepFailure


# --- Fakes ---------------------------------------------------------------------


@dataclass
class FakeStep:
    """A minimal :class:`~week1_harness.pipeline.Step` for driving the runner.

    Carries a canonical ``number``/``name`` and returns a trivial
    :class:`~week1_harness.datamodel.StepResult`. The payload embeds the step's
    identity so the test can assert the runner stored it in ``ctx.artifacts``
    under the step's name. The time cell is a ``REAL`` zero-second measurement,
    matching how a real local-CPU step reports its timing.
    """

    number: int
    name: str

    def run(self, ctx: RunContext) -> StepResult:
        return StepResult(
            number=self.number,
            name=self.name,
            how="fake step",
            time=real(0.0, unit="s"),
            payload={"step_number": self.number},
            notes=[],
        )


class _InjectedError(Exception):
    """A distinct error type raised by the failing fake step."""


@dataclass
class RaisingStep:
    """A step whose :meth:`run` always raises, to exercise the failure path (R1.5)."""

    number: int
    name: str

    def run(self, ctx: RunContext) -> StepResult:
        raise _InjectedError(f"boom in {self.name}")


def _make_context() -> RunContext:
    """A RunContext with a trivial capabilities stand-in (``None``) and no out dir."""
    return RunContext(seed=0, out_dir=None, capabilities=None)


# --- Strategies ----------------------------------------------------------------

# A non-empty list of fake steps with pairwise-distinct numbers (and hence
# distinct names), 1..8 steps long -- the canonical Data Loop has eight. A wide
# number pool exercises multi-digit numbers in the "line contains the number"
# check.
distinct_numbers = st.lists(
    st.integers(min_value=1, max_value=99),
    min_size=1,
    max_size=8,
    unique=True,
)


def _steps_from_numbers(numbers: list[int]) -> list[FakeStep]:
    """Build fake steps with distinct numbers and matching distinct names."""
    return [FakeStep(number=n, name=f"Step{n}") for n in numbers]


# --- Property 14: one progress line per step, in order -------------------------


@given(numbers=distinct_numbers)
@settings(max_examples=100, deadline=None)
def test_run_emits_one_progress_line_per_step_in_order(numbers):
    """``run`` emits exactly one progress line per step, in list order, each
    naming that step's number and canonical name; results and artifacts follow
    the same order and mapping (Property 14, R1.4)."""
    steps = _steps_from_numbers(numbers)
    ctx = _make_context()

    # Capture stdout for THIS example only, using a fresh buffer per Hypothesis
    # run so progress lines can never leak between examples. ``ctx.progress``
    # uses ``print`` -> ``sys.stdout``, which ``redirect_stdout`` intercepts.
    buf = io.StringIO()
    with redirect_stdout(buf):
        results = pipeline.run(ctx, steps)

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]

    # Exactly one progress line per step...
    assert len(lines) == len(steps)
    # ...and each line, in order, names its step's number and canonical name.
    for line, step in zip(lines, steps):
        assert str(step.number) in line
        assert step.name in line

    # One StepResult per step, in execution (list) order.
    assert [(r.number, r.name) for r in results] == [
        (s.number, s.name) for s in steps
    ]

    # Each fake real-CPU step reports a REAL time cell (sanity on the fake).
    assert all(r.time.marker is Marker.REAL for r in results)

    # Each step's payload is stored in the shared artifact store under its name.
    assert set(ctx.artifacts) == {s.name for s in steps}
    for step in steps:
        assert ctx.artifacts[step.name] == {"step_number": step.number}


# --- Failure path (example-based, R1.5) ----------------------------------------


def test_raising_step_halts_run_with_stepfailure_naming_it(capsys):
    """A step that raises stops the run and surfaces ``StepFailure`` naming that
    step (number + name), chaining the original error. Earlier payloads are
    stored; the failing step and everything after it are not (R1.5)."""
    good = FakeStep(number=1, name="Film")
    bad = RaisingStep(number=2, name="Extract frames")
    never = FakeStep(number=3, name="Zero-shot baseline")
    ctx = _make_context()

    with pytest.raises(StepFailure) as excinfo:
        pipeline.run(ctx, [good, bad, never])

    failure = excinfo.value
    # StepFailure names the failing step's number and name.
    assert failure.number == 2
    assert failure.name == "Extract frames"
    assert "2" in str(failure)
    assert "Extract frames" in str(failure)
    # The original exception is preserved as the cause.
    assert isinstance(failure.__cause__, _InjectedError)

    # The run halted at the failing step: the good step's payload was stored, the
    # failing step and the step after it never reached the artifact store.
    assert ctx.artifacts == {"Film": {"step_number": 1}}
    assert "Extract frames" not in ctx.artifacts
    assert "Zero-shot baseline" not in ctx.artifacts

    # The failing step is reported to stderr by ``ctx.fail``, naming its number
    # and name.
    err = capsys.readouterr().err
    assert "2" in err
    assert "Extract frames" in err
