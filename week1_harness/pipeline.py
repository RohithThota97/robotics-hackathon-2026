"""Step interface and the ordered pipeline runner (Requirement 1).

The harness runs the eight Data Loop steps as an ordered list of small,
isolated components. Each component implements the tiny :class:`Step` protocol:
it carries a canonical ``number`` and ``name`` and knows how to :meth:`~Step.run`
itself against the shared :class:`~week1_harness.context.RunContext`, returning a
:class:`~week1_harness.datamodel.StepResult`. Steps never talk to one another
directly; they hand off through ``ctx.artifacts``, which keeps each step
independently testable (give it a seeded context, assert its result).

:func:`run` is the thin runner that drives the list in order. It takes the
steps as an **injected argument** rather than reaching for a module-level
constant, so it is fully testable now with fake steps -- the canonical
eight-step list (``CANONICAL_STEPS``) is assembled later during CLI wiring
(task 14.2), once the eight concrete step classes exist. For each step the
runner:

* emits one progress line via ``ctx.progress`` (R1.4);
* runs the step inside ``try``/``except`` and, on any error, reports the failing
  step via ``ctx.fail`` and raises :class:`StepFailure` (R1.5) so the caller can
  map it to a non-zero exit code without running the reporters;
* on success, stores the step's ``payload`` in the shared artifact store under
  the step name and collects the :class:`~week1_harness.datamodel.StepResult`.

Requirements touched: R1.1 (ordered execution of the injected steps), R1.4 (one
progress line per step), R1.5 (a failing step is reported and halts the run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.context import RunContext
    from week1_harness.datamodel import StepResult


class Step(Protocol):
    """The tiny interface every Data Loop step implements.

    A step is a pure-ish function of the :class:`~week1_harness.context.RunContext`
    (plus the shared artifact store it reaches through ``ctx``): it returns a
    :class:`~week1_harness.datamodel.StepResult` and never calls another step
    directly.

    Attributes:
        number: the step's position ``1..8`` in canonical order (R1.1).
        name:   the human-readable step name (``"Film"``, ``"Extract frames"``,
                ...); also the key under which the step's payload is stored in
                ``ctx.artifacts``.
    """

    number: int
    name: str

    def run(self, ctx: RunContext) -> StepResult:
        """Execute the step against ``ctx`` and return its :class:`StepResult`."""
        ...


class StepFailure(Exception):
    """Raised when a Data Loop step raises during :func:`run` (R1.5).

    Carries the failing step's ``number`` and ``name`` as attributes so the
    caller (``cli.main``) can map the failure to a non-zero exit code and the
    reporters are never run. The original exception is preserved as the
    ``__cause__`` via ``raise ... from exc``.

    Attributes:
        number: the failing step's number.
        name:   the failing step's name.
    """

    def __init__(self, number: int, name: str) -> None:
        self.number = number
        self.name = name
        super().__init__(f"step {number} ({name}) failed")


def run(ctx: RunContext, steps: list[Step]) -> list[StepResult]:
    """Run the injected ``steps`` in order and collect their results (R1).

    For each step, in list order (R1.1): emit one progress line (R1.4), run the
    step, and -- on success -- hand its payload to downstream steps through the
    shared artifact store (keyed by ``step.name``) before collecting its
    :class:`~week1_harness.datamodel.StepResult`. If a step raises, report it via
    ``ctx.fail`` and re-raise as :class:`StepFailure` so the run halts before any
    further step or reporter runs (R1.5).

    Args:
        ctx:   the shared run context (seed, output dir, capabilities, the
               ``artifacts`` store, and the ``progress``/``fail`` helpers).
        steps: the ordered steps to execute, injected by the caller so the
               runner is testable with fakes.

    Returns:
        One :class:`~week1_harness.datamodel.StepResult` per step, in execution
        order.

    Raises:
        StepFailure: if any step raises; it names the failing step's number and
            name and chains the original exception as ``__cause__``.
    """
    results: list[StepResult] = []
    for step in steps:
        ctx.progress(step.number, step.name)          # R1.4: one line per step
        try:
            result = step.run(ctx)
        except Exception as exc:                        # R1.5: report and halt
            ctx.fail(step.number, step.name, exc)
            raise StepFailure(step.number, step.name) from exc
        ctx.artifacts[step.name] = result.payload       # hand off to downstream steps
        results.append(result)
    return results
