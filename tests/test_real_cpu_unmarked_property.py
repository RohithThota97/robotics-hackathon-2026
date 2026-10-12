"""Property-based test for real-CPU steps being unmarked (Task 10.8).

Feature: week1-data-loop-harness, Property 3: Real-CPU steps are unmarked -- for
any step that runs genuine CPU work this run (Film, Extract, Auto-label, Review,
and Zero-shot when torch+ultralytics+weights are present), its time cell is
``REAL`` and carries no marker.

This test exercises the property end to end over several seeds. For each seed it
runs the canonical front of the Data Loop in order --
:class:`~week1_harness.steps.film.FilmStep` (1) ->
:class:`~week1_harness.steps.extract.ExtractStep` (2) ->
:class:`~week1_harness.steps.zero_shot.ZeroShotStep` (3) ->
:class:`~week1_harness.steps.auto_label.AutoLabelStep` (4) ->
:class:`~week1_harness.steps.review.ReviewStep` (5) -- via
:func:`week1_harness.pipeline.run`, which drives the steps in list order and
publishes each step's payload under ``ctx.artifacts[step.name]`` so the next
step reads it (``"Film"`` -> ``"Extract frames"`` -> ``"Zero-shot baseline"`` ->
``"Auto-label"`` -> ``"Review"``).

The steps that do genuine local-CPU work *regardless of the environment* are
Film (draws frames with Pillow), Extract (reads/organizes the frame files),
Auto-label (rasterizes masks from the known geometry), and Review (an automated
label transform). For every seed this test asserts each of those four steps'
``StepResult.time`` cell is :data:`~week1_harness.markers.Marker.REAL` with
``reason is None`` (R4.1 -- every CPU-runnable, ungated step runs for real; R6.5
-- Extract records its real elapsed time). Zero-shot is *not* a real-CPU step on
the dev machine (``torch``/``ultralytics`` and the ungated weights are absent),
so it takes the placeholder path; the test does not assert ``REAL`` on it and,
when the capable path cannot run, additionally asserts its time cell is
``PLACEHOLDER`` -- the complement of the property in this environment.

Each example runs the full Film + Extract + Auto-label + Review pass, which
renders ~284 PNGs to disk (S1 + S2 + golden clips) -- expensive I/O -- so the
example budget is kept deliberately small (``max_examples=3``) and the
per-example deadline is disabled. A fresh ``tempfile.TemporaryDirectory`` is
created *inside* the test body per example (not via a function-scoped fixture,
which Hypothesis would share across every example) so the runs stay hermetic and
never touch the repository tree.

**Validates: Requirements 4.1, 6.5**

_Requirements: 4.1, 6.5_
"""

from __future__ import annotations

import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness import capabilities, pipeline
from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.review import ReviewStep
from week1_harness.steps.zero_shot import ZeroShotStep

# The steps that do genuine local-CPU work on *every* run, independent of the
# environment -- so their time cells are always REAL (R4.1, R6.5). Keyed by the
# canonical step name under which the runner publishes each payload.
REAL_CPU_STEP_NAMES: tuple[str, ...] = ("Film", "Extract frames", "Auto-label", "Review")

# Zero-shot is REAL only on the capable path (torch + ultralytics + ungated
# weights present and not disabled by flag); otherwise it is a PLACEHOLDER.
ZERO_SHOT_NAME = "Zero-shot baseline"


def _front_of_loop_steps() -> list:
    """The first five canonical Data Loop steps, in order (Film .. Review).

    Returned as fresh instances so each example drives its own steps; the order
    matches the harness's canonical pipeline order and the artifact hand-off
    each downstream step depends on.
    """
    return [FilmStep(), ExtractStep(), ZeroShotStep(), AutoLabelStep(), ReviewStep()]


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=3, deadline=None)
def test_real_cpu_steps_time_cells_are_real_and_unmarked(seed: int) -> None:
    """Property 3: Film, Extract, Auto-label, and Review time cells are REAL.

    For every seed, after running Film -> Extract -> Zero-shot -> Auto-label ->
    Review via the pipeline runner, each real-CPU step's ``time`` cell is
    ``Marker.REAL`` and carries no reason (R4.1, R6.5). Zero-shot takes the
    placeholder path on this dev machine (no torch/ultralytics/weights), so it is
    not asserted REAL; when the capable path cannot run, its time cell is
    asserted ``PLACEHOLDER`` instead (the property's complement here).
    """
    # Build the capabilities the harness would detect for a real run
    # (filesystem/import checks only, no network) and thread them through the
    # context, exactly as the CLI does.
    caps = capabilities.detect(None)

    with tempfile.TemporaryDirectory() as out_dir:
        ctx = RunContext(seed=seed, out_dir=out_dir, capabilities=caps)
        results = pipeline.run(ctx, _front_of_loop_steps())

    by_name = {result.name: result for result in results}

    # Guard against a vacuous pass: all five steps must have run and all four
    # real-CPU steps must be present among the results.
    assert len(results) == 5, f"expected 5 step results, got {len(results)}"
    missing = set(REAL_CPU_STEP_NAMES) - set(by_name)
    assert not missing, f"real-CPU steps missing from results: {sorted(missing)}"

    # The property: every real-CPU step's time cell is REAL with no reason.
    for name in REAL_CPU_STEP_NAMES:
        cell = by_name[name].time
        assert cell.marker is Marker.REAL, (
            f"{name!r} time cell is {cell.marker.value}, expected REAL -- a step "
            f"that runs for real on the local CPU must not be marked (R4.1/R6.5)"
        )
        assert cell.reason is None, (
            f"{name!r} time cell is REAL but carries a reason {cell.reason!r}; "
            f"REAL values must carry no reason (R4.1/R4.6)"
        )

    # Zero-shot is a real-CPU step only on the capable path. On this dev machine
    # that path cannot run, so the step is NOT asserted REAL; instead its time
    # cell must be PLACEHOLDER -- the complement of the property in this
    # environment. The guard keeps the test correct if run where weights exist.
    capable = (
        caps.torch_available
        and caps.ultralytics_available
        and caps.yoloe_weights_path is not None
        and not ctx.no_zero_shot
    )
    if not capable:
        zero_shot_cell = by_name[ZERO_SHOT_NAME].time
        assert zero_shot_cell.marker is Marker.PLACEHOLDER, (
            f"with torch/ultralytics/weights absent, Zero-shot should take the "
            f"placeholder path, but its time cell is {zero_shot_cell.marker.value}"
        )
