"""Unit tests for Step 8 -- Test live (``week1_harness.steps.live_test.LiveTestStep``).

Covers task 11.6: the eighth and final Data Loop step, which simulates the
on-site live test. On site this is where the freshly trained model would run
against a live webcam on the GPU laptop and report a real-time frame rate over a
reshuffled golden layout. The harness runs CPU-only with no webcam and no NVIDIA
GPU, so the live test is **simulated**: it emits a plausible, seed-derived live
frame rate and references a reshuffled golden layout containing all six practice
items. These example-based unit tests pin that contract:

* **fps ESTIMATE (R12.1).** Live-test's time cell is the fps ``MarkedValue``: an
  ``ESTIMATE`` whose reason is exactly ``"webcam and GPU absent"`` and whose unit
  is ``" fps"`` -- a real live rate needs the absent webcam + GPU. The payload's
  ``fps`` equals that time cell's value (the one, same simulated number).
* **Reshuffled golden layout, all six items (R12.3).** The referenced
  ``golden_layout_id`` names a golden layout (its id starts with ``"golden"``),
  and ``reshuffled_items`` is a permutation of the six practice-item classes
  (set-equal to ``CLASS_IDS``, length six, no duplicates).
* **Seed-derived determinism (R12.2).** Running the whole loop a second time with
  the **same** seed (into a fresh temp dir) yields the identical simulated fps
  and the identical reshuffled item order -- both are drawn from ``rng_for(8)``,
  so a given seed always reproduces them.
* **Canonical identity.** The step reports ``number == 8`` and
  ``name == "Test live"``.

Requirements: 12.1, 12.2, 12.3.

The tests are hermetic: each run writes under a pytest-managed temp directory
(``tmp_path_factory``, the module-scoped sibling of ``tmp_path``), so nothing is
written into the repository working tree. Live-test sits at the end of the loop
and reads Score's payload, so reaching it means running the full eight-step
chain Film -> Extract -> Zero-shot -> Auto-label -> Review -> Train -> Score ->
Test live via :func:`week1_harness.pipeline.run`, which publishes each step's
payload under ``ctx.artifacts[step.name]`` for the next step. Film renders a few
hundred PNG frames, so the full pass runs exactly **once** in a module-scoped
fixture and every per-run assertion reads from that single run. The determinism
check needs a second same-seed run (into its own fresh temp dir), kept in its
own module-scoped fixture so it, too, films only once.

On this dev machine ``torch``/``ultralytics`` and the ungated YOLOE weights are
absent, so Zero-shot (step 3) takes its placeholder path; that does not affect
Live-test's simulated fps or reshuffle, which are seed-derived from
``rng_for(8)`` and the (deterministic) ground-truth layouts.

_Requirements: 12.1, 12.2, 12.3_
"""

from __future__ import annotations

import pytest

from week1_harness import capabilities, pipeline
from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.live_test import LiveTestStep
from week1_harness.steps.review import ReviewStep
from week1_harness.steps.score import ScoreStep
from week1_harness.steps.train import TrainStep
from week1_harness.steps.zero_shot import ZeroShotStep
from week1_harness.synth.items import CLASS_IDS

# A fixed seed makes every assertion concrete and reproducible, and is what the
# determinism check reuses for its second run (the default/start-date seed used
# across the other harness tests).
SEED = 20261009

# The exact reason the fps time cell carries as an ESTIMATE (R12.1): a real live
# frame rate needs the webcam + GPU the harness does not have.
WEBCAM_GPU_ABSENT_REASON = "webcam and GPU absent"

# The display unit appended to the simulated frame rate (R12.1).
FPS_UNIT = " fps"

# Live-test is the eighth and final Data Loop step.
LIVE_TEST_NUMBER = 8
LIVE_TEST_NAME = "Test live"


def _canonical_steps() -> list:
    """The eight canonical Data Loop steps, in order (Film .. Test live).

    Returned as fresh instances so each run drives its own steps; the order is
    the harness's canonical pipeline order (R1.1) and matches the artifact
    hand-off each downstream step depends on -- Live-test (step 8) reads Score's
    payload, which depends on the whole chain in front of it.
    """
    return [
        FilmStep(),
        ExtractStep(),
        ZeroShotStep(),
        AutoLabelStep(),
        ReviewStep(),
        TrainStep(),
        ScoreStep(),
        LiveTestStep(),
    ]


def _run_live_test(out_dir) -> object:
    """Run the full eight-step loop once into ``out_dir`` and return step 8's result.

    Builds the capabilities the harness would detect for a real run
    (filesystem/import checks only, no network) and threads them through a
    ``RunContext`` seeded with :data:`SEED`, exactly as the CLI does, then drives
    the canonical steps via :func:`week1_harness.pipeline.run` -- which stores
    each payload under ``ctx.artifacts[step.name]`` so Live-test can read Score's
    output. Returns the ``StepResult`` for step 8 (``number == 8``).
    """
    caps = capabilities.detect(None)
    ctx = RunContext(seed=SEED, out_dir=out_dir, capabilities=caps)
    results = pipeline.run(ctx, _canonical_steps())
    by_number = {result.number: result for result in results}
    assert len(results) == 8, f"expected 8 step results, got {len(results)}"
    return by_number[LIVE_TEST_NUMBER]


@pytest.fixture(scope="module")
def live_result(tmp_path_factory):
    """Run the full loop once and share Live-test's result (R12).

    Film writes a few hundred PNG frames, so the per-run assertions below all
    read from this one run rather than re-filming per test. ``tmp_path_factory``
    hands back a pytest-managed temp dir (the module-scoped equivalent of
    ``tmp_path``, with the same hermetic guarantee), so nothing touches the repo
    tree.
    """
    out_dir = tmp_path_factory.mktemp("live_test_out")
    return _run_live_test(out_dir)


@pytest.fixture(scope="module")
def live_result_rerun(tmp_path_factory):
    """Re-run the full loop with the SAME seed into a fresh temp dir (R12.2).

    Used only by the determinism check: a second same-seed run in its own
    hermetic temp dir, so the fps and reshuffle order can be compared to the
    first run's. Films once (module-scoped) like :func:`live_result`.
    """
    out_dir = tmp_path_factory.mktemp("live_test_rerun_out")
    return _run_live_test(out_dir)


# --- Canonical identity --------------------------------------------------------

def test_live_test_step_number_and_name(live_result):
    """Test live is the eighth Data Loop step, named "Test live" (canonical order, R1.1)."""
    assert live_result.number == LIVE_TEST_NUMBER
    assert live_result.name == LIVE_TEST_NAME


# --- fps ESTIMATE (R12.1) ------------------------------------------------------

def test_fps_time_cell_is_estimate_with_webcam_gpu_absent_reason(live_result):
    """Live-test's time cell is an ESTIMATE reason 'webcam and GPU absent', unit ' fps' (R12.1)."""
    time_cell = live_result.time
    assert time_cell.marker is Marker.ESTIMATE, (
        f"Live-test time cell is {time_cell.marker.value}, expected ESTIMATE -- a "
        f"real live frame rate needs the webcam + GPU the harness lacks (R12.1)"
    )
    assert time_cell.reason == WEBCAM_GPU_ABSENT_REASON, (
        f"Live-test fps estimate reason is {time_cell.reason!r}, expected "
        f"{WEBCAM_GPU_ABSENT_REASON!r} (R12.1)"
    )
    assert time_cell.unit == FPS_UNIT, (
        f"Live-test fps estimate unit is {time_cell.unit!r}, expected {FPS_UNIT!r} (R12.1)"
    )


def test_payload_fps_equals_time_cell_value(live_result):
    """The payload's fps is the SAME simulated number as the time cell's value (R12.1)."""
    assert live_result.payload["fps"] == live_result.time.value, (
        f"payload fps {live_result.payload['fps']!r} != time-cell value "
        f"{live_result.time.value!r}; the fps ESTIMATE must be the time cell (R12.1)"
    )


# --- Reshuffled golden layout, all six items (R12.3) ---------------------------

def test_referenced_layout_is_a_golden_layout(live_result):
    """Live-test references a GOLDEN layout -- its id starts with 'golden' (R12.3)."""
    golden_layout_id = live_result.payload["golden_layout_id"]
    assert isinstance(golden_layout_id, str) and golden_layout_id.startswith("golden"), (
        f"referenced layout id {golden_layout_id!r} is not a golden layout (R12.3)"
    )


def test_reshuffled_items_is_a_permutation_of_the_six_classes(live_result):
    """``reshuffled_items`` is a permutation of the six practice-item classes (R12.3)."""
    reshuffled = live_result.payload["reshuffled_items"]

    # All six items are present -- the referenced golden layout contains them all.
    assert set(reshuffled) == set(CLASS_IDS), (
        f"reshuffled_items {sorted(reshuffled)} != the six classes "
        f"{sorted(CLASS_IDS)} (R12.3)"
    )
    # Exactly six, with no duplicates: a true permutation, not a multiset.
    assert len(reshuffled) == 6, (
        f"expected 6 reshuffled items, got {len(reshuffled)} (R12.3)"
    )
    assert len(set(reshuffled)) == len(reshuffled), (
        f"reshuffled_items contains duplicates: {reshuffled} (R12.3)"
    )


# --- Seed-derived determinism (R12.2) ------------------------------------------

def test_fps_and_reshuffle_are_seed_derived(live_result, live_result_rerun):
    """Same seed -> identical simulated fps and identical reshuffled order (R12.2).

    Both the simulated frame rate and the reshuffle permutation are drawn from
    ``rng_for(8)`` (an independent sub-stream keyed by the seed and the step
    number), so two runs with the same seed reproduce them exactly -- even into
    different output directories.
    """
    assert live_result.payload["fps"] == live_result_rerun.payload["fps"], (
        f"same seed produced different fps: {live_result.payload['fps']!r} vs "
        f"{live_result_rerun.payload['fps']!r} (R12.2)"
    )
    assert live_result.payload["reshuffled_items"] == live_result_rerun.payload["reshuffled_items"], (
        f"same seed produced a different reshuffle: "
        f"{live_result.payload['reshuffled_items']} vs "
        f"{live_result_rerun.payload['reshuffled_items']} (R12.2)"
    )
