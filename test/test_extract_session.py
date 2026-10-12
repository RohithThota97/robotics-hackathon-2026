"""Unit tests for Step 2 -- Extract frames session tagging and holdout.

Covers task 8.5: the second Data Loop step, which stands in for "pull a training
set out of the captured footage" by *subsampling* the short synthetic clips that
``FilmStep`` already wrote to disk. Extract does genuine local-CPU work (reads
and organizes the real frame files Film produced), so its time cell is ``REAL``
and it is never stubbed or hardware-blocked (R4.1/R6.5).

These are example-based unit tests that pin the concrete session contract of a
single Film -> Extract run (the extraction-scale property is Property 9,
task 8.4; the multi-run session-coverage property is Property 10, task 14.7).
Here we assert:

* **Session tagging (R6.4).** Every training ``ExtractedImage`` inherits the
  session of its source layout (looked up via Film's layouts) and is tagged
  ``"S1"`` or ``"S2"`` -- never ``"golden"``.
* **Golden holdout (R5.4 / R6).** Every golden layout's frames land in
  ``golden_images`` (all tagged ``"golden"``), and no frame from a golden layout
  appears among the training images -- the golden set is held out of training.
* **REAL time (R6.5).** Extract's time cell is ``REAL`` and carries no reason,
  because the read/organize pass is a genuine local-CPU measurement.

Requirements: 6.4, 6.5.

The tests are hermetic: the one shared run writes under a pytest-managed temp
directory (``tmp_path_factory``, the module-scoped sibling of ``tmp_path``), so
nothing is written into the repository working tree. Film renders ~284 PNG
frames, so Film + Extract run exactly once in a module-scoped fixture and every
assertion reads from that single run rather than re-filming per test.
"""

from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep

import pytest

# A fixed seed makes the run concrete and reproducible (the default/start-date
# seed used across the other harness tests).
SEED = 20261009

# The two training sessions; golden is deliberately excluded from training.
TRAINING_SESSIONS = {"S1", "S2"}


@pytest.fixture(scope="module")
def extract_run(tmp_path_factory):
    """Run ``FilmStep`` then ``ExtractStep`` exactly once and share the results.

    Film writes ~284 PNG frames, so the assertions below all read from this one
    run rather than re-filming per test. ``tmp_path_factory`` hands back a
    pytest-managed temp dir (the module-scoped equivalent of ``tmp_path``, with
    the same hermetic guarantee), so nothing touches the repo tree.

    The pipeline runner would store Film's payload under its step name for
    downstream steps; here we wire that handoff by hand -- ``ExtractStep.run``
    reads ``ctx.artifacts["Film"]`` -- then run Extract against the same seeded
    context. ``capabilities`` is unused by both steps and is passed as ``None``
    per the task contract.

    Returns ``(film_result, extract_result)``.
    """
    out_dir = tmp_path_factory.mktemp("extract_out")
    ctx = RunContext(seed=SEED, out_dir=out_dir, capabilities=None)

    film_result = FilmStep().run(ctx)
    ctx.artifacts["Film"] = film_result.payload
    extract_result = ExtractStep().run(ctx)

    return film_result, extract_result


# --- Session tagging (R6.4) ----------------------------------------------------

def test_training_image_session_matches_its_source_layout(extract_run):
    """Every training image inherits its source layout's session, in {S1, S2} (R6.4)."""
    film_result, extract_result = extract_run

    # Ground truth: the session of every layout Film composed.
    session_by_layout = {
        layout.layout_id: layout.session for layout in film_result.payload["layouts"]
    }

    training_images = extract_result.payload["training_images"]
    assert training_images, "Extract produced no training images"

    for image in training_images:
        # The source layout is a real Film layout...
        assert image.source_layout_id in session_by_layout, (
            f"training image {image.path} references unknown layout "
            f"{image.source_layout_id}"
        )
        # ...and the image's session is exactly that layout's session (R6.4)...
        assert image.session == session_by_layout[image.source_layout_id], (
            f"training image {image.path} tagged session {image.session!r} but its "
            f"source layout {image.source_layout_id} is "
            f"{session_by_layout[image.source_layout_id]!r}"
        )
        # ...and it is a training session, never golden.
        assert image.session in TRAINING_SESSIONS, (
            f"training image {image.path} has non-training session {image.session!r}"
        )


# --- Golden holdout (R5.4 / R6) ------------------------------------------------

def test_golden_layout_frames_are_held_out_of_training(extract_run):
    """Golden layouts' frames land in golden_images and never in training (R5.4/R6)."""
    film_result, extract_result = extract_run

    layouts = film_result.payload["layouts"]
    frames = film_result.payload["frames"]  # layout_id -> [{path, session, ...}, ...]

    golden_layout_ids = {
        layout.layout_id for layout in layouts if layout.session == "golden"
    }
    assert golden_layout_ids, "Film composed no golden layouts to hold out"

    training_images = extract_result.payload["training_images"]
    golden_images = extract_result.payload["golden_images"]
    assert golden_images, "Extract produced no golden images"

    # Every golden image is tagged golden and comes from a golden layout, and the
    # golden set covers exactly the golden layouts (each golden layout contributes
    # frames; no non-golden layout leaks in).
    for image in golden_images:
        assert image.session == "golden", (
            f"golden image {image.path} tagged non-golden session {image.session!r}"
        )
        assert image.source_layout_id in golden_layout_ids, (
            f"golden image {image.path} came from non-golden layout "
            f"{image.source_layout_id}"
        )
    assert {image.source_layout_id for image in golden_images} == golden_layout_ids

    # Every frame Film wrote for a golden layout lands in golden_images...
    golden_frame_paths = {
        record["path"] for lid in golden_layout_ids for record in frames[lid]
    }
    golden_image_paths = {image.path for image in golden_images}
    assert golden_frame_paths, "golden layouts wrote no frames"
    assert golden_frame_paths <= golden_image_paths, (
        "some golden-layout frames did not land in golden_images"
    )

    # ...and none of those golden-layout frames appears among the training images.
    training_paths = {image.path for image in training_images}
    assert golden_frame_paths.isdisjoint(training_paths), (
        "a golden-layout frame leaked into the training set"
    )


# --- REAL time cell (R6.5) -----------------------------------------------------

def test_extract_time_cell_is_real_and_unmarked(extract_run):
    """Extract's read/organize pass is a genuine CPU measurement: REAL, no reason (R6.5)."""
    _film_result, extract_result = extract_run

    # Anchor that this is the Extract step's result.
    assert extract_result.number == 2
    assert extract_result.name == "Extract frames"

    # The time cell is a real local-CPU measurement: REAL marker, no reason (R6.5).
    assert extract_result.time.marker is Marker.REAL
    assert extract_result.time.reason is None
