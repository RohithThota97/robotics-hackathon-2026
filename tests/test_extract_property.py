"""Property-based test for Step 2 -- Extract frames scale (Task 8.4).

Feature: week1-data-loop-harness, Property 9: Extraction scale -- for any seed,
the Frame_Extractor samples at an effective rate within 2-3 fps, yields a
training image count within 200-320, and produces at least 150 labelled
instances for each of the six practice items. This test additionally asserts the
held-out golden images never leak into the training set.

The property is exercised end to end: for each generated seed it runs the real
:class:`~week1_harness.steps.film.FilmStep` followed by the real
:class:`~week1_harness.steps.extract.ExtractStep`, mirroring the pipeline
hand-off (Film publishes its payload under ``ctx.artifacts["Film"]``, which
Extract reads), then checks the realized training set against the Day-1 scale
contract.

Each example runs the full Film+Extract pass, which renders ~284 PNGs to disk --
expensive I/O -- so the example budget is kept deliberately small
(``max_examples=5``) and the per-example deadline is disabled. A fresh
``tempfile.TemporaryDirectory`` is created *inside* the test body per example
(not via a function-scoped fixture, which Hypothesis would share across every
example) so the runs stay hermetic and never touch the repository tree.

**Validates: Requirements 6.1, 6.2, 6.3**
"""

from __future__ import annotations

import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness.context import RunContext
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.synth.items import CLASS_IDS

# Property 9 scale bounds, written as literals straight from the acceptance
# criteria rather than imported from the Extract module's own constants -- the
# test must check the *requirement*, not re-assert the implementation against
# itself.
MIN_FPS = 2.0  # R6.1 lower bound of the effective-rate band
MAX_FPS = 3.0  # R6.1 upper bound of the effective-rate band
MIN_TRAINING_IMAGES = 200  # R6.2 lower bound of the training image count
MAX_TRAINING_IMAGES = 320  # R6.2 upper bound of the training image count
MIN_INSTANCES_PER_ITEM = 150  # R6.3 floor of labelled instances per item
NUM_PRACTICE_ITEMS = 6  # the six practice items, look-alike pair kept separate


def _run_film_then_extract(seed: int, out_dir: str) -> dict:
    """Run Film then Extract against one seeded context and return Extract's payload.

    Reproduces the pipeline runner's hand-off: ``FilmStep`` writes its clips and
    returns a payload, which is stored under ``ctx.artifacts["Film"]`` (the key
    ``ExtractStep`` reads) before Extract subsamples it. ``capabilities`` is
    unused by both steps, so it is passed as ``None``.
    """
    ctx = RunContext(seed=seed, out_dir=out_dir, capabilities=None)
    film_result = FilmStep().run(ctx)
    ctx.artifacts["Film"] = film_result.payload
    extract_result = ExtractStep().run(ctx)
    return extract_result.payload


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=5, deadline=None)
def test_extraction_scale_holds_for_any_seed(seed: int) -> None:
    """Property 9: the realized training set lands at Day-1 scale for any seed.

    For every seed, after Film then Extract the realized training set satisfies
    all of: effective rate within 2-3 fps (R6.1); training image count within
    200-320 (R6.2); every one of the six practice items has >= 150 labelled
    instances (R6.3); and no held-out golden image appears in the training set.
    """
    with tempfile.TemporaryDirectory() as out_dir:
        payload = _run_film_then_extract(seed, out_dir)

    training_images = payload["training_images"]
    golden_images = payload["golden_images"]
    instances_per_class = payload["instances_per_class"]
    training_count = payload["training_count"]
    effective_fps = payload["effective_fps"]

    # R6.1 -- effective sample rate within the 2-3 fps band.
    assert MIN_FPS <= effective_fps <= MAX_FPS, (
        f"effective rate {effective_fps} fps outside "
        f"[{MIN_FPS}, {MAX_FPS}] fps (R6.1)"
    )

    # R6.2 -- training image count within 200-320, and the reported count agrees
    # with the realized training image list.
    assert training_count == len(training_images)
    assert MIN_TRAINING_IMAGES <= training_count <= MAX_TRAINING_IMAGES, (
        f"training image count {training_count} outside "
        f"[{MIN_TRAINING_IMAGES}, {MAX_TRAINING_IMAGES}] (R6.2)"
    )

    # R6.3 -- exactly the six practice items, each with >= 150 labelled instances.
    assert set(instances_per_class) == set(CLASS_IDS)
    assert len(instances_per_class) == NUM_PRACTICE_ITEMS
    for class_id, count in instances_per_class.items():
        assert count >= MIN_INSTANCES_PER_ITEM, (
            f"item {class_id!r} has {count} labelled instances "
            f"< {MIN_INSTANCES_PER_ITEM} (R6.3)"
        )

    # Golden isolation -- the held-out golden images never leak into the training
    # set: their paths are disjoint and no training image is a golden frame.
    assert golden_images, "expected held-out golden images to exist"
    training_paths = {img.path for img in training_images}
    golden_paths = {img.path for img in golden_images}
    assert training_paths.isdisjoint(golden_paths), (
        "a golden image path appeared in the training set"
    )
    assert all(img.session != "golden" for img in training_images), (
        "a training image is tagged as a golden-session frame"
    )
