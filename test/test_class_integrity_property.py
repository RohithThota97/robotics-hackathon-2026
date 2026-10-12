"""Property-based test for class integrity and the look-alike pair (Task 10.7).

Feature: week1-data-loop-harness, Property 5: Class integrity and the look-alike
pair -- for any generated dataset, exactly the six practice-item classes are
present, the look-alike pair is held as two distinct classes end to end
(generation, auto-label, review, and the dataset card), and every produced mask
carries exactly one class.

The property is exercised end to end over the first five Data Loop steps: for
each generated seed it runs the real
:class:`~week1_harness.steps.film.FilmStep`, then
:class:`~week1_harness.steps.extract.ExtractStep`, then
:class:`~week1_harness.steps.auto_label.AutoLabelStep`, then
:class:`~week1_harness.steps.review.ReviewStep`, mirroring the pipeline hand-off
-- each step publishes its payload under ``ctx.artifacts[step.name]`` (``"Film"``
-> ``"Extract frames"`` -> ``"Auto-label"`` -> ``"Review"``), which the next
step reads. Auto-label publishes the instance masks (each a
:class:`~week1_harness.datamodel.Mask` with a single ``class_id``) and Review
publishes the reviewed labels (each a
:class:`~week1_harness.datamodel.ReviewedLabel` with a single ``class_id``).

For every seed, after Film -> Extract -> Auto-label -> Review we assert class
integrity holds end to end:

* **Masks are a subset of the six classes, and -- because all six items are
  placed (S1 shows each item alone, S2 packs all six) -- equal exactly the six**
  (R5.1, R8.3, R14.1). No class is invented and none is dropped.
* **Every produced mask carries exactly one practice-item class** (R8.3): each
  ``Mask.class_id`` is one of the six.
* **The look-alike pair stays two distinct classes** (R5.1, R8.4, R9.4): both
  ``screwdriver_small`` and ``screwdriver_large`` are present and distinct among
  the masks *and* among the retained reviewed labels -- they are never merged
  into a single class. (S1 shows each screwdriver alone and unoccluded, so each
  is fully visible and survives Review's visibility rule, R9.2/R9.3.)
* **Every reviewed label carries one of the six classes** (R9.4): Review neither
  invents nor merges a class.

Each example runs the full Film + Extract + Auto-label + Review pass, which
renders ~284 PNGs to disk -- expensive I/O -- so the example budget is kept
deliberately small (``max_examples=3``) and the per-example deadline is
disabled. A fresh ``tempfile.TemporaryDirectory`` is created *inside* the test
body per example (not via a function-scoped fixture, which Hypothesis would
share across every example) so the runs stay hermetic and never touch the
repository tree.

**Validates: Requirements 5.1, 8.3, 8.4, 9.4, 14.1**

_Requirements: 5.1, 8.3, 8.4, 9.4, 14.1_
"""

from __future__ import annotations

import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness import capabilities
from week1_harness.context import RunContext
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.review import ReviewStep
from week1_harness.synth.items import CLASS_IDS

# The six practice-item classes and the look-alike pair, named straight from the
# requirements/design (the authoritative source) so the test checks the
# *requirement* rather than re-asserting the implementation against itself.
EXPECTED_CLASS_COUNT = 6  # R5.1 / R14.1: exactly six practice-item classes
# The look-alike pair held as two separate classes (R5.1, R8.4, R9.4).
LOOK_ALIKE_SMALL = "screwdriver_small"
LOOK_ALIKE_LARGE = "screwdriver_large"


def _run_pipeline_to_review(seed: int, out_dir: str) -> tuple[dict, dict]:
    """Run Film -> Extract -> Auto-label -> Review; return the last two payloads.

    Reproduces the pipeline runner's hand-off: each step's payload is stored
    under its canonical step name (``"Film"``, ``"Extract frames"``,
    ``"Auto-label"``, ``"Review"``) before the next step reads it.
    ``capabilities`` is built with :func:`week1_harness.capabilities.detect`
    (filesystem/import checks only, no network) to mirror how the harness builds
    the context for a real run; none of these four steps consult it. Returns
    ``(auto_label_payload, review_payload)``.
    """
    ctx = RunContext(seed=seed, out_dir=out_dir, capabilities=capabilities.detect(None))

    film_result = FilmStep().run(ctx)
    ctx.artifacts["Film"] = film_result.payload

    extract_result = ExtractStep().run(ctx)
    ctx.artifacts["Extract frames"] = extract_result.payload

    auto_label_result = AutoLabelStep().run(ctx)
    ctx.artifacts["Auto-label"] = auto_label_result.payload

    review_result = ReviewStep().run(ctx)
    ctx.artifacts["Review"] = review_result.payload

    return auto_label_result.payload, review_result.payload


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=3, deadline=None)
def test_class_integrity_and_look_alike_pair_hold_for_any_seed(seed: int) -> None:
    """Property 5: class integrity and the look-alike pair hold end to end.

    For every seed, after Film -> Extract -> Auto-label -> Review: the mask
    classes are exactly the six practice-item classes (subset and, since all six
    are placed, equal); every mask carries exactly one of the six classes; the
    look-alike pair (``screwdriver_small`` / ``screwdriver_large``) is present
    and distinct both among the masks and among the retained reviewed labels
    (never merged); and every reviewed label carries one of the six classes.
    Together these discharge R5.1, R8.3, R8.4, R9.4, and R14.1.
    """
    six_classes = set(CLASS_IDS)

    # Anchor the expectations to the requirement: exactly six classes, and the
    # look-alike pair is two of them and genuinely distinct.
    assert len(six_classes) == EXPECTED_CLASS_COUNT, (
        f"expected {EXPECTED_CLASS_COUNT} distinct practice-item classes, "
        f"got {len(six_classes)} (R5.1)"
    )
    assert LOOK_ALIKE_SMALL != LOOK_ALIKE_LARGE, "look-alike pair must be two classes"
    assert {LOOK_ALIKE_SMALL, LOOK_ALIKE_LARGE} <= six_classes, (
        "the look-alike pair must be among the six practice-item classes (R5.1)"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        auto_label_payload, review_payload = _run_pipeline_to_review(seed, out_dir)

    masks = auto_label_payload["masks"]
    reviewed_labels = review_payload["reviewed_labels"]

    # Guard against vacuous passes: there must be masks and reviewed labels for
    # the class-integrity checks to be meaningful.
    assert masks, "Auto-label produced no masks"
    assert reviewed_labels, "Review produced no labels"

    # --- Masks: exactly the six classes, one class per mask (R5.1/R8.3/R14.1) --
    mask_classes = [mask.class_id for mask in masks]
    mask_class_set = set(mask_classes)

    # Every produced mask carries exactly one of the six practice-item classes
    # (R8.3): a single, valid class_id per mask -- no class is invented.
    for mask in masks:
        assert mask.class_id in six_classes, (
            f"mask carries class {mask.class_id!r} which is not one of the six "
            f"practice-item classes (R8.3)"
        )

    # The mask classes are a subset of the six and -- because all six items are
    # placed (S1 each item alone, S2 all six) -- equal exactly the six: no
    # extras, no missing (R5.1/R8.3/R14.1).
    assert mask_class_set <= six_classes, (
        f"mask classes {sorted(mask_class_set - six_classes)} are not a subset "
        f"of the six practice-item classes (R8.3)"
    )
    assert mask_class_set == six_classes, (
        f"mask classes are not exactly the six: missing "
        f"{sorted(six_classes - mask_class_set)}, extra "
        f"{sorted(mask_class_set - six_classes)} (R5.1/R14.1)"
    )

    # --- Look-alike pair: distinct among the masks, never merged (R8.4) --------
    assert LOOK_ALIKE_SMALL in mask_class_set, (
        f"{LOOK_ALIKE_SMALL!r} is absent among the masks -- the look-alike pair "
        f"must stay a distinct class (R8.4)"
    )
    assert LOOK_ALIKE_LARGE in mask_class_set, (
        f"{LOOK_ALIKE_LARGE!r} is absent among the masks -- the look-alike pair "
        f"must stay a distinct class (R8.4)"
    )

    # --- Reviewed labels: one of the six each; pair distinct and retained ------
    # Every reviewed label carries one of the six classes -- Review neither
    # invents nor merges a class (R9.4).
    for label in reviewed_labels:
        assert label.class_id in six_classes, (
            f"reviewed label carries class {label.class_id!r} which is not one "
            f"of the six practice-item classes (R9.4)"
        )

    # The look-alike pair survives Review as two distinct retained classes: each
    # screwdriver appears alone and unoccluded in S1 (visible_fraction == 1.0),
    # so each is retained, and the two are never merged (R9.4).
    retained_classes = {label.class_id for label in reviewed_labels if label.retained}
    assert LOOK_ALIKE_SMALL in retained_classes, (
        f"{LOOK_ALIKE_SMALL!r} is absent among retained reviewed labels -- the "
        f"look-alike pair must stay a distinct retained class (R9.4)"
    )
    assert LOOK_ALIKE_LARGE in retained_classes, (
        f"{LOOK_ALIKE_LARGE!r} is absent among retained reviewed labels -- the "
        f"look-alike pair must stay a distinct retained class (R9.4)"
    )
    # Distinctness end to end: the pair is two class ids, never collapsed to one.
    assert {LOOK_ALIKE_SMALL, LOOK_ALIKE_LARGE} <= retained_classes, (
        "the look-alike pair was merged in review (R9.4)"
    )
