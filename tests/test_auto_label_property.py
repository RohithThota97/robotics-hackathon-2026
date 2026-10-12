"""Property-based test for Step 4 -- Auto-label mask-instance bijection (Task 10.2).

Feature: week1-data-loop-harness, Property 6: Mask-instance bijection -- for any
generated training image set, the Auto_Labeler produces exactly one instance
mask per labelled training instance, with a one-to-one correspondence between
masks and instances.

The property is exercised end to end: for each generated seed it runs the real
:class:`~week1_harness.steps.film.FilmStep`, then
:class:`~week1_harness.steps.extract.ExtractStep`, then
:class:`~week1_harness.steps.auto_label.AutoLabelStep`, mirroring the pipeline
hand-off -- each step publishes its payload under ``ctx.artifacts[step.name]``
(``"Film"`` -> ``"Extract frames"`` -> ``"Auto-label"``), which the next step
reads.

The *interpretation* of "labelled instance in the training image set" (R8.1)
that the implementation uses -- and that this test asserts the bijection
against -- is: a **unique** ground-truth instance (keyed by ``instance_id``)
that appears in at least one training image. The same instance recurs across
every frame of its clip, so masks are produced per unique instance, not per
frame-occurrence. Concretely, for each seed we assert that after Film -> Extract
-> Auto-label:

* **(a) mask instance ids are unique** -- no duplicate masks;
* **(b) the set of mask instance ids equals the set of unique instance ids
  referenced by Extract's ``training_images``** -- a one-to-one correspondence
  (bijection), exactly one mask per labelled training instance;
* **(c) ``mask_count == len(masks) == that set size``** -- the reported count,
  the realized mask list, and the labelled-instance set all agree; and
* **(d) no mask corresponds to a golden-only instance** -- every mask instance
  id is in the training id set and none is an id that appears only in the
  held-out golden set.

Each example runs the full Film + Extract + Auto-label pass, which renders ~284
PNGs to disk -- expensive I/O -- so the example budget is kept deliberately
small (``max_examples=4``) and the per-example deadline is disabled. A fresh
``tempfile.TemporaryDirectory`` is created *inside* the test body per example
(not via a function-scoped fixture, which Hypothesis would share across every
example) so the runs stay hermetic and never touch the repository tree.

**Validates: Requirements 8.1**

_Requirements: 8.1_
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


def _unique_instance_ids(images) -> set[str]:
    """The set of distinct instance ids referenced across ``images``.

    Each :class:`~week1_harness.datamodel.ExtractedImage` lists the instance ids
    present in it; the same instance recurs across every frame of its clip, so a
    union over the images collapses those repeated occurrences to one id per
    unique instance -- exactly the "labelled instance" notion the bijection is
    stated against (R8.1).
    """
    ids: set[str] = set()
    for image in images:
        ids.update(image.instance_ids)
    return ids


def _run_film_extract_auto_label(seed: int, out_dir: str) -> tuple[dict, dict]:
    """Run Film -> Extract -> Auto-label on one seeded context; return their payloads.

    Reproduces the pipeline runner's hand-off: each step's payload is stored
    under its canonical step name (``"Film"``, ``"Extract frames"``,
    ``"Auto-label"``) before the next step reads it. ``capabilities`` is built
    with :func:`week1_harness.capabilities.detect` (filesystem/import checks
    only, no network) to mirror how the harness builds the context for a real
    run; Auto-label does not consult it. Returns ``(extract_payload,
    auto_label_payload)``.
    """
    ctx = RunContext(seed=seed, out_dir=out_dir, capabilities=capabilities.detect(None))

    film_result = FilmStep().run(ctx)
    ctx.artifacts["Film"] = film_result.payload

    extract_result = ExtractStep().run(ctx)
    ctx.artifacts["Extract frames"] = extract_result.payload

    auto_label_result = AutoLabelStep().run(ctx)
    ctx.artifacts["Auto-label"] = auto_label_result.payload

    return extract_result.payload, auto_label_result.payload


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=4, deadline=None)
def test_mask_instance_bijection_holds_for_any_seed(seed: int) -> None:
    """Property 6: masks are in one-to-one correspondence with labelled instances.

    For every seed, after Film -> Extract -> Auto-label the produced masks are a
    bijection onto the unique instances labelled in Extract's training image
    set: ids are unique (a), the mask id set equals the training id set (b), the
    reported count agrees with both (c), and no held-out golden-only instance is
    masked (d). Together these discharge R8.1 (exactly one instance mask per
    labelled training instance, one-to-one).
    """
    with tempfile.TemporaryDirectory() as out_dir:
        extract_payload, auto_label_payload = _run_film_extract_auto_label(seed, out_dir)

    training_images = extract_payload["training_images"]
    golden_images = extract_payload["golden_images"]
    masks = auto_label_payload["masks"]
    mask_count = auto_label_payload["mask_count"]

    training_id_set = _unique_instance_ids(training_images)
    golden_id_set = _unique_instance_ids(golden_images)
    mask_ids = [mask.instance_id for mask in masks]
    mask_id_set = set(mask_ids)

    # Guard against vacuous passes: there must be masks and labelled training
    # instances for the bijection to be meaningful.
    assert masks, "Auto-label produced no masks"
    assert training_id_set, "expected labelled instances in the training image set"

    # (a) Mask instance ids are unique -- no instance is masked twice.
    assert len(mask_ids) == len(mask_id_set), (
        f"duplicate mask instance ids: {len(mask_ids)} masks collapse to "
        f"{len(mask_id_set)} distinct instance ids (R8.1)"
    )

    # (b) Bijection: the mask id set is exactly the set of unique instance ids
    # referenced by Extract's training images -- one mask per labelled training
    # instance, and no spurious masks.
    missing = training_id_set - mask_id_set  # labelled but got no mask
    extra = mask_id_set - training_id_set  # masked but not a training instance
    assert mask_id_set == training_id_set, (
        f"mask<->instance bijection broken (R8.1): "
        f"{len(missing)} labelled instance(s) without a mask, "
        f"{len(extra)} mask(s) without a labelled training instance"
    )

    # (c) The reported count, the realized mask list, and the labelled-instance
    # set all agree.
    assert mask_count == len(masks) == len(training_id_set), (
        f"count disagreement (R8.1): mask_count={mask_count}, "
        f"len(masks)={len(masks)}, |training instances|={len(training_id_set)}"
    )

    # (d) No mask corresponds to a golden-only (held-out) instance: every mask
    # id is a training id, and none is an id that appears only in the golden set.
    golden_only_ids = golden_id_set - training_id_set
    assert mask_id_set <= training_id_set, (
        "a mask labels an instance that is not in the training image set (R8.1)"
    )
    assert mask_id_set.isdisjoint(golden_only_ids), (
        "a mask labels a held-out golden-only instance (R8.1)"
    )
