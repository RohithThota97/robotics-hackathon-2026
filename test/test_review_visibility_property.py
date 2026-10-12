"""Property-based test for Step 5 -- Review visibility threshold (Task 10.5).

Feature: week1-data-loop-harness, Property 7: Review visibility threshold -- for
any labelled instance with visible fraction ``v``, the Review_Simulator
(:class:`~week1_harness.steps.review.ReviewStep`) retains its label if and only
if ``v >= 1/3``.

This exercises the visibility rule in isolation, so Film/Extract/Auto-label are
*not* run. Instead a minimal Auto-label payload is crafted directly: for a list
of visible-fraction values, one ground-truth
:class:`~week1_harness.datamodel.Instance` and one matching
:class:`~week1_harness.datamodel.Mask` are built per value (unique instance ids,
a fixed 2x2 square geometry, an all-true 2x2 raster, a known class id). Review
reads that payload from ``ctx.artifacts["Auto-label"]`` and emits one
:class:`~week1_harness.datamodel.ReviewedLabel` per mask, which lets the test
assert the retain/remove decision against each instance's visible fraction
directly and cheaply.

The threshold is stated exactly as one third (``1.0 / 3.0``) rather than a
rounded decimal, so the boundary at exactly one third is inclusive: an instance
exactly one third visible is retained (``>=``). The property is complemented by
explicit boundary examples just below (0.32 -> removed) and just above
(0.34 -> retained) the cutoff, at the cutoff itself, and at the two extremes.

**Validates: Requirements 9.2, 9.3**

_Requirements: 9.2, 9.3_
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness.context import RunContext
from week1_harness.datamodel import Instance, Mask
from week1_harness.steps.review import ReviewStep
from week1_harness.synth.items import CLASS_IDS

# The retention threshold, written as one third exactly (design Property 7,
# R9.2/R9.3) so that an instance exactly one third visible is retained (>=).
THRESHOLD: float = 1.0 / 3.0


def _build_auto_label_payload(visible_fractions: list[float]) -> dict:
    """Craft a minimal ``Auto-label`` payload for a list of visible fractions.

    One :class:`~week1_harness.datamodel.Instance` and one matching
    :class:`~week1_harness.datamodel.Mask` are built per value, with unique
    instance ids (``"I0"``, ``"I1"``, ...) so Review's ``instance_id -> Instance``
    lookup is one-to-one. Every other field is a fixed, valid stand-in: a 2x2
    square polygon/bbox, an all-true ``2x2`` boolean raster, a real practice-item
    class id (so Review's class-integrity assertion holds), layout ``"L0"`` and
    session ``"S2"``. Only ``visible_fraction`` varies -- it is the single input
    the visibility rule depends on.
    """
    instances: list[Instance] = []
    masks: list[Mask] = []
    for i, v in enumerate(visible_fractions):
        instance_id = f"I{i}"
        class_id = CLASS_IDS[i % len(CLASS_IDS)]
        instances.append(
            Instance(
                instance_id=instance_id,
                class_id=class_id,
                polygon=[(0, 0), (2, 0), (2, 2), (0, 2)],
                bbox=(0, 0, 2, 2),
                visible_fraction=v,
                layout_id="L0",
                session="S2",
            )
        )
        masks.append(
            Mask(instance_id=instance_id, class_id=class_id, raster=np.ones((2, 2), bool))
        )
    return {
        "masks": masks,
        "all_instances": instances,
        "class_ids": list(CLASS_IDS),
        "instances_per_class": {},
        "instances_by_layout": {},
        "training_images": [],
        "golden_images": [],
    }


def _reviewed_labels_by_id(visible_fractions: list[float]) -> dict:
    """Run :class:`ReviewStep` over the crafted payload; map ``instance_id -> label``.

    Builds a seeded :class:`~week1_harness.context.RunContext`, installs the
    crafted Auto-label payload under ``ctx.artifacts["Auto-label"]`` (the key the
    pipeline runner would use), runs Review, and returns its reviewed labels
    keyed by instance id for direct per-instance assertions.
    """
    ctx = RunContext(seed=0, out_dir=None, capabilities=None)
    ctx.artifacts["Auto-label"] = _build_auto_label_payload(visible_fractions)
    result = ReviewStep().run(ctx)
    return {label.instance_id: label for label in result.payload["reviewed_labels"]}


@given(
    visible_fractions=st.lists(
        st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=25
    )
)
@settings(deadline=None)
def test_review_retains_iff_visible_fraction_at_least_one_third(
    visible_fractions: list[float],
) -> None:
    """Property 7: a label is retained iff its instance is at least 1/3 visible.

    For any list of visible fractions in ``[0, 1]``, Review emits exactly one
    reviewed label per instance and ``label.retained`` equals ``v >= 1/3`` for
    every instance -- the retain/remove rule of R9.2 (retain when at least one
    third visible) and R9.3 (remove when less than one third visible).

    **Validates: Requirements 9.2, 9.3**
    """
    labels_by_id = _reviewed_labels_by_id(visible_fractions)

    # One reviewed label per crafted instance/mask, keyed 1:1 by instance id.
    assert len(labels_by_id) == len(visible_fractions)

    for i, v in enumerate(visible_fractions):
        label = labels_by_id[f"I{i}"]
        assert label.retained == (v >= THRESHOLD), (
            f"visible_fraction={v!r}: retained={label.retained}, "
            f"expected {v >= THRESHOLD} (retain iff v >= 1/3; R9.2/R9.3)"
        )


# Explicit boundary examples (not random) pinning the exact one-third cutoff:
# just below the cutoff is removed, just above is retained, exactly one third is
# retained (>= is inclusive), and the two extremes 0.0 (removed) / 1.0 (retained).
_BOUNDARY_CASES = [
    (0.32, False),
    (0.34, True),
    (1.0 / 3.0, True),
    (0.0, False),
    (1.0, True),
]


@pytest.mark.parametrize(("visible_fraction", "expected_retained"), _BOUNDARY_CASES)
def test_review_visibility_boundary_examples(
    visible_fraction: float, expected_retained: bool
) -> None:
    """Boundary examples pin the exact 1/3 cutoff (R9.2/R9.3, design Property 7).

    ``v = 0.32`` is removed and ``v = 0.34`` is retained (the cutoff sits between
    them); ``v = 1/3`` exactly is retained because the rule is ``>=``; ``v = 0.0``
    is removed and ``v = 1.0`` is retained.

    **Validates: Requirements 9.2, 9.3**
    """
    labels_by_id = _reviewed_labels_by_id([visible_fraction])
    label = labels_by_id["I0"]
    assert label.retained is expected_retained, (
        f"visible_fraction={visible_fraction!r}: retained={label.retained}, "
        f"expected {expected_retained} (retain iff v >= 1/3; R9.2/R9.3)"
    )
