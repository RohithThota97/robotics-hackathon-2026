"""Property-based test for Step 5 -- Review second-reviewer 10% flagging (Task 10.6).

Feature: week1-data-loop-harness, Property 8: Second-reviewer 10% flagging --
for any reviewed label set of size ``n`` (the number of RETAINED labels), the
number of labels flagged as checked by a second reviewer equals
``round(0.10 * n)``, the flagged subset is drawn only from the retained labels
(a removed label is never second-checked), and the selected subset is a
deterministic function of the run seed.

The test drives :class:`~week1_harness.steps.review.ReviewStep` directly against
a hand-crafted Auto-label payload (Film / Extract / Auto-label are *not* run):
for each instance it builds a matching :class:`~week1_harness.datamodel.Instance`
(whose ``visible_fraction`` decides retention via the ``>= 1/3`` rule) and
:class:`~week1_harness.datamodel.Mask`, places them under
``ctx.artifacts["Auto-label"]`` exactly as the pipeline runner would, and reads
back ``ReviewStep`` 's payload (``reviewed_labels``, ``retained_count``,
``second_check_count``).

Three facets of the property are exercised:

* **All retained** -- with every instance fully visible (``vf = 1.0``) the
  retained count equals ``n``, so the flagged count must be exactly
  ``round(0.10 * n)`` and equal to the payload's ``second_check_count``.
* **Mixed (some removed)** -- with some instances below the ``1/3`` visibility
  threshold, the flagged count is ``round(0.10 * retained_count)`` computed from
  the RETAINED labels only (not the total), and no removed label is ever
  flagged.
* **Determinism** -- the same seed yields the identical set of flagged
  instance ids across two independent runs; and the selection genuinely depends
  on the seed (different seeds do not all collapse to one fixed subset).

**Validates: Requirements 9.5**

_Requirements: 9.5_
"""

from __future__ import annotations

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from week1_harness.context import RunContext
from week1_harness.datamodel import Instance, Mask
from week1_harness.steps.review import ReviewStep
from week1_harness.synth.items import CLASS_IDS

# The spec's visibility rule (R9.2/R9.3): a label is retained iff at least one
# third of the instance is visible. Written as 1/3 (not 0.33) so the boundary at
# exactly one third is retained -- identical to the value ReviewStep uses.
_THRESHOLD: float = 1.0 / 3.0

# The spec's second-reviewer fraction (R9.5): 10 percent of the RETAINED labels.
# Hard-coded here (rather than imported from the implementation) so the test
# validates the 0.10 the specification mandates.
_SECOND_CHECK_FRACTION: float = 0.10

# Non-negative seeds only: RunContext feeds the seed into numpy SeedSequence,
# which requires non-negative entropy. The upper bound keeps np.random.seed and
# random.seed happy while still spanning a large space.
_seeds = st.integers(min_value=0, max_value=2**31 - 1)

# A pool of visibility fractions with members clearly on each side of the 1/3
# threshold plus the exact boundary 1/3 (retained, inclusive) and 0.32 (removed),
# so a sampled list produces an unambiguous, well-separated retained/removed mix.
_VISIBILITY_POOL = [0.0, 0.1, 0.2, 0.3, 0.32, _THRESHOLD, 0.4, 0.5, 0.75, 0.9, 1.0]


def _expected_second_check_count(retained_count: int) -> int:
    """The spec's flagged-count target: ``round(0.10 * retained_count)``.

    Uses Python's built-in :func:`round` (round-half-to-even) over the exact
    expression the specification states, so it tracks the reference formula
    rather than any particular implementation constant.
    """
    return round(_SECOND_CHECK_FRACTION * retained_count)


def _build_auto_label_payload(visibilities: list[float]) -> dict:
    """Craft a minimal Auto-label payload for a list of per-instance visibilities.

    One :class:`~week1_harness.datamodel.Instance` and one matching
    :class:`~week1_harness.datamodel.Mask` are built per entry, sharing the id
    ``f"i{k}"`` and cycling through the six practice-item classes. The payload
    mirrors the keys :class:`~week1_harness.steps.review.ReviewStep` reads from
    ``ctx.artifacts["Auto-label"]``.
    """
    masks: list[Mask] = []
    all_instances: list[Instance] = []
    for k, v in enumerate(visibilities):
        instance_id = f"i{k}"
        class_id = CLASS_IDS[k % 6]
        all_instances.append(
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
            Mask(
                instance_id=instance_id,
                class_id=class_id,
                raster=np.ones((2, 2), bool),
            )
        )
    return {
        "masks": masks,
        "all_instances": all_instances,
        "class_ids": list(CLASS_IDS),
        "instances_per_class": {},
        "instances_by_layout": {},
        "training_images": [],
        "golden_images": [],
    }


def _run_review(seed: int, visibilities: list[float]) -> dict:
    """Run :class:`ReviewStep` on a seeded context + crafted payload; return payload.

    Builds a :class:`~week1_harness.context.RunContext` with the given seed (no
    output dir, no capabilities are needed by Review), stashes the crafted
    Auto-label payload under its canonical artifact key, and returns the
    ``ReviewStep`` result payload.
    """
    ctx = RunContext(seed=seed, out_dir=None, capabilities=None)
    ctx.artifacts["Auto-label"] = _build_auto_label_payload(visibilities)
    return ReviewStep().run(ctx).payload


def _flagged_ids(reviewed_labels) -> set[str]:
    """The set of instance ids flagged as second-reviewer-checked."""
    return {label.instance_id for label in reviewed_labels if label.second_check}


@given(n=st.integers(min_value=0, max_value=60), seed=_seeds)
@settings(max_examples=200, deadline=None)
def test_flagged_count_is_ten_percent_when_all_retained(n: int, seed: int) -> None:
    """Property 8: with every label retained, flagged count == round(0.10 * n).

    All ``n`` instances are fully visible (``vf = 1.0``) so ``retained_count``
    equals ``n``; the number of ``second_check`` labels must then be exactly
    ``round(0.10 * n)`` and agree with the payload's ``second_check_count``.
    """
    payload = _run_review(seed, [1.0] * n)
    reviewed = payload["reviewed_labels"]
    flagged = _flagged_ids(reviewed)
    expected = _expected_second_check_count(n)

    # Every crafted instance becomes exactly one reviewed label, all retained.
    assert len(reviewed) == n
    assert payload["retained_count"] == n, (
        f"expected all {n} fully-visible labels retained, "
        f"got retained_count={payload['retained_count']}"
    )
    assert all(label.retained for label in reviewed)
    assert payload["removed_count"] == 0

    # R9.5: exactly round(10% * retained) labels are flagged, and the payload's
    # own count agrees with that target.
    assert payload["second_check_count"] == expected, (
        f"payload second_check_count={payload['second_check_count']} "
        f"!= round(0.10 * {n}) = {expected}"
    )
    assert len(flagged) == expected, (
        f"flagged {len(flagged)} of {n} retained labels, expected {expected} "
        f"(= round(0.10 * {n}))"
    )
    # The flagged set is a subset of the (all-retained) instance ids.
    assert flagged <= {label.instance_id for label in reviewed}


@given(
    visibilities=st.lists(st.sampled_from(_VISIBILITY_POOL), max_size=80),
    seed=_seeds,
)
@settings(max_examples=200, deadline=None)
# A deliberately mixed example: 30 retained + 10 removed (interleaved), so the
# retained-based target round(0.10 * 30) = 3 differs from the total-based
# round(0.10 * 40) = 4 -- proving the count is driven by RETAINED labels only.
@example(visibilities=[1.0, 0.0] * 10 + [1.0] * 20, seed=12345)
def test_flagged_count_uses_retained_not_total_with_removed(
    visibilities: list[float], seed: int
) -> None:
    """Property 8: flagged count == round(0.10 * retained_count), never over removed.

    With a mix of visible (``>= 1/3``, retained) and occluded (``< 1/3``,
    removed) instances, the flagged count is computed from the RETAINED labels
    only -- not the total label count -- and no removed label is ever flagged.
    """
    payload = _run_review(seed, visibilities)
    reviewed = payload["reviewed_labels"]
    retained = [label for label in reviewed if label.retained]
    removed = [label for label in reviewed if not label.retained]
    flagged = [label for label in reviewed if label.second_check]

    # Retention follows the spec's >= 1/3 rule applied to our crafted inputs, so
    # retained_count reflects RETAINED labels (which can be fewer than the total).
    expected_retained = sum(1 for v in visibilities if v >= _THRESHOLD)
    assert len(reviewed) == len(visibilities)
    assert payload["retained_count"] == expected_retained == len(retained), (
        f"retained_count={payload['retained_count']} disagrees with "
        f"expected {expected_retained} / actual retained {len(retained)}"
    )
    assert payload["removed_count"] == len(removed) == len(visibilities) - expected_retained

    # R9.5: the flagged count is round(10% * RETAINED), independent of the total.
    expected_flagged = _expected_second_check_count(expected_retained)
    assert payload["second_check_count"] == expected_flagged, (
        f"payload second_check_count={payload['second_check_count']} "
        f"!= round(0.10 * {expected_retained}) = {expected_flagged}"
    )
    assert len(flagged) == expected_flagged, (
        f"flagged {len(flagged)} labels; expected {expected_flagged} "
        f"(= round(0.10 * retained {expected_retained}), total was {len(visibilities)})"
    )

    # No removed label is ever second-checked, and every flagged label is retained.
    assert all(not label.second_check for label in removed), (
        "a removed label was flagged for a second review (R9.5)"
    )
    assert all(label.retained for label in flagged), (
        "a second-checked label was not retained (R9.5)"
    )


@given(n=st.integers(min_value=0, max_value=60), seed=_seeds)
@settings(max_examples=150, deadline=None)
def test_flagged_selection_is_deterministic_for_a_seed(n: int, seed: int) -> None:
    """Property 8: the flagged subset is a deterministic function of the seed.

    Two independent runs with the same seed and the same input select the
    identical set of flagged instance ids.
    """
    first = _run_review(seed, [1.0] * n)
    second = _run_review(seed, [1.0] * n)

    assert _flagged_ids(first["reviewed_labels"]) == _flagged_ids(
        second["reviewed_labels"]
    ), "same seed produced a different second-check selection across two runs"
    assert first["second_check_count"] == second["second_check_count"]


def test_flagged_selection_depends_on_the_seed() -> None:
    """Property 8: the selection genuinely varies with the seed (not a fixed subset).

    With 50 retained labels and a non-trivial target of ``round(0.10 * 50) = 5``
    flagged, scanning a fixed range of seeds must yield more than one distinct
    flagged subset -- otherwise the selection would be ignoring the seed. The
    seeds are fixed, so this check is itself deterministic.
    """
    n = 50
    expected = _expected_second_check_count(n)
    assert 0 < expected < n, "choose an n giving a non-trivial flagged count"

    selections: set[frozenset[str]] = set()
    for seed in range(24):
        payload = _run_review(seed, [1.0] * n)
        flagged = _flagged_ids(payload["reviewed_labels"])
        assert len(flagged) == expected, (
            f"seed {seed}: flagged {len(flagged)} labels, expected {expected}"
        )
        selections.add(frozenset(flagged))

    assert len(selections) > 1, (
        "the second-check selection did not vary across seeds; "
        "it may be ignoring the run seed (R9.5)"
    )
