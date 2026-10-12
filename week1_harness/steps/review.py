"""Step 5 -- Review: the label-review correction step (Requirement 9).

:class:`ReviewStep` is the fifth Data Loop step. On site this is where a human
reviewer would open the auto-labelled masks in a review GUI and correct them.
The harness has no GUI (that is an explicit non-goal), so the review is encoded
as an **automated data transformation** over the Auto-label output (R9.1): it
applies the two label-quality rules the reviewer would apply by hand -- the
visibility rule and the second-reviewer spot check -- deterministically. Because
that transform is genuine local-CPU work (it reads the auto-label masks and the
ground-truth visibility and derives a reviewed label set), its timing cell is a
``REAL`` :class:`~week1_harness.markers.MarkedValue`; nothing here is stubbed or
hardware-blocked (R4.1).

What ``run`` does
-----------------

1. **Reads Auto-label's output** from ``ctx.artifacts["Auto-label"]`` (the
   pipeline runner stores each step's payload under the step name). From it
   Review uses the ``masks`` (one per unique labelled training instance) and the
   ground-truth ``all_instances`` list, from which it builds an
   ``instance_id -> Instance`` map so each masked instance's
   ``visible_fraction`` and ``class_id`` can be looked up.
2. **Produces one** :class:`~week1_harness.datamodel.ReviewedLabel` **per mask**
   -- i.e. one per unique training instance (R9.1) -- iterating the masks in
   Auto-label's fixed, deterministic order.
3. **Applies the visibility rule (R9.2/R9.3):** a label is ``retained`` iff at
   least one third of its instance is visible
   (``visible_fraction >= 1/3``); an instance with less than one third visible
   is removed (``retained is False``).
4. **Carries exactly one class per label (R9.4):** each reviewed label keeps its
   instance's single ``class_id``. No class is merged or invented, so the
   look-alike pair (``screwdriver_small`` / ``screwdriver_large``) stays two
   distinct classes end to end.
5. **Flags the second-reviewer spot check (R9.5):** exactly
   ``round(0.10 * retained_count)`` of the **retained** labels are flagged
   ``second_check``, selected deterministically from ``ctx.rng_for(5)`` by
   permuting the retained labels' positions and taking the first ``k`` (so a
   removed label is never second-checked).
6. **Returns a** :class:`~week1_harness.datamodel.StepResult` with a ``REAL``
   time cell, the "How" text, and a payload carrying the reviewed label set, the
   retained/removed/second-check tallies, and the pass-through artifacts
   (``class_ids``, ``instances_per_class``, both image sets,
   ``instances_by_layout``, ``all_instances``) that Train, Score, and the
   reporters consume.

Determinism (R3.3)
------------------

The only randomness is the second-reviewer selection, and it draws solely from
``ctx.rng_for(5)`` -- an independent sub-stream keyed by the run seed and this
step's number. Given Auto-label's (seeded, deterministic) masks, the retained
set and its ordering are fixed, so permuting the retained positions with that
generator and taking the first ``k`` yields the same flagged instances on every
run with the same seed.

Scope (task 10.4 only): this module transforms the auto-label output into the
reviewed label set and hands it off. It does not train, score, or run the live
test.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from week1_harness.datamodel import ReviewedLabel, StepResult
from week1_harness.markers import real

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.context import RunContext
    from week1_harness.datamodel import Instance, Mask

__all__ = ["ReviewStep"]


# The "How" column text for the Timing_Table (R13.2); Review does real CPU work
# (an automated label transform), so the time cell stays REAL.
_HOW = "Visibility >=1/3 rule; 10% second-reviewer check"

# R9.2/R9.3: retain a label iff at least one third of the instance is visible.
# Expressed as 1/3 (not 0.33) so the boundary at exactly one third is retained.
VISIBILITY_THRESHOLD: float = 1.0 / 3.0

# R9.5: the fraction of RETAINED labels flagged as checked by a second reviewer.
SECOND_CHECK_FRACTION: float = 0.10


class ReviewStep:
    """Data Loop step 5 -- the automated label-review transform (R9).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that reads Auto-label's masks
    from the shared artifact store, produces one
    :class:`~week1_harness.datamodel.ReviewedLabel` per unique training instance
    (retained iff ``visible_fraction >= 1/3``, one class each with the look-alike
    pair kept separate, and 10% of retained labels flagged for a second review),
    and returns a ``REAL``-timed :class:`~week1_harness.datamodel.StepResult`.
    """

    number: int = 5
    name: str = "Review"

    def run(self, ctx: RunContext) -> StepResult:
        """Transform Auto-label's masks into the reviewed label set (R9).

        Times the real transform with :func:`time.perf_counter` (genuine
        local-CPU work -> ``REAL`` time cell, R4.1), reads Auto-label's payload
        from ``ctx.artifacts["Auto-label"]``, applies the ``>= 1/3`` visibility
        rule (R9.2/R9.3) to produce one reviewed label per mask (R9.1) with
        exactly one class each (R9.4), and flags ``round(10% * retained)`` of the
        retained labels as second-reviewer-checked, selected deterministically
        from ``ctx.rng_for(5)`` (R9.5).
        """
        start = time.perf_counter()

        auto_label = ctx.artifacts["Auto-label"]
        masks: list[Mask] = auto_label["masks"]
        all_instances: list[Instance] = auto_label["all_instances"]
        class_ids: list[str] = list(auto_label["class_ids"])
        instances_per_class = auto_label["instances_per_class"]
        instances_by_layout = auto_label["instances_by_layout"]
        training_images = auto_label["training_images"]
        golden_images = auto_label["golden_images"]

        # instance_id -> Instance, so each mask's visible_fraction and class are
        # read from the single ground-truth source (R9.2/R9.3).
        instance_by_id = {inst.instance_id: inst for inst in all_instances}

        # First pass: apply the visibility rule in the masks' fixed, deterministic
        # order. One mask -> one reviewed instance (R9.1); retained iff >= 1/3
        # visible (R9.2/R9.3); the instance's single class_id is carried through
        # unchanged, so no class is merged or invented (R9.4).
        decisions: list[tuple[str, str, bool]] = []  # (instance_id, class_id, retained)
        for mask in masks:
            instance = instance_by_id[mask.instance_id]
            retained = instance.visible_fraction >= VISIBILITY_THRESHOLD
            decisions.append((mask.instance_id, instance.class_id, retained))

        # Second-reviewer spot check (R9.5): flag exactly round(10% * retained) of
        # the RETAINED labels. Selection is deterministic -- permute the retained
        # positions with rng_for(5) and keep the first k -- and never touches a
        # removed label, so a removed label is never second-checked.
        retained_positions = [i for i, (_, _, retained) in enumerate(decisions) if retained]
        retained_count = len(retained_positions)
        second_check_count = round(SECOND_CHECK_FRACTION * retained_count)
        flagged: set[int] = set()
        if retained_positions and second_check_count > 0:
            rng = ctx.rng_for(self.number)
            permuted = rng.permutation(retained_positions)
            flagged = {int(position) for position in permuted[:second_check_count]}

        # Second pass: build the immutable reviewed labels with their final flags.
        reviewed_labels: list[ReviewedLabel] = [
            ReviewedLabel(
                instance_id=instance_id,
                class_id=class_id,
                retained=retained,
                second_check=index in flagged,
            )
            for index, (instance_id, class_id, retained) in enumerate(decisions)
        ]

        removed_count = len(reviewed_labels) - retained_count

        elapsed = time.perf_counter() - start

        # R9.4: every retained class stays one of the six item classes -- no class
        # is merged or invented -- so the look-alike pair remains two distinct
        # classes. R9.5: exactly round(10% * retained) labels are second-checked.
        assert all(label.class_id in class_ids for label in reviewed_labels), (
            "reviewed label carries an unknown class id (R9.4)"
        )
        assert (
            sum(label.second_check for label in reviewed_labels) == second_check_count
        ), "second-reviewer flag count does not match round(10% * retained) (R9.5)"

        how = (
            f"{_HOW}: {retained_count} kept, {removed_count} removed, "
            f"{second_check_count} second-checked"
        )

        # Train, Score, and the reporters need the reviewed labels and the review
        # tallies, plus the pass-through ground truth / image sets / per-class
        # counts they continue to consume downstream.
        payload = {
            "reviewed_labels": reviewed_labels,
            "retained_count": retained_count,
            "removed_count": removed_count,
            "second_check_count": second_check_count,
            "class_ids": class_ids,
            "instances_per_class": instances_per_class,
            "training_images": training_images,
            "golden_images": golden_images,
            "instances_by_layout": instances_by_layout,
            "all_instances": all_instances,
        }

        # Real local-CPU transform: the time cell is REAL and carries no reason,
        # and Review emits no placeholder/estimate notes (R4.1, R4.6).
        return StepResult(
            number=self.number,
            name=self.name,
            how=how,
            time=real(elapsed, unit="s"),
            payload=payload,
            notes=[],
        )
