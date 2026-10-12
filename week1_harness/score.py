"""Step 7 -- Score: golden-set metrics (Requirement 11).

:class:`ScoreStep` is the seventh Data Loop step. On site this is where the
freshly trained model would be evaluated on the held-out golden layouts to
produce mAP50, per-item recall, and look-alike confusion. The harness never
trains a real model (Step 6 is stubbed), so there is nothing to evaluate for
real: the Scorer **fabricates** those metrics from the run seed and labels every
one of them a ``PLACEHOLDER`` with the reason ``"training stubbed"`` (R11.2).
Because the numbers are fabricated rather than measured on the local CPU, even
the step's *time* cell is a ``PLACEHOLDER`` "training stubbed" -- timing the
fabrication would mislead (design per-step marker table) -- so, unlike the real
CPU steps, this step takes no ``time.perf_counter`` reading.

What is honest here is the **input restriction** (R11.1): the Scorer evaluates
on the held-out Golden Layouts *only* and never on the training layouts. The
harness gathers the golden-session inputs (the held-out golden images and the
golden layouts from ``instances_by_layout``), records exactly which golden
layouts were scored, and asserts no training layout leaks into the evaluation.

What ``run`` does
-----------------

1. **Reads Train's output** from ``ctx.artifacts["Train"]`` (the pipeline runner
   stores each step's payload under the step name). From it Score uses the six
   ``class_ids`` (to emit one recall per item), the held-out ``golden_images``
   and the ground-truth ``instances_by_layout`` (to restrict evaluation to the
   golden layouts), and passes ``all_instances`` and ``instances_per_class``
   through for Live-test / the reporters.
2. **Restricts evaluation inputs to the held-out golden layouts (R11.1).** It
   identifies the golden layouts (the layouts whose instances carry
   ``session == "golden"``) from ``instances_by_layout``, confines the held-out
   image inputs to those layouts, and records the golden layouts actually
   covered by that held-out image set as ``scored_golden_layouts``. It asserts
   every held-out image comes from a golden layout, so no training layout can
   enter the evaluation.
3. **Emits golden-set metrics (R11.2) as a**
   :class:`~week1_harness.datamodel.Metrics`: ``map50``, one ``recall_per_item``
   entry for **each** of the six ``class_ids`` (R11.4), and
   ``lookalike_confusion`` -- each a ``PLACEHOLDER`` ``MarkedValue`` with the
   reason ``"training stubbed"``. The underlying numbers are drawn
   deterministically from ``ctx.rng_for(7)`` (R11.3 / R3.4), so a given seed
   always yields the same clearly-marked stand-ins. They are *not* real accuracy
   numbers -- no trained model exists to measure.
4. **Returns a** :class:`~week1_harness.datamodel.StepResult` with a
   ``PLACEHOLDER`` "training stubbed" time cell, the "How" text, a payload
   carrying the golden metrics and the scored golden layouts plus the
   pass-through artifacts Live-test / the reporters consume, and ``notes``
   listing every metric placeholder (``map50``, the six recalls, and
   ``lookalike_confusion``) so the Dataset_Card known-gaps scan over
   ``StepResult.notes`` (R14.4) catches them.

Determinism (R3.4 / R11.3)
--------------------------

All metric randomness is drawn from a single generator, ``ctx.rng_for(7)`` -- an
independent sub-stream keyed by the run seed and this step's number -- obtained
once and drawn in a fixed order (``map50``, then one recall per item in
``class_ids`` order, then ``lookalike_confusion``). So the same seed yields
identical metric values on every run, and changing another step's logic cannot
shift this step's stream. The golden-layout selection draws no randomness: it is
a deterministic function of Train's (seeded, deterministic) ground truth.

Scope (task 11.3 only): this module scores the held-out golden set and hands off
the metrics and scored golden layouts. It does not run the live test (Step 8) or
render the reporters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from week1_harness.datamodel import Metrics, StepResult
from week1_harness.markers import placeholder

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    import numpy as np

    from week1_harness.context import RunContext
    from week1_harness.datamodel import ExtractedImage, Instance
    from week1_harness.markers import MarkedValue

__all__ = ["ScoreStep"]


# The "How" column text for the Timing_Table (R13.2). Scoring fabricates numbers
# from the seed, so the time cell is a PLACEHOLDER rather than a REAL measurement
# (design per-step marker table).
_HOW = "Golden-set metrics (stubbed, seed-derived)"

# The honesty-marker reason shared by the metrics and the time cell (R11.2): the
# numbers are not real because training was stubbed (Step 6 did no real training).
_STUB_REASON = "training stubbed"

# The time-cell display value; it is a PLACEHOLDER, so it reads "n/a" rather than
# a measured duration (there is no real scoring computation to time).
_TIME_PLACEHOLDER_VALUE = "n/a"

# The session tag marking a held-out golden layout (R5.4 / R6.4 / R11.1).
_GOLDEN_SESSION = "golden"

# Placeholder metric ranges. These produce plausible-looking BUT fabricated
# numbers; every one is wrapped PLACEHOLDER with the stub reason, so they can
# never be mistaken for a real measurement (R11.2). Values are drawn from
# rng_for(7) and rounded for clean, reproducible output. The ranges mirror the
# Train stub's so the fabricated numbers read consistently across the two steps.
_MAP50_RANGE: tuple[float, float] = (0.40, 0.85)
_RECALL_RANGE: tuple[float, float] = (0.35, 0.95)
_CONFUSION_RANGE: tuple[float, float] = (0.05, 0.45)
_METRIC_DIGITS: int = 3


def _draw_metric(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw one rounded placeholder metric in ``[lo, hi]`` from ``rng`` (R11.3)."""
    return round(float(rng.uniform(lo, hi)), _METRIC_DIGITS)


class ScoreStep:
    """Data Loop step 7 -- the golden-set scorer (R11).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that reads Train's payload,
    restricts evaluation to the held-out golden layouts (R11.1), emits
    ``PLACEHOLDER`` "training stubbed" mAP50 / per-item recall (one per item) /
    look-alike confusion metrics derived deterministically from
    ``ctx.rng_for(7)`` (R11.2-R11.4), and returns a ``PLACEHOLDER``-timed
    :class:`~week1_harness.datamodel.StepResult`.
    """

    number: int = 7
    name: str = "Score"

    def run(self, ctx: RunContext) -> StepResult:
        """Score the held-out golden set and return the Score result (R11).

        Reads Train's payload from ``ctx.artifacts["Train"]``, restricts the
        evaluation inputs to the held-out golden layouts (R11.1), fabricates
        ``PLACEHOLDER`` "training stubbed" mAP50, one recall per each of the six
        items (R11.4), and look-alike confusion -- all drawn deterministically
        from ``ctx.rng_for(7)`` (R11.2/R11.3) -- and returns the
        :class:`~week1_harness.datamodel.StepResult` with a ``PLACEHOLDER``
        "training stubbed" time cell.
        """
        train = ctx.artifacts["Train"]
        class_ids: list[str] = list(train["class_ids"])
        golden_images: list[ExtractedImage] = train["golden_images"]
        instances_by_layout: dict[str, list[Instance]] = train["instances_by_layout"]
        all_instances = train["all_instances"]
        instances_per_class = train["instances_per_class"]

        # R11.1: restrict evaluation to the HELD-OUT golden layouts only. The
        # golden layouts are those whose ground-truth instances carry
        # session == "golden"; the training layouts (S1/S2) are excluded. We
        # record exactly which golden layouts the held-out image set covers.
        scored_golden_layouts = self._scored_golden_layouts(
            golden_images, instances_by_layout
        )

        # Single generator for the whole step, drawn in a fixed order (map50,
        # then one recall per item in class_ids order, then look-alike
        # confusion) so the metric values are reproducible for a given seed
        # (R11.3 / R3.4).
        rng = ctx.rng_for(self.number)

        # Golden-set metrics (R11.2): each a PLACEHOLDER "training stubbed"
        # MarkedValue whose number is seed-derived. One recall entry per EACH of
        # the six items (R11.4).
        map50 = placeholder(_draw_metric(rng, *_MAP50_RANGE), reason=_STUB_REASON)
        recall_per_item: dict[str, MarkedValue] = {
            class_id: placeholder(_draw_metric(rng, *_RECALL_RANGE), reason=_STUB_REASON)
            for class_id in class_ids
        }
        lookalike_confusion = placeholder(
            _draw_metric(rng, *_CONFUSION_RANGE), reason=_STUB_REASON
        )
        metrics = Metrics(
            map50=map50,
            recall_per_item=recall_per_item,
            lookalike_confusion=lookalike_confusion,
        )

        # notes: every metric placeholder (map50, each per-item recall, and the
        # look-alike confusion) so the Dataset_Card known-gaps scan over
        # StepResult.notes catches them (R14.4). The time-cell PLACEHOLDER
        # already lives on StepResult.time.
        notes: list[MarkedValue] = [
            map50,
            *recall_per_item.values(),
            lookalike_confusion,
        ]

        how = f"{_HOW}: {len(scored_golden_layouts)} golden layouts scored"

        # Hand Live-test and the reporters the golden metrics and the scored
        # golden layouts, plus the ground truth / golden set / counts they
        # continue to consume. The pass-through keys keep the artifact chain
        # intact for Step 8 and both reporters.
        payload = {
            "metrics": metrics,
            "scored_golden_layouts": scored_golden_layouts,
            "golden_images": golden_images,
            "instances_by_layout": instances_by_layout,
            "class_ids": class_ids,
            "all_instances": all_instances,
            "instances_per_class": instances_per_class,
        }

        # The metrics are fabricated from the seed, not measured on the local
        # CPU, so even the time cell is a PLACEHOLDER "training stubbed" (design
        # per-step marker table) -- there is no real scoring computation to time.
        return StepResult(
            number=self.number,
            name=self.name,
            how=how,
            time=placeholder(_TIME_PLACEHOLDER_VALUE, reason=_STUB_REASON),
            payload=payload,
            notes=notes,
        )

    def _scored_golden_layouts(
        self,
        golden_images: list[ExtractedImage],
        instances_by_layout: dict[str, list[Instance]],
    ) -> list[str]:
        """Return the held-out golden layouts the evaluation scored (R11.1).

        The golden layouts are those whose ground-truth instances carry
        ``session == "golden"`` in ``instances_by_layout``; the training (S1/S2)
        layouts are excluded. Evaluation inputs are restricted to the held-out
        ``golden_images``, and this returns -- sorted for diff-stable output --
        the golden layouts actually covered by that held-out image set.

        It asserts the restriction holds: every held-out image comes from a
        golden layout, so no training layout can leak into the evaluation (R11.1).
        """
        golden_layout_ids = {
            layout_id
            for layout_id, instances in instances_by_layout.items()
            if instances and all(inst.session == _GOLDEN_SESSION for inst in instances)
        }

        # The held-out image inputs' source layouts; R11.1 requires every one of
        # them to be a golden layout (never a training layout).
        image_layout_ids = {image.source_layout_id for image in golden_images}
        assert image_layout_ids <= golden_layout_ids, (
            "held-out evaluation image references a non-golden (training) layout; "
            "the Scorer must evaluate on golden layouts only (R11.1)"
        )

        # The golden layouts actually covered by the held-out image set, sorted.
        return sorted(golden_layout_ids & image_layout_ids)
