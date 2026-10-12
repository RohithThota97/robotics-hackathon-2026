"""Step 8 -- Test live: the simulated live-test step (Requirement 12).

:class:`LiveTestStep` is the eighth and final Data Loop step. On site this is
where the freshly trained model would run against a live webcam on the GPU
laptop and report a real-time frame rate over a reshuffled golden layout. The
harness runs CPU-only with no webcam and no NVIDIA GPU, so the live-test step is
**simulated**: it emits a plausible live frame rate and references a reshuffled
golden layout containing all six practice items -- so the final stage of the
loop still produces the expected output field without any hardware.

Because the frame rate is not a real local-CPU measurement (it needs the absent
webcam + GPU), the honesty markers do the heavy lifting (Requirement 4):

* the **frame rate** is an ``ESTIMATE`` with reason ``"webcam and GPU absent"``
  (R12.1), and this ``MarkedValue`` *is* the ``StepResult`` time cell;
* the fps number is drawn deterministically from ``ctx.rng_for(8)`` (R12.2), so
  a given seed always yields the same (clearly-marked) rate.

What ``run`` does
-----------------

1. **Reads Score's output** from ``ctx.artifacts["Score"]`` (the pipeline runner
   stores each step's payload under the step name). From it Live-test uses the
   six ``class_ids``, the per-layout ground-truth instances
   (``instances_by_layout``), and the held-out ``golden_images``; it passes
   those three through for the reporters.
2. **Estimates a live frame rate** (R12.1) as an ``ESTIMATE`` in fps naming the
   absent hardware, the value drawn deterministically from ``ctx.rng_for(8)``
   (R12.2) in a plausible webcam range.
3. **Picks a golden layout containing all six items** (R12.3): among
   ``instances_by_layout`` it selects a layout whose instances are all from the
   ``"golden"`` session and whose class set is exactly the six ``class_ids``,
   then **verifies** the referenced layout really does contain all six items.
4. **Reshuffles that layout's items** deterministically from ``ctx.rng_for(8)``
   -- a permutation of the six class ids -- and records the referenced golden
   layout id plus the reshuffled order in the payload (R12.3).
5. **Returns a** :class:`~week1_harness.datamodel.StepResult` with an
   ``ESTIMATE`` time cell (the fps), the "How" text, and a payload carrying the
   fps, the referenced golden layout id, the reshuffled item order, and the
   pass-through artifacts the reporters consume. ``notes`` is empty: the only
   non-real value this step emits is the fps, which already lives on
   ``StepResult.time`` (so the Dataset_Card known-gaps scan over time cells +
   notes still catches it without double-counting).

Determinism (R3.3 / R12.2)
--------------------------

All randomness is drawn from a single generator, ``ctx.rng_for(8)`` -- an
independent sub-stream keyed by the run seed and this step's number -- obtained
once and drawn in a fixed order (the fps first, then the reshuffle permutation).
The golden-layout choice itself is deterministic (first match in sorted layout
order), so the same seed yields the same fps, the same referenced layout, and
the same reshuffled ordering on every run.

Scope (task 11.5 only): this module simulates the live test and hands off to the
reporters. It is the last step in the loop; it scores nothing and trains
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from week1_harness.datamodel import StepResult
from week1_harness.markers import estimate

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from collections.abc import Iterable

    import numpy as np

    from week1_harness.context import RunContext
    from week1_harness.datamodel import Instance

__all__ = ["LiveTestStep"]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    """Return the distinct ``values`` preserving first-occurrence order.

    Used to read a golden layout's item classes in the layout's own arrangement
    (its instance order) rather than an arbitrary canonical order, so the
    "reshuffle" is a permutation of *its items' order*. The order is still fully
    deterministic because the upstream instance order is seed-derived (R3.3).
    """
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# The "How" column text for the Timing_Table (R13.2); the live test is
# simulated, so its time cell is an ESTIMATE rather than a REAL measurement
# (R12.1).
_HOW = "Reshuffled golden layout, all 6 items; simulated fps"

# The honesty-marker reason for the simulated frame rate: it names the absent
# hardware a real live-test rate would require (R12.1).
_WEBCAM_GPU_ABSENT_REASON = "webcam and GPU absent"

# The session tag a golden layout's instances carry (R6.4); the referenced
# layout must be entirely from this session.
_GOLDEN_SESSION = "golden"

# Simulated live frame rate (fps), ESTIMATE "webcam and GPU absent" (R12.1).
# A plausible on-device webcam range; the value is drawn from rng_for(8) and
# rounded for clean, reproducible output (R12.2).
_FPS_RANGE: tuple[float, float] = (8.0, 30.0)
_FPS_DIGITS: int = 1


class LiveTestStep:
    """Data Loop step 8 -- the simulated live-test step (R12).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that reads Score's payload,
    emits an ``ESTIMATE`` live frame rate (seed-derived), references a reshuffled
    golden layout containing all six items, and returns the
    :class:`~week1_harness.datamodel.StepResult`.
    """

    number: int = 8
    name: str = "Test live"

    def run(self, ctx: RunContext) -> StepResult:
        """Simulate the live test and return the Test-live result (R12).

        Reads Score's payload from ``ctx.artifacts["Score"]``, emits an
        ``ESTIMATE`` "webcam and GPU absent" live frame rate drawn
        deterministically from ``ctx.rng_for(8)`` (R12.1/R12.2) as the time
        cell, references a reshuffled golden layout containing all six items
        (R12.3), and records the referenced layout id and the reshuffled order in
        the payload.
        """
        score = ctx.artifacts["Score"]
        class_ids: list[str] = list(score["class_ids"])
        golden_images = score["golden_images"]
        instances_by_layout: dict[str, list[Instance]] = score["instances_by_layout"]

        # Single generator for the whole step, drawn in a fixed order (the fps,
        # then the reshuffle permutation) so both are reproducible for a given
        # seed (R3.3/R12.2).
        rng = ctx.rng_for(self.number)

        # 1. Simulated live frame rate (R12.1): an ESTIMATE in fps naming the
        #    absent hardware, the value seed-derived (R12.2). This MarkedValue is
        #    the StepResult time cell.
        fps = round(float(rng.uniform(*_FPS_RANGE)), _FPS_DIGITS)
        time_cell = estimate(fps, reason=_WEBCAM_GPU_ABSENT_REASON, unit=" fps")

        # 2. Reference a reshuffled golden layout containing all six items
        #    (R12.3): pick a golden layout whose class set is exactly the six,
        #    verify it, then reshuffle its items deterministically from rng.
        golden_layout_id = self._choose_golden_layout(instances_by_layout, class_ids)
        layout_items = _ordered_unique(
            inst.class_id for inst in instances_by_layout[golden_layout_id]
        )
        # Verify the referenced layout really does contain all six items (R12.3).
        assert set(layout_items) == set(class_ids), (
            f"referenced golden layout {golden_layout_id!r} does not contain all "
            f"six practice items (R12.3)"
        )
        order = rng.permutation(len(layout_items))
        reshuffled_items = [layout_items[int(i)] for i in order]

        # Hand the reporters the simulated fps, the referenced golden layout and
        # its reshuffled item order, plus the ground-truth / golden pass-throughs
        # they continue to consume.
        payload = {
            "fps": fps,
            "golden_layout_id": golden_layout_id,
            "reshuffled_items": reshuffled_items,
            "class_ids": class_ids,
            "golden_images": golden_images,
            "instances_by_layout": instances_by_layout,
        }

        # notes is empty: the only non-real value is the fps ESTIMATE, which
        # already lives on StepResult.time (R12.1); duplicating it in notes would
        # double-count it in the Dataset_Card known-gaps scan (R14.4).
        return StepResult(
            number=self.number,
            name=self.name,
            how=_HOW,
            time=time_cell,
            payload=payload,
            notes=[],
        )

    def _choose_golden_layout(
        self,
        instances_by_layout: dict[str, list[Instance]],
        class_ids: list[str],
    ) -> str:
        """Return a golden layout id whose instances cover all six items (R12.3).

        Scans ``instances_by_layout`` in sorted layout-id order (so the choice is
        deterministic and independent of dict insertion order) and returns the
        first layout whose instances are all from the ``"golden"`` session and
        whose class set is exactly the six ``class_ids``. Raises
        :class:`ValueError` if no such layout exists -- a broken upstream
        contract rather than a graceful-degradation case.
        """
        want = set(class_ids)
        for layout_id in sorted(instances_by_layout):
            layout_instances = instances_by_layout[layout_id]
            if not layout_instances:
                continue
            if any(inst.session != _GOLDEN_SESSION for inst in layout_instances):
                continue
            if {inst.class_id for inst in layout_instances} == want:
                return layout_id
        raise ValueError(
            "no golden layout containing all six practice items found in "
            "instances_by_layout (R12.3)"
        )
