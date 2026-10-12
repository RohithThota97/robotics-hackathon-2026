"""Step 4 -- Auto-label: instance masks from known geometry (Requirement 8).

:class:`AutoLabelStep` is the fourth Data Loop step. On site this is where SAM 3
would be prompted (text and box prompts) to segment every labelled instance in
the training image set. The harness has no gated SAM 3 weights, so instead of
calling a segmentation model it *rasterizes the masks directly from the known
synthetic polygon geometry* that Film already produced -- genuine local-CPU
pixel work, so the timing cell is a ``REAL``
:class:`~week1_harness.markers.MarkedValue` (R4.1). The only non-real aspect is
the *method* (SAM 3 would have produced these on site), which is recorded as a
``PLACEHOLDER`` note (R8.2), **not** in the time cell -- see the design's
"Auto-label vs Score split" rationale.

What ``run`` does
-----------------

1. **Reads Extract's output** from ``ctx.artifacts["Extract frames"]`` (the
   pipeline runner stores each step's payload under the step name). From it
   Auto-label uses the training image set (``training_images``) to learn *which*
   instances are labelled in training, and the ground-truth geometry
   (``all_instances``) to rasterize them.
2. **Collects the unique labelled training instances.** A "labelled instance in
   the training image set" (R8.1) is a unique ground-truth
   :class:`~week1_harness.datamodel.Instance` (keyed by ``instance_id``) that
   appears in at least one training frame. The same instance is repeated across
   every frame of its clip, so masks are produced **per unique instance, not per
   frame-occurrence** -- exactly one mask per labelled instance (R8.1). The
   held-out golden instances never appear in ``training_images`` and so get no
   mask.
3. **Rasterizes one mask per unique instance** from the instance's known polygon
   via :func:`week1_harness.synth.render.rasterize_instance_mask`, assigning
   exactly one class per mask (``instance.class_id``, R8.3). The look-alike pair
   (``screwdriver_small`` / ``screwdriver_large``) keeps its two distinct class
   ids, so the pair stays two classes end to end (R8.4). To bound memory each
   full-canvas raster is immediately **cropped to the instance bbox** and stored
   as a small standalone array in :class:`~week1_harness.datamodel.Mask.raster`
   (the full 640x640 canvas is transient and freed each iteration).
4. **Emits the SAM 3 placeholder note** -- the text/box-prompt mask generation
   the harness deliberately did not run -- as a ``PLACEHOLDER``
   :class:`~week1_harness.markers.MarkedValue` with the reason
   ``"gated weights absent"`` (R8.2).
5. **Returns a** :class:`~week1_harness.datamodel.StepResult` with a ``REAL``
   time cell, the "How" text, the produced masks and their count, and the
   pass-through artifacts (``class_ids``, ``instances_by_layout``,
   ``all_instances``, both image sets, per-class tallies) that Review and the
   reporters consume.

Determinism (R3.3)
------------------

Auto-label draws no randomness: given Extract's (seeded, deterministic) training
set it rasterizes a fixed set of instances in Film's deterministic instance
order, so the same seed yields the same masks in the same order.

Scope (task 10.1 only): this module rasterizes masks and hands them off. It does
not apply the Review visibility rule, train, or score.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from week1_harness.datamodel import Mask, StepResult
from week1_harness.markers import placeholder, real
from week1_harness.synth import render

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.context import RunContext
    from week1_harness.datamodel import Instance

__all__ = ["AutoLabelStep"]


# The "How" column text for the Timing_Table (R13.2); Auto-label does real CPU
# work, with SAM 3 recorded separately as a placeholder note (R8.2).
_HOW = "Masks from synthetic geometry (stands in for SAM 3)"

# The SAM 3 text/box-prompt mask generation the harness deliberately did not run
# (no gated weights). Emitted as a PLACEHOLDER note, scanned into the
# Dataset_Card known-gaps list (R8.2 / R14.4).
_SAM3_PLACEHOLDER_VALUE = "SAM 3 text/box-prompt mask generation"
_SAM3_PLACEHOLDER_REASON = "gated weights absent"


def _crop_to_bbox(
    raster: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop a full-canvas boolean mask to an instance's bounding box.

    ``bbox`` is ``(x0, y0, x1, y1)`` with the max corner *inclusive* (it is the
    max polygon vertex, see ``scene._bbox``), and ``raster`` has shape
    ``(height, width)``, so the crop is ``raster[y0:y1 + 1, x0:x1 + 1]``. The
    bounds are clamped into the canvas so a vertex nudged off-canvas by rotation
    never produces a wrapped or empty slice. The result is a fresh contiguous
    copy, which both bounds memory (the big canvas is freed) and decouples the
    stored mask from the transient full raster.
    """
    height, width = raster.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    x1 = max(x0, min(int(x1), width - 1))
    y1 = max(y0, min(int(y1), height - 1))
    return np.ascontiguousarray(raster[y0 : y1 + 1, x0 : x1 + 1])


class AutoLabelStep:
    """Data Loop step 4 -- rasterize one instance mask per labelled instance (R8).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that reads Extract's training
    set from the shared artifact store, produces exactly one
    :class:`~week1_harness.datamodel.Mask` per unique labelled training instance
    (each with exactly one class, the look-alike pair kept separate), records the
    SAM 3 placeholder note, and returns a ``REAL``-timed
    :class:`~week1_harness.datamodel.StepResult`.
    """

    number: int = 4
    name: str = "Auto-label"

    def run(self, ctx: RunContext) -> StepResult:
        """Rasterize the training masks, then return the Auto-label result (R8).

        Times the real rasterization pass with :func:`time.perf_counter`
        (genuine local-CPU pixel work -> ``REAL`` time cell, R4.1), reads
        Extract's payload from ``ctx.artifacts["Extract frames"]``, produces one
        mask per unique labelled training instance (R8.1) with exactly one class
        each (R8.3) and the look-alike pair kept separate (R8.4), and records the
        SAM 3 text/box mask generation as a ``PLACEHOLDER`` note (R8.2).
        """
        start = time.perf_counter()

        extract = ctx.artifacts["Extract frames"]
        training_images = extract["training_images"]
        golden_images = extract["golden_images"]
        instances_per_class = extract["instances_per_class"]
        class_ids: list[str] = list(extract["class_ids"])
        instances_by_layout = extract["instances_by_layout"]
        all_instances: list[Instance] = extract["all_instances"]

        # The set of instance ids labelled anywhere in the TRAINING set (S1/S2).
        # Golden frames are held out of training_images, so golden instances are
        # naturally excluded. The same instance recurs across every frame of its
        # clip; collecting a set collapses those occurrences to one id (R8.1).
        training_instance_ids: set[str] = set()
        for image in training_images:
            training_instance_ids.update(image.instance_ids)

        # One mask per UNIQUE labelled training instance (R8.1). Iterate the
        # ground truth in Film's deterministic order and keep each instance whose
        # id is labelled in training; all_instances holds unique ids, so this is
        # a one-to-one mask<->instance correspondence (len(masks) == count).
        canvas = render.DEFAULT_SIZE
        masks: list[Mask] = []
        for instance in all_instances:
            if instance.instance_id not in training_instance_ids:
                continue
            # Rasterize the known polygon on the full canvas, then crop to the
            # bbox so Mask.raster stores only the small painted region (memory
            # bound) rather than a full 640x640 array per instance.
            full_raster = render.rasterize_instance_mask(instance.polygon, size=canvas)
            cropped = _crop_to_bbox(full_raster, instance.bbox)
            masks.append(
                Mask(
                    instance_id=instance.instance_id,
                    class_id=instance.class_id,  # exactly one class per mask (R8.3)
                    raster=cropped,
                )
            )

        elapsed = time.perf_counter() - start

        # SAM 3 text/box-prompt mask generation would have produced these masks
        # on site; the harness used synthetic geometry instead. Record that as a
        # PLACEHOLDER note (R8.2) -- the time cell stays REAL (R4.1).
        sam3_note = placeholder(
            _SAM3_PLACEHOLDER_VALUE,
            reason=_SAM3_PLACEHOLDER_REASON,
        )

        how = f"{_HOW}: {len(masks)} masks"

        # Hand Review and the reporters the masks and count, plus the ground
        # truth / image sets / tallies they continue to need downstream.
        payload = {
            "masks": masks,
            "mask_count": len(masks),
            "class_ids": class_ids,
            "instances_by_layout": instances_by_layout,
            "all_instances": all_instances,
            "training_images": training_images,
            "golden_images": golden_images,
            "instances_per_class": instances_per_class,
        }

        # Real local-CPU rasterization: the time cell is REAL and carries no
        # reason; the only non-real aspect (the SAM 3 method) is the note (R4.1,
        # R8.2).
        return StepResult(
            number=self.number,
            name=self.name,
            how=how,
            time=real(elapsed, unit="s"),
            payload=payload,
            notes=[sam3_note],
        )
