"""Step 3 -- Zero-shot baseline (Requirement 7).

:class:`ZeroShotStep` is the third Data Loop step. Before any training, it asks
an open-vocabulary detector which of the six practice items it can already find
from their *names* alone -- the "zero-shot baseline" that tells the team which
items work out of the box (R7). The step has two paths:

* **Capable path (R7.1).** Only when the local machine can genuinely run it --
  ``torch`` and ``ultralytics`` are importable, ungated YOLOE-26 weights are in
  the local cache, and the operator did not pass ``--no-zero-shot`` -- the step
  lazily imports ``torch``/``ultralytics`` (never at module top, R2.3 offline
  discipline), seeds ``torch`` from the run seed, loads the local weights by
  path, and runs **CPU** text-prompt inference over the six item names. That is
  real local-CPU work, so its time cell is ``REAL`` and the per-item detections
  are real (R4.1). The whole branch is guarded: any import/inference error
  degrades gracefully to the placeholder path (R2.5) with the reason
  ``"zero-shot inference unavailable: <err>"``.
* **Placeholder path (R7.2).** Otherwise the step emits a stubbed, "nothing
  detected" result as a ``PLACEHOLDER`` whose reason names the blocker --
  ``"zero-shot disabled by flag"``, ``"ultralytics or torch not importable"``,
  or ``"ungated weights not in local cache"`` -- and the time cell carries the
  same ``PLACEHOLDER`` reason. On the current dev machine ``torch``/
  ``ultralytics`` are absent, so this is the path that normally runs.

Two things are emitted **on both paths**:

* **SAM 3 concept-prompt baseline (R7.4).** SAM 3 is a gated model the harness
  never downloads or loads, so its concept-prompt baseline is *always* a
  ``PLACEHOLDER`` note with the reason ``"gated weights absent"``.
* **Per-item detection note (R7.3).** A single note whose value maps every one
  of the six item names to its detection status, so the note always covers all
  six items. On the placeholder path every item reads ``"not detected
  (stubbed)"``; on the capable path each item reads ``"detected"`` or
  ``"not detected"`` from the real inference.

Inputs
------

The six item names come from Extract's payload in the shared artifact store
(``ctx.artifacts["Extract frames"]["class_ids"]``); the capable path also reads
Extract's ``training_images`` to run inference over. The capability record
(``ctx.capabilities``) and the ``ctx.no_zero_shot`` flag select the path.

Determinism (R3)
----------------

The placeholder path draws no randomness (the stub is fixed). The capable path
seeds ``torch`` with ``ctx.seed`` before inference so a given seed + weights +
images yields the same detections. Real elapsed time is, by design, not part of
the determinism contract.

Scope (task 9.1 only): this module implements the Zero-shot step and its two
paths. It does not auto-label, review, train, or score, and it must import
cleanly with neither ``torch`` nor ``ultralytics`` installed (both are imported
lazily, inside the capable branch only).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from week1_harness.datamodel import StepResult
from week1_harness.markers import MarkedValue, placeholder, real

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.context import RunContext

__all__ = ["ZeroShotStep"]


# --------------------------------------------------------------------------- #
# Reason strings and "How" text (single source of truth).                     #
# The three placeholder reasons are the exact strings required by R7.2 and the #
# design's capability decision logic; the SAM 3 reason is required by R7.4.    #
# --------------------------------------------------------------------------- #

#: Operator passed ``--no-zero-shot`` -- forces the placeholder path.
REASON_FLAG = "zero-shot disabled by flag"
#: ``torch``/``ultralytics`` not importable on this machine.
REASON_LIBS = "ultralytics or torch not importable"
#: Libraries present but no ungated YOLOE-26 weights in the local cache (R7.2).
REASON_WEIGHTS = "ungated weights not in local cache"
#: SAM 3 is gated and never loaded; its baseline is always stubbed (R7.4).
REASON_SAM3 = "gated weights absent"

_HOW_CAPABLE = "YOLOE-26 text prompts on CPU"
_HOW_PLACEHOLDER = "stubbed (weights/libs absent)"

#: Per-item detection status strings used in the per-item note (R7.3).
_STATUS_DETECTED = "detected"
_STATUS_NOT_DETECTED = "not detected"
_STATUS_STUBBED = "not detected (stubbed)"


def _status_map(detected: dict[str, bool], *, stubbed: bool) -> dict[str, str]:
    """Build the per-item ``class_id -> status string`` map for the note (R7.3).

    When ``stubbed`` (the placeholder path), every item reads
    ``"not detected (stubbed)"``; otherwise each item reads ``"detected"`` or
    ``"not detected"`` from the real inference result. The map always has one
    entry per item in ``detected``, so the note covers all six items.
    """
    status: dict[str, str] = {}
    for class_id, hit in detected.items():
        if hit:
            status[class_id] = _STATUS_DETECTED
        elif stubbed:
            status[class_id] = _STATUS_STUBBED
        else:
            status[class_id] = _STATUS_NOT_DETECTED
    return status


class ZeroShotStep:
    """Data Loop step 3 -- open-vocabulary baseline over the six item names (R7).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that selects the capable or
    placeholder path from ``ctx.capabilities`` and ``ctx.no_zero_shot``, always
    emits the SAM 3 placeholder note (R7.4) and a per-item detection note
    covering all six items (R7.3), and returns a ``REAL``- or
    ``PLACEHOLDER``-timed :class:`~week1_harness.datamodel.StepResult`.
    """

    number: int = 3
    name: str = "Zero-shot baseline"

    def run(self, ctx: RunContext) -> StepResult:
        """Run (or stub) the zero-shot baseline and return the result (R7).

        Reads the six item names from Extract's payload, decides the path from
        the detected capabilities and the ``--no-zero-shot`` flag, and assembles
        the :class:`~week1_harness.datamodel.StepResult`. The capable path is
        wrapped in ``try``/``except`` so any import or inference failure falls
        back to the placeholder path (R2.5) rather than aborting the run.
        """
        caps = ctx.capabilities
        class_ids = list(ctx.artifacts["Extract frames"]["class_ids"])

        # The SAM 3 concept-prompt baseline is ALWAYS stubbed: gated weights are
        # never downloaded or loaded (R7.4).
        sam3_note = placeholder("SAM 3 concept-prompt baseline (not run)", reason=REASON_SAM3)

        capable = (
            caps.torch_available
            and caps.ultralytics_available
            and caps.yoloe_weights_path is not None
            and not ctx.no_zero_shot
        )

        if capable:
            start = time.perf_counter()
            try:
                detected = self._infer(ctx, class_ids)
            except Exception as exc:  # R2.5: degrade gracefully, don't abort
                return self._placeholder_result(
                    class_ids, sam3_note, reason=f"zero-shot inference unavailable: {exc}"
                )
            elapsed = time.perf_counter() - start
            return self._capable_result(class_ids, detected, elapsed, sam3_note)

        # Not capable: emit the stubbed result with the blocker's reason.
        return self._placeholder_result(
            class_ids, sam3_note, reason=self._placeholder_reason(ctx)
        )

    @staticmethod
    def _placeholder_reason(ctx: RunContext) -> str:
        """Pick the placeholder reason by precedence: flag, then libs, then weights.

        The explicit ``--no-zero-shot`` flag is the most informative blocker, so
        it wins; next, missing libraries are the more fundamental cause than a
        missing weight file; otherwise the only remaining disqualifier is that
        the ungated weights are absent from the local cache (R7.2).
        """
        if ctx.no_zero_shot:
            return REASON_FLAG
        caps = ctx.capabilities
        if not (caps.torch_available and caps.ultralytics_available):
            return REASON_LIBS
        return REASON_WEIGHTS

    def _capable_result(
        self,
        class_ids: list[str],
        detected: dict[str, bool],
        elapsed: float,
        sam3_note: MarkedValue,
    ) -> StepResult:
        """Assemble the ``REAL``-timed result for the capable path (R7.1).

        The time cell is ``REAL`` (genuine local-CPU inference) and the per-item
        note is ``REAL`` because it reports actual detections.
        """
        status = _status_map(detected, stubbed=False)
        per_item_note = real(status)
        payload = {
            "path": "capable",
            "reason": None,
            "detections": detected,
            "detected_items": [cid for cid, hit in detected.items() if hit],
            "per_item_status": status,
            "class_ids": class_ids,
        }
        return StepResult(
            number=self.number,
            name=self.name,
            how=_HOW_CAPABLE,
            time=real(elapsed, unit="s"),
            payload=payload,
            notes=[sam3_note, per_item_note],
        )

    def _placeholder_result(
        self,
        class_ids: list[str],
        sam3_note: MarkedValue,
        reason: str,
    ) -> StepResult:
        """Assemble the ``PLACEHOLDER``-timed stub result (R7.2).

        No item is detected without real inference, so every item's status is
        ``"not detected (stubbed)"``. The time cell and the per-item note both
        carry ``reason`` so the known-gaps scan records why the baseline did not
        run for real.
        """
        detected = {class_id: False for class_id in class_ids}
        status = _status_map(detected, stubbed=True)
        per_item_note = placeholder(status, reason=reason)
        payload = {
            "path": "placeholder",
            "reason": reason,
            "detections": detected,
            "detected_items": [],
            "per_item_status": status,
            "class_ids": class_ids,
        }
        return StepResult(
            number=self.number,
            name=self.name,
            how=_HOW_PLACEHOLDER,
            time=placeholder("n/a", reason=reason),
            payload=payload,
            notes=[sam3_note, per_item_note],
        )

    def _infer(self, ctx: RunContext, class_ids: list[str]) -> dict[str, bool]:
        """Run real CPU YOLOE-26 text-prompt inference over the six names (R7.1).

        Lazily imports ``torch``/``ultralytics`` *inside this method* (never at
        module top, so the module imports fine with neither installed), seeds
        ``torch`` from ``ctx.seed`` for reproducibility, loads the local weights
        by path, sets the six item names as text prompts, and runs CPU inference
        over Extract's training images. Returns ``{class_id: detected?}`` for all
        six items. May raise; the caller guards it and falls back to the
        placeholder path on any error (R2.5).
        """
        import torch
        from ultralytics import YOLOE

        torch.manual_seed(ctx.seed)

        model = YOLOE(ctx.capabilities.yoloe_weights_path)
        names = list(class_ids)
        # Prompt the open-vocabulary detector with the six item names.
        model.set_classes(names, model.get_text_pe(names))

        extract = ctx.artifacts["Extract frames"]
        image_paths = [image.path for image in extract.get("training_images", [])]

        detected = {class_id: False for class_id in class_ids}
        if not image_paths:
            return detected

        results = model.predict(image_paths, device="cpu", verbose=False)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.cls is None:
                continue
            for class_index in boxes.cls.tolist():
                index = int(class_index)
                if 0 <= index < len(names):
                    detected[names[index]] = True
        return detected
