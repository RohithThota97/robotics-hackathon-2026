"""Step 6 -- Train: the stubbed training step (Requirement 10).

:class:`TrainStep` is the sixth Data Loop step. On site this is where the
reviewed labels would fine-tune a YOLO26-seg model on an NVIDIA GPU. The harness
runs CPU-only with no GPU and does no real fine-tuning, so the Train step is
kept **in the loop but stubbed** (R10.1): it emits a placeholder model artifact,
placeholder training metrics, a simulated (estimated) GPU training time, and a
train/val split -- so the pipeline shape stays complete without a GPU or any
real training.

Because nothing here is a real local-CPU measurement, the honesty markers do the
heavy lifting (Requirement 4):

* the **model artifact** is a ``PLACEHOLDER`` value recording it is *not* a
  trained model (R10.2);
* the **metrics** -- mAP50, per-item recall (one entry per practice item), and
  look-alike confusion -- are each ``PLACEHOLDER`` values (R10.3);
* the **time cell** is an ``ESTIMATE`` with reason ``"NVIDIA GPU absent"``
  (R10.4), since a real training time needs a GPU the harness does not have.

What ``run`` does
-----------------

1. **Reads Review's output** from ``ctx.artifacts["Review"]`` (the pipeline
   runner stores each step's payload under the step name). From it Train uses
   the ``training_images`` (to derive the training layouts for the split) and
   the ``class_ids`` (to emit one recall value per item), and it passes the rest
   through for Score / Live-test / the reporters.
2. **Writes a placeholder model artifact file** to
   ``<ctx.out_dir>/model/placeholder_model.txt`` (R10.1) whose text states
   plainly that it is **not** a trained model and holds no weights, and wraps it
   in a :class:`~week1_harness.datamodel.ModelArtifact` whose ``value`` is a
   ``PLACEHOLDER`` ``MarkedValue`` (R10.2).
3. **Emits placeholder training metrics** (R10.3) as a
   :class:`~week1_harness.datamodel.Metrics` whose ``map50``, each entry of
   ``recall_per_item`` (one per the six ``class_ids``, R11.4-adjacent), and
   ``lookalike_confusion`` are ``PLACEHOLDER`` values. The underlying numbers are
   drawn deterministically from ``ctx.rng_for(6)`` so a given seed always yields
   the same (clearly-marked) stand-ins (R3.4-adjacent). They are *not* real
   accuracy numbers -- no trained model exists to measure.
4. **Estimates a GPU training time** (R10.4) as an ``ESTIMATE`` in minutes with
   reason ``"NVIDIA GPU absent"``, the minutes also drawn deterministically from
   ``ctx.rng_for(6)``.
5. **Records a train/val split defined by layout, not by frame** (R10.5):
   it collects the distinct source layout ids among the *training* images (the
   S1/S2 layouts; the held-out golden layouts never appear in
   ``training_images`` and so are excluded), partitions those **layout ids**
   into disjoint train and val sets using ``ctx.rng_for(6)``, and stores the
   split as ``{"train_layouts": [...], "val_layouts": [...]}``. Splitting on
   layout id (never on an individual frame) keeps every frame of a layout on the
   same side of the split.
6. **Returns a** :class:`~week1_harness.datamodel.StepResult` with an
   ``ESTIMATE`` time cell, the "How" text, a payload carrying the model
   artifact, metrics, and split plus the pass-through artifacts Score /
   Live-test / the reporters consume, and ``notes`` listing the model artifact's
   placeholder value and the metric placeholder values so the Dataset_Card
   known-gaps scan (which scans ``StepResult.notes``, R14.4) catches them.

Determinism (R3.3 / R3.4)
-------------------------

All randomness is drawn from a single generator, ``ctx.rng_for(6)`` -- an
independent sub-stream keyed by the run seed and this step's number -- obtained
once and drawn in a fixed order (metrics, then the estimated minutes, then the
split assignment). So the same seed yields identical metric values, the same
estimated training time, and the same train/val layout partition on every run.

Scope (task 11.1 only): this module stubs training and hands off. It does not
score the golden set (Step 7) or run the live test (Step 8).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from week1_harness.datamodel import Metrics, ModelArtifact, StepResult
from week1_harness.markers import estimate, placeholder

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    import numpy as np

    from week1_harness.context import RunContext
    from week1_harness.datamodel import ExtractedImage
    from week1_harness.markers import MarkedValue

__all__ = ["TrainStep"]


# The "How" column text for the Timing_Table (R13.2); Train is stubbed, so its
# time cell is an ESTIMATE rather than a REAL measurement (R10.4).
_HOW = "YOLO26-seg stub; split by layout"

# Shared reasons for the honesty markers (R10.2/R10.3/R10.4). The metric/model
# placeholders say what was stubbed; the time estimate names the absent hardware.
_STUB_REASON = "training stubbed"
_GPU_ABSENT_REASON = "NVIDIA GPU absent"

# Placeholder metric ranges. These produce plausible-looking BUT fabricated
# numbers; every one is wrapped PLACEHOLDER with the stub reason, so they can
# never be mistaken for a real measurement (R10.3). Values are drawn from
# rng_for(6) and rounded for clean, reproducible output.
_MAP50_RANGE: tuple[float, float] = (0.40, 0.85)
_RECALL_RANGE: tuple[float, float] = (0.35, 0.95)
_CONFUSION_RANGE: tuple[float, float] = (0.05, 0.45)
_METRIC_DIGITS: int = 3

# Simulated GPU training time (minutes), ESTIMATE "NVIDIA GPU absent" (R10.4).
_TRAIN_MINUTES_RANGE: tuple[float, float] = (15.0, 45.0)
_MINUTES_DIGITS: int = 1

# Fraction of TRAINING layouts (not frames) held out as the validation split
# (R10.5). Clamped so both sides are non-empty whenever there are >= 2 layouts.
_VAL_LAYOUT_FRACTION: float = 0.20

# The non-trained-model text written to the placeholder artifact file (R10.1).
# It states plainly that the file is not a model, so nobody can mistake it for
# trained weights.
_ARTIFACT_FILENAME = "placeholder_model.txt"
_ARTIFACT_TEXT = (
    "PLACEHOLDER MODEL ARTIFACT -- NOT A TRAINED MODEL\n"
    "\n"
    "This file is a stand-in emitted by the Week 1 Data-Loop Harness Train step.\n"
    "No fine-tuning or training was performed: the Train step is stubbed because\n"
    "the harness runs CPU-only with no NVIDIA GPU (R10.1, R10.2).\n"
    "\n"
    "This file contains NO model weights. Do not load, evaluate, or deploy it as\n"
    "a trained model. Any accuracy metrics emitted alongside it are placeholder\n"
    "stand-ins, not real measurements.\n"
)
_ARTIFACT_VALUE = "not a trained model"


def _draw_metric(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw one rounded placeholder metric in ``[lo, hi]`` from ``rng`` (R3.4)."""
    return round(float(rng.uniform(lo, hi)), _METRIC_DIGITS)


class TrainStep:
    """Data Loop step 6 -- the stubbed training step (R10).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that reads Review's payload,
    writes a ``PLACEHOLDER`` model artifact file, emits ``PLACEHOLDER`` training
    metrics and an ``ESTIMATE`` GPU training time (both seed-derived), records a
    train/val split defined by layout, and returns the
    :class:`~week1_harness.datamodel.StepResult`.
    """

    number: int = 6
    name: str = "Train"

    def run(self, ctx: RunContext) -> StepResult:
        """Stub training and return the Train result (R10).

        Reads Review's payload from ``ctx.artifacts["Review"]``, writes a
        non-trained ``PLACEHOLDER`` model artifact to ``<out_dir>/model/``
        (R10.1/R10.2), emits ``PLACEHOLDER`` mAP50 / per-item recall / look-alike
        confusion metrics (R10.3) and an ``ESTIMATE`` "NVIDIA GPU absent" GPU
        training time (R10.4) -- all drawn deterministically from
        ``ctx.rng_for(6)`` -- and records a by-layout train/val split (R10.5).
        """
        review = ctx.artifacts["Review"]
        reviewed_labels = review["reviewed_labels"]
        class_ids: list[str] = list(review["class_ids"])
        instances_per_class = review["instances_per_class"]
        instances_by_layout = review["instances_by_layout"]
        training_images: list[ExtractedImage] = review["training_images"]
        golden_images = review["golden_images"]
        all_instances = review["all_instances"]

        # Single generator for the whole step, drawn in a fixed order so the
        # metrics, the estimated minutes, and the split are reproducible for a
        # given seed (R3.3/R3.4).
        rng = ctx.rng_for(self.number)

        # 1. Placeholder metrics (R10.3): mAP50, one recall per item (R11.4
        #    shape), and look-alike confusion -- each a PLACEHOLDER MarkedValue
        #    whose number is seed-derived. Drawn before the split so the draw
        #    order is fixed.
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

        # 2. Simulated GPU training time (R10.4): an ESTIMATE in minutes naming
        #    the absent hardware, the minutes seed-derived.
        minutes = round(float(rng.uniform(*_TRAIN_MINUTES_RANGE)), _MINUTES_DIGITS)
        time_cell = estimate(minutes, reason=_GPU_ABSENT_REASON, unit=" min")

        # 3. Train/val split DEFINED BY LAYOUT, not by frame (R10.5). Partition
        #    the distinct training layout ids (S1/S2; golden is held out of
        #    training_images and so excluded) into disjoint train/val sets using
        #    rng_for(6). Sorting first makes the partition independent of set
        #    iteration order; sorting each side gives canonical, diff-stable
        #    output.
        split = self._split_by_layout(training_images, rng)

        # 4. Placeholder model artifact FILE (R10.1), explicitly labelled NOT a
        #    trained model (R10.2).
        model_artifact = self._write_model_artifact(ctx)

        # notes: the model artifact's placeholder value plus every metric
        # placeholder (map50, each per-item recall, look-alike confusion) so the
        # Dataset_Card known-gaps scan over StepResult.notes catches them
        # (R14.4). The time-cell ESTIMATE already lives on StepResult.time.
        notes: list[MarkedValue] = [
            model_artifact.value,
            map50,
            *recall_per_item.values(),
            lookalike_confusion,
        ]

        how = (
            f"{_HOW}: {len(split['train_layouts'])} train / "
            f"{len(split['val_layouts'])} val layouts"
        )

        # Hand Score, Live-test, and the reporters the stub outputs plus the
        # ground truth / golden set / counts they continue to consume.
        payload = {
            "model_artifact": model_artifact,
            "metrics": metrics,
            "split": split,
            "class_ids": class_ids,
            "instances_per_class": instances_per_class,
            "golden_images": golden_images,
            "instances_by_layout": instances_by_layout,
            "all_instances": all_instances,
            "reviewed_labels": reviewed_labels,
        }

        return StepResult(
            number=self.number,
            name=self.name,
            how=how,
            time=time_cell,
            payload=payload,
            notes=notes,
        )

    def _split_by_layout(
        self, training_images: list[ExtractedImage], rng: np.random.Generator
    ) -> dict[str, list[str]]:
        """Partition the distinct training layout ids into train/val sets (R10.5).

        The split is over **layout ids** (never individual frames): the distinct
        ``source_layout_id`` values among ``training_images`` are the S1/S2
        layouts (golden is held out of the training set and so never appears
        here). They are sorted for determinism, permuted with ``rng``, and cut
        into a validation slice of ``_VAL_LAYOUT_FRACTION`` -- clamped so both
        sides are non-empty whenever there are at least two layouts -- with the
        rest as the train set. Each side is returned sorted (canonical, stable).
        """
        layout_ids = sorted({image.source_layout_id for image in training_images})
        n_layouts = len(layout_ids)

        if n_layouts >= 2:
            n_val = round(_VAL_LAYOUT_FRACTION * n_layouts)
            n_val = max(1, min(n_val, n_layouts - 1))
        else:
            n_val = 0  # degenerate: cannot split <2 layouts; keep all for train

        order = rng.permutation(n_layouts)
        shuffled = [layout_ids[int(i)] for i in order]
        val_layouts = sorted(shuffled[:n_val])
        train_layouts = sorted(shuffled[n_val:])

        return {"train_layouts": train_layouts, "val_layouts": val_layouts}

    def _write_model_artifact(self, ctx: RunContext) -> ModelArtifact:
        """Write the placeholder model file and wrap it as a PLACEHOLDER (R10.1/R10.2).

        Writes ``<ctx.out_dir>/model/placeholder_model.txt`` (creating the
        ``model`` directory if needed) with text stating plainly it is not a
        trained model and holds no weights, then returns a
        :class:`~week1_harness.datamodel.ModelArtifact` pointing at that file
        whose ``value`` is a ``PLACEHOLDER`` ``MarkedValue``.
        """
        model_dir = os.path.join(os.fspath(ctx.out_dir), "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, _ARTIFACT_FILENAME)
        with open(model_path, "w", encoding="utf-8") as handle:
            handle.write(_ARTIFACT_TEXT)

        return ModelArtifact(
            path=model_path,
            value=placeholder(_ARTIFACT_VALUE, reason=_STUB_REASON),
        )
