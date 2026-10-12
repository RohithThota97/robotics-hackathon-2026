"""Unit tests for Step 6 -- Train (``week1_harness.steps.train.TrainStep``).

Covers task 11.2: the sixth Data Loop step, which keeps the Train step **in the
loop but stubbed** (R10.1). On site this is where the reviewed labels would
fine-tune a YOLO26-seg model on an NVIDIA GPU; the harness runs CPU-only with no
GPU and does no real fine-tuning, so every value Train emits is a visibly
labelled stand-in. These example-based unit tests pin that contract for a single
Film -> Extract -> Auto-label -> Review -> Train run:

* **Model artifact (R10.1/R10.2).** A placeholder model file is written to disk
  at ``model_artifact.path``, and its ``value`` is a ``PLACEHOLDER``
  :class:`~week1_harness.markers.MarkedValue` recording it is *not* a trained
  model.
* **Metrics placeholder (R10.3).** ``metrics.map50`` is ``PLACEHOLDER``; every
  ``recall_per_item`` entry is ``PLACEHOLDER`` and the keys are exactly the six
  practice-item classes; ``lookalike_confusion`` is ``PLACEHOLDER``.
* **GPU time estimate (R10.4).** Train's time cell is an ``ESTIMATE`` whose
  reason is exactly ``"NVIDIA GPU absent"`` -- a real training time needs a GPU
  the harness does not have.
* **Split by layout (R10.5).** The train/val split is defined over **layout
  ids** (never individual frames): ``train_layouts`` and ``val_layouts`` are
  disjoint and both non-empty, their union is exactly the set of training
  (S1/S2) layout ids, and it holds no held-out golden layout id (none starts
  with ``"golden"``).
* **Canonical identity.** The step reports ``number == 6`` and ``name == "Train"``.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5.

The tests are hermetic: the one shared run writes under a pytest-managed temp
directory (``tmp_path_factory``, the module-scoped sibling of ``tmp_path``), so
nothing is written into the repository working tree. Film renders a few hundred
PNG frames, so the full Film -> Extract -> Auto-label -> Review -> Train pass
runs exactly once in a module-scoped fixture and every assertion reads from that
single run rather than re-filming per test.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from week1_harness import capabilities
from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.review import ReviewStep
from week1_harness.steps.train import TrainStep
from week1_harness.synth.items import CLASS_IDS

# A fixed seed makes every assertion concrete and reproducible (the
# default/start-date seed used across the other harness tests).
SEED = 20261009

# The exact reason the Train time cell carries as an ESTIMATE (R10.4): a real
# training time needs an NVIDIA GPU the harness does not have.
GPU_ABSENT_REASON = "NVIDIA GPU absent"


@pytest.fixture(scope="module")
def train_run(tmp_path_factory):
    """Run Film -> Extract -> Auto-label -> Review -> Train once and share the result.

    Film writes a few hundred PNG frames, so the assertions below all read from
    this one run rather than re-filming per test. ``tmp_path_factory`` hands back
    a pytest-managed temp dir (the module-scoped equivalent of ``tmp_path``, with
    the same hermetic guarantee), so nothing touches the repo tree.

    The pipeline runner would store each step's payload under its step name for
    downstream steps; here we wire that handoff by hand -- ``ExtractStep`` reads
    ``ctx.artifacts["Film"]``, ``AutoLabelStep`` reads
    ``ctx.artifacts["Extract frames"]``, ``ReviewStep`` reads
    ``ctx.artifacts["Auto-label"]``, and ``TrainStep`` reads
    ``ctx.artifacts["Review"]`` -- then run each step against the same seeded
    context. (Zero-shot, step 3, is not on Train's artifact path and is skipped.)
    The context carries a real detected ``Capabilities`` record
    (``capabilities.detect(None)``, filesystem/import checks only, no network),
    mirroring how the harness builds the context for a real run.

    Returns a namespace with the Train ``result`` plus the ground truth the split
    assertions need, all derived from this same run: the ``training_images`` and
    their distinct ``training_layout_ids`` (the S1/S2 source layouts the split
    must cover) and the held-out ``golden_layout_ids`` it must never include.
    """
    out_dir = tmp_path_factory.mktemp("train_out")
    ctx = RunContext(seed=SEED, out_dir=out_dir, capabilities=capabilities.detect(None))

    film_result = FilmStep().run(ctx)
    ctx.artifacts["Film"] = film_result.payload

    extract_result = ExtractStep().run(ctx)
    ctx.artifacts["Extract frames"] = extract_result.payload

    auto_label_result = AutoLabelStep().run(ctx)
    ctx.artifacts["Auto-label"] = auto_label_result.payload

    review_result = ReviewStep().run(ctx)
    ctx.artifacts["Review"] = review_result.payload

    result = TrainStep().run(ctx)

    # Ground truth for the split assertions, from the SAME run. Train does not
    # pass the training images through its payload, so read them from Review's
    # (upstream) payload; golden images do pass through Train's payload.
    training_images = ctx.artifacts["Review"]["training_images"]
    training_layout_ids = {image.source_layout_id for image in training_images}
    golden_layout_ids = {
        image.source_layout_id for image in result.payload["golden_images"]
    }

    return SimpleNamespace(
        result=result,
        training_images=training_images,
        training_layout_ids=training_layout_ids,
        golden_layout_ids=golden_layout_ids,
    )


# --- Canonical identity --------------------------------------------------------

def test_train_step_number_and_name(train_run):
    """Train is the sixth Data Loop step, named "Train" (canonical order, R1.1)."""
    assert train_run.result.number == 6
    assert train_run.result.name == "Train"


# --- Model artifact (R10.1/R10.2) ----------------------------------------------

def test_model_artifact_file_exists_on_disk(train_run):
    """A placeholder model artifact FILE is written to disk (R10.1)."""
    model_artifact = train_run.result.payload["model_artifact"]
    assert os.path.isfile(model_artifact.path), (
        f"model artifact file was not written to disk: {model_artifact.path!r}"
    )


def test_model_artifact_is_placeholder(train_run):
    """The model artifact is labelled a PLACEHOLDER, not a trained model (R10.2)."""
    model_artifact = train_run.result.payload["model_artifact"]
    assert model_artifact.value.marker is Marker.PLACEHOLDER, (
        f"model artifact marker is {model_artifact.value.marker.value}, "
        f"expected PLACEHOLDER (the stub is not a trained model, R10.2)"
    )
    # The marker-reason invariant: a non-REAL value must carry a non-empty reason.
    assert model_artifact.value.reason, (
        "PLACEHOLDER model artifact must carry a non-empty reason (R10.2/R4.4)"
    )


# --- Metrics placeholder (R10.3) -----------------------------------------------

def test_map50_metric_is_placeholder(train_run):
    """mAP50 is emitted as a PLACEHOLDER metric (R10.3)."""
    metrics = train_run.result.payload["metrics"]
    assert metrics.map50.marker is Marker.PLACEHOLDER, (
        f"map50 marker is {metrics.map50.marker.value}, expected PLACEHOLDER (R10.3)"
    )


def test_recall_per_item_covers_six_classes_and_is_placeholder(train_run):
    """One PLACEHOLDER recall per practice item: keys == the six classes (R10.3)."""
    recall_per_item = train_run.result.payload["metrics"].recall_per_item

    # Exactly one entry per practice-item class -- the six classes, no more, no
    # fewer (R10.3 per-item recall; R11.4 shape).
    assert set(recall_per_item) == set(CLASS_IDS), (
        f"recall_per_item keys {sorted(recall_per_item)} != the six classes "
        f"{sorted(CLASS_IDS)}"
    )
    assert len(recall_per_item) == 6, (
        f"expected 6 per-item recall entries, got {len(recall_per_item)}"
    )

    # Every per-item recall value is a PLACEHOLDER stand-in (R10.3).
    for class_id, recall in recall_per_item.items():
        assert recall.marker is Marker.PLACEHOLDER, (
            f"recall for {class_id!r} is {recall.marker.value}, expected PLACEHOLDER"
        )


def test_lookalike_confusion_metric_is_placeholder(train_run):
    """Look-alike pair confusion is emitted as a PLACEHOLDER metric (R10.3)."""
    metrics = train_run.result.payload["metrics"]
    assert metrics.lookalike_confusion.marker is Marker.PLACEHOLDER, (
        f"lookalike_confusion marker is {metrics.lookalike_confusion.marker.value}, "
        f"expected PLACEHOLDER (R10.3)"
    )


# --- GPU time estimate (R10.4) -------------------------------------------------

def test_gpu_training_time_is_estimate_with_gpu_absent_reason(train_run):
    """Train's time cell is an ESTIMATE reason 'NVIDIA GPU absent' (R10.4)."""
    time_cell = train_run.result.time
    assert time_cell.marker is Marker.ESTIMATE, (
        f"Train time cell is {time_cell.marker.value}, expected ESTIMATE -- a "
        f"real training time needs a GPU the harness does not have (R10.4)"
    )
    assert time_cell.reason == GPU_ABSENT_REASON, (
        f"Train time estimate reason is {time_cell.reason!r}, expected "
        f"{GPU_ABSENT_REASON!r} (R10.4)"
    )


# --- Split by layout (R10.5) ---------------------------------------------------

def test_train_val_split_is_defined_by_layout(train_run):
    """The train/val split is over layout IDS -- disjoint, non-empty, golden held out (R10.5)."""
    split = train_run.result.payload["split"]
    train_set = set(split["train_layouts"])
    val_set = set(split["val_layouts"])

    # Both sides are non-empty: a real split, not everything-on-one-side.
    assert train_set, "train split is empty"
    assert val_set, "val split is empty"

    # The two sides are disjoint -- no layout is both trained and validated.
    assert train_set.isdisjoint(val_set), (
        f"train/val splits overlap on layouts: {sorted(train_set & val_set)}"
    )

    union = train_set | val_set

    # The union is EXACTLY the set of training (S1/S2) layout ids: the split
    # partitions the training layouts and nothing else (R10.5).
    assert union == train_run.training_layout_ids, (
        "split union is not exactly the training layout ids; "
        f"missing={sorted(train_run.training_layout_ids - union)}, "
        f"extra={sorted(union - train_run.training_layout_ids)}"
    )

    # No held-out golden layout leaks into the split: golden is reserved for the
    # Scorer and never trained/validated (R5.4/R10.5). Golden ids start "golden".
    assert all(not layout_id.startswith("golden") for layout_id in union), (
        f"a golden layout id leaked into the split: "
        f"{sorted(lid for lid in union if lid.startswith('golden'))}"
    )
    assert union.isdisjoint(train_run.golden_layout_ids), (
        f"golden layouts appear in the split: "
        f"{sorted(union & train_run.golden_layout_ids)}"
    )


def test_split_entries_are_layout_ids_not_frames(train_run):
    """Each split entry is a layout-id string, not a per-frame image path (R10.5)."""
    split = train_run.result.payload["split"]
    union = set(split["train_layouts"]) | set(split["val_layouts"])

    # Every entry is a layout-id string: not an ExtractedImage path (which would
    # carry a path separator, a "frame_" stem, and a ".png" suffix). This is what
    # "split by layout, not by frame" means concretely.
    for layout_id in union:
        assert isinstance(layout_id, str), (
            f"split entry {layout_id!r} is not a string layout id"
        )
        assert os.sep not in layout_id, (
            f"split entry {layout_id!r} looks like a file path, not a layout id"
        )
        assert "frame_" not in layout_id, (
            f"split entry {layout_id!r} looks like a per-frame id, not a layout id"
        )
        assert not layout_id.endswith(".png"), (
            f"split entry {layout_id!r} looks like an image file, not a layout id"
        )

    # There are strictly fewer split layouts than extracted training frames --
    # each layout contributes many frames -- confirming the split is over the
    # (few) layouts, not the (many) individual frames.
    assert len(union) < len(train_run.training_images), (
        f"split has {len(union)} entries but there are only "
        f"{len(train_run.training_images)} training frames; the split does not "
        f"look like it is over layouts"
    )
