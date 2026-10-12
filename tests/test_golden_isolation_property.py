"""Property-based test for golden isolation and split-by-layout (Task 11.7).

Feature: week1-data-loop-harness, Property 11: Golden isolation and split-by-layout
-- for any run, the number of golden layouts is one or two, the golden images are
disjoint from the train and validation images, the train/validation split is
defined so that no layout spans both sides, and the Scorer evaluates only
golden-layout inputs.

This test exercises the property end to end over several seeds. For each seed it
runs the relevant slice of the Data Loop in canonical order --
:class:`~week1_harness.steps.film.FilmStep` (1) ->
:class:`~week1_harness.steps.extract.ExtractStep` (2) ->
:class:`~week1_harness.steps.auto_label.AutoLabelStep` (4) ->
:class:`~week1_harness.steps.review.ReviewStep` (5) ->
:class:`~week1_harness.steps.train.TrainStep` (6) ->
:class:`~week1_harness.steps.score.ScoreStep` (7) -- via
:func:`week1_harness.pipeline.run`, which drives the steps in list order and
publishes each step's payload under ``ctx.artifacts[step.name]`` so the next
step reads it (``"Film"`` -> ``"Extract frames"`` -> ``"Auto-label"`` ->
``"Review"`` -> ``"Train"`` -> ``"Score"``). The Zero-shot (3) and Live-test (8)
steps are irrelevant to golden isolation and the by-layout split, so this slice
omits them; Auto-label reads Extract's payload directly, so the artifact chain
stays intact without Zero-shot.

For every seed the test asserts the three facets of Property 11:

* **SPLIT BY LAYOUT (R10.5).** Train records a train/val split over **layout
  ids**: ``train_layouts`` and ``val_layouts`` are disjoint, both non-empty,
  every entry is a layout-id string, their union contains no golden layout
  (none begins with ``"golden"``), and that union is exactly the distinct
  training (S1/S2) source layout ids of Extract's training set -- so no layout
  spans both sides and no frame-level split can leak a layout across the cut.
* **GOLDEN HELD OUT (R5.4).** The held-out golden layouts (session
  ``"golden"``) never enter training: the golden layout ids are disjoint from
  the split union, and no golden image path appears among Extract's
  ``training_images``.
* **SCORED GOLDEN ONLY (R11.1).** The Scorer's ``scored_golden_layouts`` are all
  golden layout ids (each begins with ``"golden"``) and are disjoint from the
  split's training layouts, so evaluation touches only held-out golden inputs.

Each example runs the full Film + Extract + Auto-label + Review + Train + Score
pass, which renders ~284 PNGs to disk (S1 + S2 + golden clips) -- expensive I/O
-- so the example budget is kept deliberately small (``max_examples=3``) and the
per-example deadline is disabled. A fresh ``tempfile.TemporaryDirectory`` is
created *inside* the test body per example (not via a function-scoped fixture,
which Hypothesis would share across every example) so the runs stay hermetic and
never touch the repository tree.

**Validates: Requirements 5.4, 10.5, 11.1**

_Requirements: 5.4, 10.5, 11.1_
"""

from __future__ import annotations

import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness import capabilities, pipeline
from week1_harness.context import RunContext
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.review import ReviewStep
from week1_harness.steps.score import ScoreStep
from week1_harness.steps.train import TrainStep

# The canonical step names under which the runner publishes each payload in
# ``ctx.artifacts`` (``step.name``). Extract's key is "Extract frames".
_EXTRACT_NAME = "Extract frames"
_TRAIN_NAME = "Train"
_SCORE_NAME = "Score"

# A golden (held-out) layout id always begins with this prefix (see
# ``synth/scene.py``: golden layouts are built as ``golden_{k:02d}``); S1/S2
# training layouts never do (``s1_...`` / ``s2_...``).
_GOLDEN_PREFIX = "golden"


def _golden_isolation_steps() -> list:
    """The Data Loop slice this property needs, in canonical order.

    Film -> Extract -> Auto-label -> Review -> Train -> Score, returned as fresh
    instances so each example drives its own steps. The order matches the
    harness's canonical pipeline order and the artifact hand-off each downstream
    step depends on (Auto-label reads Extract directly, so Zero-shot is not
    required for the chain to resolve).
    """
    return [
        FilmStep(),
        ExtractStep(),
        AutoLabelStep(),
        ReviewStep(),
        TrainStep(),
        ScoreStep(),
    ]


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=3, deadline=None)
def test_golden_isolation_and_split_by_layout(seed: int) -> None:
    """Property 11: golden layouts stay held out and the split is by layout.

    For every seed, after running Film -> Extract -> Auto-label -> Review ->
    Train -> Score via the pipeline runner, this asserts the split-by-layout
    (R10.5), golden-held-out (R5.4), and scored-golden-only (R11.1) facets of
    Property 11 against the Extract, Train, and Score payloads.
    """
    # Build the capabilities the harness would detect for a real run
    # (filesystem/import checks only, no network) and thread them through the
    # context, exactly as the CLI does.
    caps = capabilities.detect(None)

    with tempfile.TemporaryDirectory() as out_dir:
        ctx = RunContext(seed=seed, out_dir=out_dir, capabilities=caps)
        results = pipeline.run(ctx, _golden_isolation_steps())

    # Guard against a vacuous pass: all six steps must have run and published
    # their payloads under the expected names.
    assert len(results) == 6, f"expected 6 step results, got {len(results)}"
    for name in (_EXTRACT_NAME, _TRAIN_NAME, _SCORE_NAME):
        assert name in ctx.artifacts, f"missing payload for step {name!r}"

    extract = ctx.artifacts[_EXTRACT_NAME]
    train = ctx.artifacts[_TRAIN_NAME]
    score = ctx.artifacts[_SCORE_NAME]

    training_images = extract["training_images"]
    golden_images = extract["golden_images"]

    split = train["split"]
    train_layouts = split["train_layouts"]
    val_layouts = split["val_layouts"]

    scored_golden_layouts = score["scored_golden_layouts"]

    # ------------------------------------------------------------------ #
    # SPLIT BY LAYOUT (R10.5)                                            #
    # ------------------------------------------------------------------ #
    # Every split entry is a layout-id string.
    assert all(isinstance(lid, str) for lid in train_layouts), (
        f"train_layouts must be layout-id strings, got {train_layouts!r} (R10.5)"
    )
    assert all(isinstance(lid, str) for lid in val_layouts), (
        f"val_layouts must be layout-id strings, got {val_layouts!r} (R10.5)"
    )

    train_set = set(train_layouts)
    val_set = set(val_layouts)

    # Both sides non-empty.
    assert train_set, "train_layouts is empty; the split must have a train side (R10.5)"
    assert val_set, "val_layouts is empty; the split must have a val side (R10.5)"

    # Disjoint: no layout spans both sides of the split.
    assert train_set.isdisjoint(val_set), (
        f"train/val layouts overlap: {sorted(train_set & val_set)}; a layout "
        f"must not span both sides of the split (R10.5)"
    )

    split_union = train_set | val_set

    # The split contains no golden layout -- the split is over training layouts
    # only.
    golden_in_split = {lid for lid in split_union if lid.startswith(_GOLDEN_PREFIX)}
    assert not golden_in_split, (
        f"golden layout(s) {sorted(golden_in_split)} leaked into the train/val "
        f"split; golden layouts must be held out of training (R10.5/R5.4)"
    )

    # The split union is exactly the distinct training (S1/S2) source layout ids.
    training_source_layouts = {image.source_layout_id for image in training_images}
    assert split_union == training_source_layouts, (
        "train/val split union does not match the distinct training source "
        f"layouts; split-only={sorted(split_union - training_source_layouts)}, "
        f"training-only={sorted(training_source_layouts - split_union)} (R10.5)"
    )

    # ------------------------------------------------------------------ #
    # GOLDEN HELD OUT (R5.4)                                             #
    # ------------------------------------------------------------------ #
    golden_source_layouts = {image.source_layout_id for image in golden_images}

    # Guard against a vacuous pass and confirm the held-out layouts really are
    # golden (one or two of them, each with the golden prefix).
    assert golden_source_layouts, "no golden layouts were held out (R5.4)"
    assert 1 <= len(golden_source_layouts) <= 2, (
        f"expected 1-2 golden layouts, got {sorted(golden_source_layouts)} (R5.4)"
    )
    assert all(lid.startswith(_GOLDEN_PREFIX) for lid in golden_source_layouts), (
        f"held-out layouts are not all golden: {sorted(golden_source_layouts)} (R5.4)"
    )

    # No golden layout appears anywhere in the train/val split.
    assert golden_source_layouts.isdisjoint(split_union), (
        f"golden layout(s) {sorted(golden_source_layouts & split_union)} appear "
        f"in the training split; they must be held out (R5.4)"
    )

    # No golden image path appears among the training images.
    training_paths = {image.path for image in training_images}
    golden_paths = {image.path for image in golden_images}
    assert training_paths.isdisjoint(golden_paths), (
        f"golden image path(s) leaked into the training set: "
        f"{sorted(training_paths & golden_paths)} (R5.4)"
    )

    # ------------------------------------------------------------------ #
    # SCORED GOLDEN ONLY (R11.1)                                         #
    # ------------------------------------------------------------------ #
    scored_set = set(scored_golden_layouts)

    # Guard against a vacuous pass: the Scorer must have scored golden layouts.
    assert scored_set, "the Scorer scored no golden layouts (R11.1)"

    # Everything the Scorer evaluated is a golden layout.
    assert all(lid.startswith(_GOLDEN_PREFIX) for lid in scored_set), (
        f"the Scorer evaluated a non-golden layout: "
        f"{sorted(lid for lid in scored_set if not lid.startswith(_GOLDEN_PREFIX))} "
        f"(R11.1)"
    )

    # The scored golden layouts are disjoint from the split's training layouts,
    # so evaluation never touches a training layout.
    assert scored_set.isdisjoint(split_union), (
        f"the Scorer evaluated training layout(s) {sorted(scored_set & split_union)}; "
        f"it must evaluate held-out golden layouts only (R11.1)"
    )
