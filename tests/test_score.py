"""Unit tests for Step 7 -- Score (``week1_harness.steps.score.ScoreStep``).

Covers task 11.4: the seventh Data Loop step, the golden-set Scorer. On site
this is where the freshly fine-tuned model would be evaluated on the held-out
golden layouts to produce mAP50, per-item recall, and look-alike confusion. The
harness never trains a real model (Step 6 is stubbed), so there is nothing to
evaluate for real: the Scorer fabricates those metrics from the run seed and
labels every one -- and even its own time cell -- a ``PLACEHOLDER`` with the
reason ``"training stubbed"``. What *is* honest is the input restriction: the
Scorer evaluates on the held-out golden layouts only. These example-based unit
tests pin that contract for a single Film -> Extract -> Auto-label -> Review ->
Train -> Score run:

* **Golden-only inputs (R11.1).** ``scored_golden_layouts`` is non-empty and
  every entry is a golden layout id (starts with ``"golden"``); no training
  (S1/S2) layout appears.
* **Metrics placeholder (R11.2).** ``metrics.map50`` and
  ``metrics.lookalike_confusion`` are each ``PLACEHOLDER`` with the exact reason
  ``"training stubbed"``.
* **Six recalls (R11.4).** ``recall_per_item`` has exactly the six practice-item
  classes as keys, and every value is ``PLACEHOLDER`` ``"training stubbed"``.
* **Time cell (R11.2).** Score's time cell is a ``PLACEHOLDER`` with the reason
  ``"training stubbed"`` -- the fabricated numbers are not measured on the local
  CPU, so timing them would mislead.
* **Seed-derived determinism (R11.3).** A second full pipeline run with the same
  seed into a separate temp dir yields identical metric values (``map50``, each
  per-item recall, and ``lookalike_confusion``).
* **Canonical identity.** The step reports ``number == 7`` and ``name == "Score"``.

Requirements: 11.1, 11.2, 11.4.

The tests are hermetic: each shared run writes under a pytest-managed temp
directory (``tmp_path_factory``, the module-scoped sibling of ``tmp_path``), so
nothing is written into the repository working tree. Film renders a few hundred
PNG frames, so the full Film -> ... -> Score pass runs exactly once in a
module-scoped fixture and every assertion reads from that single run rather than
re-filming per test; the determinism test uses one additional run into a
separate temp dir.
"""

from __future__ import annotations

import pytest

from week1_harness import capabilities
from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.review import ReviewStep
from week1_harness.steps.score import ScoreStep
from week1_harness.steps.train import TrainStep
from week1_harness.synth.items import CLASS_IDS

# A fixed seed makes every assertion concrete and reproducible (the
# default/start-date seed used across the other harness tests). The determinism
# test (R11.3) re-runs the whole pipeline with this SAME seed.
SEED = 20261009

# The exact honesty-marker reason the Scorer stamps on every fabricated metric
# and on its time cell (R11.2): the numbers are not real because training was
# stubbed (Step 6 did no real training).
STUB_REASON = "training stubbed"


def _run_pipeline_to_score(out_dir) -> "object":
    """Run Film -> Extract -> Auto-label -> Review -> Train -> Score once.

    The pipeline runner would store each step's payload under its step name for
    downstream steps; here we wire that handoff by hand -- each upstream step's
    payload is stored under ``ctx.artifacts[step.name]`` (``"Film"``,
    ``"Extract frames"``, ``"Auto-label"``, ``"Review"``, ``"Train"``) before
    ``ScoreStep`` reads ``ctx.artifacts["Train"]``. (Zero-shot, step 3, is not on
    Score's artifact path and is skipped.) The context carries a real detected
    ``Capabilities`` record (``capabilities.detect(None)`` -- filesystem/import
    checks only, no network), mirroring how the harness builds the context for a
    real run.

    Returns the Score :class:`~week1_harness.datamodel.StepResult`.
    """
    ctx = RunContext(seed=SEED, out_dir=out_dir, capabilities=capabilities.detect(None))

    # Run and hand off each upstream step under its canonical name, exactly as
    # the pipeline runner does (``ctx.artifacts[step.name] = result.payload``).
    for step in (FilmStep(), ExtractStep(), AutoLabelStep(), ReviewStep(), TrainStep()):
        result = step.run(ctx)
        ctx.artifacts[step.name] = result.payload

    return ScoreStep().run(ctx)


@pytest.fixture(scope="module")
def score_run(tmp_path_factory):
    """Run the Film -> ... -> Score pass once and share the Score result.

    Film writes a few hundred PNG frames, so the assertions below all read from
    this one run rather than re-filming per test. ``tmp_path_factory`` hands back
    a pytest-managed temp dir (the module-scoped equivalent of ``tmp_path``, with
    the same hermetic guarantee), so nothing touches the repo tree.
    """
    out_dir = tmp_path_factory.mktemp("score_out")
    return _run_pipeline_to_score(out_dir)


@pytest.fixture(scope="module")
def score_run_repeat(tmp_path_factory):
    """A second, independent full pipeline run with the SAME seed (R11.3).

    Runs into its own temp dir so it shares no state with ``score_run``; because
    the seed is identical, the Scorer's seed-derived metric values must match the
    first run's exactly (determinism, R11.3 / R3.4).
    """
    out_dir = tmp_path_factory.mktemp("score_out_repeat")
    return _run_pipeline_to_score(out_dir)


# --- Canonical identity --------------------------------------------------------

def test_score_step_number_and_name(score_run):
    """Score is the seventh Data Loop step, named "Score" (canonical order, R1.1)."""
    assert score_run.number == 7
    assert score_run.name == "Score"


# --- Golden-only inputs (R11.1) ------------------------------------------------

def test_scored_golden_layouts_are_golden_only(score_run):
    """Evaluation is restricted to the held-out golden layouts only (R11.1)."""
    scored = score_run.payload["scored_golden_layouts"]

    # Non-empty: the Scorer actually evaluated on some held-out golden layout(s).
    assert scored, "scored_golden_layouts is empty; the Scorer evaluated nothing (R11.1)"

    # Every scored layout is a golden layout id (golden ids are "golden_00"...).
    assert all(layout_id.startswith("golden") for layout_id in scored), (
        f"a non-golden layout was scored: "
        f"{sorted(lid for lid in scored if not lid.startswith('golden'))} (R11.1)"
    )

    # No training (S1/S2) layout leaked into the evaluation. Training layout ids
    # start with "s1_"/"s2_"; golden is reserved for the Scorer (R5.4/R11.1).
    assert not any(layout_id.startswith(("s1", "s2")) for layout_id in scored), (
        f"a training (S1/S2) layout leaked into scoring: "
        f"{sorted(lid for lid in scored if lid.startswith(('s1', 's2')))} (R11.1)"
    )


# --- Metrics placeholder (R11.2) -----------------------------------------------

def test_map50_is_placeholder_training_stubbed(score_run):
    """mAP50 is a PLACEHOLDER with the exact reason "training stubbed" (R11.2)."""
    map50 = score_run.payload["metrics"].map50
    assert map50.marker is Marker.PLACEHOLDER, (
        f"map50 marker is {map50.marker.value}, expected PLACEHOLDER (R11.2)"
    )
    assert map50.reason == STUB_REASON, (
        f"map50 reason is {map50.reason!r}, expected {STUB_REASON!r} (R11.2)"
    )


def test_lookalike_confusion_is_placeholder_training_stubbed(score_run):
    """Look-alike confusion is PLACEHOLDER reason "training stubbed" (R11.2)."""
    confusion = score_run.payload["metrics"].lookalike_confusion
    assert confusion.marker is Marker.PLACEHOLDER, (
        f"lookalike_confusion marker is {confusion.marker.value}, "
        f"expected PLACEHOLDER (R11.2)"
    )
    assert confusion.reason == STUB_REASON, (
        f"lookalike_confusion reason is {confusion.reason!r}, "
        f"expected {STUB_REASON!r} (R11.2)"
    )


# --- Six recalls (R11.4) -------------------------------------------------------

def test_recall_per_item_covers_six_classes_and_is_placeholder(score_run):
    """One PLACEHOLDER "training stubbed" recall per practice item (R11.4/R11.2)."""
    recall_per_item = score_run.payload["metrics"].recall_per_item

    # Exactly one entry per practice-item class -- the six classes, no more, no
    # fewer (R11.4).
    assert set(recall_per_item) == set(CLASS_IDS), (
        f"recall_per_item keys {sorted(recall_per_item)} != the six classes "
        f"{sorted(CLASS_IDS)} (R11.4)"
    )
    assert len(recall_per_item) == 6, (
        f"expected 6 per-item recall entries, got {len(recall_per_item)} (R11.4)"
    )

    # Every per-item recall is a PLACEHOLDER stand-in with the stub reason (R11.2).
    for class_id, recall in recall_per_item.items():
        assert recall.marker is Marker.PLACEHOLDER, (
            f"recall for {class_id!r} is {recall.marker.value}, "
            f"expected PLACEHOLDER (R11.2)"
        )
        assert recall.reason == STUB_REASON, (
            f"recall for {class_id!r} reason is {recall.reason!r}, "
            f"expected {STUB_REASON!r} (R11.2)"
        )


# --- Time cell (R11.2) ---------------------------------------------------------

def test_score_time_cell_is_placeholder_training_stubbed(score_run):
    """Score's time cell is a PLACEHOLDER reason "training stubbed" (R11.2).

    The metrics are fabricated from the seed, not measured on the local CPU, so
    even the step's time cell is a PLACEHOLDER -- there is no real scoring
    computation to time.
    """
    time_cell = score_run.time
    assert time_cell.marker is Marker.PLACEHOLDER, (
        f"Score time cell is {time_cell.marker.value}, expected PLACEHOLDER -- "
        f"the fabricated metrics are not a real CPU measurement (R11.2)"
    )
    assert time_cell.reason == STUB_REASON, (
        f"Score time reason is {time_cell.reason!r}, expected {STUB_REASON!r} (R11.2)"
    )


# --- Seed-derived determinism (R11.3) ------------------------------------------

def test_metrics_are_seed_derived_and_deterministic(score_run, score_run_repeat):
    """Two runs with the SAME seed emit identical metric values (R11.3).

    Each metric is drawn deterministically from ``rng_for(7)`` -- a sub-stream
    keyed by the run seed and the step number -- so a given seed always yields
    the same clearly-marked stand-ins across independent runs.
    """
    first = score_run.payload["metrics"]
    second = score_run_repeat.payload["metrics"]

    # mAP50 value is identical across the two same-seed runs.
    assert first.map50.value == second.map50.value, (
        f"map50 differs across same-seed runs: "
        f"{first.map50.value} != {second.map50.value} (R11.3)"
    )

    # Each per-item recall value is identical (same keys, same seed-derived value).
    assert set(first.recall_per_item) == set(second.recall_per_item), (
        "recall_per_item keys differ across same-seed runs (R11.3)"
    )
    for class_id in first.recall_per_item:
        assert (
            first.recall_per_item[class_id].value
            == second.recall_per_item[class_id].value
        ), (
            f"recall for {class_id!r} differs across same-seed runs: "
            f"{first.recall_per_item[class_id].value} != "
            f"{second.recall_per_item[class_id].value} (R11.3)"
        )

    # Look-alike confusion value is identical across the two same-seed runs.
    assert first.lookalike_confusion.value == second.lookalike_confusion.value, (
        f"lookalike_confusion differs across same-seed runs: "
        f"{first.lookalike_confusion.value} != {second.lookalike_confusion.value} (R11.3)"
    )
