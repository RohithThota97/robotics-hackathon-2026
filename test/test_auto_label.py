"""Unit tests for Step 4 -- Auto-label (``week1_harness.steps.auto_label.AutoLabelStep``).

Covers task 10.3: the fourth Data Loop step, which stands in for SAM 3 by
*rasterizing one instance mask per labelled training instance from the known
synthetic polygon geometry* Film already produced. Because the rasterization is
genuine local-CPU pixel work, Auto-label's time cell is ``REAL``; the only
non-real aspect is the *method* (SAM 3 would have produced these masks on site),
which is recorded as a ``PLACEHOLDER`` note rather than in the time cell (R4.1).

These are example-based unit tests pinning the concrete class-assignment
contract of a single Film -> Extract -> Auto-label run (the mask<->instance
bijection is Property 6, task 10.2). Here we assert:

* **One class per mask (R8.3).** Every ``Mask.class_id`` is exactly one of the
  six practice-item classes -- a single class id, not a collection.
* **Look-alike pair kept separate (R8.4).** ``screwdriver_small`` and
  ``screwdriver_large`` both appear among the mask class ids as two distinct,
  non-empty classes; they are never merged into one.
* **Class coverage.** The set of mask class ids is a subset of ``CLASS_IDS`` and
  in fact covers all six classes, with at least one mask per class.
* **SAM 3 placeholder note (R8.2).** ``result.notes`` carries a ``PLACEHOLDER``
  ``MarkedValue`` whose reason is exactly ``"gated weights absent"``.
* **REAL time cell (R4.1).** Auto-label's time cell is ``REAL`` with no reason.

Requirements: 8.2, 8.3, 8.4.

The tests are hermetic: the one shared run writes under a pytest-managed temp
directory (``tmp_path_factory``, the module-scoped sibling of ``tmp_path``), so
nothing is written into the repository working tree. Film renders a few hundred
PNG frames, so Film + Extract + Auto-label run exactly once in a module-scoped
fixture and every assertion reads from that single run rather than re-filming
per test.
"""

from collections import Counter

import pytest

from week1_harness import capabilities
from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.auto_label import AutoLabelStep
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.synth.items import CLASS_IDS

# A fixed seed makes every assertion concrete and reproducible (the
# default/start-date seed used across the other harness tests).
SEED = 20261009

# The look-alike pair held as two separate classes end to end (R5.1/R8.4).
LOOKALIKE_PAIR = ("screwdriver_small", "screwdriver_large")

# The SAM 3 text/box-prompt mask generation the harness deliberately did not run
# (no gated weights); emitted as a PLACEHOLDER note with this exact reason (R8.2).
SAM3_REASON = "gated weights absent"


@pytest.fixture(scope="module")
def auto_label_result(tmp_path_factory):
    """Run ``FilmStep`` -> ``ExtractStep`` -> ``AutoLabelStep`` once and share the result.

    Film writes a few hundred PNG frames, so the assertions below all read from
    this one run rather than re-filming per test. ``tmp_path_factory`` hands back
    a pytest-managed temp dir (the module-scoped equivalent of ``tmp_path``, with
    the same hermetic guarantee), so nothing touches the repo tree.

    The pipeline runner would store each step's payload under its step name for
    downstream steps; here we wire that handoff by hand -- ``ExtractStep.run``
    reads ``ctx.artifacts["Film"]`` and ``AutoLabelStep.run`` reads
    ``ctx.artifacts["Extract frames"]`` -- then run each step against the same
    seeded context. The context carries a real detected ``Capabilities`` record
    (``capabilities.detect(None)``, filesystem/import checks only, no network);
    Auto-label does not consult it, but this mirrors how the harness builds the
    context for a real run.
    """
    out_dir = tmp_path_factory.mktemp("auto_label_out")
    ctx = RunContext(seed=SEED, out_dir=out_dir, capabilities=capabilities.detect(None))

    film_result = FilmStep().run(ctx)
    ctx.artifacts["Film"] = film_result.payload

    extract_result = ExtractStep().run(ctx)
    ctx.artifacts["Extract frames"] = extract_result.payload

    return AutoLabelStep().run(ctx)


def _masks(auto_label_result):
    """The produced masks, guarded to be a non-empty list (no vacuous passes)."""
    masks = auto_label_result.payload["masks"]
    assert masks, "Auto-label produced no masks"
    return masks


# --- One class per mask (R8.3) -------------------------------------------------

def test_each_mask_has_exactly_one_class_in_class_ids(auto_label_result):
    """Every mask carries a single class id drawn from the six classes (R8.3)."""
    class_ids = set(CLASS_IDS)
    for mask in _masks(auto_label_result):
        # A single class per mask: class_id is one scalar label (a str), not a
        # list/set/tuple of classes...
        assert isinstance(mask.class_id, str), (
            f"mask {mask.instance_id} class_id is not a single class label: "
            f"{mask.class_id!r}"
        )
        # ...and that one label is one of the six practice-item classes (R8.3).
        assert mask.class_id in class_ids, (
            f"mask {mask.instance_id} has class {mask.class_id!r} not in CLASS_IDS"
        )


# --- Look-alike pair kept separate (R8.4) --------------------------------------

def test_lookalike_pair_kept_as_two_distinct_classes(auto_label_result):
    """Both screwdriver classes appear among the masks as two distinct classes (R8.4)."""
    small, large = LOOKALIKE_PAIR
    counts = Counter(mask.class_id for mask in _masks(auto_label_result))

    # Both look-alikes exist among the masks (there are masks of each)...
    assert counts[small] >= 1, f"no masks for {small!r}"
    assert counts[large] >= 1, f"no masks for {large!r}"

    # ...and they are two distinct classes, never merged into one. The two ids
    # are different and both survive as separate keys in the class tally.
    assert small != large
    screwdriver_classes = {cid for cid in counts if cid.startswith("screwdriver")}
    assert screwdriver_classes == {small, large}, (
        f"look-alike pair not kept as exactly two separate classes: "
        f"{sorted(screwdriver_classes)}"
    )


# --- Class coverage: subset of CLASS_IDS and all six represented ---------------

def test_mask_class_ids_cover_all_six_classes(auto_label_result):
    """Mask classes are a subset of CLASS_IDS and cover all six, >=1 mask each."""
    class_ids = set(CLASS_IDS)
    counts = Counter(mask.class_id for mask in _masks(auto_label_result))
    present = set(counts)

    # Never a class outside the six (subset of CLASS_IDS)...
    assert present <= class_ids, (
        f"masks carry classes outside CLASS_IDS: {sorted(present - class_ids)}"
    )
    # ...and every one of the six training classes is represented (>=1 mask).
    assert present == class_ids, (
        f"not all six classes are represented among the masks; missing: "
        f"{sorted(class_ids - present)}"
    )
    for class_id in CLASS_IDS:
        assert counts[class_id] >= 1, f"class {class_id!r} has no mask"


# --- SAM 3 placeholder note (R8.2) ---------------------------------------------

def test_sam3_mask_generation_is_a_placeholder_note(auto_label_result):
    """notes carry a PLACEHOLDER MarkedValue with reason 'gated weights absent' (R8.2)."""
    sam3_notes = [
        note
        for note in auto_label_result.notes
        if note.marker is Marker.PLACEHOLDER and note.reason == SAM3_REASON
    ]
    assert sam3_notes, (
        "expected a PLACEHOLDER note with reason "
        f"{SAM3_REASON!r} in Auto-label notes, got "
        f"{[(n.marker, n.reason) for n in auto_label_result.notes]}"
    )


# --- REAL time cell (R4.1) -----------------------------------------------------

def test_auto_label_time_cell_is_real_and_unmarked(auto_label_result):
    """Auto-label rasterizes real masks on CPU: REAL time cell, no reason (R4.1)."""
    # Anchor that this is the Auto-label step's result.
    assert auto_label_result.number == 4
    assert auto_label_result.name == "Auto-label"

    # The rasterization is a genuine local-CPU measurement: REAL marker, no
    # reason. The only non-real aspect (the SAM 3 method) is a note, not here.
    assert auto_label_result.time.marker is Marker.REAL
    assert auto_label_result.time.reason is None
