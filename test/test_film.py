"""Unit tests for Step 1 -- Film (``week1_harness.steps.film.FilmStep``).

Covers task 8.2: the first Data Loop step, which stands in for real capture by
*generating* the synthetic source clips with Pillow and writing them under
``<out_dir>/frames/<layout_id>/``. Because Film does genuine local-CPU drawing,
its time cell is ``REAL`` and it emits no placeholder/estimate notes (R4.1).

These are example-based unit tests (the multi-run session-coverage property is
Property 10, task 14.7; the full-run determinism property is Property 4,
task 14.5). Here we pin the concrete contract of a single ``FilmStep.run``:

* **Result metadata (R4.1).** ``number``/``name`` are the canonical ``1``/
  ``"Film"``, the time cell is ``REAL`` with no reason, and ``notes`` is empty.
* **Six classes incl. the look-alike pair (R5.1).** The payload surfaces exactly
  the six practice-item classes, with ``screwdriver_small`` and
  ``screwdriver_large`` held as two separate classes.
* **S1/S2/golden structure (R5.2-R5.4).** S1 holds each item alone under the
  three lightings; there are >= 5 S2 layouts of all six items; there are 1-2
  golden layouts tagged ``session == "golden"``.
* **Frames written (R5.5).** PNG frames actually land on disk under
  ``<out_dir>/frames/<layout_id>/`` and are non-empty.
* **Determinism (R5.5 / R3.3).** The same seed into two separate out dirs writes
  byte-identical frame files.

Requirements: 5.1, 5.2, 5.3, 5.4.

The tests are hermetic: every ``out_dir`` is a pytest temp directory, so nothing
is written into the repository working tree.
"""

import os
from collections import defaultdict

import pytest

from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.film import FilmStep
from week1_harness.synth.items import CLASS_IDS
from week1_harness.synth.scene import LIGHTING_CONDITIONS

# A fixed seed makes every assertion concrete and reproducible (matches the
# default/start-date seed used across the other synth tests).
SEED = 20261009


def _ctx(seed: int, out_dir) -> RunContext:
    """Build a RunContext for a Film run.

    ``FilmStep.run`` reaches for ``ctx.rng_for(1)`` (via ``scene.compose``),
    ``ctx.out_dir``, and ``ctx.seed`` only; it never touches ``capabilities``, so
    it is passed as ``None`` per the task contract.
    """
    return RunContext(seed=seed, out_dir=out_dir, capabilities=None)


@pytest.fixture(scope="module")
def film_run(tmp_path_factory):
    """Run ``FilmStep`` exactly once and share the result + out dir.

    Film writes a few hundred PNG frames, so the single-run assertions below all
    read from this one run rather than re-filming per test. ``tmp_path_factory``
    hands back a pytest-managed temp dir (same hermetic guarantee as
    ``tmp_path``), so nothing touches the repo tree. The determinism test does
    its own two runs on purpose.
    """
    out_dir = tmp_path_factory.mktemp("film_out")
    result = FilmStep().run(_ctx(SEED, out_dir))
    return result, out_dir


# --- Result metadata and the REAL time cell (R4.1) -----------------------------

def test_film_result_metadata_is_real_and_unmarked(film_run):
    """number/name are canonical, the time cell is REAL (no reason), notes empty."""
    result, _ = film_run

    assert result.number == 1
    assert result.name == "Film"

    # Film is a genuine local-CPU step: REAL time cell, no reason, no notes (R4.1).
    assert result.time.marker is Marker.REAL
    assert result.time.reason is None
    assert result.notes == []


# --- Six classes incl. the look-alike pair (R5.1) ------------------------------

def test_payload_exposes_six_classes_with_lookalike_pair_separate(film_run):
    """The payload carries exactly the six classes, screwdrivers kept distinct (R5.1)."""
    result, _ = film_run
    class_ids = result.payload["class_ids"]

    # Exactly the six canonical classes, and no duplicates -> each is a separate
    # class (the look-alike pair is not collapsed into one).
    assert set(class_ids) == set(CLASS_IDS)
    assert len(class_ids) == len(set(class_ids)) == 6

    # The look-alike pair is present as two distinct classes.
    screwdrivers = sorted(c for c in class_ids if c.startswith("screwdriver"))
    assert screwdrivers == ["screwdriver_large", "screwdriver_small"]

    # The six classes also actually appear in the composed layouts, not just in
    # the declared id list.
    placed = {p.class_id for layout in result.payload["layouts"] for p in layout.placements}
    assert placed == set(CLASS_IDS)


# --- S1/S2/golden structure (R5.2-R5.4) ----------------------------------------

def test_payload_layouts_cover_s1_s2_and_golden_structure(film_run):
    """S1 each item x3 lighting, >=5 S2 of all six, 1-2 golden tagged golden."""
    result, _ = film_run
    layouts = result.payload["layouts"]

    s1 = [layout for layout in layouts if layout.session == "S1"]
    s2 = [layout for layout in layouts if layout.session == "S2"]
    golden = [layout for layout in layouts if layout.session == "golden"]

    # S1 (R5.2): each layout is a single item; all six items under three lightings.
    assert len(LIGHTING_CONDITIONS) == 3
    lightings_by_item: dict[str, set[str]] = defaultdict(set)
    for layout in s1:
        assert len(layout.placements) == 1
        lightings_by_item[layout.placements[0].class_id].add(layout.lighting)
    assert set(lightings_by_item) == set(CLASS_IDS)
    for class_id in CLASS_IDS:
        assert lightings_by_item[class_id] == set(LIGHTING_CONDITIONS)
    assert len(s1) == len(CLASS_IDS) * len(LIGHTING_CONDITIONS)

    # S2 (R5.3): at least five shuffled layouts, each containing all six items.
    assert len(s2) >= 5
    for layout in s2:
        assert sorted(p.class_id for p in layout.placements) == sorted(CLASS_IDS)

    # Golden (R5.4): one or two held-out layouts of all six items, tagged golden.
    assert len(golden) in {1, 2}
    for layout in golden:
        assert layout.session == "golden"
        assert sorted(p.class_id for p in layout.placements) == sorted(CLASS_IDS)


# --- Frames written to disk (R5.5) ---------------------------------------------

def test_png_frames_are_written_under_out_dir_frames(film_run):
    """Every layout's clip lands as non-empty PNGs under frames/<layout_id>/ (R5.5)."""
    result, out_dir = film_run

    frames_dir = result.payload["frames_dir"]
    assert frames_dir == os.path.join(os.fspath(out_dir), "frames")
    assert os.path.isdir(frames_dir)

    frames = result.payload["frames"]
    # Every composed layout has a clip of written frames.
    layout_ids = {layout.layout_id for layout in result.payload["layouts"]}
    assert set(frames) == layout_ids

    written = 0
    for layout_id, records in frames.items():
        assert records, f"layout {layout_id} wrote no frames"
        for record in records:
            path = record["path"]
            # Lives directly under <frames_dir>/<layout_id>/ and is a PNG.
            assert os.path.dirname(path) == os.path.join(frames_dir, layout_id)
            assert path.endswith(".png")
            # Actually on disk and non-empty.
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            written += 1

    assert written > 0


# --- Determinism from one seed (R5.5 / R3.3) -----------------------------------

def _read_frame_bytes(out_dir) -> dict[str, bytes]:
    """Map each written frame (path relative to ``out_dir``) to its file bytes."""
    base = os.fspath(out_dir)
    frames_dir = os.path.join(base, "frames")
    contents: dict[str, bytes] = {}
    for root, _dirs, files in os.walk(frames_dir):
        for name in files:
            abs_path = os.path.join(root, name)
            with open(abs_path, "rb") as handle:
                contents[os.path.relpath(abs_path, base)] = handle.read()
    return contents


def test_frames_are_byte_identical_for_the_same_seed(tmp_path):
    """Two runs with the same seed write byte-identical frame files (R5.5/R3.3)."""
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    FilmStep().run(_ctx(SEED, out_a))
    FilmStep().run(_ctx(SEED, out_b))

    frames_a = _read_frame_bytes(out_a)
    frames_b = _read_frame_bytes(out_b)

    # Both runs wrote something, and wrote the same set of (relative) frame paths.
    assert frames_a
    assert set(frames_a) == set(frames_b)

    # And every matching frame is byte-for-byte identical.
    for rel_path in sorted(frames_a):
        assert frames_a[rel_path] == frames_b[rel_path], (
            f"frame {rel_path} differs across two runs with the same seed"
        )
