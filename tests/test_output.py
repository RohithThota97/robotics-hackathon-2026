"""Unit tests for the atomic deliverable writers in ``week1_harness.output``.

Covers the two properties task 14.1 exists for: writes are atomic (a failed
write leaves neither a partial deliverable nor a stray temporary file), and the
marker/reason of every value survives JSON serialization, so a placeholder is
still visibly a placeholder in ``run.json``.

Also pins the payload projection: step payloads hold frozen dataclasses and
numpy mask rasters that ``json.dumps`` rejects, so ``run.json`` must store a
curated projection rather than the payload itself.

Requirements: 1.3, 4.2, 4.4, 4.5, 4.6, 14.1
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from week1_harness import output
from week1_harness.datamodel import StepResult
from week1_harness.markers import Marker, estimate, placeholder, real

SEED = 20261009


def _result(
    number: int = 1,
    name: str = "Film",
    *,
    time=None,
    payload: dict | None = None,
    notes: list | None = None,
) -> StepResult:
    """Build a StepResult with the six required fields."""
    return StepResult(
        number=number,
        name=name,
        how="how text",
        time=time if time is not None else real(1.5, "s"),
        payload=payload if payload is not None else {},
        notes=notes if notes is not None else [],
    )


def _temp_leftovers(directory) -> list[str]:
    """Names of write_atomic's temporary files still present in ``directory``."""
    return [n for n in os.listdir(directory) if n.startswith(".tmp-")]


# --- write_atomic ---------------------------------------------------------


def test_write_atomic_creates_file_and_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "doc.md"

    written = output.write_atomic(target, "hello\n")

    assert written == os.fspath(target)
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_write_atomic_overwrites_existing_file(tmp_path):
    target = tmp_path / "doc.md"
    target.write_text("old content\n", encoding="utf-8")

    output.write_atomic(target, "new content\n")

    assert target.read_text(encoding="utf-8") == "new content\n"
    assert _temp_leftovers(tmp_path) == []


def test_write_atomic_leaves_no_temp_file_on_failure(tmp_path, monkeypatch):
    """A crash mid-write must leave no partial file and no temp file (R1.3)."""
    target = tmp_path / "doc.md"

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(output.os, "replace", boom)

    with pytest.raises(OSError, match="disk full"):
        output.write_atomic(target, "partial\n")

    assert not target.exists()
    assert _temp_leftovers(tmp_path) == []


def test_write_atomic_never_leaves_a_partial_target(tmp_path, monkeypatch):
    """An existing file is left untouched when the replacement write fails."""
    target = tmp_path / "doc.md"
    target.write_text("intact\n", encoding="utf-8")

    monkeypatch.setattr(
        output.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
    )

    with pytest.raises(OSError):
        output.write_atomic(target, "replacement\n")

    assert target.read_text(encoding="utf-8") == "intact\n"
    assert _temp_leftovers(tmp_path) == []


# --- marker preservation --------------------------------------------------


@pytest.mark.parametrize(
    "marked, expected_marker, expected_reason",
    [
        (real(2.5, "s"), "REAL", None),
        (placeholder("n/a", reason="training stubbed"), "PLACEHOLDER", "training stubbed"),
        (estimate(32.6, reason="NVIDIA GPU absent", unit=" min"), "ESTIMATE", "NVIDIA GPU absent"),
    ],
    ids=["real", "placeholder", "estimate"],
)
def test_marked_values_keep_marker_and_reason_in_json(marked, expected_marker, expected_reason):
    """The honesty guarantee must survive serialization (R4.2, R4.4-R4.6)."""
    encoded = json.loads(json.dumps(output._jsonable(marked)))

    assert encoded["marker"] == expected_marker
    assert encoded["reason"] == expected_reason


def test_jsonable_coerces_numpy_scalars():
    assert output._jsonable(np.float64(1.25)) == pytest.approx(1.25)
    assert output._jsonable(np.int64(7)) == 7


def test_jsonable_falls_back_instead_of_raising():
    """An unexpected object degrades to repr() rather than breaking the run."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert output._jsonable(Opaque()) == "<opaque>"
    # A numpy mask raster is the real-world case: it must not raise.
    json.dumps(output._jsonable(np.zeros((2, 2), dtype=np.uint8)))


# --- payload projection ---------------------------------------------------


def test_projection_keeps_scalars_and_counts_bulk_collections():
    payload = {
        "class_ids": ["a", "b"],
        "instances_per_class": {"a": 3},
        "training_images": [object(), object(), object()],
        "masks": [object()],
        "scene": object(),  # neither scalar nor counted: dropped
    }

    projected = output._project_payload(payload)

    assert projected["class_ids"] == ["a", "b"]
    assert projected["instances_per_class"] == {"a": 3}
    assert projected["training_images_count"] == 3
    assert projected["masks_count"] == 1
    assert "scene" not in projected
    assert "training_images" not in projected


# --- write_deliverables --------------------------------------------------


def test_write_deliverables_writes_three_files(tmp_path):
    paths = output.write_deliverables(
        tmp_path,
        timing_table="# timing\n",
        dataset_card="# card\n",
        results=[_result()],
        capabilities=None,
        seed=SEED,
    )

    assert set(paths) == {"timing_table", "dataset_card", "run_json"}
    for name in (
        output.TIMING_TABLE_FILENAME,
        output.DATASET_CARD_FILENAME,
        output.RUN_JSON_FILENAME,
    ):
        assert (tmp_path / name).is_file()
    assert (tmp_path / output.TIMING_TABLE_FILENAME).read_text() == "# timing\n"
    assert _temp_leftovers(tmp_path) == []


def test_run_json_is_parseable_and_records_seed_and_markers(tmp_path):
    results = [
        _result(1, "Film", time=real(0.5, "s")),
        _result(
            3,
            "Zero-shot baseline",
            time=placeholder("n/a", reason="zero-shot disabled by flag"),
            payload={"class_ids": ["x"]},
            notes=[placeholder("n/a", reason="gated weights absent")],
        ),
    ]

    output.write_deliverables(
        tmp_path,
        timing_table="t\n",
        dataset_card="c\n",
        results=results,
        capabilities=None,
        seed=SEED,
    )
    run = json.loads((tmp_path / output.RUN_JSON_FILENAME).read_text(encoding="utf-8"))

    assert run["seed"] == SEED
    assert [s["number"] for s in run["steps"]] == [1, 3]
    zero_shot = run["steps"][1]
    assert zero_shot["time"]["marker"] == Marker.PLACEHOLDER.value
    assert zero_shot["time"]["reason"] == "zero-shot disabled by flag"
    assert zero_shot["notes"][0]["reason"] == "gated weights absent"
    assert zero_shot["payload"]["class_ids"] == ["x"]


def test_run_json_survives_unserializable_payloads(tmp_path):
    """Real payloads carry numpy arrays and dataclasses; this must not raise."""
    results = [_result(payload={"masks": [np.zeros((4, 4))], "scene": object()})]

    output.write_deliverables(
        tmp_path,
        timing_table="t\n",
        dataset_card="c\n",
        results=results,
        capabilities=None,
        seed=SEED,
    )

    run = json.loads((tmp_path / output.RUN_JSON_FILENAME).read_text(encoding="utf-8"))
    assert run["steps"][0]["payload"]["masks_count"] == 1
