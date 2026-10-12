"""Atomic writers for the three Week 1 run outputs (task 14.1).

This module owns every file the harness leaves behind: the two deliverables
(``timing_table.md``, ``dataset_card.md``) and the structured ``run.json``.

Two properties matter here.

**Atomicity (R1.3).** Each file is written to a temporary file in the *same*
directory and then :func:`os.replace`-d into place. A rename within one
filesystem is atomic, so a crash or a full disk can never leave a half-written
deliverable that someone might quote from. A partially written temporary file
is removed on failure.

**The honesty guarantee reaches JSON too.** ``run.json`` stores every
:class:`~week1_harness.markers.MarkedValue` as an object carrying its
``marker`` and ``reason``, so a placeholder stays visibly a placeholder after
serialization, not just in the rendered Markdown (R4.2, R4.4-R4.6).

Step payloads are deliberately *not* serialized wholesale. They hold frozen
dataclasses, ``Scene``/``ClipSpec`` objects and numpy mask rasters, none of
which ``json.dumps`` accepts. Instead each step contributes a curated
projection: the scalar payload fields in :data:`SCALAR_PAYLOAD_KEYS`, plus a
length for each bulk collection in :data:`COUNTED_PAYLOAD_KEYS`. Anything
unexpected still degrades through :func:`_jsonable` rather than raising, so a
future payload change cannot break a run at the final step.

Requirements: 1.3, 4.2, 4.4, 4.5, 4.6, 13.1, 14.1
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import Any

from week1_harness.datamodel import StepResult
from week1_harness.markers import MarkedValue

__all__ = [
    "TIMING_TABLE_FILENAME",
    "DATASET_CARD_FILENAME",
    "RUN_JSON_FILENAME",
    "SCALAR_PAYLOAD_KEYS",
    "COUNTED_PAYLOAD_KEYS",
    "write_atomic",
    "write_deliverables",
]

TIMING_TABLE_FILENAME = "timing_table.md"
DATASET_CARD_FILENAME = "dataset_card.md"
RUN_JSON_FILENAME = "run.json"

# Payload fields that are already JSON-safe (scalars, or collections of
# scalars) and worth keeping in run.json verbatim.
SCALAR_PAYLOAD_KEYS = frozenset(
    {
        "canvas_size",
        "class_ids",
        "detected_items",
        "effective_fps",
        "fps",
        "frames_dir",
        "golden_count",
        "golden_layout_id",
        "instances_per_class",
        "mask_count",
        "per_item_status",
        "removed_count",
        "reshuffled_items",
        "retained_count",
        "scored_golden_layouts",
        "second_check_count",
        "source_fps",
        "stride",
        "train_layouts",
        "training_count",
        "val_layouts",
    }
)

# Bulk payload fields holding dataclasses or numpy data. We record how many
# there were, never their contents.
COUNTED_PAYLOAD_KEYS = frozenset(
    {
        "all_instances",
        "clips",
        "frames",
        "golden_images",
        "layouts",
        "masks",
        "reviewed_labels",
        "training_images",
    }
)


def _jsonable(value: Any) -> Any:
    """Coerce ``value`` into something :func:`json.dumps` accepts.

    Known shapes are converted faithfully; anything unrecognised falls back to
    ``repr()`` so serializing a run can never fail outright.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, MarkedValue):
        # Marker subclasses str, so asdict() already yields a JSON-safe label.
        return {
            "value": _jsonable(value.value),
            "marker": value.marker.value,
            "reason": value.reason,
            "unit": value.unit,
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    # numpy scalars (and anything else exposing a 0-d .item()).
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        try:
            return _jsonable(item())
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {
                f.name: _jsonable(getattr(value, f.name))
                for f in dataclasses.fields(value)
            }
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    return repr(value)


def _project_payload(payload: dict) -> dict:
    """Reduce one step's payload to its JSON-safe projection."""
    projected: dict[str, Any] = {}
    for key in sorted(payload):
        if key in SCALAR_PAYLOAD_KEYS:
            projected[key] = _jsonable(payload[key])
        elif key in COUNTED_PAYLOAD_KEYS:
            try:
                projected[f"{key}_count"] = len(payload[key])
            except TypeError:  # pragma: no cover - defensive
                pass
    return projected


def _project_result(result: StepResult) -> dict:
    """Reduce one :class:`StepResult` to its run.json entry."""
    return {
        "number": result.number,
        "name": result.name,
        "how": result.how,
        "time": _jsonable(result.time),
        "notes": [_jsonable(note) for note in result.notes],
        "payload": _project_payload(result.payload),
    }


def _project_capabilities(capabilities: Any) -> Any:
    """Reduce the detected capabilities to a JSON-safe mapping."""
    if capabilities is None:
        return None
    return _jsonable(capabilities)


def write_atomic(path: str | os.PathLike[str], text: str) -> str:
    """Write ``text`` to ``path`` atomically, creating parent directories.

    The content lands in a temporary file beside the target and is renamed over
    it, so readers only ever see the complete file (R1.3). Returns the written
    path, matching :func:`week1_harness.synth.render.save_png`.
    """
    target = os.fspath(path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        # Never leave a stray partial file behind.
        try:
            os.unlink(tmp)
        except OSError:  # pragma: no cover - defensive
            pass
        raise
    return target


def write_deliverables(
    out_dir: str | os.PathLike[str],
    timing_table: str,
    dataset_card: str,
    results: list[StepResult],
    capabilities: Any = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Write both deliverables and ``run.json`` into ``out_dir`` atomically.

    Returns a mapping of logical name (``"timing_table"``, ``"dataset_card"``,
    ``"run_json"``) to the path written.
    """
    base = os.fspath(out_dir)
    run = {
        "seed": seed,
        "capabilities": _project_capabilities(capabilities),
        "steps": [_project_result(result) for result in results],
    }
    run_json = json.dumps(run, indent=2, sort_keys=False) + "\n"

    return {
        "timing_table": write_atomic(
            os.path.join(base, TIMING_TABLE_FILENAME), timing_table
        ),
        "dataset_card": write_atomic(
            os.path.join(base, DATASET_CARD_FILENAME), dataset_card
        ),
        "run_json": write_atomic(os.path.join(base, RUN_JSON_FILENAME), run_json),
    }
