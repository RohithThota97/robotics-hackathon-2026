"""Reporters -- the Dataset_Card deliverable (Requirement 14).

:func:`render_dataset_card` renders Week 1 deliverable (b): the one-page dataset
card. It is a **pure function** of the collected
:class:`~week1_harness.datamodel.StepResult`s -- it does no I/O and reads nothing
but the ``results`` list -- so it is trivially testable and can never depend on
the filesystem or run order.

The card is a single Markdown section (R14.5) that opens with the honesty notice
(R4.3) and then reports the four required fields:

* **Classes (R14.1).** The six practice-item classes, including the look-alike
  pair (``screwdriver_small`` / ``screwdriver_large``) held as *two* separate
  classes. The class list is sourced from the "Extract frames" step's payload
  (``class_ids``, falling back to the keys of ``instances_per_class``), never
  hard-coded here, so the card always reflects what the pipeline actually
  produced.
* **Labelled-instance count per class (R14.2).** One count per class, read from
  the "Extract frames" payload's ``instances_per_class`` tally.
* **Sessions (R14.3).** S1, S2, and Golden with an image count for each. The
  S1/S2 counts are derived from the sessions tagged on the training images (each
  :class:`~week1_harness.datamodel.ExtractedImage` carries ``session``); the
  golden count is ``len(golden_images)``. The training total is ``S1 + S2``.
* **Known gaps (R14.4).** Built by **scanning every non-``REAL``
  :class:`~week1_harness.markers.MarkedValue` across all**
  :class:`~week1_harness.datamodel.StepResult`s -- each result's ``time`` cell
  and each :class:`~week1_harness.markers.MarkedValue` in its ``notes`` -- and
  keeping the ``PLACEHOLDER`` / ``ESTIMATE`` ones, each listed with its marker
  and reason. Because the list is derived from exactly what the steps emitted, it
  cannot drift from the run's real honesty markers (this is the whole point of
  Requirement 4's key tension: nobody can read the card as a real measurement).

The "Extract frames" result is located by name among ``results``. The function
is defensive about a missing step or payload key (so a partial result list still
renders a well-formed card), though a full eight-step run always carries every
field.

Scope (task 13.2 only): this module renders the Dataset_Card. The Timing_Table
(task 13.1), structured ``run.json`` / atomic writes (task 14.1), and the CLI
wiring (task 14.2) live elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from week1_harness.markers import Marker

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.datamodel import ExtractedImage, StepResult
    from week1_harness.markers import MarkedValue

__all__ = ["render_dataset_card", "HONESTY_NOTICE"]


# The honesty notice shown at the start of BOTH deliverables (R4.3). Kept
# byte-for-byte identical to the Timing_Table's notice and to design.md's
# "Reporters and output format" notice, so the two reports open exactly the same
# way and nobody can mistake either for a real Week 1 measurement. (This module
# may define it locally per task 13.2; the Timing_Table uses the same text.)
HONESTY_NOTICE = (
    "> **HONESTY NOTICE:** This run used **synthetic inputs**, "
    "**CPU-only execution**, and **stubbed training**. "
    "It does **not** report "
    "a real Week 1 measurement. "
    "Values tagged `[PLACEHOLDER: \u2026]` or "
    "`[ESTIMATE: \u2026]` are fabricated stand-ins; the reason is shown on each."
)

# The canonical name of the Extract step (its ``StepResult.name`` and the key
# under which its payload is stored in the artifact store). Its payload carries
# the class ids, the per-class tallies, and both image sets that the card's
# first three fields are built from.
EXTRACT_STEP_NAME = "Extract frames"

# The three session tags an extracted image / golden image can carry (R6.4).
_SESSION_S1 = "S1"
_SESSION_S2 = "S2"

# Markdown document title. The notice precedes it so the card still *opens* with
# the honesty notice (R4.3); the whole thing below the notice is one section
# (R14.5) -- a single ``#`` heading with bold field labels, never sub-headers.
_TITLE = "# Dataset Card - Week 1 Practice (SYNTHETIC)"


def render_dataset_card(results: list[StepResult]) -> str:
    """Render the one-page Dataset_Card (deliverable b) as Markdown (R14).

    Pure function: reads only ``results`` and performs no I/O. Opens with the
    honesty notice (R4.3) and renders the four required fields -- classes
    (R14.1), per-class labelled-instance counts (R14.2), S1/S2/Golden image
    counts (R14.3), and the scanned known-gaps list (R14.4) -- all within one
    section (R14.5).

    Args:
        results: the collected per-step :class:`StepResult`s from a run. The
            "Extract frames" result supplies the classes, counts, and sessions;
            every result is scanned for the known-gaps list.

    Returns:
        The Dataset_Card as a Markdown string, beginning with
        :data:`HONESTY_NOTICE`.
    """
    extract = _find_result(results, EXTRACT_STEP_NAME)
    payload = extract.payload if extract is not None else {}

    class_ids = _class_ids(payload)
    instances_per_class = payload.get("instances_per_class", {}) or {}
    training_images = payload.get("training_images", []) or []
    golden_images = payload.get("golden_images", []) or []

    lines: list[str] = [HONESTY_NOTICE, "", _TITLE, ""]
    lines += _classes_block(class_ids)
    lines.append("")
    lines += _counts_block(class_ids, instances_per_class)
    lines.append("")
    lines += _sessions_block(training_images, golden_images)
    lines.append("")
    lines += _known_gaps_block(results)

    return "\n".join(lines) + "\n"


def _find_result(results: list[StepResult], name: str) -> StepResult | None:
    """Return the :class:`StepResult` named ``name``, or ``None`` if absent.

    Locating the step by name (rather than by index) keeps the card robust to a
    partial or re-ordered result list; in a full eight-step run the match is
    always present.
    """
    for result in results:
        if result.name == name:
            return result
    return None


def _class_ids(payload: dict) -> list[str]:
    """Source the six class ids from the Extract payload (R14.1).

    Prefers the explicit ``class_ids`` list; falls back to the keys of the
    ``instances_per_class`` tally so the card still lists every class even if the
    explicit list is missing. Returns ``[]`` when neither is present (defensive).
    """
    class_ids = payload.get("class_ids")
    if class_ids:
        return list(class_ids)
    return list((payload.get("instances_per_class") or {}).keys())


def _lookalike_pairs(class_ids: list[str]) -> list[list[str]]:
    """Group class ids that share a shape prefix into look-alike groups (R14.1).

    The look-alike pair (``screwdriver_small`` / ``screwdriver_large``) shares
    the ``screwdriver`` prefix before the final ``_``; every other item resolves
    to a singleton group. Groups with two or more members are the look-alike
    groups, surfaced so the card can call out that the pair is held as two
    separate classes without hard-coding the item names. First-occurrence order
    is preserved so the output is stable.
    """
    groups: dict[str, list[str]] = {}
    for class_id in class_ids:
        prefix = class_id.rsplit("_", 1)[0] if "_" in class_id else class_id
        groups.setdefault(prefix, []).append(class_id)
    return [members for members in groups.values() if len(members) >= 2]


def _classes_block(class_ids: list[str]) -> list[str]:
    """Render the Classes field: the six classes incl. the pair as two (R14.1)."""
    count = len(class_ids)
    listing = ", ".join(class_ids) if class_ids else "(none)"
    block = [
        f"**Classes ({count}, incl. look-alike pair held as two classes):**",
        "",
        f"{listing}.",
    ]
    for pair in _lookalike_pairs(class_ids):
        block.append("")
        block.append(
            "Look-alike pair (held as two separate classes): " + " / ".join(pair) + "."
        )
    return block


def _counts_block(class_ids: list[str], instances_per_class: dict) -> list[str]:
    """Render the per-class labelled-instance count table (R14.2).

    One row per class in canonical order, each with its count from the Extract
    step's ``instances_per_class`` tally (defaulting to ``0`` if a class is
    somehow missing from the tally).
    """
    block = [
        "**Labelled instances per class:**",
        "",
        "| Class | Labelled instances |",
        "| --- | --- |",
    ]
    for class_id in class_ids:
        block.append(f"| {class_id} | {instances_per_class.get(class_id, 0)} |")
    return block


def _sessions_block(
    training_images: list[ExtractedImage],
    golden_images: list[ExtractedImage],
) -> list[str]:
    """Render the Sessions table: S1, S2, Golden with image counts (R14.3).

    S1/S2 image counts come from the sessions tagged on the training images;
    the golden count is ``len(golden_images)``. The training total is ``S1 + S2``
    (the golden layouts are held out of training). The Notes column reports the
    distinct source-layout count per session, derived from the images' source
    layout ids.
    """
    s1_images = [img for img in training_images if _session_of(img) == _SESSION_S1]
    s2_images = [img for img in training_images if _session_of(img) == _SESSION_S2]

    s1_count = len(s1_images)
    s2_count = len(s2_images)
    golden_count = len(golden_images)
    training_total = s1_count + s2_count

    return [
        "**Sessions:**",
        "",
        "| Session | Images | Notes |",
        "| --- | --- | --- |",
        f"| S1 (each item alone, 3 lighting) | {s1_count} "
        f"| {_layout_count(s1_images)} layouts |",
        f"| S2 (5+ shuffled layouts, all items) | {s2_count} "
        f"| {_layout_count(s2_images)} layouts |",
        f"| Golden (held out for scoring) | {golden_count} "
        f"| {_layout_count(golden_images)} layouts |",
        f"| Training total (S1+S2) | {training_total} | target 200-320 |",
    ]


def _session_of(image: ExtractedImage) -> str | None:
    """Return an extracted image's session tag, defensively (R6.4)."""
    return getattr(image, "session", None)


def _layout_count(images: list[ExtractedImage]) -> int:
    """Count the distinct source layouts covered by a set of images."""
    return len({getattr(img, "source_layout_id", None) for img in images})


def _known_gaps_block(results: list[StepResult]) -> list[str]:
    """Render the Known gaps list by scanning every non-``REAL`` value (R14.4).

    Scans each :class:`StepResult`'s ``time`` cell and each
    :class:`~week1_harness.markers.MarkedValue` in its ``notes``, keeps those
    whose marker is ``PLACEHOLDER`` or ``ESTIMATE``, and lists each with its
    emitting step, marker, and reason. Because the list is derived from exactly
    the values the steps emitted (not hand-authored here), it cannot drift from
    the run's real honesty markers.
    """
    block = ["**Known gaps:**", ""]
    gaps = [
        (result.name, marked)
        for result in results
        for marked in _non_real_values(result)
    ]
    if not gaps:
        block.append("- none")
        return block
    block += [f"- {name}: {_render_gap(marked)}" for name, marked in gaps]
    return block


def _non_real_values(result: StepResult) -> list[MarkedValue]:
    """Yield the result's non-``REAL`` marked values: its time cell then notes.

    Order matters only for stable output: the time cell (if non-real) comes
    first, then each non-real note in the order the step emitted them.
    """
    found: list[MarkedValue] = []
    time_cell = getattr(result, "time", None)
    if time_cell is not None and time_cell.marker is not Marker.REAL:
        found.append(time_cell)
    for note in getattr(result, "notes", None) or []:
        if note.marker is not Marker.REAL:
            found.append(note)
    return found


def _render_gap(marked: MarkedValue) -> str:
    """Render one known-gap entry carrying its marker and reason (R14.4).

    Scalar/string values render in full via :meth:`MarkedValue.render`
    (``value[unit] [MARKER: reason]``), so distinct fabricated numbers stay
    visible. Container values (e.g. the zero-shot per-item status map) would be
    noise if dumped, so only their ``[MARKER: reason]`` tag is shown -- still
    carrying the marker and reason the requirement asks for.
    """
    if isinstance(marked.value, (dict, list, tuple, set)):
        return f"[{marked.marker.value}: {marked.reason}]"
    return marked.render()
