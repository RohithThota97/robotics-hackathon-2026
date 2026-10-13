"""Timing_Table reporter -- Week 1 deliverable (a) (Requirements 4.3, 13).

:func:`render_timing_table` turns the collected :class:`StepResult`s into the
Markdown Timing_Table, in the table style of ``prep/week1/week1.md`` section 4.
It is a **pure function** of the results -- no file I/O, no printing, no clock
reads -- so the CLI (task 14) and the tests can both call it and get the same
string for the same input.

What the rendered document contains:

* **The honesty notice (R4.3).** The document opens with
  :data:`HONESTY_NOTICE`, a Markdown blockquote stating the run used synthetic
  inputs, CPU-only execution, and stubbed training, and does not report a real
  Week 1 measurement. The same text opens the Dataset_Card (task 13.2), so the
  constant is defined once here and reused there.
* **One row per step (R13.1), columns ``# | Step | How | Time`` (R13.2).** Rows
  are sorted by step number. The ``Time`` cell is produced by
  :meth:`week1_harness.markers.MarkedValue.render`, so a ``REAL`` time renders
  untagged (R13.3) while a ``PLACEHOLDER``/``ESTIMATE`` time renders with its
  ``[MARKER: reason]`` tag inline in the time column (R13.4).
* **A total row (R13.5).** The total sums **only** the ``REAL`` step times
  (whose ``value`` is a float in seconds) and is labelled "real CPU steps only",
  stating how many of the steps were placeholder or estimate and therefore
  excluded. Labelling it this way is deliberate: the number can never be read as
  a full-loop turnaround (the honesty tension at the heart of the harness).

Table safety: cell text is passed through :func:`_cell`, which collapses stray
newlines and escapes any ``|`` so a value can never break the Markdown table.
None of the step names, "How" strings, or marker reasons currently contain
either character, but the escaping keeps the table well-formed if one ever does.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from week1_harness.markers import Marker, MarkedValue

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.datamodel import StepResult

__all__ = ["HONESTY_NOTICE", "render_timing_table"]


# The honesty notice shown at the start of both deliverables (R4.3). Defined
# once here; the Dataset_Card reporter (task 13.2) imports and reuses it so the
# two deliverables cannot drift. Kept as a single literal so the exact wording
# (including the `[PLACEHOLDER: ...]` / `[ESTIMATE: ...]` code spans) is stable.
HONESTY_NOTICE = (
    "> **HONESTY NOTICE:** This run used **synthetic inputs**, **CPU-only "
    "execution**, and **stubbed training**. It does **not** report a real "
    "Week 1 measurement. Values tagged `[PLACEHOLDER: ...]` or `[ESTIMATE: ...]` "
    "are fabricated stand-ins; the reason is shown on each."
)

# Timing_Table column header and its separator (R13.2): step number, name, how,
# and time.
_HEADER = "| # | Step | How | Time |"
_SEPARATOR = "|---|------|-----|------|"


def _format_seconds(value: float) -> str:
    """Format a duration for display without flattening it to zero.

    Two decimals suit the step times people compare, but the fast synthetic
    steps run in well under 10 ms, and ``f"{0.000134:.2f}"`` is ``"0.00"`` --
    which reads as "took no time at all". Sub-centisecond values therefore fall
    back to two significant figures, so a real measurement is never displayed
    as zero.
    """
    if value and abs(value) < 0.01:
        return f"{value:.2g}"
    return f"{value:.2f}"


def _time_cell(time: MarkedValue) -> str:
    """Render a time cell, rounding float seconds to 2 decimals.

    Raw ``time.perf_counter()`` deltas render as ``2.9141306629999804s``, which
    is noise in a document people read. Only the *rendered* value is rounded --
    the ``MarkedValue`` itself keeps full precision for ``run.json`` -- and the
    marker tag is preserved, so a rounded estimate is still visibly an estimate.
    """
    if isinstance(time.value, float):
        time = dataclasses.replace(time, value=_format_seconds(time.value))
    return time.render()


def _cell(text: object) -> str:
    """Return ``text`` made safe to drop into a single Markdown table cell.

    Collapses any newline (which would otherwise end the table row early) to a
    space and escapes ``|`` as ``\\|`` so it is not read as a column separator.
    Defensive only -- no step name, "How" text, or marker reason is expected to
    contain either character.
    """
    s = str(text)
    for nl in ("\r\n", "\r", "\n"):
        s = s.replace(nl, " ")
    return s.replace("|", "\\|")


def render_timing_table(results: list[StepResult]) -> str:
    """Render the Timing_Table deliverable as a Markdown document (R13, R4.3).

    Pure function of ``results``: it reads each step's number, name, "How" text,
    and time cell, and returns the Markdown string without any I/O.

    The document opens with :data:`HONESTY_NOTICE` (R4.3), then a table with one
    row per step sorted by step number (R13.1) and the columns ``# | Step | How
    | Time`` (R13.2). Each ``Time`` cell is rendered via
    :meth:`~week1_harness.markers.MarkedValue.render`, so ``REAL`` times appear
    untagged (R13.3) and ``PLACEHOLDER``/``ESTIMATE`` times appear with their
    ``[MARKER: reason]`` tag (R13.4). A final total row (R13.5) sums only the
    ``REAL`` step times (seconds) and states how many steps were placeholder or
    estimate and thus excluded, so the figure cannot be mistaken for a full-loop
    turnaround.

    Args:
        results: the collected :class:`~week1_harness.datamodel.StepResult`s
            (any order; sorted here by step number).

    Returns:
        The Timing_Table as a Markdown string, terminated by a newline.
    """
    rows = sorted(results, key=lambda r: r.number)

    lines: list[str] = [HONESTY_NOTICE, "", _HEADER, _SEPARATOR]

    total_seconds = 0.0
    n_real = n_placeholder = n_estimate = 0
    for result in rows:
        marker = result.time.marker
        if marker is Marker.REAL:
            n_real += 1
            total_seconds += float(result.time.value)  # REAL time values are float seconds
        elif marker is Marker.PLACEHOLDER:
            n_placeholder += 1
        elif marker is Marker.ESTIMATE:
            n_estimate += 1
        lines.append(
            f"| {result.number} | {_cell(result.name)} | {_cell(result.how)} "
            f"| {_cell(_time_cell(result.time))} |"
        )

    # Total row (R13.5): REAL CPU steps only, never presentable as a full-loop
    # turnaround -- it names how many of the steps were placeholder/estimate and
    # therefore excluded from the sum.
    total_how = (
        f"real CPU steps only ({n_real} of {len(rows)}; "
        f"{n_placeholder} placeholder, {n_estimate} estimate excluded)"
    )
    lines.append(f"|  | **Total** | {_cell(total_how)} | **{total_seconds:.2f}s** |")

    return "\n".join(lines) + "\n"
