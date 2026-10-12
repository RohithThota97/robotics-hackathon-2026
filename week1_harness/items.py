"""The six synthetic practice items and their pure-geometry shape drawers (R5.1).

This module is the single source of truth for *what* the harness draws: the six
:class:`~week1_harness.datamodel.PracticeItem` definitions (including the
look-alike pair held as two distinct classes) and, for each ``shape_family``, a
**pure-geometry** "shape drawer" that turns placement parameters into an outline
polygon plus named sub-part polygons.

Scope (task 7.1 only). Everything here is math over coordinates:

* **No rasterization.** Drawers return polygons as ``list[(x, y)]`` integer
  points; painting those polygons with Pillow and rasterizing per-instance masks
  is ``synth/render.py`` (task 7.3).
* **No scene composition.** Choosing positions / scales / rotations / lighting
  and computing occlusion and ``visible_fraction`` is ``synth/scene.py``
  (task 7.2). The drawers here are stateless and take placement parameters from
  the caller.

Because of that, the module imports only the standard library (``math``,
``dataclasses``, ``functools``) plus the light-weight domain model; it never
needs ``numpy`` or ``Pillow`` to be importable.

The six items (authoritative table in design.md, "Practice items (R5.1)"):

===================  ==============  ==========================================
class_id             shape_family    look-alike
===================  ==============  ==========================================
``screwdriver_small``  ``screwdriver``  pair with ``screwdriver_large``
``screwdriver_large``  ``screwdriver``  pair with ``screwdriver_small``
``wrench``             ``wrench``       open-end "U" bar
``hammer``             ``hammer``       head rectangle + handle
``pliers``             ``pliers``       two crossed bars
``tape_measure``       ``tape_measure`` rounded square body
===================  ==============  ==========================================

The two screwdrivers **share** the ``screwdriver`` shape family and are
deliberately confusable: they differ only in handle length (and the scene may
further vary their overall scale), so a detector can mix them up -- exactly the
look-alike confusion the Scorer reports. They remain two distinct ``class_id``s
end to end (R5.1, R8.4, R9.4, R14.1). :func:`shape_drawer_for` encodes the
per-item handle difference while keeping one geometry function per family.

Palette convention: each ``PracticeItem.palette`` is ordered so that
``palette[i]`` is the colour of the part whose ``ShapePart.palette_index`` is
``i``. ``render.py`` resolves a part's colour by indexing the item's palette.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

from week1_harness.datamodel import RGB, PracticeItem

__all__ = [
    "Point",
    "ShapePart",
    "ShapeGeometry",
    "ShapeDrawer",
    "PRACTICE_ITEMS",
    "PRACTICE_ITEMS_BY_ID",
    "CLASS_IDS",
    "SHAPE_DRAWERS",
    "practice_items",
    "practice_item",
    "shape_drawer_for",
    "draw_item",
    "draw_screwdriver",
    "draw_wrench",
    "draw_hammer",
    "draw_pliers",
    "draw_tape_measure",
]

# An integer pixel coordinate on the canvas (what a drawer returns).
Point = tuple[int, int]
# A local, pre-transform coordinate (float math before placement).
_LocalPoint = tuple[float, float]


@dataclass(frozen=True)
class ShapePart:
    """One named, filled sub-region of a shape (e.g. a screwdriver's handle).

    Fields:
        name:          sub-part name (``"shaft"``, ``"handle"``, ``"head"``...).
        polygon:       the part's outline as integer canvas points.
        palette_index: index into the owning ``PracticeItem.palette`` giving the
                       colour ``render.py`` should fill this part with.
    """

    name: str
    polygon: list[Point]
    palette_index: int


@dataclass(frozen=True)
class ShapeGeometry:
    """The geometry a shape drawer produces for one placement.

    Fields:
        outline: the whole shape's silhouette as integer canvas points. Used by
                 ``scene.py`` for bounding boxes / coarse occlusion and by
                 ``render.py`` as the instance's ground-truth polygon.
        parts:   the filled sub-parts, in paint order (first painted first).
                 Painting the parts reproduces the shape; the outline is the
                 overall boundary.
    """

    outline: list[Point]
    parts: list[ShapePart]


# A per-item drawer: (center, scale, rotation_degrees) -> geometry.
ShapeDrawer = Callable[[Point, float, float], ShapeGeometry]


# --------------------------------------------------------------------------- #
# Pure-geometry helpers (local float math, then a single placement transform). #
# --------------------------------------------------------------------------- #


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[_LocalPoint]:
    """Return the four corners of an axis-aligned rectangle, clockwise."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _rotate_local(point: _LocalPoint, angle_deg: float) -> _LocalPoint:
    """Rotate a local point about the local origin by ``angle_deg`` degrees."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y = point
    return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)


def _bar(back: float, front: float, half_w: float, angle_deg: float) -> list[_LocalPoint]:
    """A rectangular bar spanning x in [-back, front], rotated about the pivot.

    Used to build the two crossed bars of the pliers; the pivot sits at the
    local origin so the two bars rotate against each other to form an X.
    """
    corners = [(-back, -half_w), (front, -half_w), (front, half_w), (-back, half_w)]
    return [_rotate_local(c, angle_deg) for c in corners]


def _rounded_rect(hw: float, hh: float, radius: float, per_corner: int) -> list[_LocalPoint]:
    """A rounded rectangle outline sampled as a polygon (tape-measure body).

    ``hw``/``hh`` are the half-width/half-height; ``radius`` is the corner
    radius (clamped to the box); ``per_corner`` arc segments approximate each
    rounded corner.
    """
    radius = min(radius, hw, hh)
    corners = [
        ((hw - radius, hh - radius), 0.0, 90.0),
        ((-(hw - radius), hh - radius), 90.0, 180.0),
        ((-(hw - radius), -(hh - radius)), 180.0, 270.0),
        ((hw - radius, -(hh - radius)), 270.0, 360.0),
    ]
    pts: list[_LocalPoint] = []
    for (cx, cy), a0, a1 in corners:
        for i in range(per_corner + 1):
            angle = math.radians(a0 + (a1 - a0) * i / per_corner)
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return _dedupe_consecutive(pts)


def _dedupe_consecutive(points: list[_LocalPoint]) -> list[_LocalPoint]:
    """Drop consecutive duplicate points (incl. the wrap-around pair)."""
    out: list[_LocalPoint] = []
    for p in points:
        if not out or out[-1] != p:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _convex_hull(points: list[_LocalPoint]) -> list[_LocalPoint]:
    """Monotone-chain convex hull of local points (used for the pliers outline).

    The pliers are two crossed bars whose exact union boundary is awkward; the
    convex hull is a clean, well-defined enclosing silhouette for placement and
    bounding-box use, while the two bars are preserved as sub-parts.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o: _LocalPoint, a: _LocalPoint, b: _LocalPoint) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[_LocalPoint] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[_LocalPoint] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _transform(
    points: Iterable[_LocalPoint],
    center: Point,
    scale: float,
    rotation: float,
) -> list[Point]:
    """Apply scale, rotation (degrees), then translation; round to int pixels."""
    cx, cy = center
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    result: list[Point] = []
    for x, y in points:
        xs, ys = x * scale, y * scale
        xr = xs * cos_t - ys * sin_t
        yr = xs * sin_t + ys * cos_t
        result.append((int(round(cx + xr)), int(round(cy + yr))))
    return result


# --------------------------------------------------------------------------- #
# Per-shape-family drawers. Each builds the shape in a local frame centred on   #
# the origin, then places it with a single scale/rotate/translate transform.    #
# --------------------------------------------------------------------------- #

# Nominal local dimensions (in local units; a scale of ~1.0 yields ~100px items).
_SD_SHAFT_LEN = 64.0
_SD_SHAFT_HALF = 4.0
_SD_HANDLE_LEN = 24.0
_SD_HANDLE_HALF = 13.0
_SD_TIP_TAPER = 10.0


def draw_screwdriver(
    center: Point,
    scale: float,
    rotation: float,
    *,
    handle_ratio: float = 1.0,
) -> ShapeGeometry:
    """A screwdriver: a thin tapered shaft plus a fatter handle.

    ``handle_ratio`` scales the handle length only, keeping the shaft fixed.
    The two screwdriver classes pass different ratios so they share a nearly
    identical shaft (making them look-alikes) but differ in handle length.
    """
    handle_len = _SD_HANDLE_LEN * handle_ratio
    total = handle_len + _SD_SHAFT_LEN
    x_left = -total / 2.0
    x_join = x_left + handle_len
    x_tip = total / 2.0
    hh = _SD_HANDLE_HALF
    sh = _SD_SHAFT_HALF
    taper = _SD_TIP_TAPER

    outline_local = [
        (x_left, -hh),
        (x_join, -hh),
        (x_join, -sh),
        (x_tip - taper, -sh),
        (x_tip, 0.0),
        (x_tip - taper, sh),
        (x_join, sh),
        (x_join, hh),
        (x_left, hh),
    ]
    handle_local = _rect(x_left, -hh, x_join, hh)
    shaft_local = [
        (x_join, -sh),
        (x_tip - taper, -sh),
        (x_tip, 0.0),
        (x_tip - taper, sh),
        (x_join, sh),
    ]
    return ShapeGeometry(
        outline=_transform(outline_local, center, scale, rotation),
        parts=[
            ShapePart("shaft", _transform(shaft_local, center, scale, rotation), 0),
            ShapePart("handle", _transform(handle_local, center, scale, rotation), 1),
        ],
    )


def draw_wrench(center: Point, scale: float, rotation: float) -> ShapeGeometry:
    """An open-end wrench: a long bar with a "U" jaw open at one end."""
    total = 92.0
    bar_h = 6.0
    head_w = 34.0
    head_h = 18.0
    notch_h = 7.0
    notch_depth = 20.0
    x_left = -total / 2.0
    x_right = total / 2.0
    x_head = x_right - head_w

    outline_local = [
        (x_left, -bar_h),
        (x_head, -bar_h),
        (x_head, -head_h),
        (x_right, -head_h),
        (x_right, -notch_h),
        (x_right - notch_depth, -notch_h),
        (x_right - notch_depth, notch_h),
        (x_right, notch_h),
        (x_right, head_h),
        (x_head, head_h),
        (x_head, bar_h),
        (x_left, bar_h),
    ]
    outline = _transform(outline_local, center, scale, rotation)
    return ShapeGeometry(outline=outline, parts=[ShapePart("body", outline, 0)])


def draw_hammer(center: Point, scale: float, rotation: float) -> ShapeGeometry:
    """A hammer: a wide head rectangle at the top with a narrow handle below."""
    total = 104.0
    head_half_w = 22.0
    head_h = 18.0
    handle_half_w = 5.0
    top = -total / 2.0
    head_bottom = top + head_h
    bottom = total / 2.0

    outline_local = [
        (-head_half_w, top),
        (head_half_w, top),
        (head_half_w, head_bottom),
        (handle_half_w, head_bottom),
        (handle_half_w, bottom),
        (-handle_half_w, bottom),
        (-handle_half_w, head_bottom),
        (-head_half_w, head_bottom),
    ]
    head_local = _rect(-head_half_w, top, head_half_w, head_bottom)
    handle_local = _rect(-handle_half_w, head_bottom, handle_half_w, bottom)
    return ShapeGeometry(
        outline=_transform(outline_local, center, scale, rotation),
        parts=[
            ShapePart("head", _transform(head_local, center, scale, rotation), 0),
            ShapePart("handle", _transform(handle_local, center, scale, rotation), 1),
        ],
    )


def draw_pliers(center: Point, scale: float, rotation: float) -> ShapeGeometry:
    """Pliers: two bars crossed about a pivot, forming an X (jaws + handles)."""
    back = 18.0
    front = 64.0
    half_w = 6.0
    angle = 14.0

    bar_a_local = _bar(back, front, half_w, angle)
    bar_b_local = _bar(back, front, half_w, -angle)
    outline_local = _convex_hull(bar_a_local + bar_b_local)
    return ShapeGeometry(
        outline=_transform(outline_local, center, scale, rotation),
        parts=[
            ShapePart("bar_a", _transform(bar_a_local, center, scale, rotation), 0),
            ShapePart("bar_b", _transform(bar_b_local, center, scale, rotation), 1),
        ],
    )


def draw_tape_measure(center: Point, scale: float, rotation: float) -> ShapeGeometry:
    """A tape measure: a rounded square body with a small tape slot tab."""
    hw = 36.0
    hh = 32.0
    radius = 12.0
    per_corner = 5

    body_local = _rounded_rect(hw, hh, radius, per_corner)
    slot_local = _rect(-9.0, hh - 4.0, 9.0, hh + 4.0)
    outline = _transform(body_local, center, scale, rotation)
    return ShapeGeometry(
        outline=outline,
        parts=[
            ShapePart("body", outline, 0),
            ShapePart("slot", _transform(slot_local, center, scale, rotation), 1),
        ],
    )


# --------------------------------------------------------------------------- #
# The six practice items and the public accessors.                             #
# --------------------------------------------------------------------------- #

# Stable order used everywhere the six items are enumerated (reporters, counts).
PRACTICE_ITEMS: list[PracticeItem] = [
    PracticeItem(
        class_id="screwdriver_small",
        shape_family="screwdriver",
        palette=((176, 178, 184), (198, 58, 52)),  # steel shaft, red handle
        look_alike_of="screwdriver_large",
    ),
    PracticeItem(
        class_id="screwdriver_large",
        shape_family="screwdriver",
        palette=((168, 170, 176), (44, 92, 196)),  # steel shaft, blue handle
        look_alike_of="screwdriver_small",
    ),
    PracticeItem(
        class_id="wrench",
        shape_family="wrench",
        palette=((122, 126, 134),),  # chrome body
        look_alike_of=None,
    ),
    PracticeItem(
        class_id="hammer",
        shape_family="hammer",
        palette=((92, 94, 100), (170, 112, 58)),  # steel head, wood handle
        look_alike_of=None,
    ),
    PracticeItem(
        class_id="pliers",
        shape_family="pliers",
        palette=((58, 142, 138), (236, 152, 42)),  # teal grip, orange grip
        look_alike_of=None,
    ),
    PracticeItem(
        class_id="tape_measure",
        shape_family="tape_measure",
        palette=((242, 206, 72), (58, 58, 62)),  # yellow body, dark slot
        look_alike_of=None,
    ),
]

# Fast lookup and the canonical class-id order.
PRACTICE_ITEMS_BY_ID: dict[str, PracticeItem] = {item.class_id: item for item in PRACTICE_ITEMS}
CLASS_IDS: tuple[str, ...] = tuple(item.class_id for item in PRACTICE_ITEMS)

# One geometry function per shape family (two screwdrivers share "screwdriver").
SHAPE_DRAWERS: dict[str, Callable[..., ShapeGeometry]] = {
    "screwdriver": draw_screwdriver,
    "wrench": draw_wrench,
    "hammer": draw_hammer,
    "pliers": draw_pliers,
    "tape_measure": draw_tape_measure,
}

# Per-item handle length for the look-alike pair: same shaft, different handle.
_SCREWDRIVER_HANDLE_RATIO: dict[str, float] = {
    "screwdriver_small": 0.72,
    "screwdriver_large": 1.55,
}


def practice_items() -> list[PracticeItem]:
    """Return the six practice items in canonical order (a fresh list copy)."""
    return list(PRACTICE_ITEMS)


def practice_item(class_id: str) -> PracticeItem:
    """Return the :class:`PracticeItem` for ``class_id`` (raises if unknown)."""
    try:
        return PRACTICE_ITEMS_BY_ID[class_id]
    except KeyError:
        raise KeyError(f"unknown practice item class_id: {class_id!r}") from None


def shape_drawer_for(class_id: str) -> ShapeDrawer:
    """Return the per-item drawer ``(center, scale, rotation) -> ShapeGeometry``.

    For the two screwdrivers the family drawer is bound to the item's handle
    ratio, so the look-alike pair differs by handle length while every item
    still routes through exactly one geometry function per shape family.
    """
    item = practice_item(class_id)
    base = SHAPE_DRAWERS[item.shape_family]
    handle_ratio = _SCREWDRIVER_HANDLE_RATIO.get(class_id)
    if handle_ratio is not None:
        return partial(base, handle_ratio=handle_ratio)
    return base


def draw_item(
    class_id: str,
    center: Point,
    scale: float = 1.0,
    rotation: float = 0.0,
) -> ShapeGeometry:
    """Convenience: look up ``class_id``'s drawer and draw it at one placement."""
    return shape_drawer_for(class_id)(center, scale, rotation)
