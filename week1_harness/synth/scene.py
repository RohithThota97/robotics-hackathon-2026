"""Scene composition and ground truth for the synthetic frames (task 7.2).

This module turns the six practice items (:mod:`week1_harness.synth.items`) into
the harness's *scenes*: ordered compositions of placed items (``Layout`` +
``Placement``) across the three capture sessions, plus the ground-truth
``Instance`` list for every placed item. It is **pure composition and
geometry** -- it never rasterizes a canvas, never writes a PNG, and never
imports :mod:`week1_harness.synth.render`. Painting layouts to PNG and
rasterizing per-instance masks is task 7.3 (``render.py``); this module only
produces the structures and the ground truth that drive it.

What this module owns
---------------------

* **Session composition (R5.2-R5.4).**
    - **S1** -- each of the six items *alone*, under the three lighting
      conditions ``"dim"``, ``"normal"``, ``"bright"`` (R5.2).
    - **S2** -- several *shuffled* layouts (``S2_LAYOUT_COUNT``, which is >= 5),
      each containing *all six* items in randomized positions, rotations, and
      scales (R5.3).
    - **Golden** -- one to two held-out layouts containing all six items,
      tagged ``session="golden"`` and reserved for the Scorer (R5.4).
* **Draw order and occlusion.** Each placement gets a ``draw_order``; a higher
  ``draw_order`` is painted later and therefore occludes lower ones. The
  ground-truth ``visible_fraction`` of each instance is computed *geometrically*
  (see below), so the Review step's visibility rule (R9.2/R9.3) and the dataset
  counts rest on real geometry rather than on a rendered image.
* **Per-layout clip spec.** Each layout stands in for a short video: a
  :class:`ClipSpec` records how many near-identical frames to emit and the tiny
  positional/rotational jitter to apply. The *actual* per-frame jitter values
  and the fps/stride subsampling belong to Film (task 8.1) and Extract
  (task 8.3); here we only record the spec those steps consume.

Determinism (R5.5)
------------------

**Every** random choice -- S1 poses, S2/golden item order, positions,
rotations, scales, and per-layout lighting -- is drawn from a single generator,
``ctx.rng_for(1)``. :func:`build_layouts` draws that generator once and threads
it through composition, so two runs with the same seed produce byte-identical
layouts. :func:`compose` builds on :func:`build_layouts` and derives instances
and clip specs with no further randomness (``visible_fraction`` is geometric and
the clip spec is constant per session), so the whole scene is reproducible.

Visible fraction (geometric, no rendering)
------------------------------------------

For an instance with outline polygon ``P`` and the set ``O`` of outline polygons
of strictly-higher ``draw_order`` placements in the same layout,

    visible_fraction(P) = area(P \\ union(O)) / area(P)

is estimated with a small, self-contained numpy helper
(:func:`_visible_fraction`): it samples a fixed-resolution grid over ``P``'s
bounding box, counts the grid cells inside ``P``, and counts how many of those
are *not* covered by any occluder. Because numerator and denominator are counted
on the identical grid, the ratio is unbiased by cell shape. No image, no Pillow,
no ``render.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from week1_harness.datamodel import Instance, Layout, Placement
from week1_harness.synth.items import CLASS_IDS, draw_item

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.context import RunContext

__all__ = [
    "CANVAS_SIZE",
    "LIGHTING_CONDITIONS",
    "S2_LAYOUT_COUNT",
    "GOLDEN_LAYOUT_COUNT",
    "FRAMES_PER_CLIP",
    "ClipSpec",
    "SceneData",
    "build_layouts",
    "instances_for_layout",
    "compose",
]

# --------------------------------------------------------------------------- #
# Scene constants (single source of truth; downstream steps and the card read  #
# these rather than hard-coding numbers).                                      #
# --------------------------------------------------------------------------- #

# Fixed RGB canvas size (width, height). render.py (task 7.3) paints onto this.
CANVAS_SIZE: tuple[int, int] = (640, 640)

# The three S1 lighting conditions, in canonical order (R5.2).
LIGHTING_CONDITIONS: tuple[str, ...] = ("dim", "normal", "bright")

# S2: number of shuffled all-six-item layouts. Must be >= 5 (R5.3); the design
# count plan uses 8 so that extraction clears the >=150-instances/item floor.
S2_LAYOUT_COUNT: int = 8

# Golden: number of held-out all-six-item layouts; 1 or 2 (R5.4).
GOLDEN_LAYOUT_COUNT: int = 2

# Per-session "short clip" frame counts (the design count plan). These are the
# number of near-identical jittered frames a layout stands in for; Film emits
# them and Extract subsamples them. S1: 18 clips x 6 = 108 images; S2: 8 x 20 =
# 160; golden: 2 x 8 = 16. Training total S1+S2 = 268 (within 200-320), and
# instances/item = 18 (S1) + 160 (S2) = 178 (>= 150).
FRAMES_PER_CLIP: dict[str, int] = {"S1": 6, "S2": 20, "golden": 8}

# Per-clip jitter spec: the maximum tiny per-frame positional/rotational wobble
# that stands in for "a short video". The concrete per-frame values are drawn by
# Film; here we only record the spec.
_JITTER_PX: float = 2.0
_JITTER_DEG: float = 1.5

# S1 single-item placement: centred with a small wobble so the item is always
# fully on-canvas (and therefore unoccluded -> visible_fraction == 1.0).
_S1_CENTER_LO, _S1_CENTER_HI = 300, 340
_S1_SCALE_LO, _S1_SCALE_HI = 0.95, 1.25

# S2/golden packing: all six items get centres inside this central box. The box
# is deliberately tight relative to the ~100-150px item sizes so items cluster
# and overlap (a tray of tools), which exercises occlusion and visible_fraction.
_PACK_LO, _PACK_HI = 185, 455
_PACK_SCALE_LO, _PACK_SCALE_HI = 0.90, 1.45

# Grid resolution (NxN) for the geometric visible-fraction estimate.
_GRID_RES: int = 96


# --------------------------------------------------------------------------- #
# Public bundle types.                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClipSpec:
    """Per-layout "short clip" spec consumed by Film/Extract (not frozen state).

    A layout is rendered as ``frame_count`` near-identical frames with tiny
    jitter, standing in for a short video so the Frame_Extractor has something to
    subsample at 2-3 fps (R6.1) without any real footage.

    Fields:
        layout_id:   the layout this clip renders.
        session:     ``"S1"`` | ``"S2"`` | ``"golden"`` (matches the layout).
        frame_count: number of near-identical frames to emit (``FRAMES_PER_CLIP``).
        jitter_px:   max per-frame positional jitter in pixels.
        jitter_deg:  max per-frame rotational jitter in degrees.
    """

    layout_id: str
    session: str
    frame_count: int
    jitter_px: float
    jitter_deg: float


@dataclass(frozen=True)
class SceneData:
    """The fully composed scene: layouts, their ground truth, and clip specs.

    Fields:
        layouts:             every layout across S1, S2, and golden, in build
                             order (S1 first, then S2, then golden).
        instances_by_layout: ``layout_id -> list[Instance]`` ground truth for the
                             placed items in that layout (R5 ground truth).
        clips:               ``layout_id -> ClipSpec`` for Film/Extract.
    """

    layouts: list[Layout]
    instances_by_layout: dict[str, list[Instance]]
    clips: dict[str, ClipSpec]

    def all_instances(self) -> list[Instance]:
        """Flatten the per-layout ground truth into one list, in layout order."""
        out: list[Instance] = []
        for layout in self.layouts:
            out.extend(self.instances_by_layout[layout.layout_id])
        return out


# --------------------------------------------------------------------------- #
# Layout composition (all randomness from the single rng_for(1) generator).    #
# --------------------------------------------------------------------------- #


def build_layouts(ctx: "RunContext") -> list[Layout]:
    """Compose every S1, S2, and golden layout for the run (R5.2-R5.4, R5.5).

    All randomness is drawn from ``ctx.rng_for(1)``, obtained once here and
    threaded through composition, so the returned layouts are identical across
    runs with the same seed (R5.5). The order is deterministic: the six S1 items
    (each under three lightings), then ``S2_LAYOUT_COUNT`` S2 layouts, then
    ``GOLDEN_LAYOUT_COUNT`` golden layouts.
    """
    rng = ctx.rng_for(1)
    layouts: list[Layout] = []
    layouts.extend(_build_s1_layouts(rng))
    layouts.extend(_build_s2_layouts(rng))
    layouts.extend(_build_golden_layouts(rng))
    return layouts


def _build_s1_layouts(rng: np.random.Generator) -> list[Layout]:
    """S1: each item alone, shown under all three lighting conditions (R5.2).

    A single pose (centre/rotation/scale) is drawn per item and reused across the
    three lightings, so lighting is the only variable within an item's S1 clips.
    A lone item is never occluded, so its instance's visible fraction is 1.0.
    """
    layouts: list[Layout] = []
    for class_id in CLASS_IDS:
        center = (
            int(rng.integers(_S1_CENTER_LO, _S1_CENTER_HI + 1)),
            int(rng.integers(_S1_CENTER_LO, _S1_CENTER_HI + 1)),
        )
        rotation = float(rng.uniform(0.0, 360.0))
        scale = float(rng.uniform(_S1_SCALE_LO, _S1_SCALE_HI))
        for lighting in LIGHTING_CONDITIONS:
            placement = Placement(
                class_id=class_id,
                position=center,
                rotation=rotation,
                scale=scale,
                draw_order=0,
            )
            layouts.append(
                Layout(
                    layout_id=f"s1_{class_id}_{lighting}",
                    session="S1",
                    lighting=lighting,
                    placements=[placement],
                )
            )
    return layouts


def _build_s2_layouts(rng: np.random.Generator) -> list[Layout]:
    """S2: >= 5 shuffled layouts, each containing all six items (R5.3)."""
    layouts: list[Layout] = []
    for k in range(S2_LAYOUT_COUNT):
        lighting = LIGHTING_CONDITIONS[int(rng.integers(0, len(LIGHTING_CONDITIONS)))]
        placements = _compose_all_items(rng)
        layouts.append(
            Layout(
                layout_id=f"s2_{k:02d}",
                session="S2",
                lighting=lighting,
                placements=placements,
            )
        )
    return layouts


def _build_golden_layouts(rng: np.random.Generator) -> list[Layout]:
    """Golden: 1-2 held-out layouts, each containing all six items (R5.4)."""
    layouts: list[Layout] = []
    for k in range(GOLDEN_LAYOUT_COUNT):
        lighting = LIGHTING_CONDITIONS[int(rng.integers(0, len(LIGHTING_CONDITIONS)))]
        placements = _compose_all_items(rng)
        layouts.append(
            Layout(
                layout_id=f"golden_{k:02d}",
                session="golden",
                lighting=lighting,
                placements=placements,
            )
        )
    return layouts


def _compose_all_items(rng: np.random.Generator) -> list[Placement]:
    """Place all six items in a shuffled, randomized, overlap-prone arrangement.

    The item order is shuffled (so ``draw_order`` -- and hence which item ends up
    on top -- varies per layout), and each item gets a randomized centre inside
    the tight central packing box plus a random rotation and scale. Returned in
    ``draw_order`` order (0 painted first / bottom, 5 painted last / top).
    """
    order = [CLASS_IDS[i] for i in rng.permutation(len(CLASS_IDS))]
    placements: list[Placement] = []
    for draw_order, class_id in enumerate(order):
        center = (
            int(rng.integers(_PACK_LO, _PACK_HI + 1)),
            int(rng.integers(_PACK_LO, _PACK_HI + 1)),
        )
        rotation = float(rng.uniform(0.0, 360.0))
        scale = float(rng.uniform(_PACK_SCALE_LO, _PACK_SCALE_HI))
        placements.append(
            Placement(
                class_id=class_id,
                position=center,
                rotation=rotation,
                scale=scale,
                draw_order=draw_order,
            )
        )
    return placements


# --------------------------------------------------------------------------- #
# Ground-truth instances (pure function of a layout; no randomness).           #
# --------------------------------------------------------------------------- #


def instances_for_layout(layout: Layout) -> list[Instance]:
    """Build the ground-truth ``Instance`` list for one layout.

    For each placement this computes the placed outline polygon (from
    :func:`week1_harness.synth.items.draw_item`), its bounding box, and its
    ``visible_fraction`` after subtracting every strictly-higher ``draw_order``
    polygon. Pure and deterministic: given the same layout it always returns the
    same instances, so :func:`compose` adds no randomness beyond
    :func:`build_layouts`.
    """
    placements = sorted(layout.placements, key=lambda p: p.draw_order)
    polygons = [
        draw_item(p.class_id, p.position, p.scale, p.rotation).outline for p in placements
    ]

    instances: list[Instance] = []
    for idx, placement in enumerate(placements):
        polygon = polygons[idx]
        occluders = [
            polygons[k]
            for k, other in enumerate(placements)
            if other.draw_order > placement.draw_order
        ]
        visible_fraction = _visible_fraction(polygon, occluders)
        instances.append(
            Instance(
                instance_id=f"{layout.layout_id}#{placement.draw_order}",
                class_id=placement.class_id,
                polygon=polygon,
                bbox=_bbox(polygon),
                visible_fraction=visible_fraction,
                layout_id=layout.layout_id,
                session=layout.session,
            )
        )
    return instances


def compose(ctx: "RunContext") -> SceneData:
    """Compose the whole scene: layouts + ground-truth instances + clip specs.

    Builds the layouts via :func:`build_layouts` (the only source of randomness,
    ``ctx.rng_for(1)``), then derives the per-layout ground-truth instances and
    the per-layout clip spec with no further randomness. Deterministic for a
    fixed seed (R5.5).
    """
    layouts = build_layouts(ctx)
    instances_by_layout: dict[str, list[Instance]] = {}
    clips: dict[str, ClipSpec] = {}
    for layout in layouts:
        instances_by_layout[layout.layout_id] = instances_for_layout(layout)
        clips[layout.layout_id] = ClipSpec(
            layout_id=layout.layout_id,
            session=layout.session,
            frame_count=FRAMES_PER_CLIP[layout.session],
            jitter_px=_JITTER_PX,
            jitter_deg=_JITTER_DEG,
        )
    return SceneData(layouts=layouts, instances_by_layout=instances_by_layout, clips=clips)


# --------------------------------------------------------------------------- #
# Geometry helpers (self-contained; no Pillow, no render.py).                  #
# --------------------------------------------------------------------------- #


def _bbox(polygon: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box ``(x0, y0, x1, y1)`` of a polygon."""
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _points_in_polygon(
    xs: np.ndarray, ys: np.ndarray, polygon: list[tuple[int, int]]
) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test over sample points ``(xs, ys)``.

    Returns a boolean array (same shape as ``xs``) that is ``True`` where the
    sample point lies inside ``polygon``. Horizontal edges are skipped (they can
    never cross a horizontal ray), which also avoids any divide-by-zero. Handles
    convex and concave polygons, which covers every practice-item outline.
    """
    px = np.asarray([p[0] for p in polygon], dtype=np.float64)
    py = np.asarray([p[1] for p in polygon], dtype=np.float64)
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(polygon)
    for i in range(n):
        j = i - 1  # previous vertex (wraps to the last at i == 0)
        yi, yj = py[i], py[j]
        if yi == yj:
            continue  # horizontal edge: cannot cross a horizontal ray
        xi, xj = px[i], px[j]
        straddles = (yi > ys) != (yj > ys)
        x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside ^= straddles & (xs < x_cross)
    return inside


def _visible_fraction(
    polygon: list[tuple[int, int]], occluders: list[list[tuple[int, int]]]
) -> float:
    """Fraction of ``polygon`` still visible after subtracting the occluders.

    Estimated geometrically on a fixed ``_GRID_RES`` x ``_GRID_RES`` grid over
    ``polygon``'s bounding box: it is ``(cells inside polygon and inside no
    occluder) / (cells inside polygon)``. With no occluders the result is exactly
    ``1.0`` (fast path). The value is always in ``[0.0, 1.0]``.
    """
    if not occluders:
        return 1.0

    x0, y0, x1, y1 = _bbox(polygon)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return 1.0  # degenerate outline; treat as fully visible

    # Cell-centre sample coordinates across the bounding box.
    xs_axis = x0 + (np.arange(_GRID_RES) + 0.5) * (width / _GRID_RES)
    ys_axis = y0 + (np.arange(_GRID_RES) + 0.5) * (height / _GRID_RES)
    grid_x, grid_y = np.meshgrid(xs_axis, ys_axis)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    inside_self = _points_in_polygon(grid_x, grid_y, polygon)
    total = int(inside_self.sum())
    if total == 0:
        return 1.0  # grid missed a very thin outline; treat as fully visible

    occluded = np.zeros_like(inside_self)
    for occ in occluders:
        occluded |= _points_in_polygon(grid_x, grid_y, occ)

    visible = int((inside_self & ~occluded).sum())
    return visible / total
