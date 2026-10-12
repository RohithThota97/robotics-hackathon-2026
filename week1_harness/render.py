"""Pillow rendering of synthetic layouts and per-instance mask rasterization (R5.5).

This module is the harness's only pixel-level code: it turns the pure-geometry
output of :mod:`week1_harness.synth.items` into actual RGB canvases (PNG frames)
and rasterizes per-instance ground-truth masks. It is the painting half of
synthetic frame generation; the composition half (choosing placements, lighting,
sessions, and computing ``visible_fraction``) is :mod:`week1_harness.synth.scene`.

Scope (task 7.3 only). This module **does not** import ``synth/scene.py`` -- it
consumes only the domain types from :mod:`week1_harness.datamodel`
(:class:`~week1_harness.datamodel.Layout`,
:class:`~week1_harness.datamodel.Placement`,
:class:`~week1_harness.datamodel.Instance`) plus the shape geometry from
:mod:`week1_harness.synth.items`. Scene composition is a separate task and
produces the ``Layout`` / ``Instance`` objects this module renders.

What it provides:

* :func:`render_layout` -- paint one :class:`~week1_harness.datamodel.Layout`
  onto a fixed-size RGB canvas. Placements are painted in ``draw_order`` (lower
  first), so later placements occlude earlier ones -- the pixel-space mirror of
  the occlusion ``scene.py`` records as ``visible_fraction``. Each item's parts
  are filled with the colour taken from the item's palette, and the layout's
  lighting (``"dim"`` / ``"normal"`` / ``"bright"``) is applied as a
  brightness/contrast adjustment over the finished canvas.
* :func:`save_png` -- write an image to disk as a deterministic PNG, so two runs
  with the same seed produce byte-identical frames (R3.3).
* :func:`rasterize_instance_mask` -- fill one instance's known ground-truth
  polygon into a boolean mask the Auto_Labeler reuses (one mask per instance).

Determinism (R3.3): every function here is pure given its inputs. Rendering does
no randomness of its own (all jitter/placement choices are made upstream in
``scene.py`` from ``ctx.rng_for(1)``), and :func:`save_png` writes PNGs without
timestamps, so identical layouts render to identical bytes.

The item lookup is injectable (default
:data:`week1_harness.synth.items.PRACTICE_ITEMS_BY_ID`) so tests can render with
custom palettes/classes without monkeypatching the module.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from week1_harness.synth import items

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from collections.abc import Mapping, Sequence

    from week1_harness.datamodel import RGB, Instance, Layout, PracticeItem

__all__ = [
    "DEFAULT_SIZE",
    "DEFAULT_BG",
    "render_layout",
    "save_png",
    "rasterize_instance_mask",
]

# Default canvas: a fixed-size square RGB frame (design "Scene model": e.g. 640x640).
DEFAULT_SIZE: tuple[int, int] = (640, 640)

# Default background: a neutral slate "tray/bin" tone (design "Scene model").
# A mid tone keeps headroom for both the "dim" and "bright" lighting passes and
# contrasts with the light-metal and dark item palettes alike.
DEFAULT_BG: RGB = (96, 100, 108)

# Lighting applied over the finished canvas as (brightness, contrast) multipliers
# (design "Scene model": the three S1 conditions, R5.2). ``normal`` is identity.
# An unknown lighting value falls back to identity rather than failing a render.
_LIGHTING_FACTORS: dict[str, tuple[float, float]] = {
    "dim": (0.62, 0.90),
    "normal": (1.00, 1.00),
    "bright": (1.38, 1.08),
}


def _apply_lighting(image: Image.Image, lighting: str) -> Image.Image:
    """Apply a layout's lighting as a brightness then contrast enhancement.

    Returns the input image unchanged for ``"normal"`` (and for any unknown
    lighting), since both factors are ``1.0`` and enhancing by ``1.0`` is a
    no-op. Deterministic: the same image and lighting always yield the same
    pixels.
    """
    brightness, contrast = _LIGHTING_FACTORS.get(lighting, (1.0, 1.0))
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def render_layout(
    layout: Layout,
    items_by_id: Mapping[str, PracticeItem] | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    bg: RGB = DEFAULT_BG,
) -> Image.Image:
    """Paint one layout onto an RGB canvas, honouring draw-order occlusion (R5.5).

    Placements are painted in ascending ``draw_order`` so that later (higher
    ``draw_order``) placements are drawn on top of and occlude earlier ones --
    the pixel-space counterpart of the ``visible_fraction`` occlusion recorded
    upstream in ``scene.py``. For each placement the shape geometry is obtained
    from :func:`week1_harness.synth.items.draw_item`, and every
    :class:`~week1_harness.synth.items.ShapePart` is filled with the colour at
    ``item.palette[part.palette_index]``. Finally the layout's lighting is
    applied over the finished canvas.

    Args:
        layout:      the :class:`~week1_harness.datamodel.Layout` to render.
        items_by_id: class-id -> :class:`~week1_harness.datamodel.PracticeItem`
                     lookup supplying each item's palette. Defaults to
                     :data:`week1_harness.synth.items.PRACTICE_ITEMS_BY_ID`.
        size:        canvas size as ``(width, height)`` in pixels.
        bg:          background fill colour (the tray/bin tone).

    Returns:
        A new ``PIL.Image.Image`` in ``"RGB"`` mode of the given ``size``.
    """
    lookup = items.PRACTICE_ITEMS_BY_ID if items_by_id is None else items_by_id
    canvas = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(canvas)

    # Lower draw_order first => later placements paint over (occlude) earlier ones.
    for placement in sorted(layout.placements, key=lambda p: p.draw_order):
        item = lookup[placement.class_id]
        geometry = items.draw_item(
            placement.class_id,
            placement.position,
            placement.scale,
            placement.rotation,
        )
        for part in geometry.parts:
            if len(part.polygon) >= 3:  # a fillable region needs >= 3 vertices
                draw.polygon(part.polygon, fill=item.palette[part.palette_index])

    return _apply_lighting(canvas, layout.lighting)


def save_png(image: Image.Image, path: str | os.PathLike[str]) -> str:
    """Write ``image`` to ``path`` as a deterministic PNG; return the path.

    PNGs are written without a timestamp chunk and with optimization disabled, so
    the same image saved twice produces byte-identical files -- the file-level
    half of the determinism contract (R3.3). Any missing parent directories are
    created.
    """
    target = os.fspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    image.save(target, format="PNG", optimize=False)
    return target


def _polygon_of(instance_or_polygon: Instance | Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coerce an ``Instance`` (or a raw point sequence) to integer polygon points."""
    raw = getattr(instance_or_polygon, "polygon", instance_or_polygon)
    return [(int(x), int(y)) for x, y in raw]


def rasterize_instance_mask(
    instance_or_polygon: Instance | Sequence[tuple[int, int]],
    size: tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Rasterize one instance's ground-truth polygon into a boolean mask (R5.5).

    Produces exactly one mask per instance by filling the known polygon geometry
    (the instance's silhouette) at the given canvas size -- no segmentation model
    is involved; the geometry is already known because the harness drew it.

    Args:
        instance_or_polygon: an :class:`~week1_harness.datamodel.Instance` (its
            ``polygon`` is used) or a raw sequence of ``(x, y)`` points.
        size: mask size as ``(width, height)``; use the same value as the
            canvas the instance was rendered on so pixels line up.

    Returns:
        A boolean ``numpy.ndarray`` of shape ``(height, width)`` that is ``True``
        inside the filled polygon. A degenerate polygon (< 3 points) yields an
        all-``False`` mask rather than raising.
    """
    polygon = _polygon_of(instance_or_polygon)
    mask_img = Image.new("L", size, 0)
    if len(polygon) >= 3:
        ImageDraw.Draw(mask_img).polygon(polygon, fill=255)
    return np.asarray(mask_img, dtype=np.uint8) > 0
