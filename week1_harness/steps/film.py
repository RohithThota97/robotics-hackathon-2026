"""Step 1 -- Film: synthetic frame generation (Requirement 5).

:class:`FilmStep` is the first Data Loop step. It stands in for the real "film
the practice items" capture step by *generating* the footage synthetically, so
the whole loop runs on any machine with zero input footage (R5.5). Because the
work is genuine local-CPU drawing (Pillow rasterization), its timing cell is a
``REAL`` :class:`~week1_harness.markers.MarkedValue` -- there is nothing stubbed
or hardware-blocked about drawing shapes (R4.1).

What ``run`` does
-----------------

1. **Times the real work** with :func:`time.perf_counter` around the whole
   generate-and-write pass.
2. **Composes the scene** via :func:`week1_harness.synth.scene.compose`, which
   owns all layout randomness (drawn once from ``ctx.rng_for(1)``) and returns
   the layouts, the per-layout ground-truth :class:`~week1_harness.datamodel.Instance`
   list, and the per-layout :class:`~week1_harness.synth.scene.ClipSpec`:

     - **S1** -- each of the six items alone, under three lighting conditions
       (R5.2).
     - **S2** -- several shuffled layouts (>= 5), each with all six items (R5.3).
     - **Golden** -- one to two held-out layouts, all six items, reserved for
       the Scorer (R5.4).
3. **Writes each layout's short "clip"** as ``clip.frame_count`` PNG frames under
   ``<ctx.out_dir>/frames/<layout_id>/frame_000.png`` ..., painting the layout
   with :func:`week1_harness.synth.render.render_layout` and writing it with
   :func:`week1_harness.synth.render.save_png`.
4. **Returns a** :class:`~week1_harness.datamodel.StepResult` carrying a ``REAL``
   time cell, the "How" text, and a generous ``payload`` of everything the
   downstream steps need (layouts, per-layout ground truth, clip specs, the
   written frame paths tagged with their session, the canvas size, and the six
   class ids).

Determinism (R3.3 / R5.5)
-------------------------

Scene composition is already deterministic (``scene.compose`` consumes
``ctx.rng_for(1)``), and :func:`render_layout` / :func:`save_png` are pure and
write timestamp-free PNGs, so clips are byte-identical across runs with the same
seed. Each layout stands in for a short video, so frames after the first get a
*tiny* per-frame jitter (bounded by ``clip.jitter_px`` / ``clip.jitter_deg``) to
wobble like real footage. That jitter is drawn from its **own** independent
stream -- ``SeedSequence([ctx.seed, 1, layout_index, frame_index])`` -- and
never from ``ctx.rng_for(1)`` (which the scene already consumed), so it stays
reproducible without perturbing the scene. Frame 0 of every clip is the
un-jittered base pose, which matches the ground-truth geometry exactly; the
ground-truth :class:`~week1_harness.datamodel.Instance` objects are kept as-is
(base geometry) regardless of jitter, and the jitter is deliberately sub-pixel
small so that match stays honest for the downstream mask/visibility steps.

Scope (task 8.1 only): this module generates frames and hands off artifacts. It
does **not** subsample frames (that is Extract, task 8.3) or touch any other
step.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from week1_harness.datamodel import Layout, Placement, StepResult
from week1_harness.markers import real
from week1_harness.synth import render
from week1_harness.synth.items import CLASS_IDS
from week1_harness.synth.scene import CANVAS_SIZE, compose

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from week1_harness.context import RunContext
    from week1_harness.synth.scene import ClipSpec

__all__ = ["FilmStep"]


# The "How" column text for the Timing_Table (R13.2); Film does real CPU work.
_HOW = (
    "Synthetic frames (Pillow): S1 each item x3 lighting, "
    "S2 shuffled layouts, golden held out"
)


class FilmStep:
    """Data Loop step 1 -- generate the synthetic source clips (R5).

    Implements the :class:`week1_harness.pipeline.Step` protocol with the
    canonical ``number``/``name`` and a :meth:`run` that draws the scene, writes
    every layout's clip to ``out/frames/``, and returns a ``REAL``-timed
    :class:`~week1_harness.datamodel.StepResult` with the ground truth and frame
    index downstream steps consume.
    """

    number: int = 1
    name: str = "Film"

    def run(self, ctx: RunContext) -> StepResult:
        """Generate and write every clip, then return the Film result (R5).

        Times the whole generate-and-write pass with :func:`time.perf_counter`
        (real CPU work -> ``REAL`` time cell, R4.1), composes the scene once, and
        writes ``clip.frame_count`` PNG frames per layout under
        ``<ctx.out_dir>/frames/<layout_id>/``.
        """
        start = time.perf_counter()

        scene = compose(ctx)
        frames_dir = os.path.join(os.fspath(ctx.out_dir), "frames")
        frames: dict[str, list[dict]] = {}
        for layout_index, layout in enumerate(scene.layouts):
            clip = scene.clips[layout.layout_id]
            frames[layout.layout_id] = self._render_clip(
                ctx, layout, layout_index, clip, frames_dir
            )

        elapsed = time.perf_counter() - start

        # Be generous and explicit: hand downstream steps the composed scene plus
        # the decomposed parts they index into, the written frame paths (each
        # tagged with its session so sessions are recoverable per frame), the
        # canvas size, and the six class ids.
        payload = {
            "scene": scene,
            "layouts": scene.layouts,
            "instances_by_layout": scene.instances_by_layout,
            "all_instances": scene.all_instances(),
            "clips": scene.clips,
            "frames": frames,
            "frames_dir": frames_dir,
            "canvas_size": CANVAS_SIZE,
            "class_ids": list(CLASS_IDS),
        }

        # Real local-CPU drawing: the time cell is REAL and carries no reason,
        # and Film emits no placeholder/estimate notes (R4.1, R4.6).
        return StepResult(
            number=self.number,
            name=self.name,
            how=_HOW,
            time=real(elapsed, unit="s"),
            payload=payload,
            notes=[],
        )

    def _render_clip(
        self,
        ctx: RunContext,
        layout: Layout,
        layout_index: int,
        clip: ClipSpec,
        frames_dir: str,
    ) -> list[dict]:
        """Write one layout's clip as ``clip.frame_count`` PNG frames.

        Returns one record per written frame -- ``{path, layout_id, session,
        lighting, frame_index}`` -- so Extract can subsample the clip and recover
        each frame's session/layout without re-deriving them.
        """
        layout_dir = os.path.join(frames_dir, layout.layout_id)
        records: list[dict] = []
        for frame_index in range(clip.frame_count):
            frame_layout = self._frame_layout(ctx, layout, layout_index, frame_index, clip)
            image = render.render_layout(frame_layout, size=CANVAS_SIZE)
            path = os.path.join(layout_dir, f"frame_{frame_index:03d}.png")
            render.save_png(image, path)
            records.append(
                {
                    "path": path,
                    "layout_id": layout.layout_id,
                    "session": layout.session,
                    "lighting": layout.lighting,
                    "frame_index": frame_index,
                }
            )
        return records

    def _frame_layout(
        self,
        ctx: RunContext,
        layout: Layout,
        layout_index: int,
        frame_index: int,
        clip: ClipSpec,
    ) -> Layout:
        """Return the layout to paint for one frame of a clip.

        Frame 0 is the un-jittered base pose (it matches the ground-truth
        geometry exactly). Later frames get a tiny deterministic per-frame jitter
        bounded by the clip spec, drawn from an independent sub-stream keyed by
        ``[ctx.seed, 1, layout_index, frame_index]`` so the same seed yields
        byte-identical frames and the scene's ``rng_for(1)`` stream is never
        re-drawn.
        """
        if frame_index == 0 or (clip.jitter_px <= 0.0 and clip.jitter_deg <= 0.0):
            return layout
        seed_seq = np.random.SeedSequence([ctx.seed, self.number, layout_index, frame_index])
        rng = np.random.default_rng(seed_seq)
        return _jittered_layout(layout, rng, clip.jitter_px, clip.jitter_deg)


def _jittered_layout(
    layout: Layout,
    rng: np.random.Generator,
    jitter_px: float,
    jitter_deg: float,
) -> Layout:
    """Build a copy of ``layout`` with a tiny per-placement positional/rotational wobble.

    Each placement's anchor is nudged by up to ``+/- jitter_px`` pixels and its
    rotation by up to ``+/- jitter_deg`` degrees; ``class_id``, ``scale``, and
    ``draw_order`` are preserved so occlusion ordering (and hence the ground
    truth) is unchanged. Pure in ``rng``: same generator state -> same wobble.
    """
    placements: list[Placement] = []
    for p in layout.placements:
        dx = float(rng.uniform(-jitter_px, jitter_px))
        dy = float(rng.uniform(-jitter_px, jitter_px))
        dr = float(rng.uniform(-jitter_deg, jitter_deg))
        placements.append(
            Placement(
                class_id=p.class_id,
                position=(int(round(p.position[0] + dx)), int(round(p.position[1] + dy))),
                rotation=p.rotation + dr,
                scale=p.scale,
                draw_order=p.draw_order,
            )
        )
    return Layout(
        layout_id=layout.layout_id,
        session=layout.session,
        lighting=layout.lighting,
        placements=placements,
    )
