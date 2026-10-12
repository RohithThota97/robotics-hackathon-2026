"""Domain data models threaded through the harness artifact store.

Every Data Loop step hands its outputs to downstream steps and to the two
reporters through a small set of plain data objects. ``Marker`` and
``MarkedValue`` live in :mod:`week1_harness.markers`; this module defines the
pipeline's per-step result carrier (:class:`StepResult`) and the immutable
domain objects that describe the synthetic dataset.

The domain objects are **frozen** dataclasses: once a step produces a
``PracticeItem``, ``Layout``, ``Instance``, ``Mask`` (and so on), nothing
downstream can rebind its fields, which keeps the determinism guarantee
(Requirement 3) structurally honest. :class:`StepResult` is intentionally
**not** frozen -- it carries a mutable ``payload`` dict that the pipeline runner
stores in the shared artifact store for later steps.

Requirements touched by these models:

* **R4.2**  -- every fabricated field is a :class:`~week1_harness.markers.MarkedValue`
  (``ModelArtifact.value`` and every field of :class:`Metrics`), so the
  marker invariant holds structurally rather than by convention.
* **R5.1**  -- :class:`PracticeItem` carries the class id and the look-alike
  pairing, so the six classes (including the look-alike pair held as two
  separate classes) are explicit end to end.
* **R6.4**  -- :class:`Instance` and :class:`ExtractedImage` carry the
  ``session`` of their source layout (``"S1"`` | ``"S2"`` | ``"golden"``).
* **R11.4** -- :class:`Metrics.recall_per_item` holds one entry per practice
  item.

Note on the numpy dependency: numpy is a mandatory dependency of the harness,
but this module stays import-light. ``numpy`` is imported only under
``TYPE_CHECKING`` (used solely for the :class:`Mask.raster` type hint), and
``from __future__ import annotations`` keeps every annotation lazy, so importing
this module never forces numpy to load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from week1_harness.markers import MarkedValue

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    import numpy as np

# An RGB colour triple (0-255 per channel) used in a practice item's palette.
RGB = tuple[int, int, int]


@dataclass
class StepResult:
    """One Data Loop step's result: timing plus marker metadata and payload.

    Not frozen: ``payload`` is a mutable handoff dict the pipeline runner stores
    in the shared artifact store for downstream steps and the reporters.

    Fields:
        number:  step number ``1..8`` in canonical order.
        name:    human-readable step name (``"Film"``, ``"Extract frames"``...).
        how:     the "How" column text for the Timing_Table (R13.2).
        time:    the "Time" column cell as a ``MarkedValue`` (R13.3/R13.4).
        payload: outputs for downstream steps / reporters.
        notes:   extra marked sub-values (the SAM 3 stub note, per-item
                 detections, metrics...) scanned by the Dataset_Card
                 known-gaps list (R14.4).
    """

    number: int
    name: str
    how: str
    time: MarkedValue
    payload: dict
    notes: list[MarkedValue]


@dataclass(frozen=True)
class PracticeItem:
    """One of the six synthetic practice items (R5.1).

    The look-alike pair is held as two separate items whose ``look_alike_of``
    point at each other; every other item has ``look_alike_of is None``.

    Fields:
        class_id:      unique class id, e.g. ``"screwdriver_small"``.
        shape_family:  parametric shape family used to draw it, e.g.
                       ``"screwdriver"``.
        palette:       the colours used to render the item (body, handle, ...).
        look_alike_of: the paired item's class id, or ``None`` when unpaired.
    """

    class_id: str
    shape_family: str
    palette: tuple[RGB, ...]
    look_alike_of: str | None = None


@dataclass(frozen=True)
class Placement:
    """A single item placed into a layout; draw order creates occlusion.

    Fields:
        class_id:   the placed item's class id.
        position:   ``(x, y)`` anchor position on the canvas.
        rotation:   rotation in degrees.
        scale:      uniform scale factor.
        draw_order: paint order; higher numbers are painted later (on top),
                    occluding lower ones.
    """

    class_id: str
    position: tuple[int, int]
    rotation: float
    scale: float
    draw_order: int


@dataclass(frozen=True)
class Layout:
    """A composed scene: an ordered set of placements under one lighting.

    Fields:
        layout_id:  unique layout id.
        session:    ``"S1"`` | ``"S2"`` | ``"golden"`` (R6.4).
        lighting:   ``"dim"`` | ``"normal"`` | ``"bright"`` (R5.2).
        placements: the items composed into this layout, in draw order.
    """

    layout_id: str
    session: str
    lighting: str
    placements: list[Placement]


@dataclass(frozen=True)
class Instance:
    """A single labelled item occurrence with its ground-truth geometry.

    Fields:
        instance_id:      unique instance id.
        class_id:         one of the six practice-item class ids (R5.1, R8.3).
        polygon:          outline vertices as ``(x, y)`` points.
        bbox:             bounding box ``(x0, y0, x1, y1)``.
        visible_fraction: ``0..1``; drives the Review 1/3 rule (R9.2/R9.3).
        layout_id:        the layout this instance belongs to.
        session:          ``"S1"`` | ``"S2"`` | ``"golden"`` (R6.4).
    """

    instance_id: str
    class_id: str
    polygon: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    visible_fraction: float
    layout_id: str
    session: str


@dataclass(frozen=True)
class ExtractedImage:
    """A frame kept by the Frame_Extractor, tagged with its source session.

    Fields:
        path:             path to the written image file.
        source_layout_id: the layout the frame was rendered from.
        session:          ``"S1"`` | ``"S2"`` | ``"golden"``, inherited from the
                          source layout (R6.4).
        instance_ids:     ids of the instances visible in this frame.
    """

    path: str
    source_layout_id: str
    session: str
    instance_ids: list[str]


@dataclass(frozen=True)
class Mask:
    """One instance segmentation mask produced by the Auto_Labeler (R8.1/R8.3).

    Fields:
        instance_id: the instance this mask labels (one mask per instance, R8.1).
        class_id:    exactly one practice-item class (R8.3).
        raster:      the mask raster as a numpy array of the painted region.
    """

    instance_id: str
    class_id: str
    raster: np.ndarray


@dataclass(frozen=True)
class ReviewedLabel:
    """A label after the Review step's visibility and second-check transform (R9).

    Fields:
        instance_id:  the reviewed instance.
        class_id:     the retained class (the look-alike pair stays separate,
                      R9.4).
        retained:     ``True`` iff ``visible_fraction >= 1/3`` (R9.2/R9.3).
        second_check: ``True`` iff flagged as checked by a second reviewer
                      (R9.5).
    """

    instance_id: str
    class_id: str
    retained: bool
    second_check: bool


@dataclass(frozen=True)
class ModelArtifact:
    """The stubbed model file emitted by the Train step (R10.1/R10.2).

    Fields:
        path:  path to the placeholder artifact file on disk.
        value: a ``PLACEHOLDER`` ``MarkedValue`` recording that this is not a
               trained model.
    """

    path: str
    value: MarkedValue


@dataclass(frozen=True)
class Metrics:
    """Golden-set metrics emitted by Train (placeholder) and the Scorer (R11).

    Every field is a ``MarkedValue`` so the marker invariant holds structurally
    (R4.2). ``recall_per_item`` holds exactly one entry per practice item
    (R11.4).

    Fields:
        map50:               mAP@50 for the run.
        recall_per_item:     ``class_id -> recall`` ``MarkedValue``, one entry
                             per practice item (R11.4).
        lookalike_confusion: confusion between the look-alike pair.
    """

    map50: MarkedValue
    recall_per_item: dict[str, MarkedValue]
    lookalike_confusion: MarkedValue
