"""Unit tests for the scene model (``week1_harness.synth.scene``).

Covers task 7.4: the pure composition / ground-truth layer that turns the six
practice items into S1, S2, and golden layouts plus per-instance ground truth.
These are example-based unit tests (the multi-run session-coverage property is
Property 10, task 14.7); here we pin the concrete contract of :func:`compose`:

* **Determinism (R5.5).** Two runs with the same seed produce identical layout
  ids, identical placements, and identical instance ``visible_fraction``s; a
  different seed changes the composition.
* **S1 coverage (R5.2).** Each of the six items appears alone under all three
  lighting conditions; each S1 layout holds exactly one placement, and that lone
  instance is fully visible (``visible_fraction == 1.0``).
* **S2 coverage (R5.3).** There are at least five S2 layouts (the design count
  plan uses ``S2_LAYOUT_COUNT``), and each contains exactly the six classes.
* **Golden (R5.4).** ``GOLDEN_LAYOUT_COUNT`` is one or two; each golden layout
  contains all six items and its instances are tagged ``session == "golden"``.
* **Occlusion.** Every instance's ``visible_fraction`` lies in ``[0.0, 1.0]``,
  and draw-order occlusion drives at least one lower S2 instance below ``1.0``
  while the top-most (unoccluded) instance stays at ``1.0``.

Requirements: 5.1, 5.2, 5.3, 5.4.
"""

from collections import defaultdict

from week1_harness.context import RunContext
from week1_harness.synth.items import CLASS_IDS
from week1_harness.synth.scene import (
    GOLDEN_LAYOUT_COUNT,
    LIGHTING_CONDITIONS,
    S2_LAYOUT_COUNT,
    compose,
)

# A fixed seed makes every assertion below concrete and reproducible; ALT_SEED
# is simply a different seed used to show the composition is seed-dependent.
SEED = 20261009
ALT_SEED = 1234


def _ctx(seed: int) -> RunContext:
    """Build a RunContext for scene composition.

    ``compose`` only ever reaches for ``ctx.rng_for(1)``, so ``out_dir`` and
    ``capabilities`` are unused here and passed as ``None`` (mirroring
    ``test_context.py``).
    """
    return RunContext(seed=seed, out_dir=None, capabilities=None)


def _layouts_for(scene, session: str):
    """Return the composed layouts belonging to one session, in build order."""
    return [layout for layout in scene.layouts if layout.session == session]


def _placement_signature(scene):
    """A fully-ordered, comparable fingerprint of every placement in the scene.

    Two scenes compare equal here iff they share the same layouts (by id and
    order) and the same per-placement class, position, rotation, scale, and draw
    order -- i.e. the complete composition geometry.
    """
    return [
        (
            layout.layout_id,
            placement.class_id,
            placement.position,
            placement.rotation,
            placement.scale,
            placement.draw_order,
        )
        for layout in scene.layouts
        for placement in layout.placements
    ]


# --- Determinism (R5.5) --------------------------------------------------------

def test_compose_is_deterministic_for_a_fixed_seed():
    """Same seed -> identical layout ids, placements, and visible fractions (R5.5)."""
    scene_a = compose(_ctx(SEED))
    scene_b = compose(_ctx(SEED))

    # Identical layout ids, in the same build order.
    assert [l.layout_id for l in scene_a.layouts] == [l.layout_id for l in scene_b.layouts]

    # Identical placements (Placement is a frozen dataclass, so == is field-wise).
    for layout_a, layout_b in zip(scene_a.layouts, scene_b.layouts):
        assert layout_a.placements == layout_b.placements

    # Identical ground-truth visible fractions across the whole scene.
    vf_a = [inst.visible_fraction for inst in scene_a.all_instances()]
    vf_b = [inst.visible_fraction for inst in scene_b.all_instances()]
    assert vf_a == vf_b


def test_compose_differs_for_a_different_seed():
    """A different seed changes the randomized composition (R5.5)."""
    scene_a = compose(_ctx(SEED))
    scene_c = compose(_ctx(ALT_SEED))

    # The layout *ids* are structural (s1_<item>_<lighting>, s2_NN, golden_NN)
    # and therefore stable across seeds; the geometry they carry must not be.
    assert _placement_signature(scene_a) != _placement_signature(scene_c)


# --- S1 coverage (R5.2) --------------------------------------------------------

def test_s1_covers_each_item_alone_under_three_lightings_fully_visible():
    """S1: every item appears alone under all three lightings, fully visible (R5.2)."""
    scene = compose(_ctx(SEED))
    s1 = _layouts_for(scene, "S1")

    # 6 items x 3 lighting conditions.
    assert len(s1) == len(CLASS_IDS) * len(LIGHTING_CONDITIONS)

    lightings_by_item: dict[str, set[str]] = defaultdict(set)
    for layout in s1:
        # Each S1 layout is a single item alone.
        assert len(layout.placements) == 1
        placement = layout.placements[0]
        lightings_by_item[placement.class_id].add(layout.lighting)

        # A lone item has no occluders, so its single instance is fully visible.
        instances = scene.instances_by_layout[layout.layout_id]
        assert len(instances) == 1
        assert instances[0].class_id == placement.class_id
        assert instances[0].visible_fraction == 1.0

    # Every one of the six items shows up, each under all three lightings.
    assert set(lightings_by_item) == set(CLASS_IDS)
    for class_id in CLASS_IDS:
        assert lightings_by_item[class_id] == set(LIGHTING_CONDITIONS)


# --- S2 coverage (R5.3) --------------------------------------------------------

def test_s2_has_at_least_five_layouts_each_with_all_six_items():
    """S2: >= 5 shuffled layouts (expect S2_LAYOUT_COUNT), each all six items (R5.3)."""
    scene = compose(_ctx(SEED))
    s2 = _layouts_for(scene, "S2")

    assert S2_LAYOUT_COUNT >= 5
    assert len(s2) == S2_LAYOUT_COUNT

    for layout in s2:
        # Exactly the six classes, one placement each.
        assert sorted(p.class_id for p in layout.placements) == sorted(CLASS_IDS)


# --- Golden (R5.4) -------------------------------------------------------------

def test_golden_layouts_contain_all_six_items_and_are_tagged_golden():
    """Golden: 1-2 held-out layouts of all six items, tagged session=golden (R5.4)."""
    scene = compose(_ctx(SEED))

    assert GOLDEN_LAYOUT_COUNT in {1, 2}
    golden = _layouts_for(scene, "golden")
    assert len(golden) == GOLDEN_LAYOUT_COUNT

    for layout in golden:
        assert sorted(p.class_id for p in layout.placements) == sorted(CLASS_IDS)

        instances = scene.instances_by_layout[layout.layout_id]
        assert sorted(inst.class_id for inst in instances) == sorted(CLASS_IDS)
        assert all(inst.session == "golden" for inst in instances)


# --- Occlusion -----------------------------------------------------------------

def test_visible_fraction_is_within_the_unit_interval_for_every_instance():
    """Every ground-truth instance has 0.0 <= visible_fraction <= 1.0."""
    scene = compose(_ctx(SEED))
    instances = scene.all_instances()

    assert instances  # the scene is non-empty
    assert all(0.0 <= inst.visible_fraction <= 1.0 for inst in instances)


def test_draw_order_occlusion_pushes_a_lower_s2_instance_below_full_visibility():
    """Occlusion: a lower-draw-order S2 item under a higher one is < 1.0 visible."""
    scene = compose(_ctx(SEED))
    s2_layout_ids = {layout.layout_id for layout in _layouts_for(scene, "S2")}

    occluded = [
        inst
        for inst in scene.all_instances()
        if inst.layout_id in s2_layout_ids and inst.visible_fraction < 1.0
    ]
    assert occluded, "expected draw-order occlusion to reduce some S2 instance below 1.0"

    # In a layout where occlusion happened, the top-most placement (highest
    # draw_order) has no occluders, so it must remain fully visible. Instances
    # are built in ascending draw_order, so the last one is the top.
    top_instance = scene.instances_by_layout[occluded[0].layout_id][-1]
    assert top_instance.visible_fraction == 1.0
