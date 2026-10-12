"""Unit tests for Step 3 -- Zero-shot baseline, placeholder branch (task 9.2).

These example-based unit tests pin the *placeholder* path of
:class:`week1_harness.steps.zero_shot.ZeroShotStep` -- the path that runs on the
current dev machine, where ``torch``/``ultralytics`` and the ungated YOLOE-26
weights are absent. They assert:

* **The three placeholder reasons (R7.2).** The step picks its blocker reason by
  precedence -- the explicit ``--no-zero-shot`` flag first, then missing
  libraries, then absent weights:

  - weights absent (libs present, no flag)  -> ``REASON_WEIGHTS``
  - libraries missing (flag off)            -> ``REASON_LIBS``
  - ``--no-zero-shot`` (even if libs+weights present) -> ``REASON_FLAG``

  and that the time cell is a ``PLACEHOLDER`` carrying that same reason while the
  payload records ``path == "placeholder"``.
* **The SAM 3 note (R7.4).** On every path the step emits a ``PLACEHOLDER`` note
  whose reason is ``REASON_SAM3`` ("gated weights absent"), because SAM 3 is a
  gated model the harness never loads.
* **The per-item note (R7.3).** The per-item status map covers exactly the six
  practice items, and on the placeholder path every item reads as not detected.

The tests are hermetic by construction: they craft a :class:`Capabilities`
record and a minimal Extract payload directly and never run Film or Extract, so
no frames are rendered and neither ``torch`` nor ``ultralytics`` nor any weight
file is needed. None of these three cases reaches the capable branch, so the
lazy ``torch``/``ultralytics`` import is never triggered. The capable
(real-inference) path is task 9.3 and is intentionally not exercised here.

Requirements: 7.2, 7.3, 7.4.
"""

import pytest

from week1_harness.capabilities import Capabilities
from week1_harness.context import RunContext
from week1_harness.markers import Marker, estimate
from week1_harness.steps.zero_shot import (
    REASON_FLAG,
    REASON_LIBS,
    REASON_SAM3,
    REASON_WEIGHTS,
    ZeroShotStep,
)
from week1_harness.synth.items import CLASS_IDS

# A fixed seed keeps construction deterministic; the placeholder path draws no
# randomness, so the exact value is immaterial.
SEED = 0

# A plausible local weight path used by the cases where weights are "present".
# It is never opened: those cases are blocked by the flag or by missing libs
# before the capable branch (which would load it) is reached.
WEIGHTS_PATH = "/tmp/weights/yoloe-26s-seg.pt"


# --------------------------------------------------------------------------- #
# Builders: craft capabilities + context and run the step, with no Film/Extract #
# --------------------------------------------------------------------------- #

def _capabilities(*, torch_available, ultralytics_available, yoloe_weights_path):
    """Build a crafted ``Capabilities`` record (no real detection runs).

    ``gpu_in_docker`` is always an ``ESTIMATE`` in a real run (R2.6); here it is
    an arbitrary estimate because the Zero-shot step never reads it.
    """
    return Capabilities(
        torch_available=torch_available,
        ultralytics_available=ultralytics_available,
        yoloe_weights_path=yoloe_weights_path,
        gpu_in_docker=estimate(
            "unavailable (simulated)",
            reason="GPU-in-Docker check simulated; NVIDIA GPU absent",
        ),
    )


def _context(caps, *, no_zero_shot):
    """Build a ``RunContext`` wired with the minimal Extract payload the step reads.

    The Zero-shot step reads the six item names from
    ``ctx.artifacts["Extract frames"]["class_ids"]`` (and, only on the capable
    path, ``training_images``). Both are supplied directly -- the six canonical
    class ids and an empty image list -- so no Film/Extract run is needed.
    """
    ctx = RunContext(seed=SEED, out_dir=None, capabilities=caps, no_zero_shot=no_zero_shot)
    ctx.artifacts["Extract frames"] = {
        "class_ids": list(CLASS_IDS),
        "training_images": [],
    }
    return ctx


def _run(*, torch_available, ultralytics_available, yoloe_weights_path, no_zero_shot):
    """Run ``ZeroShotStep`` against a crafted capability record and return the result."""
    caps = _capabilities(
        torch_available=torch_available,
        ultralytics_available=ultralytics_available,
        yoloe_weights_path=yoloe_weights_path,
    )
    ctx = _context(caps, no_zero_shot=no_zero_shot)
    return ZeroShotStep().run(ctx)


# The three placeholder cases, keyed by the blocker that forces the stub path.
CASES = {
    "weights-absent": dict(
        torch_available=True,
        ultralytics_available=True,
        yoloe_weights_path=None,
        no_zero_shot=False,
    ),
    "import-missing": dict(
        torch_available=False,
        ultralytics_available=False,
        yoloe_weights_path=None,
        no_zero_shot=False,
    ),
    "flag": dict(
        torch_available=True,
        ultralytics_available=True,
        yoloe_weights_path=WEIGHTS_PATH,
        no_zero_shot=True,
    ),
}


def _sam3_note(result):
    """Return the single SAM 3 note (the note whose reason is ``REASON_SAM3``)."""
    notes = [note for note in result.notes if note.reason == REASON_SAM3]
    assert len(notes) == 1, f"expected exactly one SAM 3 note, found {len(notes)}"
    return notes[0]


def _per_item_note(result):
    """Return the single per-item detection note (the note whose value is the map)."""
    notes = [note for note in result.notes if isinstance(note.value, dict)]
    assert len(notes) == 1, f"expected exactly one per-item note, found {len(notes)}"
    return notes[0]


# --------------------------------------------------------------------------- #
# R7.2 -- the three placeholder reasons.                                        #
# --------------------------------------------------------------------------- #

def test_weights_absent_is_placeholder_with_weights_reason():
    """Libs present but weights absent -> PLACEHOLDER "ungated weights..." (R7.2)."""
    result = _run(
        torch_available=True,
        ultralytics_available=True,
        yoloe_weights_path=None,
        no_zero_shot=False,
    )

    # Anchor: this is Data Loop step 3.
    assert result.number == 3
    assert result.name == "Zero-shot baseline"

    # The time cell is a PLACEHOLDER naming the missing weights (R7.2)...
    assert result.time.marker is Marker.PLACEHOLDER
    assert result.time.reason == REASON_WEIGHTS
    # ...and the payload took the stub path with the same reason.
    assert result.payload["path"] == "placeholder"
    assert result.payload["reason"] == REASON_WEIGHTS


@pytest.mark.parametrize(
    "torch_available, ultralytics_available",
    [(False, True), (True, False), (False, False)],
    ids=["torch-missing", "ultralytics-missing", "both-missing"],
)
@pytest.mark.parametrize(
    "yoloe_weights_path",
    [None, WEIGHTS_PATH],
    ids=["no-weights", "weights-present"],
)
def test_import_missing_is_placeholder_with_libs_reason(
    torch_available, ultralytics_available, yoloe_weights_path
):
    """Missing torch and/or ultralytics -> PLACEHOLDER "...not importable" (R7.2).

    The libraries-missing reason outranks the weights reason: it is reported even
    when a weight path is set, because missing libraries are the more fundamental
    blocker.
    """
    result = _run(
        torch_available=torch_available,
        ultralytics_available=ultralytics_available,
        yoloe_weights_path=yoloe_weights_path,
        no_zero_shot=False,
    )

    assert result.time.marker is Marker.PLACEHOLDER
    assert result.time.reason == REASON_LIBS
    assert result.payload["path"] == "placeholder"
    assert result.payload["reason"] == REASON_LIBS


def test_no_zero_shot_flag_forces_placeholder_with_flag_reason():
    """``--no-zero-shot`` wins even with libs present and weights available (R7.2)."""
    result = _run(
        torch_available=True,
        ultralytics_available=True,
        yoloe_weights_path=WEIGHTS_PATH,
        no_zero_shot=True,
    )

    assert result.time.marker is Marker.PLACEHOLDER
    assert result.time.reason == REASON_FLAG
    assert result.payload["path"] == "placeholder"
    assert result.payload["reason"] == REASON_FLAG


# --------------------------------------------------------------------------- #
# R7.4 -- SAM 3 concept-prompt baseline is always a stubbed placeholder.        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_sam3_note_is_placeholder_gated_in_every_case(case):
    """On every placeholder case the SAM 3 note is PLACEHOLDER "gated weights absent"."""
    result = _run(**CASES[case])

    note = _sam3_note(result)
    assert note.marker is Marker.PLACEHOLDER
    assert note.reason == REASON_SAM3


# --------------------------------------------------------------------------- #
# R7.3 -- per-item detection note covers all six items.                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_per_item_note_covers_all_six_items_not_detected(case):
    """The per-item status covers the six items; on the stub path none is detected (R7.3)."""
    result = _run(**CASES[case])

    # The per-item note mirrors the payload's per-item status map.
    status = _per_item_note(result).value
    assert status == result.payload["per_item_status"]

    # It has exactly one entry per practice item, keyed by the canonical ids.
    assert len(status) == 6
    assert set(status) == set(CLASS_IDS)

    # On the placeholder path nothing is detected: every status reads not-detected,
    # and the payload's detection views agree.
    for class_id, value in status.items():
        assert "not detected" in value, f"{class_id}: {value!r} does not read not-detected"
    assert result.payload["detected_items"] == []
    assert result.payload["detections"] == {class_id: False for class_id in CLASS_IDS}
