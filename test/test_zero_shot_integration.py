"""Integration test for Step 3 -- real YOLOE-26 zero-shot inference (R7.1).

Covers task 9.3: the *optional* real-inference path of
:class:`~week1_harness.steps.zero_shot.ZeroShotStep`. The Zero-shot step has two
paths (see the module docstring in ``steps/zero_shot.py``); this test exercises
the **capable path (R7.1)**, which runs genuine CPU YOLOE-26 text-prompt
inference over the six Practice Item names and records a ``REAL`` time cell.

That path only runs when the local machine can *genuinely* do the work:

* ``torch`` is importable,
* ``ultralytics`` is importable, and
* ungated YOLOE-26 weights are present in a local filesystem cache.

On the target development machine none of that is present (``torch`` /
``ultralytics`` are absent and no weight is cached), so this test is expected to
**skip** -- skipping here is the normal, passing outcome and proves the guard
works; it must never *fail* just because the optional path is unavailable. The
end-to-end loop and both deliverables are produced with neither library
installed; only this test touches the real-inference path (R7.1 / R7.2).

The guard mirrors exactly the capable-path condition in ``ZeroShotStep.run``:
``torch_available and ultralytics_available and yoloe_weights_path is not None
and not ctx.no_zero_shot``. We detect the libraries with
:func:`importlib.util.find_spec` (an import-only presence check, no execution,
no network) and the cached weight with
``week1_harness.capabilities.detect(None).yoloe_weights_path``, and reuse that
same :class:`~week1_harness.capabilities.Capabilities` record for the run so the
step's own decision is consistent with what the guard checked.

Requirements: 7.1.
"""

import importlib.util

import pytest

from week1_harness import capabilities, pipeline
from week1_harness.context import RunContext
from week1_harness.markers import Marker
from week1_harness.steps.extract import ExtractStep
from week1_harness.steps.film import FilmStep
from week1_harness.steps.zero_shot import ZeroShotStep
from week1_harness.synth.items import CLASS_IDS

# The default/start-date seed used across the harness tests (R3.2).
SEED = 20261009


def test_real_yoloe26_inference_capable_path(tmp_path):
    """Run one real CPU inference and assert real detections + a REAL time cell (R7.1).

    OPTIONAL real-inference path. Skipped (the normal case on the dev machine)
    unless ``torch``, ``ultralytics``, and a locally-cached ungated YOLOE-26
    weight are all present. When they are, Film -> Extract -> Zero-shot takes the
    capable path and the step reports genuine local-CPU work.
    """
    # --- Runtime guard: only run when the real capable path is genuinely available.
    caps = capabilities.detect(None)  # import-presence + local-cache lookup, no network
    missing: list[str] = []
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if importlib.util.find_spec("ultralytics") is None:
        missing.append("ultralytics")
    if caps.yoloe_weights_path is None:
        missing.append("a locally-cached ungated YOLOE-26 weight")
    if missing:
        pytest.skip(
            "optional real-inference path (R7.1) unavailable: missing "
            + ", ".join(missing)
            + " -- expected on the CPU-only dev machine, so this test SKIPs "
            "rather than fails (the capable-path guard works)."
        )

    # --- Capable path: run Film -> Extract -> Zero-shot through the real pipeline.
    # Reuse the detected capabilities so the step's capable decision matches the
    # guard above; the default no_zero_shot=False keeps the capable path enabled.
    ctx = RunContext(seed=SEED, out_dir=tmp_path, capabilities=caps)
    results = pipeline.run(ctx, [FilmStep(), ExtractStep(), ZeroShotStep()])
    result = results[-1]  # the Zero-shot StepResult

    # Anchor that this is the Zero-shot step's result.
    assert result.number == 3
    assert result.name == "Zero-shot baseline"

    # The capable path was taken (R7.1), not the placeholder stub.
    assert result.payload["path"] == "capable"

    # Real local-CPU inference -> a REAL time cell carrying no reason (R4.1 / R7.1).
    assert result.time.marker is Marker.REAL
    assert result.time.reason is None

    # Real detections: a dict over exactly the six items, each a boolean.
    detections = result.payload["detections"]
    assert isinstance(detections, dict)
    assert set(detections) == set(CLASS_IDS)
    assert all(isinstance(hit, bool) for hit in detections.values())
