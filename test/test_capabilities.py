"""Unit tests for capability detection (``week1_harness.capabilities``).

Covers task 4.2: the environment check is filesystem- and import-only, never a
network call. Specifically:

* ``detect()`` always reports the GPU-in-Docker check as an ``ESTIMATE`` with a
  non-empty reason, independent of the real hardware (R2.6).
* ``_find_yoloe_weights`` searches local locations in a fixed order, returns the
  first match, prefers the ``s`` weight variant, and returns ``None`` when no
  ungated YOLOE-26 weight is present anywhere (R7.1/R7.2).
* ``_can_import`` is a *presence* check: it returns the right booleans and, for
  heavy optional libraries like ``torch``/``ultralytics``, never executes the
  module (so nothing is imported and no network call is made) (R7.1/R7.2).

To stay hermetic the ambient weight-search locations (the Ultralytics config
dir, the Hugging Face hub cache, and the process cwd) are redirected to empty
temp dirs via the ``isolated`` fixture, so results never depend on whatever
caches happen to exist on the machine running the suite.

Requirements: 2.6, 7.1, 7.2.
"""

import sys
from types import SimpleNamespace

import pytest

from week1_harness import capabilities
from week1_harness.capabilities import (
    Capabilities,
    _can_import,
    _find_yoloe_weights,
    detect,
)
from week1_harness.markers import Marker

WEIGHT_S = "yoloe-26s-seg.pt"
WEIGHT_N = "yoloe-26n-seg.pt"
WEIGHT_M = "yoloe-26m-seg.pt"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect every ambient weight-search location to an empty temp dir.

    The real Ultralytics config dir and Hugging Face hub cache helpers are
    monkeypatched to point at empty directories, and the process cwd is moved to
    an empty dir. After this fixture runs, ``_find_yoloe_weights`` searches only
    empty locations unless a test explicitly drops a weight file in one of them,
    making the search fully deterministic and independent of the host machine.

    (The alternative sanctioned by the task -- setting ``YOLO_CONFIG_DIR`` /
    ``HF_HUB_CACHE`` / ``HF_HOME`` / ``XDG_*`` to empty dirs -- would isolate the
    same locations; patching the helpers directly is simplest and does not
    depend on the env-resolution details.)
    """
    ultra = tmp_path / "ultralytics"
    ultra_weights = ultra / "weights"
    hf = tmp_path / "hf"
    cwd = tmp_path / "cwd"
    for directory in (ultra, ultra_weights, hf, cwd):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        capabilities, "_ultralytics_weights_dirs", lambda: [ultra, ultra_weights]
    )
    monkeypatch.setattr(capabilities, "_hf_hub_cache_dir", lambda: hf)
    monkeypatch.chdir(cwd)

    return SimpleNamespace(
        ultra=ultra, ultra_weights=ultra_weights, hf=hf, cwd=cwd, root=tmp_path
    )


# --- detect(): GPU-in-Docker is always an ESTIMATE (R2.6) ----------------------

def test_detect_gpu_in_docker_is_estimate_with_reason(isolated):
    """detect() reports GPU-in-Docker as an ESTIMATE carrying a non-empty reason."""
    caps = detect(weights_dir=None)

    assert isinstance(caps, Capabilities)
    assert caps.gpu_in_docker.marker is Marker.ESTIMATE
    # A non-empty reason naming the absent asset must accompany the estimate.
    assert isinstance(caps.gpu_in_docker.reason, str)
    assert caps.gpu_in_docker.reason.strip()


# --- _find_yoloe_weights(): absent everywhere -> None (R7.2) -------------------

def test_find_yoloe_weights_returns_none_when_absent(isolated, tmp_path):
    """With an empty --weights-dir and empty ambient caches, the lookup is None."""
    empty_weights = tmp_path / "empty_weights"
    empty_weights.mkdir()

    # Explicit empty weights_dir, and every ambient location is empty too.
    assert _find_yoloe_weights(empty_weights) is None
    # No weights_dir at all: the (isolated, empty) cwd + caches still yield None.
    assert _find_yoloe_weights(None) is None
    # And detect() surfaces that as yoloe_weights_path = None.
    assert detect(weights_dir=None).yoloe_weights_path is None


# --- _find_yoloe_weights(): present in weights_dir -> that path (R7.1) ----------

def test_find_yoloe_weights_found_in_weights_dir_and_via_detect(isolated, tmp_path):
    """A weight dropped in --weights-dir is returned, and surfaced by detect()."""
    weights_dir = tmp_path / "weights_dir"
    weights_dir.mkdir()
    weight = weights_dir / WEIGHT_S
    weight.write_bytes(b"")  # content is irrelevant; only presence is checked

    found = _find_yoloe_weights(weights_dir)
    assert found == str(weight)

    caps = detect(weights_dir=weights_dir)
    assert caps.yoloe_weights_path == str(weight)


def test_find_yoloe_weights_prefers_s_variant(isolated, tmp_path):
    """When s/n/m all exist in one dir, the first-listed (s) variant wins."""
    weights_dir = tmp_path / "weights_dir"
    weights_dir.mkdir()
    for name in (WEIGHT_S, WEIGHT_N, WEIGHT_M):
        (weights_dir / name).write_bytes(b"")

    assert _find_yoloe_weights(weights_dir) == str(weights_dir / WEIGHT_S)


def test_find_yoloe_weights_variant_order_prefers_n_over_m(isolated, tmp_path):
    """First-match follows WEIGHT_NAMES order: n is preferred over m when s is absent."""
    weights_dir = tmp_path / "weights_dir"
    weights_dir.mkdir()
    (weights_dir / WEIGHT_N).write_bytes(b"")
    (weights_dir / WEIGHT_M).write_bytes(b"")

    assert _find_yoloe_weights(weights_dir) == str(weights_dir / WEIGHT_N)


def test_find_yoloe_weights_prefers_weights_dir_over_cwd(isolated, tmp_path):
    """Search order: --weights-dir is consulted before the cwd, so its hit wins."""
    weights_dir = tmp_path / "weights_dir"
    weights_dir.mkdir()
    (weights_dir / WEIGHT_S).write_bytes(b"")
    # An equally-named weight also sits in the (isolated) cwd, searched later.
    (isolated.cwd / WEIGHT_S).write_bytes(b"")

    assert _find_yoloe_weights(weights_dir) == str(weights_dir / WEIGHT_S)


# --- _can_import(): presence check only, no execution/network (R7.1/R7.2) -------

def test_can_import_true_for_stdlib():
    """A always-present stdlib module probes True."""
    assert _can_import("os") is True


def test_can_import_false_for_missing_module():
    """A module that cannot exist probes False (and does not raise)."""
    assert _can_import("totally_not_a_module_xyz") is False


def test_probing_optional_libs_does_not_import_them():
    """Probing torch/ultralytics locates a spec but never executes the module.

    ``find_spec`` resolves the import spec without running the module, so a heavy
    optional library is not pulled into the process by a mere presence check.
    """
    torch_already = "torch" in sys.modules
    ultralytics_already = "ultralytics" in sys.modules

    assert isinstance(_can_import("torch"), bool)
    assert isinstance(_can_import("ultralytics"), bool)

    # The probe must not add the module to sys.modules (it was never executed).
    if not torch_already:
        assert "torch" not in sys.modules
    if not ultralytics_already:
        assert "ultralytics" not in sys.modules


def test_detection_makes_no_network_call(isolated, monkeypatch):
    """Capability detection opens no socket: it is find_spec + filesystem only."""
    import socket

    def _no_sockets(*args, **kwargs):
        raise AssertionError("capability detection must not open a socket (R7.1/R7.2)")

    monkeypatch.setattr(socket, "socket", _no_sockets)

    # Each of these must complete without touching the network.
    _can_import("torch")
    _can_import("ultralytics")
    assert _find_yoloe_weights(None) is None
    caps = detect(weights_dir=None)
    assert caps.gpu_in_docker.marker is Marker.ESTIMATE
