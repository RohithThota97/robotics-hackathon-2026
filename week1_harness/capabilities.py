"""Capability detection and offline-mode support (Requirements 2, 7).

``detect()`` runs once at startup and returns an immutable :class:`Capabilities`
record describing what the local machine can do for this run: whether ``torch``
and ``ultralytics`` are importable, whether ungated YOLOE-26 weights are present
in a local cache, and the (always simulated) GPU-in-Docker check.

Two hard rules govern everything in this module (R2.3, R2.4, R2.5, R7.1, R7.2):

* **No network, ever.** Presence of the optional libraries is checked with
  ``importlib.util.find_spec`` (an import-only *presence* check that does not
  execute the module), and weights are located by inspecting the local
  filesystem only. Cache locations are derived from well-known conventions and
  environment variables; neither ``torch``, ``ultralytics`` nor
  ``huggingface_hub`` is imported here.
* **No heavy imports at module load.** This module imports only the standard
  library and the lightweight :mod:`week1_harness.markers`. The optional ML
  libraries are imported lazily (and only if present) by the Zero-shot step.

The downstream Zero-shot step uses this record to decide between running real
YOLOE-26 inference and emitting a marked placeholder (R7.1/R7.2); the GPU-in-
Docker value is always an estimate (R2.6).
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from week1_harness.markers import MarkedValue, estimate

# Ungated YOLOE-26 segmentation weight filenames, in preferred order. The
# ``s`` (small) variant is tried first, then the ``n`` (nano) and ``m`` (medium)
# variants; within any one directory the first of these that exists wins.
WEIGHT_NAMES: tuple[str, ...] = (
    "yoloe-26s-seg.pt",
    "yoloe-26n-seg.pt",
    "yoloe-26m-seg.pt",
)


@dataclass(frozen=True)
class Capabilities:
    """Immutable snapshot of what the local environment can do this run.

    Fields:
        torch_available:       ``torch`` is importable (presence check only).
        ultralytics_available: ``ultralytics`` is importable (presence check only).
        yoloe_weights_path:    path to a local ungated YOLOE-26 weight file, or
                               ``None`` when such weights are absent from the
                               local cache.
        gpu_in_docker:         the simulated GPU-in-Docker check; ALWAYS an
                               ``ESTIMATE`` (R2.6), independent of whether a GPU
                               happens to exist.
    """

    torch_available: bool
    ultralytics_available: bool
    yoloe_weights_path: str | None
    gpu_in_docker: MarkedValue


def _can_import(module_name: str) -> bool:
    """Return whether ``module_name`` is importable, without importing it.

    Uses :func:`importlib.util.find_spec`, which locates the module's import
    spec by consulting the import machinery but does **not** execute the module
    (so a heavy library like ``torch`` is never actually loaded) and never
    touches the network. Any error raised while locating the spec -- for
    example a broken or partially installed package -- is treated as "not
    importable".
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # ImportError: a parent package could not be resolved.
        # ValueError: the module is present but has no spec (e.g. __main__).
        return False


def _ultralytics_weights_dirs() -> list[Path]:
    """Candidate Ultralytics weight directories, derived by convention only.

    Ultralytics keeps its user configuration under a platform-specific config
    directory, overridable with the ``YOLO_CONFIG_DIR`` environment variable and
    otherwise ``$XDG_CONFIG_HOME/Ultralytics`` or ``~/.config/Ultralytics`` on
    Linux. Downloaded weights live either in that directory or in a ``weights``
    subdirectory of it. These paths are reconstructed from the environment
    without importing ``ultralytics``.
    """
    env = os.environ.get("YOLO_CONFIG_DIR")
    if env:
        config_dir = Path(env)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
        config_dir = base / "Ultralytics"
    return [config_dir, config_dir / "weights"]


def _hf_hub_cache_dir() -> Path:
    """The Hugging Face hub cache directory, derived by convention only.

    Resolution follows the standard Hugging Face precedence, using environment
    variables and well-known defaults -- ``HF_HUB_CACHE`` or
    ``HUGGINGFACE_HUB_CACHE`` directly, else ``$HF_HOME/hub``, else
    ``$XDG_CACHE_HOME/huggingface/hub``, else ``~/.cache/huggingface/hub``.
    ``huggingface_hub`` is never imported and no network call is made.
    """
    for env_var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_var)
        if value:
            return Path(value)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def _search_dir(directory: Path, *, recursive: bool) -> str | None:
    """Return the first YOLOE-26 weight found in ``directory``, else ``None``.

    Each weight name in :data:`WEIGHT_NAMES` is checked in order. A direct
    (top-level) match is always preferred; when ``recursive`` is set and no
    direct match exists, the directory tree is walked as a fallback -- this is
    needed for the Hugging Face hub cache, which nests files under
    ``models--.../snapshots/<hash>/``. All filesystem access is defensive: a
    missing or unreadable directory simply yields ``None`` rather than raising.
    """
    try:
        if not directory.is_dir():
            return None
    except OSError:
        return None

    # Prefer a direct hit at the top of the directory.
    for name in WEIGHT_NAMES:
        candidate = directory / name
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue

    if not recursive:
        return None

    # Fallback: walk the tree (used for the nested HF hub cache layout).
    for name in WEIGHT_NAMES:
        try:
            for match in directory.rglob(name):
                if match.is_file():
                    return str(match)
        except OSError:
            continue
    return None


def _find_yoloe_weights(weights_dir: str | os.PathLike[str] | None = None) -> str | None:
    """Locate ungated YOLOE-26 weights on the local filesystem (no network).

    Searches, in order and filesystem-only:

    1. ``weights_dir`` if provided (the ``--weights-dir`` CLI flag),
    2. the current working directory (Ultralytics' "run from the same folder"
       download behaviour noted in week1.md),
    3. the Ultralytics default weights directory,
    4. the Hugging Face hub cache directory.

    Within each location the weight names in :data:`WEIGHT_NAMES` are tried in
    order and the first match's path is returned as a string. Returns ``None``
    when no weight is found (meaning the Zero-shot step takes its placeholder
    path, R7.2). No weights are ever downloaded.
    """
    searches: list[tuple[Path, bool]] = []
    if weights_dir is not None:
        searches.append((Path(weights_dir), False))
    searches.append((Path.cwd(), False))
    searches.extend((d, False) for d in _ultralytics_weights_dirs())
    searches.append((_hf_hub_cache_dir(), True))

    for directory, recursive in searches:
        hit = _search_dir(directory, recursive=recursive)
        if hit is not None:
            return hit
    return None


def detect(weights_dir: str | os.PathLike[str] | None = None) -> Capabilities:
    """Assemble the :class:`Capabilities` record for this run.

    Performs only import-presence checks and local filesystem lookups -- never a
    network call. The GPU-in-Docker check is always emitted as an ``ESTIMATE``
    because the harness's contract is to simulate it regardless of the actual
    hardware (R2.6).
    """
    torch_available = _can_import("torch")
    ultralytics_available = _can_import("ultralytics")
    yoloe_weights_path = _find_yoloe_weights(weights_dir)  # local cache only, no download
    gpu_in_docker = estimate(
        "unavailable (simulated)",
        reason="GPU-in-Docker check simulated; NVIDIA GPU absent",
    )
    return Capabilities(
        torch_available=torch_available,
        ultralytics_available=ultralytics_available,
        yoloe_weights_path=yoloe_weights_path,
        gpu_in_docker=gpu_in_docker,
    )
