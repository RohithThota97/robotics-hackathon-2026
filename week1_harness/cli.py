"""Command-line entrypoint for the Week 1 Data-Loop Harness.

This module is the single-command entry point (Requirement 1.1). Its first and
most important job happens at import time: it forces Hugging Face / ML
libraries into offline mode by setting the relevant environment variables
*before* any code path can import ``torch`` or ``ultralytics`` (Requirement
2.3). Those libraries are only ever imported lazily, deep inside the Zero-shot
step, so setting the variables here guarantees they are already in place when
that import eventually happens.

The actual pipeline wiring -- capability detection, ``RunContext``
construction, and ``pipeline.run`` -- is intentionally NOT done here yet; it is
assembled in task 14.2. For now :func:`main` only parses arguments, which keeps
the entrypoint importable and callable without side effects.
"""

# --- Offline enforcement (R2.3) -------------------------------------------
# Import ``os`` first and set the offline flags before importing anything that
# might (even transitively) pull in an ML library. Hugging Face reads these
# environment variables at import time, so they must be set first. The imports
# below are deliberately placed after this block; E402 is expected here.
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# --------------------------------------------------------------------------

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Fixed default seed (R3.2): the Fri 9 Oct 2026 Week 1 start date. Once
# context.py exists (task 3.2) this same value lives there as ``DEFAULT_SEED``;
# it is duplicated here only so the parser has a default without importing the
# not-yet-built context module.
DEFAULT_SEED = 20261009


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the harness CLI."""
    parser = argparse.ArgumentParser(
        prog="week1-harness",
        description=(
            "Run the Week 1 practice data loop end-to-end on CPU, fully "
            "offline, and emit the timing table and dataset card."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for all pseudo-random generation (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./out"),
        help="Output directory for deliverables and artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--no-zero-shot",
        action="store_true",
        help="Skip real zero-shot inference and emit a marked placeholder instead.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Optional directory searched first for local YOLOE-26 weights.",
    )
    return parser


def main(argv=None) -> int:
    """Parse arguments and run the harness.

    Parameters
    ----------
    argv:
        Optional argument list; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        The process exit code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # TODO(task 14.2): wire the full pipeline here -- run capabilities.detect(),
    # build the RunContext from ``args``, call pipeline.run(ctx, CANONICAL_STEPS),
    # render both reporters, write the deliverables + run.json via output.py,
    # echo both deliverables to stdout, and map StepFailure to a non-zero exit
    # code. Until then this is a stub that only parses arguments, so the
    # entrypoint stays importable and callable without side effects.
    _ = args  # Parsed args are consumed by the wiring added in task 14.2.

    return 0


if __name__ == "__main__":
    sys.exit(main())
