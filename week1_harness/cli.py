"""Command-line entrypoint for the Week 1 Data-Loop Harness.

This module is the single-command entry point (Requirement 1.1). Its first and
most important job happens at import time: it forces Hugging Face / ML
libraries into offline mode by setting the relevant environment variables
*before* any code path can import ``torch`` or ``ultralytics`` (Requirement
2.3). Those libraries are only ever imported lazily, deep inside the Zero-shot
step, so setting the variables here guarantees they are already in place when
that import eventually happens.

:func:`main` then does the wiring: detect capabilities, build the
:class:`~week1_harness.context.RunContext`, run the eight
:data:`CANONICAL_STEPS` through :func:`week1_harness.pipeline.run`, render both
reporters and write them -- plus the structured ``run.json`` -- through
:mod:`week1_harness.output`. A failing step maps to a non-zero exit code and no
deliverables are written, so a partial run never leaves quotable output.

Requirements: 1.1, 1.2, 1.3, 2.3, 3.2
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

from week1_harness import capabilities, output, pipeline  # noqa: E402
from week1_harness.context import DEFAULT_SEED, RunContext  # noqa: E402
from week1_harness.pipeline import StepFailure  # noqa: E402
from week1_harness.reporters.dataset_card import render_dataset_card  # noqa: E402
from week1_harness.reporters.timing_table import render_timing_table  # noqa: E402
from week1_harness.steps.auto_label import AutoLabelStep  # noqa: E402
from week1_harness.steps.extract import ExtractStep  # noqa: E402
from week1_harness.steps.film import FilmStep  # noqa: E402
from week1_harness.steps.live_test import LiveTestStep  # noqa: E402
from week1_harness.steps.review import ReviewStep  # noqa: E402
from week1_harness.steps.score import ScoreStep  # noqa: E402
from week1_harness.steps.train import TrainStep  # noqa: E402
from week1_harness.steps.zero_shot import ZeroShotStep  # noqa: E402


# The eight Data Loop steps in canonical order (R1.2). ``pipeline.run`` takes
# step *instances*; every step is stateless (no ``__init__``, no instance
# attributes), so one module-level instance each is safe to reuse across runs.
CANONICAL_STEPS: tuple[pipeline.Step, ...] = (
    FilmStep(),
    ExtractStep(),
    ZeroShotStep(),
    AutoLabelStep(),
    ReviewStep(),
    TrainStep(),
    ScoreStep(),
    LiveTestStep(),
)


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
        The process exit code: ``0`` on success, ``1`` when a step failed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    caps = capabilities.detect(args.weights_dir)
    ctx = RunContext(
        seed=args.seed,
        out_dir=args.out,
        capabilities=caps,
        no_zero_shot=args.no_zero_shot,
    )

    try:
        results = pipeline.run(ctx, list(CANONICAL_STEPS))
    except StepFailure:
        # ctx.fail() has already reported the offending step on stderr. Render
        # nothing: a failed run must not produce quotable deliverables (R1.2).
        return 1

    timing_table = render_timing_table(results)
    dataset_card = render_dataset_card(results)
    output.write_deliverables(
        args.out,
        timing_table=timing_table,
        dataset_card=dataset_card,
        results=results,
        capabilities=caps,
        seed=args.seed,
    )

    # Both deliverables already end in a newline.
    print(timing_table, end="")
    print()
    print(dataset_card, end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
