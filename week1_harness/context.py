"""Deterministic run context for the Week 1 Data-Loop Harness.

The :class:`RunContext` owns the single seed that drives an entire run
(Requirement 3). It seeds the global pseudo-random generators once at
construction so that any library reaching for a global RNG behaves
deterministically, and it hands each Data Loop step an isolated, reproducible
numpy ``Generator`` derived from the seed and the step number. It also carries
the shared artifact store and the two small progress/error reporting helpers
used by the pipeline runner (Requirements 1.4 and 1.5).
"""

import random
import sys

import numpy as np

DEFAULT_SEED = 20261009  # fixed default seed (R3.2): the Fri 9 Oct 2026 start date


class RunContext:
    """Shared, deterministic context threaded through every Data Loop step.

    ``artifacts`` is a plain ``dict`` into which each step stores its payload for
    downstream steps; it intentionally carries no typed schema here. ``seed`` is
    the single source of randomness for the run (R3.1).
    """

    def __init__(self, seed: int, out_dir, capabilities, *, no_zero_shot=False):
        self.seed = seed
        self.out_dir = out_dir
        self.capabilities = capabilities
        self.no_zero_shot = no_zero_shot
        self.artifacts: dict = {}
        # Seed the global generators so any library that reaches for them is
        # deterministic for this run (R3.1). torch is NOT seeded here; that is
        # done lazily in the zero-shot step, and only if torch is present.
        random.seed(seed)
        np.random.seed(seed % (2**32))

    def rng_for(self, step_number: int) -> np.random.Generator:
        """Return an independent, reproducible numpy ``Generator`` for a step.

        Each step draws from its own sub-stream derived from the run seed and the
        step number via ``SeedSequence``, so per-step randomness is independent of
        execution order and identical across runs with the same seed
        (R3.1, R3.3).
        """
        return np.random.default_rng(np.random.SeedSequence([self.seed, step_number]))

    def progress(self, number: int, name: str) -> None:
        """Emit one progress line identifying the step number and name (R1.4)."""
        print(f"[{number}/8] {name} ...")

    def fail(self, number: int, name: str, exc: Exception) -> None:
        """Report a failing step's number and name to stderr (R1.5)."""
        print(f"ERROR at step {number} ({name}): {exc}", file=sys.stderr)
