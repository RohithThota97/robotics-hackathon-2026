"""Unit tests for the deterministic run context (``week1_harness.context``).

Covers task 3.3: the per-step RNG factory is reproducible across runs with the
same seed and independent across step numbers (R3.1), the fixed default seed is
available and used (R3.2), and ``progress`` emits a single step line carrying
the step number and name (R1.4).
"""

import numpy as np

from week1_harness.context import DEFAULT_SEED, RunContext


def _make_context(seed: int) -> RunContext:
    """Build a RunContext for RNG/progress tests.

    ``capabilities`` is a trivial stand-in (``None``) and ``out_dir`` is unused
    by these tests; neither is exercised here.
    """
    return RunContext(seed=seed, out_dir=None, capabilities=None)


def test_rng_for_is_reproducible_across_contexts():
    """Two contexts built with the same seed yield identical streams for a given k (R3.1)."""
    ctx_a = _make_context(seed=4242)
    ctx_b = _make_context(seed=4242)

    for k in (1, 2, 5, 8):
        seq_a = ctx_a.rng_for(k).random(16)
        seq_b = ctx_b.rng_for(k).random(16)
        assert np.array_equal(seq_a, seq_b)

    # Each call rebuilds the sub-stream from scratch, so repeated calls for the
    # same step number on one context are identical too.
    assert np.array_equal(ctx_a.rng_for(3).random(8), ctx_a.rng_for(3).random(8))


def test_rng_for_streams_are_independent_across_step_numbers():
    """Different step numbers produce different streams; rng_for(1) != rng_for(2) (R3.1)."""
    ctx = _make_context(seed=777)

    sequences = [tuple(ctx.rng_for(k).random(16)) for k in range(1, 9)]

    # All eight canonical step streams are pairwise distinct.
    assert len(set(sequences)) == len(sequences)
    # The specific pair called out by the task.
    assert sequences[0] != sequences[1]


def test_default_seed_constant_is_available_and_used():
    """The fixed default seed is the documented constant and threads into the context (R3.2)."""
    assert DEFAULT_SEED == 20261009

    ctx = _make_context(seed=DEFAULT_SEED)
    assert ctx.seed == DEFAULT_SEED == 20261009


def test_progress_emits_exactly_one_line_with_number_and_name(capsys):
    """progress() prints exactly one line that contains the step number and name (R1.4)."""
    ctx = _make_context(seed=DEFAULT_SEED)

    ctx.progress(3, "Zero-shot baseline")

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "3" in lines[0]
    assert "Zero-shot baseline" in lines[0]
