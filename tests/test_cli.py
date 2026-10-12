"""End-to-end tests for the wired single-command entrypoint (task 14.2).

These are the first tests to exercise ``main()`` itself. They cover the four
guarantees the CLI makes: one command writes all three outputs, the same seed
reproduces the same dataset card, a failing step exits non-zero *without*
leaving deliverables anyone could quote, and ``--no-zero-shot`` yields a
placeholder zero-shot row carrying the flag reason.

The failure path substitutes a raising step into ``CANONICAL_STEPS`` rather than
breaking a real one, following the ``FakeStep`` / ``RaisingStep`` pattern in
tests/test_pipeline_property.py.

Requirements: 1.1, 1.2, 1.3, 2.3, 3.2, 7.2
"""

from __future__ import annotations

import json

import pytest

from week1_harness import cli, output
from week1_harness.context import DEFAULT_SEED
from week1_harness.datamodel import StepResult
from week1_harness.markers import Marker, real
from week1_harness.steps.zero_shot import REASON_FLAG

SEED = 20261009

ZERO_SHOT_STEP_NAME = "Zero-shot baseline"


class _InjectedError(RuntimeError):
    """Raised by _RaisingStep so the test can assert on the cause."""


class _RaisingStep:
    """A step that always fails, to drive the StepFailure path."""

    number: int = 1
    name: str = "Film"

    def run(self, ctx) -> StepResult:  # noqa: ANN001 - matches the Step protocol
        raise _InjectedError("injected failure")


class _TrivialStep:
    """A step that succeeds without touching the filesystem."""

    def __init__(self, number: int, name: str) -> None:
        self.number = number
        self.name = name

    def run(self, ctx) -> StepResult:  # noqa: ANN001 - matches the Step protocol
        return StepResult(
            number=self.number,
            name=self.name,
            how="trivial",
            time=real(0.0, "s"),
            payload={},
            notes=[],
        )


def _deliverables(out_dir) -> list[str]:
    """The deliverable filenames present in ``out_dir``."""
    if not out_dir.exists():
        return []
    expected = {
        output.TIMING_TABLE_FILENAME,
        output.DATASET_CARD_FILENAME,
        output.RUN_JSON_FILENAME,
    }
    return sorted(expected.intersection(p.name for p in out_dir.iterdir()))


def _zero_shot_row(out_dir) -> str:
    """The Zero-shot line from the rendered timing table."""
    text = (out_dir / output.TIMING_TABLE_FILENAME).read_text(encoding="utf-8")
    for line in text.splitlines():
        if ZERO_SHOT_STEP_NAME in line and line.startswith("|"):
            return line
    raise AssertionError(f"no zero-shot row in:\n{text}")


# --- 1. one command writes all three files --------------------------------


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """One full default run, shared by the tests that only read its output."""
    out_dir = tmp_path_factory.mktemp("cli_default")
    exit_code = cli.main(["--out", str(out_dir)])
    return exit_code, out_dir


def test_single_command_exits_zero(default_run):
    exit_code, _ = default_run
    assert exit_code == 0


def test_single_command_writes_all_three_files(default_run):
    _, out_dir = default_run
    assert _deliverables(out_dir) == sorted(
        [
            output.DATASET_CARD_FILENAME,
            output.RUN_JSON_FILENAME,
            output.TIMING_TABLE_FILENAME,
        ]
    )


def test_single_command_prints_both_deliverables(tmp_path, capsys):
    cli.main(["--out", str(tmp_path)])

    captured = capsys.readouterr().out
    assert "| # | Step | How | Time |" in captured
    assert "Dataset Card" in captured
    # All eight progress lines were emitted.
    assert captured.count("[8/8]") == 1


def test_run_json_records_all_eight_steps(default_run):
    _, out_dir = default_run
    run = json.loads((out_dir / output.RUN_JSON_FILENAME).read_text(encoding="utf-8"))

    assert run["seed"] == DEFAULT_SEED
    assert [s["number"] for s in run["steps"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert run["capabilities"] is not None


def test_canonical_steps_are_the_eight_in_order():
    assert [s.number for s in cli.CANONICAL_STEPS] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [s.name for s in cli.CANONICAL_STEPS] == [
        "Film",
        "Extract frames",
        ZERO_SHOT_STEP_NAME,
        "Auto-label",
        "Review",
        "Train",
        "Score",
        "Test live",
    ]


def test_timing_table_rounds_times(default_run):
    """Raw perf_counter floats must not reach the rendered table."""
    import re

    _, out_dir = default_run
    text = (out_dir / output.TIMING_TABLE_FILENAME).read_text(encoding="utf-8")

    assert not re.search(r"\d+\.\d{6,}", text)


def test_run_json_keeps_full_precision(default_run):
    """Rounding is display-only: run.json keeps the measurement intact."""
    _, out_dir = default_run
    run = json.loads((out_dir / output.RUN_JSON_FILENAME).read_text(encoding="utf-8"))

    real_times = [
        s["time"]["value"]
        for s in run["steps"]
        if s["time"]["marker"] == Marker.REAL.value
    ]
    assert real_times, "expected at least one REAL step time"
    assert all(isinstance(t, float) for t in real_times)


# --- 2. same seed reproduces the same card --------------------------------


def test_same_seed_gives_identical_dataset_card(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"

    assert cli.main(["--seed", str(SEED), "--out", str(first)]) == 0
    assert cli.main(["--seed", str(SEED), "--out", str(second)]) == 0

    card = output.DATASET_CARD_FILENAME
    assert (first / card).read_text(encoding="utf-8") == (second / card).read_text(
        encoding="utf-8"
    )


def test_same_seed_gives_identical_counts_and_markers(tmp_path):
    """Compare the JSON projection with times removed; CPU times legitimately differ."""
    first, second = tmp_path / "a", tmp_path / "b"
    cli.main(["--seed", str(SEED), "--out", str(first)])
    cli.main(["--seed", str(SEED), "--out", str(second)])

    # Fields that are a function of --out rather than of --seed, so they
    # legitimately differ between two runs writing to different directories.
    path_derived = {"frames_dir"}

    def comparable(path):
        run = json.loads((path / output.RUN_JSON_FILENAME).read_text(encoding="utf-8"))
        for step in run["steps"]:
            marker = step["time"]["marker"]
            # REAL times are wall-clock and will differ between runs; the
            # marker and reason must not.
            step["time"] = marker if marker == Marker.REAL.value else step["time"]
            for key in path_derived:
                step["payload"].pop(key, None)
        return run

    assert comparable(first) == comparable(second)


# --- 3. a failing step exits non-zero and writes nothing ------------------


def test_failing_step_exits_non_zero_and_writes_no_deliverables(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CANONICAL_STEPS", (_RaisingStep(),))

    exit_code = cli.main(["--out", str(tmp_path)])

    assert exit_code == 1
    assert _deliverables(tmp_path) == []


def test_failing_step_reports_the_step_on_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "CANONICAL_STEPS", (_RaisingStep(),))

    cli.main(["--out", str(tmp_path)])

    err = capsys.readouterr().err
    assert "ERROR at step 1 (Film)" in err


def test_failure_after_a_successful_step_still_writes_nothing(tmp_path, monkeypatch):
    """A late failure must not leave the deliverables from the steps that passed."""
    monkeypatch.setattr(
        cli,
        "CANONICAL_STEPS",
        (_TrivialStep(1, "Film"), _RaisingStep()),
    )

    assert cli.main(["--out", str(tmp_path)]) == 1
    assert _deliverables(tmp_path) == []


# --- 4. --no-zero-shot gives the flagged placeholder ----------------------


def test_no_zero_shot_renders_placeholder_with_flag_reason(tmp_path):
    assert cli.main(["--no-zero-shot", "--out", str(tmp_path)]) == 0

    row = _zero_shot_row(tmp_path)
    assert f"[{Marker.PLACEHOLDER.value}: {REASON_FLAG}]" in row


def test_no_zero_shot_reason_reaches_run_json(tmp_path):
    cli.main(["--no-zero-shot", "--out", str(tmp_path)])
    run = json.loads((tmp_path / output.RUN_JSON_FILENAME).read_text(encoding="utf-8"))

    zero_shot = next(s for s in run["steps"] if s["name"] == ZERO_SHOT_STEP_NAME)
    assert zero_shot["time"]["marker"] == Marker.PLACEHOLDER.value
    assert zero_shot["time"]["reason"] == REASON_FLAG


# --- parser defaults ------------------------------------------------------


def test_parser_uses_the_shared_default_seed():
    """cli.py must no longer keep its own duplicate of DEFAULT_SEED."""
    args = cli.build_parser().parse_args([])

    assert args.seed == DEFAULT_SEED
    assert not hasattr(cli, "DEFAULT_SEED") or cli.DEFAULT_SEED is DEFAULT_SEED
