# Dynamic Storage Picking · Hackathon workspace

Planning workspace and code for Rohith's team: Challenge 3 of the Robotics & AI Hackathon 2026 at the Siemens Healthineers Innovation Center Erlangen, 22–24 Oct 2026 (confirm the final day). Most of this repository is process plans and checklists; the one piece of code is the Week 1 practice harness described below.

## Where to start

1. [hackathon-plan.md](hackathon-plan.md): the full plan, from dataset to demo and pitch, with sources.
2. Before the event: [prep/week1/week1.md](prep/week1/week1.md) (9–15 Oct) and [prep/week2/week2.md](prep/week2/week2.md) (16–21 Oct).
3. On site: [Day 1](day1/day1.md), [Day 2](day2/day2.md), [Day 3](day3/day3.md), [Day 4 / Final day](day4/day4.md).
4. Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) — branch, open a pull request, get it reviewed. `main` is protected, so read this before your first change.

## Documents

Maintained by hand; keep this table in step when you add a document.

| Document | Checkpoint or goal |
|---|---|
| [Dynamic Storage Picking](hackathon-plan.md) | A worker asks for an item by voice. The robot works out where the item probably is, finds it even though the storage has been rearranged, grasps it and hands it over. |
| [Week 1 · Environments, practice loop, templates](prep/week1/week1.md) | Every laptop runs the full data loop with the network off. You know your real turnaround from filming to a model running live. |
| [Week 2 · Rehearse, task brain, pitch, pack](prep/week2/week2.md) | You can run hand-eye calibration and a collision-aware pick sequence from your notes. |
| [Day 1 · Understand and build a walking skeleton](day1/day1.md) | The camera detects an item and the robot moves its gripper to touch it. That proves the whole chain from pixels to robot coordinates. |
| [Day 2 · First full voice-to-handover run](day2/day2.md) | One complete run from a spoken request to a handover. |
| [Day 3 · Dynamic layouts and reliability](day3/day3.md) | At least 80% success over the last 20 trials on reshuffled layouts, or a clear plan to reach it by scoping down the item set. |
| [Day 4 · Final day: reliability, rehearsal, pitch](day4/day4.md) | Reliability runs, rehearsal and the pitch. Sun 25 Oct 2026 if the event runs four days (the FAU Innovation Days page lists 22–25 Oct). |

[Storage_Picking_Hackathon_Plan.pdf](Storage_Picking_Hackathon_Plan.pdf) is the source document that `hackathon-plan.md` was written from.

For what changed and when, use the git history — `git log --oneline` or the commit list on GitHub.

## Week 1 harness

`week1_harness/` is a CPU-only, offline runner for the Week 1 practice data loop. It generates synthetic frames, walks them through the eight loop steps, and renders the two Week 1 deliverables — the timing table and the dataset card — with every non-real value labelled as a placeholder or an estimate.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

Expect 104 passed and 1 skipped. The skip is an optional real-inference path that needs `torch` and `ultralytics`; it skips rather than fails on a CPU-only machine.

```
week1_harness/
  steps/        the eight loop steps: film, extract, auto_label, review,
                train, zero_shot, score, live_test
  synth/        synthetic frames: practice items, scenes, Pillow rendering
  reporters/    renders the timing table and the dataset card
tests/          unit tests and Hypothesis property tests
```

Run the whole loop with one command:

```bash
week1-harness --out ./out
```

It runs all eight steps on CPU, prints both deliverables, and writes three files into `--out`:

| File | What it holds |
|---|---|
| `timing_table.md` | One row per step with its time, each tagged `REAL`, `PLACEHOLDER` or `ESTIMATE` |
| `dataset_card.md` | Classes, counts per class, sessions and known gaps |
| `run.json` | The structured results, every value keeping its marker and reason |

Each file is written atomically, so a crash never leaves a half-written deliverable. `--no-zero-shot` forces the zero-shot placeholder; `--seed` makes a run reproducible; a failing step exits non-zero and writes nothing at all.

**Every number here is synthetic.** The times are real CPU measurements of synthetic work, and training, scoring and the live test are stubs — which is why each value carries a marker. These are not Week 1 answers; real numbers come from the real loop on real footage ([prep/week1/week1.md](prep/week1/week1.md) §4).

## Conventions

- Day and week files link back to the plan in their first line.
- Sources are numbered across all files: [1]–[10] in the plan, [11]–[15] in week 1, [16] onward in week 2. Reuse the number when you cite the same source again.
- Numbers marked as targets are team goals, not standards.
- This repository is public. Nothing shared privately by the organizers goes in it, and no model weights or datasets — those are gitignored and copied on site.
