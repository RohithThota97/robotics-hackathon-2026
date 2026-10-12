# Contributing

The workflow in one line: **branch → commit → push → pull request → Rohith reviews and merges into `main`.**

Nobody pushes to `main` directly, including Rohith. Everything arrives as a pull request so there is one place to see what changed and why.

## One-time setup

Clone the repo:

```bash
git clone git@github.com:RohithThota97/robotics-hackathon-2026.git
cd robotics-hackathon-2026
```

If you have not set up an SSH key for GitHub, use HTTPS instead:

```bash
git clone https://github.com/RohithThota97/robotics-hackathon-2026.git
cd robotics-hackathon-2026
```

Tell git who you are, using the email that is verified on your GitHub account
(otherwise your commits will not link to your profile):

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

Set up the Python environment. Python 3.10 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Check it works. You should see 104 passed and 1 skipped:

```bash
pytest
```

The one skip is expected: an optional real-inference path that needs torch and
ultralytics, which are not installed by default.

## Making a change

Start from an up-to-date `main` every time:

```bash
git switch main
git pull
```

Create your branch:

```bash
git switch -c perception/zero-shot-baseline
```

Name it `<workstream>/<short-description>`, using the workstream you own from
section 2 of [hackathon-plan.md](hackathon-plan.md):

| Workstream | Prefix | Example |
|---|---|---|
| Data | `data/` | `data/session-s1-capture` |
| Perception models | `perception/` | `perception/yoloe-baseline` |
| Grasp + motion | `manip/` | `manip/hand-eye-calibration` |
| Task brain | `brain/` | `brain/request-parser` |
| Integration + testing | `integration/` | `integration/state-machine` |
| Pitch | `pitch/` | `pitch/demo-script` |

Make your edits, then run the tests before you commit:

```bash
pytest
```

Commit. Write a subject line that says what changed, not "update":

```bash
git add -A
git commit -m "Add synonym table to the item catalog"
```

Push your branch:

```bash
git push -u origin perception/zero-shot-baseline
```

The `-u` is only needed the first time you push a given branch. After that,
`git push` is enough.

## Opening the pull request

With the [GitHub CLI](https://cli.github.com):

```bash
gh pr create --base main --title "Add synonym table to the item catalog" --body "What this changes and why. Any numbers you measured. Anything you are unsure about."
```

Or just open the link that `git push` prints in your terminal and click
**Compare & pull request**.

Then request a review:

```bash
gh pr edit --add-reviewer RohithThota97
```

In the description, say what you measured if you measured anything, and flag
what you are unsure about. A PR that says "I am not sure the threshold is
right" gets a more useful review than one that says "done".

## While your PR is open

If Rohith asks for changes, commit on the same branch and push again. The PR
updates itself; do not open a new one:

```bash
git add -A
git commit -m "Lower the confidence threshold to 0.4"
git push
```

If `main` has moved on and GitHub reports a conflict:

```bash
git switch main
git pull
git switch perception/zero-shot-baseline
git merge main
# fix any conflicted files, then:
git add -A
git commit
git push
```

## After it is merged

Clean up your local branches:

```bash
git switch main
git pull
git branch -d perception/zero-shot-baseline
```

## Useful commands

```bash
git status                  # what have I changed
git diff                    # the actual changes, unstaged
git diff --cached           # the changes you have staged
git log --oneline -10       # recent history
git branch -a               # all branches, local and remote
git switch <branch>         # move to an existing branch
git restore <file>          # throw away your changes to one file
gh pr list                  # open PRs
gh pr status                # your PRs and ones awaiting your review
gh pr checkout 7            # check out PR #7 locally to try it
gh pr diff 7                # read PR #7's changes in the terminal
```

## House rules

- **Never commit model weights or datasets.** `.pt`, `.pth`, `.onnx`,
  `.safetensors` and friends are already in [.gitignore](.gitignore). The repo
  stays small; weights are downloaded or copied on site.
- **Never commit `.env` files, API keys or Hugging Face tokens.** If you push a
  secret by accident, tell Rohith immediately and rotate it — rewriting history
  is not enough on a public repo.
- **This repository is public.** Anyone can read it. Do not add anything from
  the organizers that was shared privately, and do not add teammates' personal
  details.
- **Keep one PR to one concern.** Two unrelated fixes are two PRs; they are far
  faster to review.
- **Run `pytest` before you push.** It takes about 25 seconds.

## Current state of the code

Read this before you report the harness as broken:

`week1_harness/cli.py` has a working argument parser, but `main()` is still a
stub — see the `TODO(task 14.2)` at
[week1_harness/cli.py:90](week1_harness/cli.py:90). Running `week1-harness`
parses your arguments and exits 0 **without producing the timing table or the
dataset card.** The pipeline wiring and the `output.py` module it calls for have
not been written yet. The eight step components, the synthetic frame
generation and both reporters are implemented and tested; only the top-level
wiring is missing.

## Layout

```
hackathon-plan.md     the full plan: dataset, models, robot, demo, pitch
prep/week1/           9-15 Oct: environments, practice loop, templates
prep/week2/           16-21 Oct: rehearsal, task brain, pitch, packing
day1/ .. day4/        on-site daily checkpoints
week1_harness/        the Week 1 data-loop harness (CPU-only, offline)
  steps/              the eight loop steps, film through live_test
  synth/              synthetic frames: items, scenes, Pillow rendering
  reporters/          renders the timing table and dataset card
tests/                unit tests and Hypothesis property tests
```
