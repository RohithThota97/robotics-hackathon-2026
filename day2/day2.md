# Day 2 · First full voice-to-handover run

Robotics & AI Hackathon 2026 · Challenge 3: Dynamic Storage Picking · Fri 23 Oct 2026 · Full plan: [hackathon-plan.md](../hackathon-plan.md)

> **Day 2 checkpoint**
>
> One complete run from a spoken request to a handover.

## Schedule

| When | Who | What |
|---|---|---|
| Morning | Manipulation | Tier-1 grasps on 3 items from bins; place in tray. |
| | Perception | Score v1 on the golden set; capture S3 (clutter) and S4 (robot views). |
| | Task brain | Voice → parser → skill call. Memory updated from scan results. |
| | Integration | State machine chains perceive → grasp → deliver, with timeouts and retries. |
| Afternoon | Perception | Label and train v2; check 3D accuracy with the touch test. |
| | All | First voice-to-handover run. Start the trial log (10+ trials). |
| Evening | Task brain | Dashboard v1: camera view, memory map, task status. |
| | All | Review the trial log. Choose Day 3 scope (Must and Should only). |

Stand-ups at 09:00, 14:00 and 20:00, 10 minutes each.

## Manipulation · Tier-1 grasps (morning)

Analytic grasp: approach top-down over the object centre. A parallel gripper closes across the short axis; suction targets the flattest area near the centre. Add a pre-grasp offset and a slow approach.

- Works when items lie in open bins or trays.
- Start with 3 items from bins, placed in the tray.
- Target: grasp success per item ≥ 85% over 10 tries, measured in the trial log.

## Perception

### Morning · Score v1 and capture S3–S4

Score v1 on the golden test set: mAP, recall per item, and confusion between look-alikes. Build a failure gallery: look at every miss and every false detection.

| Session | Setup | Images | Purpose |
|---|---|---|---|
| S3 Clutter | Items touching, partly hidden, stacked; non-catalog distractors | 100–150 | Robustness |
| S4 Robot views | From the robot's real scan poses (and while moving, if mobile) | 100–150 | Matches deployment |

Capture targeted data for the top failure causes, not just "more data".

### Afternoon · Train v2 and check 3D accuracy

- Dataset v2 = v1 + S3–S4. Auto-label, review, then retrain from pretrained weights.
- Re-score on the golden set and keep the better model. Never deploy a model you have not scored.
- 3D localization: mask + aligned depth gives an object point cloud. Remove outliers, compute centre and main axis, then convert to robot coordinates with the hand-eye calibration. If that is not accurate enough, fuse several frames or add a second viewpoint.
- Check 3D accuracy with the touch test against a tape measure.

| Metric | Target |
|---|---|
| Detection mAP50 on the golden set | ≥ 0.85 |
| Recall per item | ≥ 0.90 |
| 3D position error | ≤ 1 cm |

## Task brain (Rohith)

### Morning · Voice → parser → skill call

- Local speech-to-text (e.g. Whisper), then a language model turns the sentence into a structured request: item, attributes, quantity, destination.
- Match to the catalog using synonyms and embeddings, in English and German.
- The language model only chooses from fixed skills (find, pick, deliver, scan, cancel) with checked parameters. It never drives motors.
- Update memory from scan results on every observation, even while looking for something else.

### Evening · Dashboard v1

Camera view, memory map, task status.

## Integration · Pick-and-deliver state machine (morning)

Chain perceive → grasp → deliver as a state machine or behaviour tree. Every step gets a timeout and a clear failure message that the dashboard shows and the voice says.

1. Receive the request: item, attributes, quantity, destination.
2. Ask memory for ranked candidate locations.
3. Move to the observation pose for the top location.
4. Capture RGB-D and fuse 3–5 frames.
5. Detect, segment, identify. Not found: try the next location. Nowhere: full scan, update memory, tell the user.
6. Compute the 3D pose and grasp candidates. Choose the best collision-free one.
7. Pre-grasp, approach, grasp, lift.
8. Verify with gripper width or force (or the vacuum sensor), plus a camera check that the item left the bin.
9. If it failed: re-perceive and retry up to 2 times with the next candidate, then ask for help.
10. Move to the delivery pose. Place in a tray, or hand over and release only when the person pulls or confirms.
11. Update memory and log the trial with timings per stage.

## Afternoon · First voice-to-handover run (all)

- Run the full chain from a spoken request to a handover.
- Start the trial log with 10+ trials. One row per attempt: trial ID · time · layout ID · requested item and exact phrasing · success or failure · failure stage · time per stage · total time · notes · video file.
- After every change, run at least 5 trials on a freshly reshuffled layout.
- Report only numbers from logged trials.

## Evening · Review and choose Day 3 scope (all)

Review the trial log, then take Day 3 scope from Must and Should only.

| Priority | Items |
|---|---|
| Must | Request → find → pick → deliver for 5+ items in reshuffled bins. ≥ 80% success over 20+ logged trials. Safety. Trial log. |
| Should | Memory-guided search. Clarifying questions. Dashboard. Enroll a new item. |
| Could | German voice. VLM verification. Driving between shelves. Learned grasping. Multi-item orders. |
| Won't (this event) | End-to-end VLA policy. Transparent items. Deformable items (unless required). |

## End of day

- [ ] Day 2 checkpoint: one complete run from a spoken request to a handover.
- [ ] 10+ trials in the trial log.
- [ ] v2 trained and scored on the golden set.
- [ ] Day 3 scope agreed.
