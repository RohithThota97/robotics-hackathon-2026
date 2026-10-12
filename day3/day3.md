# Day 3 · Dynamic layouts and reliability

Robotics & AI Hackathon 2026 · Challenge 3: Dynamic Storage Picking · Sat 24 Oct 2026 · Full plan: [hackathon-plan.md](../hackathon-plan.md)

> **Day 3 checkpoint**
>
> At least 80% success over the last 20 trials on reshuffled layouts, or a clear plan to reach it by scoping down the item set.

> **If the event is only 3 days:** freeze features at lunch today and use this afternoon as the final day. Follow [day4.md](../day4/day4.md) from then on.

## Schedule

| When | Who | What |
|---|---|---|
| Morning | Manipulation + Perception | All catalog items. Reshuffled-layout runs. Fix the top failure causes. |
| | Task brain | Memory-guided search and recovery lines, clarifying questions, enroll-new-item flow. |
| Afternoon | All | 30+ logged trials. Collect v3 data only for observed failures. |
| | Pitch | Measure the human baseline. Record backup video of the best runs. Draft slides. |
| Evening | All | Feature freeze. From here on: only reliability fixes. |

Stand-ups at 09:00, 14:00 and 20:00, 10 minutes each.

## Decision rules

- Grasp success on an item below 60% by Day 3 noon: drop it from the demo set.
- The VLM adds more than 3 seconds to a pick: move it off the critical path.
- Touch-test error above 1 cm: stop and recalibrate before anything else.
- A feature is not working by the Day 3 feature freeze: cut it, do not patch it at night.

## Morning · Reshuffled layouts (Manipulation + Perception)

Run all catalog items on reshuffled layouts and fix the top failure causes.

| Symptom | Likely cause | Fix |
|---|---|---|
| Item missed from one view | That viewpoint is missing from the data | Capture that view and retrain; add a second observation pose. |
| Wrong item picked | Look-alikes, weak identification | More reference crops; ask when confidence is low. |
| Holes in depth on shiny or black items | Depth sensor limits | Fuse frames; change the view angle; grasp the visible part. |
| Gripper misses by about 1 cm | Calibration drift | Touch test; recalibrate; check the tape marks. |
| Item slips | Grasp point or force | Grasp nearer the centre of mass; lift slower; suction for flat items. |
| Planner fails or collides | Planning scene incomplete | Pad shelf boxes; add bin walls. |
| Long delays | Heavy models in the loop | Smaller model size; call the VLM only as a fallback. |

## Morning · Task brain (Rohith)

### Memory-guided search

- Search order: last-seen location first, then other locations ranked by recency, category priors (e.g. consumables usually in drawer A) and travel distance.
- Not found anywhere: run a full scan, say so honestly, and offer the closest alternative.
- Show memory on the dashboard as a live shelf map with ages ("seen 2 min ago"). The jury watches the robot think.

### Recovery lines and clarifying questions

- Ambiguous? Ask one short question: "Two torque wrenches here, 10 or 20 Nm?"
- Unknown item? Say so and offer to enroll it.
- Speak short status updates: "Not in bin B2 any more. Checking B4."

### Enroll-new-item flow

- Embedding gallery: crop, compute an image embedding (DINOv2/v3 or CLIP), match to the nearest reference.
- Enroll a new item with 5–10 photos, no retraining. Ask the user when similarity is low.
- For detection, YOLOE-26 open-vocabulary can find a new item by text prompt.

## Afternoon · 30+ logged trials (all)

- After every change, run at least 5 trials on a freshly reshuffled layout.
- Collect v3 data only for observed failures. Score v3 on the golden set before you deploy it.
- Track the end-to-end targets in the trial log:

| Metric | Target |
|---|---|
| Identification accuracy, including look-alikes | ≥ 95% |
| Grasp success per item | ≥ 85% over 10 tries |
| End-to-end success on reshuffled layouts | ≥ 80% over 20+ trials |
| Time from request to delivery | Median ≤ 90 s (robot-dependent) |

## Afternoon · Pitch

- Human baseline: time a person (ideally not from your team) fetching the same items from the same reshuffled storage, once knowing the layout and once not. This comparison goes into the pitch.
- Record backup video of the best runs.
- Draft slides. The pitch structure is in [day4.md](../day4/day4.md).

## Evening · Feature freeze (all)

From here on: only reliability fixes. Sleep rota: whoever drives the robot during the demo must be rested.

## End of day

- [ ] 30+ logged trials.
- [ ] Day 3 checkpoint: ≥ 80% over the last 20 trials on reshuffled layouts, or a scoped-down item set with a plan to get there.
- [ ] Items below 60% grasp success dropped from the demo set.
- [ ] Human baseline measured and backup video recorded.
- [ ] Feature freeze in place.
