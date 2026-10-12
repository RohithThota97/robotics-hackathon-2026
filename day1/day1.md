# Day 1 · Understand and build a walking skeleton

Robotics & AI Hackathon 2026 · Challenge 3: Dynamic Storage Picking · Thu 22 Oct 2026 · Full plan: [hackathon-plan.md](../hackathon-plan.md)

> **Day 1 checkpoint**
>
> The camera detects an item and the robot moves its gripper to touch it. That proves the whole chain from pixels to robot coordinates.

## Schedule

| Time | Who | What |
|---|---|---|
| 09:00–11:00 | All | Kick-off and briefing. Inspect robot, storage and items; photograph everything. Ask the open Section 1 questions and get the judging rubric. |
| 11:00–12:00 | All | Freeze architecture, interfaces and owners. Set up the trial log and dataset card. |
| 12:00–15:00 | Perception | Zero-shot baseline; capture S1–S2 from robot views. |
| | Manipulation | Bring-up, e-stop test, hand-eye calibration, touch test. |
| | Task brain | Catalog and synonyms, memory schema, parser working on typed text. |
| | Integration | Skeleton connecting all modules with dummy data. |
| 15:00–20:00 | Perception | Auto-label and review v1; start training. |
| | Manipulation | Pick one item from a hard-coded position and place it in the tray. |
| Night | Perception | v1 training runs; score it first thing in the morning. |

Stand-ups at 09:00, 14:00 and 20:00, 10 minutes each.

## Deliverable per role

| Role | Day 1 deliverable |
|---|---|
| Perception lead | Zero-shot baseline and first data capture from robot views |
| Manipulation lead | Calibrated robot that touches a detected point within a few mm |
| Integration lead | Interface contract and a skeleton that runs end to end with dummy data |
| Task brain + UX (Rohith) | Item catalog with synonyms, memory schema, parser working on typed text |
| Pitch + product (shared) | Storyboard and trial log template |

## 09:00–11:00 · Kick-off and briefing (all)

Inspect the robot, storage and items, and photograph everything. Get the judging rubric. Ask whichever of these are still open:

- [ ] Which robot exactly? Arm model, gripper or hand, mobile base, humanoid?
- [ ] Which cameras, mounted where? Is there depth (RGB-D)?
- [ ] Which items, how many?
- [ ] What does "dynamic" mean here? Items moved between bins, bins moved, new items added?
- [ ] How is it judged? Rubric, weights, demo format, minutes per team?
- [ ] Is robot time shared? Are there booking slots?
- [ ] On-site GPU? Internet? Any API restrictions?
- [ ] Is pre-written code allowed? Which open-source models and APIs are allowed?
- [ ] Prizes per challenge or overall? Can Siemens Healthineers employees win prizes?
- [ ] Simulation assets? Robot URDF/USD, storage dimensions or CAD?

Once you know the hardware, pick the matching branch in plan §7.3 (Hardware branches) and adjust the targets in plan §6.4.

## 11:00–12:00 · Freeze architecture, interfaces and owners (all)

- Pipeline: request → task brain → spatial memory → go to location → perceive → plan grasp → execute + verify → deliver. Steps 1 to 4 decide, steps 5 to 8 act, and every run updates memory and the trial log.
- One owner per interface.
- The language model never drives motors. It only chooses from fixed skills (find, pick, deliver, scan, cancel) with checked parameters. Speed and workspace limits sit below it.
- Tag the furniture, not the items. AprilTags go on shelves and bins; items stay untagged.

Set up the trial log, one row per attempt: trial ID · time · layout ID · requested item and exact phrasing · success or failure · failure stage (understanding, search, detection, identification, localization, grasp, transport, handover) · time per stage · total time · notes · video file.

Set up the one-page dataset card: classes, counts per class, sessions, known gaps.

## Perception

### 12:00–15:00 · Zero-shot baseline and first capture

- Zero-shot baseline: run open-vocabulary detection with just the item names, using YOLOE-26 text prompts and SAM 3 concept prompts. It shows which items already work and which need training.
- Capture S1 and S2 with the exact camera, resolution and viewpoints the robot will use:

| Session | Setup | Images | Purpose |
|---|---|---|---|
| S1 Clean | Each item alone in each bin type; 3 lighting conditions | 80–120 | Base appearance |
| S2 Shuffled | 5+ random layouts with all items present | 120–200 | Layout variety |

- Record short videos and keep 2–3 frames per second to avoid near-duplicates.
- Lock camera exposure and resolution. Capture and deployment settings must match.
- Set aside 1–2 layouts as the golden test set. They are never used for training.

### 15:00–20:00 · Auto-label, review and train v1

- Auto-label: SAM 3 text or box prompts produce masks. A person fixes them in CVAT, Label Studio or Roboflow.
- Labelling rules:
  - Label an item when roughly a third or more of it is visible. Box and mask the whole object, including handles.
  - One class per catalog item. Look-alikes get separate classes.
  - Treat non-catalog objects consistently: either always unlabelled or always one "other" class.
  - A second person reviews 10% of labels. Fix systematic errors before training.
- Dataset v1 = S1–S2. Plan for at least ~150 labelled instances per item (a planning target, not a rule).
- Split by session or layout, never by random frames. Neighbouring frames leak and inflate scores.
- Fine-tune a YOLO26 segmentation model (n, s or m size) from pretrained weights. Use image size 640–1024; larger helps small items.

### Night · v1 training

v1 trains overnight. Score it on the golden test set first thing in the morning.

## Manipulation

### 12:00–15:00 · Bring-up and calibration

1. Robot driver running. Move to named poses, open and close the gripper, test the e-stop.
2. Camera: check depth-to-colour alignment and resolution; lock exposure.
3. Hand-eye calibration with the ChArUco board, 15–25 varied poses.
4. Touch test: command the gripper tip to a detected point and measure the error. Aim for under ~5 mm.
5. Tape-mark the camera and robot mounts. Repeat the touch test before every demo.
6. Measure shelves and bins, put AprilTags on each, and add them as collision boxes in the motion planner.

### 15:00–20:00 · First pick

Pick one item from a hard-coded position and place it in the tray.

### Safety from the first motion

- An e-stop within an operator's reach at all times. Reduced speeds. Workspace limits set in the controller.
- Nobody inside the arm's reach during motion.

## Task brain (Rohith)

### Item catalog (first hour)

- Use the organizers' items. Typically 6–10. Mix easy shapes (boxes, bottles) and hard ones (thin handles).
- Include 1–2 look-alike pairs if they exist, such as two wrench sizes. They showcase fine-grained identification and the clarifying question.
- Per item, record: name, synonyms (EN/DE), size, colour, weight, grasp type (pinch, suction, top-down) and possible storage locations.
- Healthcare angle for the story: service tools (torque wrench, multimeter, cables) and clinical consumables (glove box, gel bottle, syringe pack).

### Memory schema

Per item, store: location (shelf, bin), 3D position, time last seen, confidence, and how it was seen (scan, pick, delivery).

### Parser on typed text

Turn a typed sentence into a structured request: item, attributes, quantity, destination. Match it to the catalog using synonyms and embeddings, in English and German.

## Integration

- Interface contract between the modules.
- A skeleton that runs end to end with dummy data. Connect all eight steps crudely today, then improve the weakest link.

## Pitch + product

Storyboard and trial log template.

## End of day

- [ ] Day 1 checkpoint: the camera detects an item and the robot touches it.
- [ ] Touch test done. Error above 1 cm: stop and recalibrate before anything else.
- [ ] Targets adjusted to the real robot (plan §6.4).
- [ ] v1 training running overnight.
