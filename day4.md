# Day 4 · Final day: reliability, rehearsal, pitch

Robotics & AI Hackathon 2026 · Challenge 3: Dynamic Storage Picking · Full plan: [hackathon-plan.md](../hackathon-plan.md)

Date: Sun 25 Oct 2026 if the event runs four days (the FAU Innovation Days page lists 22–25 Oct). The event page lists 24 Oct 19:00 as the end; if that holds, run this sheet on the afternoon of Day 3.

## Schedule

| When | Who | What |
|---|---|---|
| Morning | All | Reliability runs only. Tune speeds. Compute the final metrics from the trial log. |
| | Pitch | Put the real numbers into the slides; cut the backup video. |
| Midday | All | Rehearse the pitch 3 times with a timer. Rehearse the live reshuffle with an outsider playing the jury. |
| Before pitch | Safety operator | Run the pre-demo checklist (below). |

## Morning · Final metrics

Report only numbers from logged trials. Use the golden layout for the final numbers.

| Metric | Target | How you measure it |
|---|---|---|
| Detection mAP50 on the golden set | ≥ 0.85 | Model evaluation |
| Recall per item | ≥ 0.90 | Model evaluation |
| Identification accuracy, including look-alikes | ≥ 95% | Gallery test on held-out crops |
| 3D position error | ≤ 1 cm | Touch test against a tape measure |
| Grasp success per item | ≥ 85% over 10 tries | Trial log |
| End-to-end success on reshuffled layouts | ≥ 80% over 20+ trials | Trial log |
| Time from request to delivery | Median ≤ 90 s (robot-dependent) | Trial log |

## Live demo script (about 4 minutes)

1. 20 seconds on the problem, with one number.
2. A jury member reshuffles the storage.
3. Spoken request in German, then one in English.
4. The robot checks memory, finds the item is not where it was last seen, says so, searches the next likely spot, then finds, grasps and delivers it.
5. An ambiguous request. The robot asks a clarifying question.
6. Enroll a new item with a few photos. The robot fetches it.
7. The dashboard: memory map and trial statistics.

## Pitch structure (5–7 minutes)

Likely judging dimensions (confirm the real rubric): working prototype, innovation, value for Siemens Healthineers and its customers, technical depth, presentation.

1. Problem and cost: the external number plus your measured human baseline.
2. Solution in one sentence.
3. Live demo.
4. Measured results: success rate, median time, number of trials, comparison with the human baseline.
5. Why it scales: modular, new items without retraining, works with any robot that has a camera and a gripper.
6. Where Siemens Healthineers could use it: service tool storage, manufacturing, hospital supply rooms.
7. Next steps: a concrete pilot proposal. Then the team.

External numbers for the problem slide:

- In a survey by software vendor Infor, 32% of nurses said they spend more than 20 minutes per shift searching for supplies. <https://www.infor.com/blog/optimizing-supply-availability-for-nurses>
- In an Irish hospital study, reorganizing a treatment room cut the time to gather IV cannulation equipment from 118 to 29 seconds. <https://map.amegroups.org/article/view/4155/html>

## Backups

- A recorded video of the best full run, ready to play.
- Metric slides saved as images.
- A laptop demo of perception, memory and voice if the robot fails.

## Before every demo run

- [ ] E-stop test and touch test.
- [ ] Battery or power. Lighting and camera exposure.
- [ ] Memory reset or seeded. Dashboard, logging and video recording on.
- [ ] Safety operator in position.

## Safety during the demo

- An e-stop within an operator's reach at all times. Reduced speeds. Workspace limits set in the controller.
- Nobody inside the arm's reach during motion. Hand over via a tray, or slowly with confirmed release.
- One team member is the dedicated safety operator in every demo.
- Whoever drives the robot during the demo must be rested.
