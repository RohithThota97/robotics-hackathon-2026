# Week 2 · Rehearse, task brain, pitch, pack

Robotics & AI Hackathon 2026 · Challenge 3: Dynamic Storage Picking · Prep week 2: Fri 16 – Wed 21 Oct 2026 · Full plan: [hackathon-plan.md](../../hackathon-plan.md) · Week 1: [week1.md](../week1/week1.md)

> **Week 2 checkpoint**
>
> You can run hand-eye calibration and a collision-aware pick sequence from your notes. A spoken request in German or English turns into a checked request, with a clarifying question when needed. The storyboard, demo script and trial log template are drafted, and the bag is packed.

## Checklist

- [ ] Rehearse hand-eye calibration and MoveIt 2 planning with collision objects. Add Nav2 waypoints if a mobile base is likely (section 1).
- [ ] If the robot model is known: build a simple simulation of shelf, bins and robot (Isaac Sim, Gazebo or MuJoCo) and rehearse the pick sequence (section 2).
- [ ] Task brain: speech-to-text, request parsing, clarification rules, dashboard layout. Reuse the voice-agent patterns you already know (section 3).
- [ ] Pitch: storyboard, problem slide with sources, demo script draft, trial log template (section 4).
- [ ] Print and pack: a rigid matte ChArUco calibration board, AprilTag sheets in 2–3 sizes, plus the Section 13 packing list (section 5).

Pre-written code (week 1 §2): if it is not allowed, everything here is rehearsal and the code stays at home. Voice lines, clarification rules, test phrases, storyboard, demo script and the trial log template are notes, so they come along either way.

Anything you install this week goes through the week 1 offline test again (week 1 §3).

## Start here: organizer answers

Week 2 branches on the plan §1 answers. Fill in what has arrived.

| Question (plan §1) | Answer | Changes this week |
|---|---|---|
| 1 Robot: arm, gripper, mobile base? | | ROS 2 version, MoveIt config, simulation, Nav2 yes or no |
| 2 Cameras, mount, depth? | | Eye-in-hand or eye-to-hand calibration |
| 3 Items | | Catalog entries; practice with similar items |
| 5 Judging, minutes per team | | Storyboard and demo time budget |
| 8 Pre-written code | | What you bring (week 1 §2) |
| 10 Simulation assets | | Whether section 2 is possible |

## 1. Hand-eye calibration, MoveIt 2, Nav2 · Manipulation lead

### Hand-eye calibration

The camera mount (question 2) decides the type:

| Type | Camera | Board | Result |
|---|---|---|---|
| Eye-in-hand | On the wrist | Fixed on the table | Camera pose relative to the gripper |
| Eye-to-hand (eye-on-base) | Fixed, looking at the robot | Held by the gripper | Camera pose relative to the robot base |

- Rehearse without hardware: easy_handeye2 is a ROS 2 tool that wraps OpenCV's hand-eye methods (e.g. Tsai-Lenz), and its demo package runs in a simulator [16].
- It reads the board pose from tf, so a detector node must publish it. Decide which ChArUco detector you will run in ROS 2; an AprilTag through apriltag_ros is the fallback target.
- Its automatic-move GUI is work in progress, so plan to move the robot yourself and accept samples by hand [16].
- Run the whole procedure once: the four frame names (robot base, end effector, camera, marker), sampling, computing, saving, publishing at startup, and the evaluation.
- Pose tips: rotate the end effector as much as possible (up to 90°) about each axis in both directions, keep the board close to the camera, keep translations small, and take more poses than the minimum [16]. The plan uses 15–25 varied poses.
- Depth cameras usually ship with factory intrinsics; a webcam does not. Calibrating the webcam with the new board is a good check that the board works (section 5).
- Write the procedure as a numbered checklist with the frame names, and time it. On Day 1 it shares 12:00–15:00 with bring-up and the touch test.
- Touch test (plan §7.1): command the gripper tip to a detected point and measure the error. Under ~5 mm is the aim; above 1 cm, recalibrate. In simulation the error is zero, so rehearse the steps, not the number.

### MoveIt 2 with collision objects

Use the MoveIt 2 tutorials' demo robot on mock hardware; the tutorials also run from a Docker setup. "Planning Around Objects" adds a box to the planning scene and plans around it, and its next tutorial covers pick and place with MoveIt Task Constructor [17].

Rehearse the pick motion from plan §7.2:

1. Shelf and bin walls as collision boxes with measured sizes, padded a little.
2. Named poses: home, one observation pose per bin, delivery pose.
3. Pre-grasp above the item, straight-line approach, grasp, lift.
4. Attach the grasped item to the gripper in the planning scene, so the planner avoids collisions with what it holds.
5. Reduced velocity and acceleration scaling, as on site.
6. Time each plan and note what makes planning fail (plan §9.3: pad shelf boxes, add bin walls).

### Nav2 waypoints · Integration lead, only if a mobile base is likely

- Rehearse in Nav2's simulation: one waypoint in front of each shelf and a short, fixed route between them (plan §7.3).
- The Simple Commander API offers goToPose, goThroughPoses and followWaypoints. They are non-blocking, so you poll until the task completes, and you must cancel the current task before switching between them [18].
- After arriving, re-localize on the shelf's AprilTag before the arm moves.

## 2. Simulation, if the robot model is known · Manipulation + Integration

Only worth it with the real robot's model (question 10: URDF, USD or CAD). If the model is still unknown on Sun 18 Oct, skip this and rehearse the sequence on the MoveIt 2 demo robot from section 1.

| Simulator | Good for | Watch out |
|---|---|---|
| Gazebo | The same ROS 2 interfaces as the real robot, through ros2_control | Use the release paired with your ROS 2 version: Humble with Fortress, Jazzy with Harmonic, Lyrical with Jetty [19] |
| MuJoCo | Light, fast physics on a laptop. Ready robot models in MuJoCo Menagerie (Franka, UR, KUKA iiwa, Kinova and more) [21]. mujoco_ros2_control connects it to ros2_control and has RGB-D camera plugins [20]. | Vendor URDFs need converting to MuJoCo's format; mujoco_ros2_control has tools for that [20] |
| Isaac Sim | Photoreal cameras and synthetic data (plan §5.2) | Minimum is a GeForce RTX 4080 with 16 GB VRAM and 32 GB RAM; many laptop GPUs have less. It loads assets online by default [22]. |

- Model the robot from the vendor's files, the shelf and bins as boxes with real dimensions, and the items as simple boxes and cylinders.
- Add AprilTags on shelves and bins if the simulator renders camera images.
- Rehearse the pick sequence from plan §7.2 the way the state machine will run it: observation pose → perceive → pre-grasp → approach → grasp → lift → verify → deliver → memory update. Every step gets a timeout and a failure message.

## 3. Task brain (Rohith)

### Speech-to-text

- Push-to-talk with a close-talking USB mic. With about 200 hackers the hall can be loud, and push-to-talk also keeps the robot from hearing its own voice.
- Multilingual Whisper (week 1 §3). Language detection is less reliable on short commands, so limit it to German and English, or set the language in the UI.
- Put the catalog names into Whisper's initial prompt. It is meant for custom vocabulary and makes those words more likely [15].
- Test with people outside the team and with background noise. Note the time from end of speech to text.

### Text-to-speech

- The robot speaks its status (plan §7.5). Use a local engine with German and English voices, e.g. Piper, which has several German voices including Thorsten [23]. Check each voice's licence.
- The voices are in the week 1 weights list and offline test (week 1 §3).

### Request parsing

- Output: the request format from week 1 §5.1.
- Fast path first: an exact name or synonym match against the catalog needs no language model. Only harder phrasing goes to the model.
- Constrain the model's output to the request format (structured output), then check every field: item in the catalog, skill in the fixed list, quantity a positive number, destination tray or hand-over. Anything invalid gets a question or a refusal, never motion.
- Give the model the catalog (IDs, names and synonyms in both languages), the skill list, the request format and a few examples.

Test sheet: about 40 phrasings, half German, half English. The examples assume the screwdriver pair and a pair of pliers in the practice set.

| Type | Example | Expected |
|---|---|---|
| Clear | "Bring mir den kleinen Schraubendreher" | screwdriver_small, clear |
| Ambiguous | "Bring me the screwdriver" | ambiguous, asks small or large |
| Answer to the question | "The big one, please" / "Den großen, bitte" | screwdriver_large |
| Destination | "Leg die Zange in die Ablage" | pliers, tray |
| Unknown item | "Bring mir den Tacker" | unknown, offers to enroll |
| Cancel | "Stopp, abbrechen" | cancel |
| Off-topic | "Wie spät ist es?" | polite refusal, no motion |

Score each phrasing on item, status and latency. Every clear request must be right; ambiguous ones must ask, never guess.

### Clarification rules

| Trigger | Robot says (EN) | Robot says (DE) | Then |
|---|---|---|---|
| Two or more catalog matches | "Two screwdrivers here, small or large?" | "Hier gibt es zwei Schraubendreher: klein oder groß?" | Accept short answers ("small", "the small one", "den kleinen") |
| Low identification similarity at the bin | "Is this the small screwdriver?" | "Ist das der kleine Schraubendreher?" | Show the crop on the dashboard; yes or no |
| Unknown item | "I don't know 'stapler' yet. Do you want to show it to me?" | "Den Begriff „Tacker“ kenne ich noch nicht. Möchten Sie mir den Gegenstand zeigen?" | Yes: enroll flow (plan §6.1 B) |
| Not found after a full scan | "I can't find the small screwdriver. The large one is in B4. Bring that instead?" | "Ich finde den kleinen Schraubendreher nirgends. Der große liegt in B4. Soll ich den bringen?" | Yes or no |
| Outside the skills | "I can only find, pick and deliver catalog items." | "Ich kann nur Gegenstände aus dem Katalog suchen, greifen und bringen." | No motion |
| No answer (e.g. 10 s) | "No answer, so I'll stop here." | "Keine Antwort, ich breche ab." | Ask once more first, then cancel |

- One question per request, always name the options, never guess between look-alikes.
- Use "Sie" in German. It suits a hospital or service setting, and a jury.
- German lines change with the item's gender and case ("der kleine", "den kleinen"). The catalog holds both phrases per item (week 1 §5.2); use them, or phrase lines so the name stands alone.

### Status lines

| Moment | EN | DE |
|---|---|---|
| Search | "Looking in B2." | "Ich schaue in B2 nach." |
| Moved | "Not in B2 any more. Checking B4." | "Nicht mehr in B2. Ich schaue in B4 nach." |
| Found | "Found it." | "Gefunden." |
| Retry | "Missed. Trying again." | "Daneben. Ich versuche es noch einmal." |
| Needs help | "That didn't work. Can you help me?" | "Das hat nicht geklappt. Können Sie mir helfen?" |
| Delivering | "On the way to the tray." | "Unterwegs zur Ablage." |
| Hand-over | "Please take it." | "Bitte abnehmen." |
| Done | "Done." | "Erledigt." |

### Voice-agent patterns to reuse

- Half-duplex: the mic stays closed while the robot speaks.
- Short turns: one sentence per status, one question per request.
- "Cancel" works at any time, but it is not a safety function. The e-stop is (plan §7.6).
- Captions: the dashboard shows what the robot heard and what it said.

### Dashboard layout

One screen, no scrolling, readable on a projector from the back of the room.

| Position | Panel | Shows |
|---|---|---|
| Top left, largest | Live camera | Detections, with the target item highlighted |
| Top right | Shelf map | Every place with its items and their age ("seen 2 min ago"); the place being checked now |
| Bottom left | Task status | Current step of the pick sequence, the request as understood, captions of what it heard and said |
| Bottom right | Numbers | Stage timings for this run; success rate, trial count and median time from the trial log |

- Big text and high contrast. Never use colour alone: pair it with a word or icon, e.g. "stale" next to an old age.
- Captions also help the jury follow when the hall is loud.
- Local assets only. Fonts or scripts loaded from a CDN break the page offline.
- Day 2 evening builds v1 (camera, memory map, task status). The numbers panel can follow.

## 4. Pitch · Pitch lead

### Storyboard

Plan §10.2 as a time budget. Fit it to the real slot once question 5 is answered.

| # | Beat | Time | On screen | Who |
|---|---|---|---|---|
| 1 | Problem and cost | 0:30 | Problem slide | Pitch lead |
| 2 | Solution in one sentence | 0:15 | One line and the 8-step pipeline | Pitch lead |
| 3 | Live demo | 4:00 | Robot and dashboard | Speaker, operator, safety operator |
| 4 | Measured results | 0:45 | Success rate, median time, number of trials, human baseline | Pitch lead |
| 5 | Why it scales | 0:20 | Modular blocks; new items without retraining | |
| 6 | Where Siemens Healthineers could use it | 0:20 | Service tool storage, manufacturing, hospital supply rooms | |
| 7 | Next steps and team | 0:20 | Pilot proposal, team | |
| | **Total** | **6:30** | | |

### Problem slide

One number on the slide, the source in small print, the rest in the speaker notes.

- In a nationwide survey by software vendor Infor, 32% of nurses said they spend more than 20 minutes per shift searching for supplies [8]. Say that it is a vendor survey.
- In an Irish study across 19 wards, a standardized treatment-room layout cut the mean time to gather IV cannulation equipment from 118 to 29 seconds. Asking a nurse or care staff for help finding equipment dropped from 30% to 0% [10]. It is a conference abstract.
- Bridge to your solution: a tidy, fixed layout saves time, but storage keeps changing. The robot keeps track so people don't have to.
- Your measured human baseline joins on Day 3 (plan §9.4).

### Demo script draft

Plan §10.1, about 4 minutes; the problem beat is already in the storyboard. Each full fetch costs your median request-to-delivery time, so this draft merges beats to show something new in every fetch.

| # | Beat | Time | What happens | If it fails |
|---|---|---|---|---|
| 1 | Reshuffle | 0:20 | A jury member reshuffles the storage. The shelf map still shows the old positions. | Keep it short |
| 2 | German request, memory is wrong | 1:20 | "Bring mir …". The item is not where it was last seen. The robot says so, checks the next likely place, picks and delivers. | Retry line, then backup video of this beat |
| 3 | English request, ambiguous | 1:05 | "Bring me the screwdriver." The robot asks small or large, then fetches. | Backup video of this beat |
| 4 | Enroll a new item | 1:00 | A few photos of a new item, then the robot fetches it. | Show enrolment only, skip the fetch |
| 5 | Dashboard | 0:15 | Memory map and trial statistics | Metric slides as images |

- Times assume a median fetch of about 60 s. At 90 s, cut beat 4 to enrolment only.
- Three roles: a speaker, an operator at the laptop, and a safety operator who only watches the robot with the e-stop in reach (plan §7.6).

### Trial log template

One row per attempt (plan §9.1). The software should write it; the same columns work on paper.

| Field | Meaning | Example |
|---|---|---|
| trial ID | Running number | T017 |
| time | Start of the request | 14:32 |
| layout ID | Which reshuffle | L05 |
| item | Requested catalog ID | screwdriver_small |
| phrasing | Exact words, from the request's text field (week 1 §5.1) | "Bring mir den kleinen Schraubendreher" |
| result | success or failure | failure |
| failure stage | understanding, search, detection, identification, localization, grasp, transport, handover | grasp |
| time per stage | understand / search / perceive / grasp / deliver, in seconds | 3 / 21 / 6 / 34 / – |
| total time | Request to delivery or to failure, in seconds | 64 |
| notes | What happened | slipped on lift |
| video file | Recording of the attempt | T017.mp4 |

From the log, compute the success rate over the last 20 trials, the median total time of successful trials, failures per stage, and grasp success per item (plan §6.4).

Add a second sheet for the human baseline (plan §9.4, measured on Day 3): person, knows the layout (yes or no), item, time.

## 5. Print and pack · All

### ChArUco board

- Rigid, flat and matte: matte paper glued to a flat rigid board, or a ready-made rigid target.
- Choose an odd number of rows. OpenCV 4.6 changed the layout of boards with an even row count; odd-row boards work with every OpenCV version [24].
- Print at 100% scale, measure a square with calipers, and use the measured size rather than the nominal one.
- Write the parameters on the back: squares across and down, square size, marker size, ArUco dictionary.
- Size it for the working distance, so it fills a good part of the image where the camera will see it.

### AprilTags

- Family 36h11, the default of the ROS 2 apriltag node [25].
- Print 2–3 sizes for different viewing distances, plus spares. Keep the white margin around each tag.
- Tag size is the edge where the black and white borders meet, not the outside of the printed tag [26]. Measure it after printing and enter it per tag ID [25].
- One unique ID per shelf and bin. The tag map (ID, place, size) feeds the places table in the memory schema (week 1 §5.3).
- Matte paper on stiff card, so the tags stay flat.

### Packing list

From plan §13:

- [ ] Laptops and chargers. GPU environment tested offline. Docker images. All model weights downloaded.
- [ ] Printed ChArUco board and AprilTags. Tape measure, painter's tape, velcro.
- [ ] Extension cord, USB hub, display and USB-C adapters, Ethernet cable. Phone tripod for video.
- [ ] This plan printed or on a phone.

Added by weeks 1 and 2:

- [ ] USB SSD with the saved Docker images and model weights (week 1 §3).
- [ ] Close-talking USB microphone and a small speaker (section 3).
- [ ] Calipers for the board and tags.
- [ ] Spare AprilTags.
- [ ] USB-C to Ethernet adapter, if a laptop has no Ethernet port.
- [ ] Printed notes: catalog, voice lines, clarification rules, demo script, trial log template.

## End of week

- [ ] Hand-eye calibration procedure written and timed; MoveIt 2 plans around shelf and bin boxes.
- [ ] Simulation running with the real robot model, or skipped on purpose.
- [ ] Spoken requests in both languages pass the test sheet; clarification rules and voice lines written.
- [ ] Dashboard layout agreed.
- [ ] Storyboard, problem slide, demo script draft and trial log template ready.
- [ ] Board and tags printed and measured; bag packed.
- [ ] Final offline test on every laptop after this week's installs (Wed 21 Oct).

Next: [Day 1](../../day1/day1.md), Thu 22 Oct.

## Sources

[1]–[10] match the plan, [11]–[15] match week 1. New sources start at [16].

- [8] Infor: Optimizing supply availability for nurses. <https://www.infor.com/blog/optimizing-supply-availability-for-nurses>
- [10] Boyle S. et al. Ward efficiency via a modified treatment room layout. Mesentery Peritoneum 2018;2:AB130. <https://map.amegroups.org/article/view/4155/html>
- [15] OpenAI Whisper (GitHub); initial_prompt is documented in whisper/transcribe.py. <https://github.com/openai/whisper>
- [16] easy_handeye2 (GitHub) and its simulator demo. <https://github.com/marcoesposito1988/easy_handeye2> · <https://github.com/marcoesposito1988/easy_handeye2_demo>
- [17] MoveIt 2 docs: Planning Around Objects. <https://moveit.picknik.ai/main/doc/tutorials/planning_around_objects/planning_around_objects.html>
- [18] Nav2 docs: Simple Commander API (Jazzy). <https://docs.nav2.org/jazzy/configuration_and_development/simple_commander_api/simple_commander_api/>
- [19] Gazebo docs: ROS 2 and Gazebo pairings. <https://gazebosim.org/docs/latest/ros_installation/>
- [20] ros-controls: mujoco_ros2_control (GitHub). <https://github.com/ros-controls/mujoco_ros2_control>
- [21] Google DeepMind: MuJoCo Menagerie (GitHub). <https://github.com/google-deepmind/mujoco_menagerie>
- [22] NVIDIA Isaac Sim: requirements. <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>
- [23] Piper TTS: engine and voice list. <https://github.com/OHF-Voice/piper1-gpl> · <https://github.com/rhasspy/piper/blob/master/VOICES.md>
- [24] calib.io: ChArUco targets (OpenCV 4.6 note). <https://calib.io/products/charuco-targets>
- [25] apriltag_ros, ROS 2 node (GitHub). <https://github.com/christianrauch/apriltag_ros>
- [26] AprilTag library (GitHub): tag size definition. <https://github.com/AprilRobotics/apriltag>
