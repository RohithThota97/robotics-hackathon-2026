# Week 1 · Environments, practice loop, templates

Robotics & AI Hackathon 2026 · Challenge 3: Dynamic Storage Picking · Prep week 1: Fri 9 – Thu 15 Oct 2026 · Full plan: [hackathon-plan.md](../../hackathon-plan.md)

> **Week 1 checkpoint**
>
> Every laptop runs the full data loop with the network off. You know your real turnaround from filming to a model running live. The request format, catalog template and memory schema are written and tried on the practice items.

## Checklist

- [ ] Request access to gated model weights now. SAM 3 downloads need approval on Hugging Face (section 1).
- [ ] Read the rules on pre-written code. If it is not allowed, prepare environments, notes and practice only (section 2).
- [ ] Set up one or two Linux laptops with an NVIDIA GPU: ROS 2 (match the robot vendor's version), a Python ML environment, and Docker images built and tested offline. Download every model weight locally (section 3).
- [ ] Practice the full data loop at home with 6 household tools: film, extract frames, auto-label, review, train, test live on a webcam. Time every step. This gives you your real turnaround on site (section 4).
- [ ] Write the item request format, the catalog template (names, English and German synonyms, attributes) and the memory schema (section 5).

Also due this weekend (plan §4): register (admission is rolling), send the plan §1 questions to the organizers, lock in team and roles.

## 1. Gated model weights · today

Approval may not be instant, so request before anything else. Approval is per Hugging Face account: request from the account you will download with. Once approved, download in the browser or log in on the laptop with an access token.

| Model | Used for | Request on | Status |
|---|---|---|---|
| SAM 3 · `sam3.pt`, about 3.45 GB | Auto-labelling with text or box prompts | [facebook/sam3](https://huggingface.co/facebook/sam3) [12] | |
| SAM 3.1 · `sam3.1_multiplex.pt` (optional) | Same use, faster multi-object tracking in video | [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) [12] | |
| DINOv3 (if you pick it over DINOv2 or CLIP) | Identification gallery embeddings | [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) [13]; other sizes have their own pages | |

- Ultralytics never downloads `sam3.pt` by itself. Download it from the model page once approved, then put it in the working folder or load it by full path. `sam3.1_multiplex.pt` loads the same way [3].
- If the parser's local language model will be a Llama or Gemma model, request that too. Both families are gated on Hugging Face.

## 2. Rules on pre-written code

Status 9 Oct: no rule found in a web search. Read the event page [1] in a browser and the registration emails or participant terms. If none of them says anything, email the contacts named on the event page (Sebastian Schroth, Nicolas Butterwegge) so the answer is in writing. This is question 8 in plan §1.

"Pre-written code" can mean several things, so ask about each:

- Your own code written before the event: parser, memory module, data-loop scripts
- Docker images, install scripts and config files
- Open-source libraries and pretrained model weights
- Notes and templates, such as the ones in section 5

| Answer | Prepare before the event |
|---|---|
| Allowed | Environments, notes, practice, and your own code |
| Not allowed | Environments, notes and practice only. Practice code stays at home; bring the steps, settings and timings as notes. |
| No answer yet | Treat it as not allowed |

Rule found: ________ · Source: ________ · Date: ________

## 3. Laptops and environments

One or two Linux laptops with an NVIDIA GPU. With two, keep them identical so either can take over.

### ROS 2

- Match the robot vendor's version. Vendor drivers usually target an LTS release: Humble (Ubuntu 22.04), Jazzy (Ubuntu 24.04) or Lyrical (Ubuntu 26.04, released May 2026) [14].
- Until the vendor confirms, run ROS 2 in Docker. Changing distro then means changing images, not reinstalling the laptop.
- Include MoveIt 2 and hand-eye calibration tooling (Week 2 rehearses both), AprilTag detection, and Nav2 if a mobile base is likely.
- Keep `ROS_DOMAIN_ID` configurable. Other teams may run ROS 2 on the same network, and you need to match the robot's setting.

### Python ML environment

- PyTorch with CUDA. Confirm it sees the GPU, inside Docker too (the host needs the NVIDIA Container Toolkit).
- Ultralytics 8.4.0 or newer, needed for YOLOE-26 [11].
- Ultralytics' CLIP package from GitHub. Text prompts in YOLOE-26 and SAM 3 need it, and YOLOE installs it on first use, which fails offline [3][11].
- Speech-to-text: a multilingual Whisper model (small, medium or turbo). The `.en` models are English-only and will not handle German. Whisper also needs ffmpeg [15].
- Text-to-speech: Piper, a local engine, with one German and one English voice [23].
- A labelling tool that runs locally: CVAT or Label Studio. Roboflow is a cloud service; use it only if on-site internet is confirmed.
- The embedding model, a local language model for the parser, and optionally a local VLM (weights below).

### Docker images

Three images: ROS 2, ML, and the labelling tool. Save them to files and keep them on a USB SSD together with the model weights, so the second laptop or a reinstall needs no internet.

### Model weights to download

| Model | Files | Used for | Note |
|---|---|---|---|
| YOLO26 segmentation | `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt` | Detect + segment, fine-tuned on the items | AGPL-3.0 [4] |
| YOLOE-26 | `yoloe-26s-seg.pt` (n and m too, for options) | Zero-shot baseline; new items by text prompt | Text prompts also need `mobileclip2_b.ts` (about 254 MB). It downloads on first use into the current folder, so run from the same folder on site or copy it there [11]. |
| SAM 3 | `sam3.pt` | Auto-labelling only, not the live loop | Gated (section 1) |
| Identification | DINOv2, DINOv3 or CLIP | Embedding gallery | DINOv3 is gated |
| Speech-to-text | Multilingual Whisper | Voice in English and German | Downloads on first load |
| Text-to-speech | Piper voices: one German (e.g. Thorsten), one English | Spoken status lines and questions | Each voice is a model file plus a config file; check each voice's licence [23] |
| Language model | Your choice | Request parser without internet | Llama and Gemma are gated |
| Optional | Local VLM; GraspGen checkpoints [7] | Vague requests and checks; tier-2 grasping | Check the GraspGen licence |

Keep a list of each model's file, version, source and licence next to the weights.

### Offline test

Some tools fetch extra files the first time a feature runs, so a list of weights is not enough. The real test is a full run with the network off.

1. While online, run each step once from the folders you will use on site: a YOLOE-26 text-prompt prediction, a SAM 3 text-prompt prediction, a short training run, a Whisper transcription in German and in English, a spoken line from each Piper voice, a parser call.
2. Turn Wi-Fi off and unplug Ethernet. Set `HF_HUB_OFFLINE=1` so Hugging Face libraries use the local cache instead of trying to download.
3. Run the whole data loop (section 4) again. Anything that tries to download is a bug to fix now.
4. Check the GPU inside each container, open RViz from the ROS 2 container, and repeat on the second laptop.

## 4. Practice the data loop at home

The goal is your real turnaround: from filming to a scored model running live. Use the same scale and rules as Day 1 (dataset v1 = sessions S1 + S2, plan §5), otherwise the timing is not real.

### Setup

- 6 household tools. Mix easy shapes and thin handles, and include one look-alike pair, such as two screwdriver or wrench sizes.
- 2–3 boxes or trays as bins, so you can reshuffle layouts.
- Write their catalog entries first, using the template in section 5.2. That tests the template too.
- Lock camera exposure and resolution. Use the same settings for filming and the live test.

### Steps and timings

| # | Step | How (same rules as on site) | Time |
|---|---|---|---|
| 1 | Film | Short videos. S1: each item alone in each bin, 3 lighting conditions. S2: 5+ shuffled layouts with all items. Film 1–2 extra layouts as the golden test set. | |
| 2 | Extract frames | Keep 2–3 frames per second. Aim for 200–320 training images and about 150 labelled instances per item. | |
| 3 | Zero-shot baseline | YOLOE-26 text prompts and SAM 3 concept prompts with just the item names. Note which items already work. | |
| 4 | Auto-label | SAM 3 text prompts; box prompts for the misses. | |
| 5 | Review | Fix masks in CVAT or Label Studio. Label an item when about a third or more is visible; mask the whole object, including handles; one class per item, look-alikes separate. A second person checks 10%. | |
| 6 | Train | YOLO26-seg (n or s) from pretrained weights, image size 640–1024. Split by layout, never by random frames. | |
| 7 | Score | On the golden layouts: mAP50, recall per item, confusion between the look-alikes. Look at every miss. | |
| 8 | Test live | Webcam, a freshly reshuffled layout, all 6 items. Note the frame rate. | |
| | **Total** | | |

Write the one-page dataset card for the practice set (classes, counts per class, sessions, known gaps). Day 1 needs one too.

### Does it fit Day 1?

On Day 1, Perception has 12:00–15:00 for the zero-shot baseline and the S1–S2 capture, and 15:00–20:00 to auto-label, review and start v1 training overnight ([day1.md](../../day1/day1.md)). If steps 1–5 do not fit those windows, fix the slowest step this week.

Also write down the training time on each laptop. It decides how many retrains fit into an evening.

## 5. Request format, catalog template, memory schema

Draft them here, try them on the practice items, and fill in the real items on Day 1. These are notes, not code.

### 5.1 Item request format

What the parser produces from one spoken or typed sentence. The language model fills it in and only chooses from fixed skills (find, pick, deliver, scan, cancel). Every field is checked against the catalog before anything moves.

| Field | Meaning | Example: clear (DE) | Example: ambiguous (EN) |
|---|---|---|---|
| text | Exact phrasing as heard; goes into the trial log | "Bring mir den kleinen Schraubendreher" | "Bring me the screwdriver" |
| language | de or en | de | en |
| item | Catalog ID after matching; all candidates if ambiguous; empty if unknown | screwdriver_small | screwdriver_small, screwdriver_large |
| attributes | Words that narrow the item: size, colour, type | size: small | none |
| quantity | How many; default 1 | 1 | 1 |
| destination | tray or hand-over | hand-over | tray (default) |
| status | clear · ambiguous: ask one short question · unknown: say so and offer to enroll | clear | ambiguous: "Two screwdrivers here, small or large?" |

Normalize attributes to the catalog's words. German endings change ("klein", "kleine", "kleinen"), so "kleinen" and "small" both become size: small.

### 5.2 Catalog template

One entry per item; look-alikes get separate entries. The examples are a practice look-alike pair.

| Field | Meaning | Example A | Example B |
|---|---|---|---|
| ID | One class name used everywhere: labels, memory, trial log | screwdriver_small | screwdriver_large |
| Name (EN) | Spoken name | flat-head screwdriver, small | flat-head screwdriver, large |
| Name (DE) | Spoken name | Schlitzschraubendreher, klein | Schlitzschraubendreher, groß |
| Spoken (DE), nominative | What the robot says when the item is the subject: "… liegt in B4" | der kleine Schraubendreher | der große Schraubendreher |
| Spoken (DE), accusative | What the robot says when the item is the object: "Ich suche …" | den kleinen Schraubendreher | den großen Schraubendreher |
| Synonyms (EN) | Other words people use | screwdriver, flathead | screwdriver, flathead |
| Synonyms (DE) | Other words people use | Schraubendreher, Schraubenzieher | Schraubendreher, Schraubenzieher |
| Category | Feeds the search priors (e.g. consumables usually in drawer A) | hand tool | hand tool |
| Size | Rough dimensions | about 15 cm long | about 25 cm long |
| Colour | Main colours | red handle | red handle |
| Weight | Rough weight | about 40 g | about 120 g |
| Grasp type | pinch, suction or top-down | top-down pinch on the handle | top-down pinch on the handle |
| Storage locations | Where it may be stored | bins B1–B3 | bins B1–B3 |
| Look-alikes | Items it can be confused with, and the attribute that tells them apart | screwdriver_large · size | screwdriver_small · size |
| Gallery | 5–10 clean reference crops from several angles, for identification and enrolling | gallery/screwdriver_small | gallery/screwdriver_large |

Shared synonyms such as "screwdriver" are intentional. They match both entries, which triggers the clarifying question.

German needs both spoken forms because the article and the adjective ending change with the item's role in the sentence: "Der kleine Schraubendreher liegt in B4", but "Ich suche den kleinen Schraubendreher". The voice lines in [week 2 §3](../week2/week2.md) use them.

### 5.3 Memory schema

A small graph: item → bin → shelf → zone, with a time stamp and confidence on every sighting.

**Places.** Set up once on Day 1, when shelves and bins get their AprilTags.

| Field | Meaning | Example |
|---|---|---|
| place ID | Zone, shelf or bin | B2 |
| type | zone, shelf or bin | bin |
| part of | Parent place | shelf A |
| AprilTag ID | Tag on that shelf or bin | 12 |
| observation pose | Named robot pose for looking at it | look_B2 |
| usual categories | Prior for the search | hand tool |
| travel cost | Rough distance or time to reach it | about 5 s |

**Sightings.** Add a row on every observation. The latest row with found = yes is the item's last-seen place; a later miss there means it has moved.

| Field | Meaning | Example |
|---|---|---|
| item | Catalog ID | screwdriver_small |
| place | Bin, tray, or handed over | B2 |
| found | yes, or no when the robot looked and the item was not there | yes |
| position | 3D position in the robot base frame, in metres | 0.42, -0.10, 0.05 |
| time | When it was seen | 10:42:13 |
| confidence | How sure detection and identification were | 0.91 |
| seen by | scan, pick or delivery | scan |

**Rules**

- Update memory on every observation, even while looking for something else.
- Search order: last-seen place first, then other places ranked by how recently the item was seen there, category priors and travel cost. Skip places already checked in this search.
- Not found anywhere: run a full scan, say so honestly, and offer the closest alternative.
- The dashboard shows each item's place and age ("seen 2 min ago"), and the voice says what changed ("Not in bin B2 any more. Checking B4.").

## End of week

- [ ] Gated weights approved; `sam3.pt` on every laptop.
- [ ] Written answer on pre-written code.
- [ ] Offline test passed on every laptop.
- [ ] Data loop timed end to end; slowest step known and improved.
- [ ] Request format, catalog template and memory schema written and tried on the practice items.

Next: [week2.md](../week2/week2.md), 16–21 Oct.

## Sources

[1]–[10] match the plan's numbering. New sources start at [11].

- [1] Siemens Healthineers SHIFT: Robotics and AI Hackathon 2026 (event page). <https://innovation.siemens-healthineers.com/shift/robotics-and-ai-hackathon-2026/overview>
- [3] Ultralytics docs: SAM 3. <https://docs.ultralytics.com/models/sam-3>
- [4] Ultralytics docs: YOLO26 and YOLOE-26. <https://docs.ultralytics.com/models/yolo26/>
- [7] NVlabs GraspGen (GitHub). <https://github.com/NVlabs/GraspGen>
- [11] Ultralytics docs: YOLOE (checkpoints, text encoder, offline use). <https://docs.ultralytics.com/models/yoloe/>
- [12] Hugging Face: SAM 3 and SAM 3.1 model pages. <https://huggingface.co/facebook/sam3> · <https://huggingface.co/facebook/sam3.1>
- [13] Hugging Face: DINOv3 model card (ViT-B/16). <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- [14] Open Robotics Discourse: ROS 2 Lyrical Luth released. <https://discourse.openrobotics.org/t/ros-2-lyrical-luth-released/55021>
- [15] OpenAI Whisper (GitHub). <https://github.com/openai/whisper>
- [23] Piper TTS: engine and voice list (shared with week 2). <https://github.com/OHF-Voice/piper1-gpl> · <https://github.com/rhasspy/piper/blob/master/VOICES.md>
