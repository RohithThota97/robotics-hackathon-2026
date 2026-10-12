# Tasks 1–8 · what still needs a human and a GPU

Tasks 9–14 are done (see [practice/](practice/) and the harness). The eight tasks below cannot be done from a MacBook: they need an NVIDIA GPU, real footage, a Hugging Face account, or a reply from the organizers. This page records exactly what each one needs so none of it has to be rediscovered.

Section numbers point to [week1.md](week1.md).

## Do task 1 today

**Task 1 is the critical path for the whole week.** `facebook/sam3` approval is manual, per Hugging Face account, and can take days. It gates task 4 (downloads), which gates task 5 (offline test) and task 6 (the real data loop) — and task 6 is what produces the Week 1 checkpoint numbers. Every day task 1 waits, the whole chain waits.

Request it from the account you will actually download with, not a spare one.

## Hardware you do not have

Your machine is an **Apple M5, 10 cores, 16 GB, Metal 4**. That is a capable GPU, but it is not CUDA, and three things in week 1 need CUDA specifically:

| Need | Why Apple silicon cannot substitute |
|---|---|
| ML Docker image | Requires the NVIDIA Container Toolkit for GPU passthrough. No equivalent exists on macOS. |
| `PyTorch sees the GPU inside the container` (task 3's done-when) | That check *is* a CUDA check. |
| Task 6 training times | The number must come from the hardware you will use on site. |

So tasks 3, 5 and 6 need the Linux + NVIDIA laptop. Borrow, rent or use a cloud GPU, but do not substitute the Mac and write the result into the §4 table.

## Task-by-task

### 1. Gated model weights (§1) — **start now**

- Request `facebook/sam3` (`sam3.pt`, ~3.45 GB) from your download account.
- Optional: `facebook/sam3.1` (`sam3.1_multiplex.pt`) for faster multi-object video tracking.
- Request DINOv3 ViT-B/16 if you pick it over DINOv2 or CLIP for the identification gallery.
- Request Llama or Gemma if the parser will use one — both are gated.
- Download `sam3.pt` by hand. **Ultralytics never downloads it.** Put it in the working folder or load it by full path.
- Fill in the Status column in §1.

**Done when** access is approved and `sam3.pt` is on every laptop.

### 2. Pre-written code rule (§2) — **start now**

Decides what you build in advance, so it is worth an early answer.

- Read the event page, the registration emails and the participant terms.
- If none of them says anything, email the contacts on the event page so the answer is in writing.
- Ask about each kind **separately**: your own pre-event code; Docker images, install scripts and config; open-source libraries and pretrained weights; notes and templates.

**Until there is an answer, assume it is not allowed.** That is the expensive assumption — it means the harness, the catalog and the parser stay at home and you bring only steps, settings and timings as notes. Getting this in writing early may give you that work back.

**Done when** there is a written answer with its source and date in §2.

### 3. Laptop environments (§3) — needs the NVIDIA laptop

One or two Linux laptops with an NVIDIA GPU; if two, keep them identical so either can take over. Three images, all saved to a USB SSD:

- **ROS 2** — match the robot vendor's distro (Humble/22.04, Jazzy/24.04, Lyrical/26.04). MoveIt 2, hand-eye calibration tooling, AprilTag detection, Nav2 if a mobile base is likely. Keep `ROS_DOMAIN_ID` configurable.
- **ML** — PyTorch with CUDA (host needs the NVIDIA Container Toolkit), Ultralytics ≥ 8.4.0 for YOLOE-26, Ultralytics' CLIP package from GitHub, a **multilingual** Whisper (small/medium/turbo — never `.en`), ffmpeg, Piper for TTS.
- **Labelling** — CVAT or Label Studio running locally. Roboflow only if on-site internet is confirmed.

**Done when** all three images are built and saved, and PyTorch sees the GPU inside the ML container.

### 4. Download all model weights (§3) — needs task 1, task 3

Download into the folders you will use on site, then copy everything to the SSD. Roughly 10–15 GB total; you have plenty of disk.

| Model | Files | For |
|---|---|---|
| YOLO26-seg | `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt` | Detector, fine-tuned (AGPL-3.0) |
| YOLOE-26 | `yoloe-26s-seg.pt` (n, m too) + `mobileclip2_b.ts` (~254 MB) | Zero-shot baseline, new items by text |
| SAM 3 | `sam3.pt` (task 1) | Auto-labelling only, not the live loop |
| Identification | DINOv2 / DINOv3 / CLIP | Embedding gallery |
| Speech-to-text | Multilingual Whisper | Voice, DE + EN |
| Text-to-speech | Piper: one German (e.g. Thorsten), one English — model **and** config file each | Spoken status and questions |
| Language model | Team's choice | Offline request parser |
| Optional | Local VLM; GraspGen checkpoints | Vague requests; tier-2 grasping |

Two traps: `mobileclip2_b.ts` downloads into the **current folder** on first use, so run from the same folder on site or copy it there. And check the licence of each Piper voice and of GraspGen.

**Done when** every weight is local on every laptop and on the SSD, with a `file / version / source / licence` list beside them.

### 5. Offline test (§3) — needs tasks 3, 4, 6

A list of weights proves nothing: some tools fetch extra files the first time a feature runs. Only a full run with the network off shows what breaks on site.

1. While **online**, run each feature once from the on-site folders: a YOLOE-26 text-prompt prediction, a SAM 3 text-prompt prediction, a short training run, Whisper in German and English, a line from each Piper voice, a parser call.
2. Turn Wi-Fi off, unplug Ethernet, set `HF_HUB_OFFLINE=1`.
3. Run the whole data loop again. **Anything that tries to download is a bug to fix now.**
4. Check the GPU inside each container and open RViz from the ROS 2 container.
5. Repeat on the second laptop.

**Done when** the whole loop runs with the network off on every laptop.

### 6. Real data loop with 6 household tools, timed (§4) — needs tasks 3, 4, 10

This produces the Week 1 checkpoint: your real turnaround from filming to a model running live. It decides whether v1 can train overnight on Day 1. **The harness cannot give these numbers** — its times are synthetic and labelled as such.

Setup: 6 household tools mixing easy shapes and thin handles, including one look-alike pair; 2–3 trays as bins; catalog entries written first (done — [practice/catalog.yaml](practice/catalog.yaml)); camera exposure and resolution **locked**, same settings for filming and the live test.

Run the 8 steps with Day 1's rules and time each into the §4 table:

1. **Film** — S1: each item alone in each bin, 3 lighting conditions. S2: 5+ shuffled layouts with all items. Plus 1–2 extra layouts as the golden set.
2. **Extract frames** — 2–3 fps. Target 200–320 training images, ~150 instances per item.
3. **Zero-shot baseline** — YOLOE-26 text prompts and SAM 3 concept prompts, item names only. Note which items already work.
4. **Auto-label** — SAM 3 text prompts, box prompts for the misses.
5. **Review** — in CVAT or Label Studio. Label at ≥⅓ visible; mask the whole object including handles; one class per item, look-alikes separate; a second person checks 10%.
6. **Train** — YOLO26-seg (n or s) from pretrained, image size 640–1024. **Split by layout, never by random frames** — random splits leak and will flatter you.
7. **Score** — on the golden layouts only: mAP50, recall per item, look-alike confusion. Look at every miss.
8. **Test live** — webcam, freshly reshuffled layout, all 6 items. Note the frame rate.

Use [practice/catalog.yaml](practice/catalog.yaml) for the class names and [practice/memory-schema.md](practice/memory-schema.md) to log sightings during step 8.

**Done when** all 8 rows and the Total in §4 hold real times. Plan targets for comparison: mAP50 ≥ 0.85, recall per item ≥ 0.90 on the golden set.

### 7. Day 1 fit check (§4) — needs task 6

On Day 1, Perception has **12:00–15:00** for the zero-shot baseline and S1–S2 capture, and **15:00–20:00** to auto-label, review and start v1 training overnight.

Check task 6's times against those two windows. If steps 1–5 do not fit, fix the slowest step this week and time it again. Write down the training time on **each** laptop — it decides how many retrains fit into an evening.

**Done when** steps 1–5 fit the windows and per-laptop training time is written down.

### 8. Practice dataset card (§4) — needs task 6

One page for the practice set: classes, counts per class, sessions, known gaps. The harness's generated `dataset_card.md` has the same layout and is the template — run `week1-harness --out ./out` and fill the real counts over the synthetic ones.

**Done when** the card is written from task 6's real counts.

---

## Appendix · making zero-shot real on the M5

Optional, and **not** a substitute for task 6. The harness emits a labelled `PLACEHOLDER` for the zero-shot step when torch, Ultralytics or the weights are missing, which is a designed branch, not a failure. If you want that row to be a real measurement on your Mac:

```bash
pip install torch ultralytics
```

Verified to resolve on this machine: torch **2.14.1**, torchvision 0.29.1, Ultralytics **8.4.175** for Python 3.14 on arm64. Roughly 3 GB.

Then fetch the **ungated** YOLOE-26 weights — unlike SAM 3 these need no approval — into one folder:

- `yoloe-26s-seg.pt`
- `mobileclip2_b.ts` (~254 MB, needed for text prompts)

Point the harness at them:

```bash
week1-harness --weights-dir /path/to/weights --out ./out
```

`capabilities.detect()` searches `--weights-dir` first, then the current directory, then Ultralytics' config dir, then the Hugging Face cache. With all three present the zero-shot step runs real inference and its row becomes `REAL`.

To use the M5 GPU rather than CPU, PyTorch's Metal backend is `torch.device("mps")`; Ultralytics takes `device="mps"`.

### Read this before you quote any number from it

**An M5 timing is an `ESTIMATE`, never a task 6 answer.** Task 6 exists to tell you what the loop costs on the hardware you will use on site. A Metal number is not that, it is not comparable to a CUDA number, and putting one in the §4 table would manufacture exactly the kind of fake-real figure [week1_harness/markers.py](../../week1_harness/markers.py) exists to prevent. If you record an M5 time anywhere, label it and say which machine produced it.
