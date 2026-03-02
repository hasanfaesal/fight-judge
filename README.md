# fight-judge



https://github.com/user-attachments/assets/b98d7962-321f-40e1-9638-cb4691a1a56b




An AI system for scoring boxing and MMA fights according to official judging criteria.
The project builds a full computer-vision pipeline — from raw fight footage to pose-based
action recognition — with the eventual goal of automated, rule-compliant fight scoring.

![System Design](system-design-dark.png)

---

## Models & Datasets

### Trained Models

| Model | Task | Key Metric | Download |
|-------|------|------------|----------|
| **YOLOv8s** (finetuned) | Fighter detection | mAP50-95: **0.984** | [Hugging Face](https://huggingface.co/hasanfaesal/fight-judge-yolov8s) |
| **YOLO11x-pose** (finetuned) | Pose estimation (17 keypoints) | Pose mAP50-95: **0.920** | [Hugging Face](https://huggingface.co/hasanfaesal/fight-judge-yolov11x-pose) |

### Datasets

| Dataset | Contents | Format | Source |
|---------|----------|--------|--------|
| **MMA Fighter Detection** | 5,106 images, 10,186 fighter bboxes, 20 UFC fights | YOLO | [Mendeley Data](https://data.mendeley.com/datasets/c456bnk8bm/1) |
| **MMA Fighter Pose Estimation** | 5,106 images, 10,155 fighters with 17 COCO keypoints | YOLO-Pose | Auto-generated ([methodology](#pose-dataset-auto-generation)) |

Both datasets use 640×640 px images covering diverse weight classes and fighting styles
(Poirier, Adesanya, Pereira, Topuria, Gaethje, and others). Licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Table of Contents

- [Models & Datasets](#models--datasets)
- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Results & Metrics](#results--metrics)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)

---

## Quick Start

```python
from ultralytics import YOLO

# Fighter detection
detector = YOLO("hasanfaesal/fight-judge-yolov8s")
detections = detector("fight_frame.jpg")

# Pose estimation
pose_model = YOLO("hasanfaesal/fight-judge-yolov11x-pose")
poses = pose_model("fight_frame.jpg")

keypoints = poses[0].keypoints.xy     # [N, 17, 2] — pixel coordinates
confidence = poses[0].keypoints.conf  # [N, 17]    — per-keypoint confidence
boxes = poses[0].boxes.xyxy           # [N, 4]     — fighter bounding boxes
```

---

## Project Overview

Combat sports judging is notoriously subjective and inconsistent. This project explores
whether a pose-estimation-based AI can objectively quantify fighter output — effective
striking, aggression, ring generalship — to assist or replace human judges.

The current phase focuses on building a high-quality, labeled dataset of fighter poses
across diverse UFC matchups, and training robust detection and keypoint models as the
foundation for downstream action recognition.

---

## Pipeline Architecture

```
Stage 1 — Frame Extraction
  scripts/data_preparation/extract-frames.py
  └─ MP4 fight footage → JPEG frames (all frames, no sampling)

Stage 2 — Fighter Detection
  Model: YOLOv8s (finetuned) → mAP50-95: 0.984
  Dataset: MMA Fighter Detection Dataset (5,106 images, YOLO format)
  └─ Manual annotation via Roboflow, published on Mendeley Data

Stage 3 — Pose Dataset Auto-Generation  ← KEY CONTRIBUTION
  scripts/data_preparation/yolo-to-coco-bbox.py
  └─ Converts YOLO bbox labels → COCO JSON (required by mmpose)
  scripts/pose_estimation/create-pose-dataset-from-object-detection.ipynb
  └─ Runs YOLOv11x-pose on each frame, matches detections to annotated
     fighter bboxes via IoU (threshold 0.6), extracts 17 COCO keypoints
  └─ 99.7% match rate (10,155 / 10,186 fighters)

Stage 4 — Annotation Fixing
  scripts/data_preparation/fix_annotations.py
  └─ Normalizes COCO JSON output from ViTPose inference pipeline

Stage 5 — Pose Model Training
  Model: YOLO11x-pose (finetuned) → Pose mAP50-95: 0.920
  Dataset: MMA Fighter Pose Estimation Dataset (YOLO-Pose format)

Stage 6 — Visualization & Validation
  scripts/pose_estimation/visualize_pose.py
  └─ Renders YOLO-Pose labels as annotated images for quality checks

Stage 7 — Action Recognition  [PLANNED]
  └─ GCN / Transformer on multi-frame keypoint sequences
  └─ Strike classification, landing detection, scoring

Stage 8 — Scoring System  [PLANNED]
  └─ Rule-based + learned scoring engine (10-point must system)
```

---

## Results & Metrics

### Fighter Detection (YOLOv8s, finetuned)

| Metric      | Value      |
|-------------|------------|
| mAP50-95    | **0.984**  |
| mAP50       | **0.995**  |
| Precision   | 0.998      |
| Recall      | 0.999      |

Trained for 100 epochs on 2× GPUs (Kaggle), batch size 64.
Download: [hasanfaesal/fight-judge-yolov8s](https://huggingface.co/hasanfaesal/fight-judge-yolov8s)

### Pose Dataset Auto-Generation

| Metric                        | Value      |
|-------------------------------|------------|
| Total images processed        | 5,106      |
| Total fighter instances       | 10,186     |
| Fighters matched with poses   | **10,155 (99.7%)** |
| Fighters without pose match   | 31 (0.3%) |
| Matching method               | IoU ≥ 0.6  |
| Pose model used for labeling  | YOLOv11x-pose (pretrained) |

### Pose Estimation (YOLO11x-pose, finetuned)

| Metric            | Value      |
|-------------------|------------|
| Pose mAP50-95     | **0.920**  |
| Pose mAP50        | **0.993**  |
| Box mAP50-95      | **0.986**  |
| Precision (pose)  | 0.992      |
| Recall (pose)     | 0.991      |

Trained for 150 epochs (resumed from epoch 90) on Tesla P100, AdamW, batch size 8.
Download: [hasanfaesal/fight-judge-yolov11x-pose](https://huggingface.co/hasanfaesal/fight-judge-yolov11x-pose)

---

## Repository Structure

```
fight-judge/
├── scripts/
│   ├── data_preparation/
│   │   ├── extract-frames.py                    # Stage 1: video → frames
│   │   ├── yolo-to-coco-bbox.py                 # Stage 3a: YOLO → COCO JSON
│   │   └── fix_annotations.py                   # Stage 4: fix COCO annotations
│   └── pose_estimation/
│       ├── create-pose-dataset-from-object-detection.ipynb  # Stage 3b: pose labeling
│       ├── run-inference-batch.py               # Stage 5: ViTPose batch inference
│       └── visualize_pose.py                    # Stage 6: annotation visualization
├── dataset/
│   ├── mma-fighter-detection-dataset/           # Detection dataset (metadata)
│   └── mma-fighter-pose-estimation-dataset/     # Pose dataset (metadata)
├── annotations/                                 # COCO JSON annotations
├── notes/                                       # Technical notes
│   ├── PoseEstimation.md
│   └── ActionRecognition.md
├── system-design-dark.png                       # Architecture diagram (dark)
├── system-design-light.png                      # Architecture diagram (light)
├── METHODOLOGY.md                               # Technical write-up
└── pyproject.toml
```

---

## Tech Stack

| Component            | Technology                          |
|----------------------|-------------------------------------|
| Object detection     | YOLOv8s (Ultralytics), finetuned    |
| Pose estimation      | YOLO11x-pose (Ultralytics), finetuned |
| Pose labeling        | YOLOv11x-pose (pretrained, Ultralytics) |
| Dataset management   | Roboflow                            |
| Annotation format    | YOLO (.txt), COCO JSON              |
| Training platform    | Kaggle (Tesla P100-PCIE-16GB)       |
| Model hosting        | Hugging Face                        |
| Language             | Python 3.12                         |
| Core libraries       | OpenCV, NumPy, Ultralytics, mmpose, tqdm |

---

## Roadmap

| Stage | Task | Status |
|-------|------|--------|
| 1 | Frame extraction | Done |
| 2 | Fighter detection annotation (Roboflow) | Done |
| 2 | Fighter detection model training (YOLOv8s) | Done — mAP50-95: 0.984 |
| 3 | Pose dataset auto-generation | Done — 99.7% match rate |
| 5 | Pose model finetuning (YOLO11x-pose) | Done — Pose mAP50-95: 0.920 |
| 5 | ViTPose inference pipeline | Done |
| 7 | Action recognition (GCN / Transformer) | Planned |
| 7 | Strike classification (jab, cross, hook, kick, ...) | Planned |
| 7 | Landing detection (clean strike vs. blocked/missed) | Planned |
| 8 | Scoring engine (10-point must system) | Planned |
| — | Re-ID for fighter identity tracking | Planned |
| — | Ground game: takedowns, submissions, control time | Planned |

---

## Citation

```bibtex
@misc{faisal2025mmafighter,
  author    = {Hasan Faisal},
  title     = {MMA Fighter Detection Dataset},
  year      = {2025},
  publisher = {Mendeley Data},
  version   = {V1},
  doi       = {10.17632/c456bnk8bm.1}
}
```

## License

Code & models: [MIT License](LICENSE)
Dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
