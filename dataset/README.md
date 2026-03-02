# Datasets

This directory contains the datasets used in the **fight-judge** project. Each sub-folder holds dataset metadata (configuration, license, documentation) committed to the repository, while the heavy image and label files are excluded via `.gitignore` and hosted on **Mendeley Data**.

---

## Included Datasets

### 1. Object Detection — `mma-fighter-detection-dataset/`

| Attribute | Value |
|---|---|
| **Task** | Object detection (bounding boxes) |
| **Images** | 5,106 |
| **Classes** | 1 (`fighter`) |
| **Format** | YOLOv8 |
| **Resolution** | 640×640 |
| **Download** | [Mendeley Data — Version 1](https://data.mendeley.com/datasets/c456bnk8bm/1) |

Bounding box annotations for MMA fighters in UFC stand-up striking footage. See the [full README](mma-fighter-detection-dataset/README.md) for details.

### 2. Pose Estimation — `mma-fighter-pose-estimation-dataset/`

| Attribute | Value |
|---|---|
| **Task** | Pose estimation (17 COCO keypoints) |
| **Images** | 5,109 |
| **Classes** | 1 (`fighter`) |
| **Format** | YOLO Pose |
| **Resolution** | 640×640 |
| **Download** | [Mendeley Data — Version 2](https://data.mendeley.com/datasets/c456bnk8bm/2) |

17-point skeletal keypoint annotations derived from the object detection dataset by running YOLOv11x-pose inference and filtering keypoints within ground-truth bounding boxes. See the [full README](mma-fighter-pose-estimation-dataset/README.md) for details.

---

## Setup

1. Clone this repository.
2. Download the desired dataset version from Mendeley Data (links above).
3. Extract the contents so that `train/`, `valid/`, and `test/` folders sit inside the corresponding dataset directory.

```
dataset/
├── README.md                             ← this file
├── mma-fighter-detection-dataset/
│   ├── data.yaml
│   ├── LICENSE.txt
│   ├── README.md
│   ├── train/  (images/ + labels/)       ← download from Mendeley V1
│   ├── valid/  (images/ + labels/)
│   └── test/   (images/ + labels/)
└── mma-fighter-pose-estimation-dataset/
    ├── data.yaml
    ├── LICENSE.txt
    ├── README.md
    ├── train/  (images/ + labels/)       ← download from Mendeley V2
    ├── valid/  (images/ + labels/)
    └── test/   (images/ + labels/)
```
