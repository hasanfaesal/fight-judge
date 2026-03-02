# Pose Detection Dataset — UFC/MMA Fighter Keypoints

## Overview

This dataset is used for **pose estimation** of MMA fighters in UFC fight footage.
It provides **17 COCO-style keypoints** per fighter bounding box, annotated in YOLO pose format.

It is the companion dataset to the [object detection dataset](../mma-fighter-detection-dataset/README.md),
which first localises fighters in each frame. This dataset builds on that by adding full body pose.

---

## Dataset Statistics

| Split     | Images | Labels |
|-----------|--------|--------|
| Train     | 3,636  | 3,636  |
| Valid     | 981    | 981    |
| Test      | 492    | 492    |
| **Total** | **5,109** | **5,109** |

- **Classes**: 1 (`fighter`)
- **Keypoints per instance**: 17 (COCO format)
- **Annotation format**: YOLO Pose (`.txt`)

---

## Keypoint Schema (COCO-17)

| Index | Keypoint       | Index | Keypoint        |
|-------|----------------|-------|-----------------|
| 0     | nose           | 9     | left wrist      |
| 1     | left eye       | 10    | right wrist     |
| 2     | right eye      | 11    | left hip        |
| 3     | left ear       | 12    | right hip       |
| 4     | right ear      | 13    | left knee       |
| 5     | left shoulder  | 14    | right knee      |
| 6     | right shoulder | 15    | left ankle      |
| 7     | left elbow     | 16    | right ankle     |
| 8     | right elbow    |       |                 |

Each keypoint is stored as `x y v` where `v` is visibility: `0` = not labeled, `1` = labeled but occluded, `2` = labeled and visible.

---

## Annotation Format

Each `.txt` label file follows YOLO pose format, one instance per line:

```
<class> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <kp0_v> ... <kp16_x> <kp16_y> <kp16_v>
```

All values are **normalised** to `[0, 1]` relative to image dimensions.

---

## Folder Structure

```
mma-fighter-pose-estimation-dataset/
├── data.yaml          ← YOLO pose training config (committed)
├── README.md          ← this file (committed)
├── LICENSE.txt        ← dataset license (committed)
├── train/
│   ├── images/        ← gitignored
│   └── labels/        ← gitignored
├── valid/
│   ├── images/        ← gitignored
│   └── labels/        ← gitignored
└── test/
    ├── images/        ← gitignored
    └── labels/        ← gitignored
```

> **Note**: `images/` and `labels/` folders are excluded from this repository via `.gitignore`.
> Download the full dataset from Mendeley Data (see link below).

---

## Source & Provenance

- **Platform**: [Roboflow Universe](https://universe.roboflow.com/deep-learning-zhrus/my-first-project-pucm2/dataset/4)
- **Workspace**: `deep-learning-zhrus`
- **Project**: `my-first-project-pucm2`
- **Version**: 4
- **License**: Creative Commons BY-NC-SA 4.0

---

## Downloading the Full Dataset

Download the full dataset (images + labels) from **Mendeley Data**:

> **[https://data.mendeley.com/datasets/c456bnk8bm/2](https://data.mendeley.com/datasets/c456bnk8bm/2)**

After downloading, extract the contents so that the `train/`, `valid/`, and `test/` folders sit alongside this `README.md`.

---

## Usage (Ultralytics YOLOv11)

```python
from ultralytics import YOLO

model = YOLO("yolo11x-pose.pt")
model.train(
    data="dataset/mma-fighter-pose-estimation-dataset/data.yaml",
    epochs=100,
    imgsz=640,
)
```

---

## Citation

If you use this dataset in your research, please cite it as:

Faisal, Hasan (2025), "MMA Fighter Pose Estimation Dataset: Keypoint-Annotated UFC Stand-Up Combat Images for Computer Vision", Mendeley Data, V2, doi: 10.17632/c456bnk8bm.2

```bibtex
@misc{faisal_2025_mma_fighter_pose_dataset,
  author       = {Faisal, Hasan},
  title        = {{MMA Fighter Pose Estimation Dataset: Keypoint-Annotated UFC Stand-Up Combat Images for Computer Vision}},
  year         = {2025},
  publisher    = {Mendeley Data},
  version      = {V2},
  doi          = {10.17632/c456bnk8bm.2},
  url          = {https://doi.org/10.17632/c456bnk8bm.2}
}
```

## Related

- [`mma-fighter-detection-dataset/`](../mma-fighter-detection-dataset/README.md) — Object detection dataset (fighter bounding boxes)
