# MMA Fighter Detection Dataset: Annotated UFC Stand-Up Combat Images for Computer Vision

## 1. Description

This dataset contains **5,106 images** designed for object detection tasks, specifically for identifying mixed martial arts (MMA) fighters inside an octagon. All images are sourced from high-definition fight footage from the Ultimate Fighting Championship (UFC).

The dataset is annotated with a single class, `fighter`, making it suitable for training models to locate and track fighters during a match. The annotations are provided in the **YOLOv8** format.

This dataset was created and exported using the [Roboflow](https://roboflow.com/) platform.

## 2. Scope and Limitations

**A key characteristic of this version of the dataset is its focus on the "stand-up" or striking aspects of MMA.** All annotated images capture fighters primarily engaged in striking exchanges from a distance.

Scenes involving prolonged clinching against the cage or grappling on the ground ("ground game") have been intentionally excluded from this version. This makes the dataset particularly well-suited for models focused on stand-up fighter detection but less suitable for analyzing grappling or clinch scenarios without further data collection.

## 3. Dataset Details

| Attribute             | Value                        |
| --------------------- | ---------------------------- |
| **Total Images**      | 5,106                        |
| **Annotation Format** | YOLOv8 (`.txt` files)        |
| **Image Resolution**  | Resized to 640x640 (Stretch) |
| **Class Name**        | `fighter`                    |
| **Class ID**          | 0                            |
| **Augmentations**     | None applied                 |
| **Data Splits**       | Train, Validation, and Test  |

## 4. Data Collection and Source

The images were extracted from official, full-fight videos published on the official UFC YouTube channels (including regional counterparts like UFC Español). The use of high-quality broadcast footage ensures a diverse range of camera angles, lighting conditions, and fighter appearances.

The training data is composed of frames from the following 20 UFC fights:

1.  Dustin Poirier vs. Michael Chandler
2.  Israel Adesanya vs. Robert Whittaker 2
3.  Alex Pereira vs. Khalil Rountree Jr.
4.  Justin Gaethje vs. Michael Chandler
5.  Dustin Poirier vs. Conor McGregor 2
6.  Petr Yan vs. Cory Sandhagen
7.  Petr Yan vs. Sean O'Malley
8.  Petr Yan vs. Song Yadong
9.  Petr Yan vs. José Aldo
10. Israel Adesanya vs. Paulo Costa
11. Sharabutdin Magomedov vs. Michal Oleksiejczuk
12. Israel Adesanya vs. Alex Pereira 1
13. Israel Adesanya vs. Alex Pereira 2
14. Ilia Topuria vs. Alexander Volkanovski
15. Ilia Topuria vs. Max Holloway
16. Dricus du Plessis vs. Sean Strickland
17. Cody Garbrandt vs. Dominick Cruz
18. Max Holloway vs. Brian Ortega
19. Dustin Poirier vs. Justin Gaethje 1
20. Dustin Poirier vs. Dan Hooker

## 5. Folder Structure

The dataset is organized into a standard YOLOv8 format. The expected structure is as follows:
dataset/
├── train/
│ ├── images/
│ │ ├── image001.jpg
│ │ └── ...
│ └── labels/
│ ├── image001.txt
│ └── ...
├── valid/
│ ├── images/
│ │ └── ...
│ └── labels/
│ └── ...
├── test/
│ ├── images/
│ │ └── ...
│ └── labels/
│ └── ...
├── data.yaml
├── LICENSE.txt
└── README.md

The `data.yaml` file contains the configuration for training with YOLOv8.

## 6. Downloading the Full Dataset

The images and labels are not included in this repository. Download the full dataset from **Mendeley Data**:

> **[https://data.mendeley.com/datasets/c456bnk8bm/1](https://data.mendeley.com/datasets/c456bnk8bm/1)**

After downloading, extract the contents so that the `train/`, `valid/`, and `test/` folders sit alongside this `README.md`.

## 7. How to Use

This dataset can be used to train object detection models with the [Ultralytics YOLO framework](https://github.com/ultralytics/ultralytics) or any other framework that supports the YOLO annotation format. Simply point the training script to the included `data.yaml` file.

## 8. Citation

If you use this dataset in your research, please cite it as:

Faisal, Hasan (2025), "MMA Fighter Detection Dataset: Annotated UFC Stand-Up Combat Images for Computer Vision", Mendeley Data, V1, doi: 10.17632/c456bnk8bm.1

**BibTeX format:**

```bibtex
@misc{faisal_2025_mma_fighter_dataset,
  author       = {Faisal, Hasan},
  title        = {{MMA Fighter Detection Dataset: Annotated UFC Stand-Up Combat Images for Computer Vision}},
  year         = {2025},
  publisher    = {Mendeley Data},
  version      = {V1},
  doi          = {10.17632/c456bnk8bm.1},
  url          = {https://doi.org/10.17632/c456bnk8bm.1}
}
```
