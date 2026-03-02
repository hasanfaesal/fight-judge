# Methodology

This document describes the technical approach, design decisions, and experimental
results for the fight-judge project.

---

## 1. Problem Framing

Combat sports judging requires interpreting a continuous video stream and mapping it
onto a discrete scoring rubric (the 10-point must system). The key judging criteria
under unified MMA rules are:

1. **Effective striking** — clean, significant strikes landed
2. **Effective grappling** — takedowns, submission attempts, control
3. **Aggression** — forward pressure and output
4. **Octagon/ring control** — dictating the pace and position of the fight

Automating this requires solving several sub-problems in order:
1. **Detect** the fighters in each frame (object detection)
2. **Localize** their body pose (keypoint estimation)
3. **Recognize** their actions across time (action recognition)
4. **Classify** whether strikes land cleanly (impact detection)
5. **Aggregate** per-action scores into a round-level score (scoring engine)

This project addresses stages 1–2 fully and lays the groundwork for stage 3.

---

## 2. Data Collection

### 2.1 Source Material

Twenty UFC fights were selected to cover a range of weight classes, fighting styles
(striker vs. wrestler, orthodox vs. southpaw), and camera angles. Only stand-up
phases were used in the initial dataset to reduce class complexity.

Fights were recorded at broadcast quality (HD) and trimmed to relevant action
sequences using standard video editing tools.

### 2.2 Frame Extraction

All frames were extracted without temporal subsampling using `scripts/data_preparation/
extract-frames.py`. This preserves mid-technique frames that are important for pose
estimation training. For a typical 3-round fight, this yields ~30,000–60,000 frames.

A representative 5,106-frame subset was selected for annotation, balancing coverage
across fights and action diversity.

### 2.3 Bounding Box Annotation

Fighter bounding boxes were annotated manually on Roboflow using a single-class
schema (`fighter`, class ID 0). Each frame was reviewed for quality; frames with
severe motion blur, extreme occlusion, or partial fighters at frame edges were
excluded.

The resulting dataset has **10,186 fighter instances** across 5,106 images, giving
an average of approximately 2.0 fighters per frame (expected, since all selected
clips are 1-vs-1 matchups).

Dataset published on Mendeley Data: https://data.mendeley.com/datasets/c456bnk8bm/1

---

## 3. Fighter Detection

### 3.1 Model Selection

YOLOv8s was chosen as the detection backbone for its balance of speed and accuracy.
A small model is sufficient here because the detection task is simple — one class
with large, non-overlapping objects in a constrained environment.

### 3.2 Training

The model was finetuned on the MMA fighter detection dataset (5,106 images) using
Ultralytics' default YOLOv8 training configuration. No custom augmentations were
added; Ultralytics' built-in augmentation pipeline (mosaic, flipping, scaling) was
used as-is.

### 3.3 Results

| Metric   | Value |
|----------|-------|
| mAP50-95 | **0.983** |
| mAP50    | >0.99 |

The near-perfect detection performance reflects that the task is well-constrained:
two large human figures in a clearly bounded arena with a consistent camera setup.
These detections serve as inputs to the top-down pose estimation pipeline.

---

## 4. Pose Dataset Generation

### 4.1 Approach: Pseudo-Label with IoU Matching

Rather than manually annotating 17 keypoints per fighter across 10,186 instances
(which would be prohibitively expensive), a **pseudo-labeling** approach was used:
a pretrained off-the-shelf pose model generates candidate keypoint annotations,
which are then matched and filtered against the known ground-truth fighter bounding
boxes.

This is the core technical contribution of this phase.

### 4.2 Matching Algorithm

For each frame:
1. Load the ground-truth fighter bounding boxes from the detection dataset labels
2. Run `YOLOv11x-pose` (pretrained, no finetuning) on the full frame to generate
   pose detections (bounding box + 17 keypoints) for all visible persons
3. For each ground-truth fighter box, find the pose detection with the highest
   Intersection-over-Union (IoU) score
4. If `IoU ≥ 0.6`, accept the match and copy the 17 keypoints to the output label
5. If no pose detection achieves `IoU ≥ 0.6`, the fighter is marked as unmatched

### 4.3 Results

| Metric                      | Value    |
|-----------------------------|----------|
| Total fighter instances     | 10,186   |
| Successfully matched        | **10,155 (99.7%)** |
| Unmatched (no keypoints)    | 31 (0.3%) |

The 99.7% match rate validates that the pretrained YOLOv11x-pose model generalizes
well to UFC footage without domain-specific finetuning. The 31 unmatched instances
correspond to heavily occluded fighters (e.g., one fighter completely covering the
other during a clinch or ground transition).

### 4.4 Output Format

The generated dataset uses YOLO-Pose format:
```
class_id  x_c  y_c  w  h  kp0_x  kp0_y  kp0_v  kp1_x  kp1_y  kp1_v  ...  kp16_x  kp16_y  kp16_v
```
All coordinates are normalized to [0, 1]. Visibility flags: `1` = labeled but
occluded (confidence < 0.3), `2` = visible (confidence ≥ 0.3).

The dataset is split 71% / 19% / 10% (train / valid / test):
- Train: 3,635 images
- Valid: 980 images
- Test: 491 images

---

## 5. Pose Estimation Model

### 5.1 Model Selection

Two approaches were evaluated:

**Top-down (ViTPose via mmpose)**
Uses the ground-truth fighter bounding boxes to crop each fighter, then runs a
dedicated pose estimation model on each crop. Achieves high accuracy but requires
a separate detection stage and is slower at inference time.

**One-stage (YOLO11x-pose, finetuned)**
Jointly detects and estimates pose in a single forward pass. Faster and simpler
to deploy, at a small accuracy cost. This is the primary model.

### 5.2 Training

YOLO11x-pose was finetuned on the auto-generated `pose-detection-yolov11x-dataset`
using Ultralytics' default training configuration on a Kaggle Tesla P100-PCIE-16GB.

### 5.3 Results

| Metric         | Value  |
|----------------|--------|
| Pose mAP50-95  | 0.920  |

### 5.4 Inference Considerations

For the eventual scoring application, inference latency is a constraint. "Real-time"
in broadcast sports analysis typically means processing at or above the source frame
rate (25–60 FPS). The tradeoffs for available models are:

| Model         | Approach    | Accuracy | Speed |
|---------------|-------------|----------|-------|
| YOLO11x-pose  | One-stage   | 0.920 mAP50-95 | Fast  |
| ViTPose-B     | Top-down    | Higher   | Slower |
| RTMO          | One-stage   | Comparable | Fastest |
| DETRPose      | End-to-end  | High     | Moderate |

RTMO is a strong candidate for a production deployment given its speed-accuracy
tradeoff. Further ablation is planned once the action recognition stage requires
real-time input.

---

## 6. Action Recognition (Planned)

### 6.1 Input Representation

The output of the pose estimation stage is a sequence of skeleton graphs: one graph
per frame, where nodes are keypoints and edges are limb connections. For action
recognition, a window of consecutive frames is used as input, giving a
spatio-temporal graph.

### 6.2 Architecture Candidates

The spatio-temporal nature of the data makes the following architectures applicable:

**Graph Convolutional Networks (GCNs)**
The human skeleton is naturally a graph. Spatial-Temporal GCNs (ST-GCN, MS-G3D,
CTR-GCN) operate directly on the adjacency structure, capturing both within-frame
joint relationships and cross-frame temporal dynamics. This is the preferred
approach: the inductive bias of the graph structure should improve data efficiency
on the relatively small action dataset.

**Transformers**
Skeleton-based transformers (PoseFormer, ST-TR) treat joints as tokens and model
global dependencies via self-attention. Strong performance on large benchmarks
(NTU RGB+D) but may require more data.

**TCNs / LSTMs**
Simpler sequence models that flatten the skeleton into a feature vector per frame.
Lower capacity but faster to train and easier to interpret. Suitable as a baseline.

### 6.3 Proposed Labels

Initial action categories to recognize:
- **Strike types:** jab, cross, hook, uppercut, body shot, roundhouse kick, front kick, teep
- **Defensive actions:** slip, parry, block, clinch
- **Grappling:** takedown attempt, takedown defense, clinch work
- **Outcome:** landed clean, landed blocked, missed

---

## 7. Scoring Engine (Planned)

The scoring engine maps per-action classifications and counts into a 10-point must
system score for each round. The primary judging criteria (effective striking,
effective grappling, aggression, control) each map to quantifiable metrics:

- **Effective striking:** total significant strikes landed per round, weighted by
  strike type and landing quality (clean vs. partial vs. blocked)
- **Effective grappling:** takedown completions, submission attempts, control time
- **Aggression:** net strike output and forward pressure (estimated from position
  time series)
- **Control:** proportion of round spent dictating pace (attacker role in exchanges)

The scoring function is initially rule-based (hand-tuned weights per criterion) and
later replaced or augmented by a learned model trained against official judge scorecards.
