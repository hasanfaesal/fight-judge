# fight-judge
This repository is a work in progress.

The end goal is to make an AI judge that can score boxing and MMA fights according to official rules.

## Potential Features:
- Detect fighters and their positions in the ring/octagon and apply ReID if the model fails
- Recognize strikes like punches, kicks, elbows, knees, etc.
- Classify strikes into categories (e.g., jab, cross, hook, uppercut for punches)
- Determine whether strikes land cleanly or are blocked/missed
- Track ground game actions like takedowns, submissions, and control time

## Current Progress:
I've finetuned a pose estimation model using a top down approach to detect keypoints of fighters in boxing and MMA videos. The model is trained on a custom dataset with annotated keypoints.

Dataset: https://data.mendeley.com/datasets/c456bnk8bm/1