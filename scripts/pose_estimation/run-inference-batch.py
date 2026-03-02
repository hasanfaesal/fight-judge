"""
Batch pose inference using mmpose (ViTPose top-down approach).

Given a COCO-format annotations JSON (containing image metadata and fighter
bounding boxes) and a trained pose model checkpoint, this script runs
top-down keypoint inference on every image and saves per-image results.

Pipeline position:
    yolo-to-coco-bbox.py  →  [this script]  →  fix_annotations.py

For each image the script:
  1. Loads pre-annotated fighter bounding boxes from the COCO JSON
  2. Crops each bounding box and passes it through the pose model
  3. Saves the 17 COCO keypoints (x, y, score) to a JSON file per image
  4. Optionally saves an overlaid visualization image

Dependencies:
    mmpose, xtcocotools, tqdm, numpy, PyTorch

Usage:
    python run-inference-batch.py <pose_config> <pose_checkpoint> \\
        --img-root <image_dir> --json-file <coco_annotations.json> \\
        --out-json-root <output_json_dir> \\
        [--out-img-root <output_vis_dir>] [--save-vis] \\
        [--device cuda:0] [--kpt-thr 0.3]

Example:
    python run-inference-batch.py \\
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \\
        vitpose_base.pth \\
        --img-root mma-fighter-detection-dataset/train/images \\
        --json-file annotations/train_coco.json \\
        --out-json-root annotations/pose_results_train \\
        --device cuda:0
"""

import os
import json
import warnings
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from xtcocotools.coco import COCO

from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo


def save_keypoints_to_json(pose_results, output_path):
    """
    Save pose estimation results to a JSON file.

    Args:
        pose_results: List of pose results from inference
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in pose_results:
        json_result = {
            "keypoints": result[
                "keypoints"
            ].tolist(),  # Shape: [num_keypoints, 3] (x, y, score)
            "bbox": result["bbox"].tolist() if "bbox" in result else None,
        }
        json_results.append(json_result)

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)


def main():
    """
    Run batch inference on images with pre-existing bounding boxes.
    Saves both visualizations and keypoint predictions.
    """
    parser = ArgumentParser()
    parser.add_argument("pose_config", help="Config file for pose model")
    parser.add_argument("pose_checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--img-root", type=str, required=True, help="Image root directory"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        required=True,
        help="COCO format JSON file containing image info and bboxes",
    )
    parser.add_argument(
        "--out-img-root",
        type=str,
        default="",
        help="Root directory to save visualization images",
    )
    parser.add_argument(
        "--out-json-root",
        type=str,
        default="",
        help="Root directory to save keypoint JSON files",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for inference (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint score threshold for visualization",
    )
    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=0.0,
        help="Bounding box score threshold (set to 0 for pre-annotated boxes)",
    )
    parser.add_argument(
        "--radius", type=int, default=4, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Whether to show images (not recommended for batch processing)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        default=False,
        help="Whether to save visualization images",
    )

    args = parser.parse_args()

    # Create output directories
    if args.out_img_root:
        os.makedirs(args.out_img_root, exist_ok=True)
    if args.out_json_root:
        os.makedirs(args.out_json_root, exist_ok=True)

    # Load COCO format annotations
    print(f"Loading annotations from {args.json_file}")
    coco = COCO(args.json_file)

    # Initialize pose model
    print(f"Initializing pose model from {args.pose_config}")
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower()
    )

    # Get dataset info for visualization
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning,
        )
    else:
        dataset_info = DatasetInfo(dataset_info)

    # Get all image IDs
    img_keys = list(coco.imgs.keys())
    print(f"\nProcessing {len(img_keys)} images...")

    # Process each image
    for i, image_id in enumerate(tqdm(img_keys)):
        # Get image info
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image["file_name"])

        # Check if image exists
        if not os.path.exists(image_name):
            print(f"Warning: Image not found: {image_name}")
            continue

        # Get bounding box annotations for this image
        ann_ids = coco.getAnnIds(image_id)

        if len(ann_ids) == 0:
            print(f"Warning: No annotations found for image {image['file_name']}")
            continue

        # Prepare person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            person = {
                "bbox": ann["bbox"]  # COCO format: [x, y, width, height]
            }
            person_results.append(person)

        # Run pose estimation inference
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=args.bbox_thr,
            format="xywh",  # COCO bbox format
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        # Save keypoint predictions to JSON
        if args.out_json_root:
            json_filename = os.path.splitext(image["file_name"])[0] + "_keypoints.json"
            json_path = os.path.join(args.out_json_root, json_filename)
            save_keypoints_to_json(pose_results, json_path)

        # Save visualization if requested
        if args.save_vis and args.out_img_root:
            vis_filename = os.path.splitext(image["file_name"])[0] + "_vis.jpg"
            out_file = os.path.join(args.out_img_root, vis_filename)

            vis_pose_result(
                pose_model,
                image_name,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=args.show,
                out_file=out_file,
            )

    print("\n" + "=" * 50)
    print("Inference complete!")
    print(f"Processed {len(img_keys)} images")
    if args.out_json_root:
        print(f"Keypoint predictions saved to: {args.out_json_root}")
    if args.save_vis and args.out_img_root:
        print(f"Visualizations saved to: {args.out_img_root}")
    print("=" * 50)


if __name__ == "__main__":
    main()
