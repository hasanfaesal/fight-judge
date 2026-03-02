"""
Convert YOLO format bounding boxes to COCO format JSON.

This script converts the MMA fighter detection dataset (YOLO .txt labels) into
COCO-format annotation JSON files required by mmpose's top-down inference pipeline.

YOLO format:  class_id  x_center  y_center  width  height  (all normalized to [0,1])
COCO format:  {"bbox": [x_min, y_min, width, height]}  (absolute pixel coordinates)

Pipeline position:
    [Roboflow dataset download]  →  [this script]  →  run-inference-batch.py

The script expects the standard Roboflow directory layout:
    mma-fighter-detection-dataset/
        train/images/   train/labels/
        valid/images/   valid/labels/
        test/images/    test/labels/

Output:
    annotations/train_coco.json
    annotations/valid_coco.json
    annotations/test_coco.json

Usage:
    python yolo-to-coco-bbox.py
    (No arguments needed — paths are configured in __main__ block below)
"""

import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm


def convert_yolo_to_coco(image_dir, label_dir, output_json, categories=None):
    """
    Convert YOLO format annotations to COCO format JSON.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO .txt annotation files
        output_json: Path to save the output COCO JSON file
        categories: List of category names (default: ["person"])
    """

    if categories is None:
        categories = ["person"]

    # Initialize COCO format dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(categories)
        ],
    }

    image_id = 0
    annotation_id = 0

    # Get all image files - support multiple extensions
    valid_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if any(f.endswith(ext) for ext in valid_extensions)
        ]
    )

    print(f"Processing {len(image_files)} images from {image_dir}")

    for img_file in tqdm(image_files, desc="Converting"):
        # Read image to get dimensions
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue

        height, width = img.shape[:2]

        # Add image info to COCO format
        coco_format["images"].append(
            {"id": image_id, "file_name": img_file, "width": width, "height": height}
        )

        # Read corresponding YOLO annotation file
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: No annotation file found for {img_file}")
            image_id += 1
            continue

        # Parse YOLO annotations
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Parse YOLO format: class_id x_center y_center width height
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            # Convert from normalized YOLO format to absolute COCO format
            # COCO uses [x_min, y_min, width, height] where x_min, y_min is top-left corner
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            abs_width = bbox_width * width
            abs_height = bbox_height * height

            # Clamp values to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            abs_width = min(abs_width, width - x_min)
            abs_height = min(abs_height, height - y_min)

            # Add annotation to COCO format (COCO uses 1-based category IDs)
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,  # YOLO is 0-indexed, COCO is 1-indexed
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(abs_width),
                        float(abs_height),
                    ],
                    "area": float(abs_width * abs_height),
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    # Save to JSON file
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Conversion complete!")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in coco_format['categories']]}")
    print(f"Saved to: {output_json}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # Hardcoded paths - no command line arguments needed
    base_dir = "mma-fighter-detection-dataset"
    categories = ["person"]

    # Create annotations directory
    os.makedirs("annotations", exist_ok=True)

    # Convert training set
    print("\n" + "=" * 60)
    print("CONVERTING TRAINING SET")
    print("=" * 60)
    convert_yolo_to_coco(
        image_dir=f"{base_dir}/train/images",
        label_dir=f"{base_dir}/train/labels",
        output_json="annotations/train_coco.json",
        categories=categories,
    )

    # Convert validation set
    print("\n" + "=" * 60)
    print("CONVERTING VALIDATION SET")
    print("=" * 60)
    convert_yolo_to_coco(
        image_dir=f"{base_dir}/valid/images",
        label_dir=f"{base_dir}/valid/labels",
        output_json="annotations/valid_coco.json",
        categories=categories,
    )

    # Convert test set
    print("\n" + "=" * 60)
    print("CONVERTING TEST SET")
    print("=" * 60)
    convert_yolo_to_coco(
        image_dir=f"{base_dir}/test/images",
        label_dir=f"{base_dir}/test/labels",
        output_json="annotations/test_coco.json",
        categories=categories,
    )

    print("\n" + "=" * 60)
    print("ALL CONVERSIONS COMPLETE!")
    print("=" * 60)
    print("Output files:")
    print("  - annotations/train_coco.json")
    print("  - annotations/valid_coco.json")
    print("  - annotations/test_coco.json")
    print("=" * 60)
