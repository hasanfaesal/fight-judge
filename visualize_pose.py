import cv2
import numpy as np
import os
import argparse
from pathlib import Path

# --- CONFIGURATION ---
# This section is for easy customization.

# This defines the connections between keypoints to form the skeleton.
# It assumes a 17-point model like COCO (nose, eyes, ears, shoulders, etc.).
# MODIFY THIS if your model has a different number or order of keypoints.
# The indices correspond to the order of keypoints in your label files (0-indexed).
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# Colors for the skeleton connections (in BGR format)
# You can add more colors if your skeleton has more connections.
LIMB_COLORS = [
    [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
    [0, 255, 0], [255, 128, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
    [0, 255, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
    [255, 128, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]
]

# Color for the keypoints (in BGR format)
KEYPOINT_COLOR = [0, 0, 255] # Red

# Color for the bounding box (in BGR format)
BBOX_COLOR = [255, 0, 0] # Blue

# --- SCRIPT LOGIC ---

def draw_annotations(image_path, label_path, output_path):
    """
    Reads an image and its corresponding label file, draws bounding boxes and
    keypoints, and saves the result.
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    img_h, img_w, _ = image.shape

    # Check if a label file exists
    if not label_path.exists():
        print(f"Info: No label file for {image_path.name}. Copying image as is.")
        cv2.imwrite(str(output_path), image)
        return

    # Read the label file
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue # Skip malformed lines

            # --- 1. Draw Bounding Box ---
            # Parse normalized bbox coordinates
            x_center, y_center, width, height = map(float, parts[1:5])

            # Denormalize coordinates
            abs_x_center = x_center * img_w
            abs_y_center = y_center * img_h
            abs_width = width * img_w
            abs_height = height * img_h

            # Calculate top-left and bottom-right corners
            x1 = int(abs_x_center - abs_width / 2)
            y1 = int(abs_y_center - abs_height / 2)
            x2 = int(abs_x_center + abs_width / 2)
            y2 = int(abs_y_center + abs_height / 2)

            # Draw the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), BBOX_COLOR, thickness=2)

            # --- 2. Draw Keypoints and Skeleton ---
            # Check if there are keypoints in the file
            if len(parts) > 5:
                keypoints_data = parts[5:]
                keypoints = []

                # Parse and denormalize keypoints
                for i in range(0, len(keypoints_data), 2):
                    kpt_x_norm = float(keypoints_data[i])
                    kpt_y_norm = float(keypoints_data[i+1])
                    kpt_x = int(kpt_x_norm * img_w)
                    kpt_y = int(kpt_y_norm * img_h)
                    keypoints.append((kpt_x, kpt_y))

                    # Draw the keypoint as a circle
                    cv2.circle(image, (kpt_x, kpt_y), radius=5, color=KEYPOINT_COLOR, thickness=-1)

                # Draw the skeleton lines
                for i, (p1_idx, p2_idx) in enumerate(SKELETON):
                    # Adjust indices to be 0-based from YOLO's 1-based standard
                    # YOLO format usually outputs keypoints in order. Here we assume the indices in SKELETON are 0-based.
                    # For COCO, keypoint indices are: 0-Nose, 1-LEye, 2-REye, ..., 16-RAnkle
                    # The SKELETON list uses different indices based on a common mapping.
                    # We map them back to the 0-16 range for indexing our `keypoints` list.
                    # Note: Ultralytics keypoint indices are: 0-nose, 1-left_eye, ..., 16-right_ankle.
                    # The SKELETON provided is compatible with this ordering.
                    p1_idx_mapped = p1_idx - 1
                    p2_idx_mapped = p2_idx - 1
                    
                    if p1_idx_mapped < len(keypoints) and p2_idx_mapped < len(keypoints):
                        p1 = keypoints[p1_idx_mapped]
                        p2 = keypoints[p2_idx_mapped]
                        
                        # Draw line if both points are detected (not at 0,0)
                        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                            color = LIMB_COLORS[i % len(LIMB_COLORS)]
                            cv2.line(image, p1, p2, color, thickness=2)


    # Save the final image
    cv2.imwrite(str(output_path), image)

def main():
    """
    Main function to parse arguments and run the visualization process.
    """
    parser = argparse.ArgumentParser(description="Visualize YOLO Pose annotations on a dataset.")
    parser.add_argument("data_dir", help="Path to the dataset split directory (e.g., 'test', 'train', or 'valid').")
    parser.add_argument("--output_dir", default="output_visualizations", help="Directory to save the visualized images.")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    images_path = data_path / "images"
    labels_path = data_path / "labels"
    output_path = Path(args.output_dir)

    # Validate paths
    if not images_path.is_dir() or not labels_path.is_dir():
        print(f"Error: 'images' or 'labels' subdirectories not found in '{data_path}'")
        return

    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"Saving visualized images to: {output_path.resolve()}")

    # Process each image
    image_files = sorted(list(images_path.glob("*")))
    for image_file in image_files:
        label_file = labels_path / (image_file.stem + ".txt")
        output_file = output_path / image_file.name

        print(f"Processing {image_file.name}...")
        draw_annotations(image_file, label_file, output_file)

    print("\nVisualization complete!")

if __name__ == "__main__":
    main()