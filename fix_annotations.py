import json
import os

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

def convert_bbox_to_coco(bbox):
    """Convert bbox from [x1, y1, x2, y2] to COCO format [x, y, width, height]"""
    if len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        width = x2 - x1
        height = y2 - y1
        return [float(x1), float(y1), float(width), float(height)]
    return [0.0, 0.0, 0.0, 0.0]

def count_visible_keypoints(keypoints):
    """Count keypoints with visibility > 0"""
    count = 0
    for i in range(2, len(keypoints), 3):
        if keypoints[i] > 0:
            count += 1
    return count

def fix_coco_annotations(input_json, output_json):
    """Fix COCO annotation format issues"""
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Processing: {input_json}")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    
    # Fix categories - use proper keypoint names
    data['categories'] = [{
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
        'keypoints': COCO_KEYPOINT_NAMES,
        'skeleton': COCO_SKELETON
    }]
    
    # Add info section
    data['info'] = {
        'description': 'MMA Fighter Pose Dataset',
        'version': '1.0',
        'year': 2025,
        'contributor': 'Your Name',
        'date_created': '2025-12-05'
    }
    
    # Add licenses section
    data['licenses'] = [{
        'id': 1,
        'name': 'Attribution-NonCommercial License',
        'url': 'https://creativecommons.org/licenses/by-nc/4.0/'
    }]
    
    # Fix each annotation
    bbox_fixed_count = 0
    for ann in data['annotations']:
        # Convert bbox from [x1,y1,x2,y2] to [x,y,w,h]
        old_bbox = ann['bbox']
        new_bbox = convert_bbox_to_coco(old_bbox)
        bbox_fixed_count += 1
        ann['bbox'] = new_bbox
        
        # Add area (width * height)
        ann['area'] = ann['bbox'][2] * ann['bbox'][3]
        
        # Add iscrowd
        ann['iscrowd'] = 0
        
        # Add num_keypoints
        if 'keypoints' in ann:
            ann['num_keypoints'] = count_visible_keypoints(ann['keypoints'])
    
    print(f"  Bboxes converted: {bbox_fixed_count}")
    
    # Save fixed annotations
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved to: {output_json}\n")

def main():
    annotations_dir = 'annotations'
    
    # Create backup directory
    backup_dir = 'annotations_backup'
    os.makedirs(backup_dir, exist_ok=True)
    
    files = [
        'pose_results_train.json',
        'pose_results_valid.json', 
        'pose_results_test.json'
    ]
    
    for filename in files:
        input_path = os.path.join(annotations_dir, filename)
        backup_path = os.path.join(backup_dir, filename)
        
        if os.path.exists(input_path):
            # Backup original
            with open(input_path, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)
            print(f"Backed up: {backup_path}")
            
            # Fix and overwrite
            fix_coco_annotations(input_path, input_path)
    
    print("=" * 50)
    print("All annotations fixed!")
    print(f"Backups saved in: {backup_dir}/")
    
    # Verify fix
    print("\n=== Verification ===")
    with open('annotations/pose_results_train.json', 'r') as f:
        data = json.load(f)
    
    ann = data['annotations'][0]
    img = data['images'][0]
    bbox = ann['bbox']
    print(f"Image size: {img['width']}x{img['height']}")
    print(f"Sample bbox [x,y,w,h]: {bbox}")
    print(f"  x2 = x + w = {bbox[0] + bbox[2]:.1f} (should be <= {img['width']})")
    print(f"  y2 = y + h = {bbox[1] + bbox[3]:.1f} (should be <= {img['height']})")

if __name__ == '__main__':
    main()