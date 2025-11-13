import cv2
import os
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(output_folder):
    """Set up logging to both file and console."""
    log_filename = os.path.join(output_folder, f"extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger('FrameExtractor')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def extract_video_frames(video_path, output_folder="extracted_frames"):
    """
    Extract all frames from a video file and save them as images.
    
    Args:
        video_path: Path to the input video file
        output_folder: Directory where frames will be saved
    """
    
    # Create output directory first
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_folder)
    logger.info("=" * 60)
    logger.info("VIDEO FRAME EXTRACTION STARTED")
    logger.info("=" * 60)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file '{video_path}' not found!")
        return
    
    logger.info(f"Input video file: {video_path}")
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        logger.error("Could not open video file!")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    codec_fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    
    # Display and log video information
    logger.info("=" * 60)
    logger.info("VIDEO PROPERTIES")
    logger.info("=" * 60)
    logger.info(f"Resolution:       {width} x {height} pixels")
    logger.info(f"Frame Rate (FPS): {fps:.2f}")
    logger.info(f"Total Frames:     {frame_count}")
    logger.info(f"Duration:         {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"Codec (FourCC):   {codec_fourcc}")
    logger.info(f"File Size:        {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    logger.info("=" * 60)
    
    # Log extraction settings
    logger.info(f"Output folder: {os.path.abspath(output_folder)}")
    logger.info(f"Frame format: JPEG")
    logger.info("Starting frame extraction...")
    logger.info("")
    
    frame_number = 0
    extracted_count = 0
    start_time = datetime.now()
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Save frame as image
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        extracted_count += 1
        
        # Progress indicator
        if (frame_number + 1) % 100 == 0 or (frame_number + 1) == frame_count:
            progress = ((frame_number + 1) / frame_count) * 100
            logger.info(f"Progress: {frame_number + 1}/{frame_count} frames ({progress:.1f}%)")
        
        frame_number += 1
    
    # Release video object
    video.release()
    
    # Calculate extraction time
    end_time = datetime.now()
    extraction_duration = (end_time - start_time).total_seconds()
    
    # Log completion summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total frames extracted: {extracted_count}")
    logger.info(f"Extraction time: {extraction_duration:.2f} seconds ({extraction_duration/60:.2f} minutes)")
    logger.info(f"Average speed: {extracted_count/extraction_duration:.2f} frames/second")
    logger.info(f"Frames saved in: {os.path.abspath(output_folder)}/")
    logger.info("=" * 60)
    
    print(f"\nLog file saved: {os.path.join(output_folder, [h.baseFilename for h in logger.handlers if isinstance(h, logging.FileHandler)][0])}")


if __name__ == "__main__":
    # Get video path from user
    video_path = input("Enter the path to your video file: ").strip()
    # /home/hasan-faisal/code/AIJudge/results/pose_estimation_videos_20251108_175541/24_trimmed_fixed_pose.mp4



    # Optional: Get custom output folder
    custom_output = input("Enter output folder name (press Enter for 'extracted_frames'): ").strip()
    output_folder = custom_output if custom_output else "extracted_frames"
    
    # Extract frames
    extract_video_frames(video_path, output_folder)