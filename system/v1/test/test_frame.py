import cv2
import os
from pathlib import Path

def frame(n=0, indoors=True):

    # Get absolute path to sample_frames directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sample_frames_dir = os.path.join(test_dir, 'sample_frames')
    
    # Choose subdirectory based on indoors flag
    subdir = 'indoors' if indoors else 'outdoors'
    frames_dir = os.path.join(sample_frames_dir, subdir)
    
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Directory not found at {frames_dir}")
    
    # Load all image files
    all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not all_frames:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    
    if n < 0 or n >= len(all_frames):
        raise IndexError(f"Frame index {n} out of range (0-{len(all_frames)-1})")
    
    # Load and return the nth frame
    frame_path = os.path.join(frames_dir, all_frames[n])
    img = cv2.imread(frame_path)
    
    if img is None:
        raise ValueError(f"Failed to load image from {frame_path}")
    
    return img