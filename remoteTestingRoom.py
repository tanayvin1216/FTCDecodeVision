#%%
# Remote Testing Room - Capture frames for Claude to analyze
# Press 's' to save a snapshot, 'q' to quit

import cv2
import numpy as np
import os
from datetime import datetime

# Import detection from vision.py
from vision import runPipeline, PURPLE_YCRCB_LOWER, PURPLE_YCRCB_UPPER, GREEN_HSV_LOWER, GREEN_HSV_UPPER, GREEN_YCRCB_LOWER, GREEN_YCRCB_UPPER

# Create output directory
OUTPUT_DIR = "/Users/tanayvinaykya/Desktop/robotics/vision/test_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_debug_frames(frame, frame_id):
    """Save frame and all debug masks for Claude to analyze"""

    # Convert to color spaces
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Create masks (same as vision.py)
    mask_green_hsv = cv2.inRange(img_hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    mask_green_ycrcb = cv2.inRange(img_ycrcb, GREEN_YCRCB_LOWER, GREEN_YCRCB_UPPER)
    mask_green = cv2.bitwise_and(mask_green_hsv, mask_green_ycrcb)

    mask_purple = cv2.inRange(img_ycrcb, PURPLE_YCRCB_LOWER, PURPLE_YCRCB_UPPER)

    # Run full pipeline
    motif, output, mask_combined, _, _ = runPipeline(frame)

    # Find contours for debug
    green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Print contour info
    print(f"\n{'='*50}")
    print(f"Frame {frame_id} Analysis:")
    print(f"{'='*50}")
    print(f"Detected motif: {motif}")
    print(f"\nGreen contours found: {len(green_contours)}")
    for i, c in enumerate(green_contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"  G{i+1}: area={area:.0f}, pos=({x},{y}), size={w}x{h}")

    print(f"\nPurple contours found: {len(purple_contours)}")
    for i, c in enumerate(purple_contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"  P{i+1}: area={area:.0f}, pos=({x},{y}), size={w}x{h}")

    # Save all images
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_1_original.png", frame)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_2_output.png", output)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_3_mask_green.png", mask_green)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_4_mask_purple.png", mask_purple)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_5_mask_combined.png", mask_combined)

    print(f"\nSaved to {OUTPUT_DIR}/{frame_id}_*.png")
    print(f"Share these images with Claude for analysis!")

    return motif

def main():
    cap = cv2.VideoCapture(0)

    print("="*50)
    print("REMOTE TESTING ROOM")
    print("="*50)
    print("Press 's' to save snapshot for Claude analysis")
    print("Press 'q' to quit")
    print("="*50)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pipeline for live preview
        motif, output, mask_combined, mask_green, mask_purple = runPipeline(frame)

        # Show windows
        cv2.imshow("Live Output", output)
        cv2.imshow("Green Mask", mask_green)
        cv2.imshow("Purple Mask", mask_purple)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            frame_count += 1
            timestamp = datetime.now().strftime("%H%M%S")
            frame_id = f"frame_{timestamp}_{frame_count}"
            save_debug_frames(frame, frame_id)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# %%
# Quick single-frame test - run this cell to capture one frame
def capture_single_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        timestamp = datetime.now().strftime("%H%M%S")
        save_debug_frames(frame, f"single_{timestamp}")
        return frame
    return None

# Uncomment to capture:
# capture_single_frame()
