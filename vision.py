#%%
import cv2
import numpy as np
from collections import deque

# Temporal smoothing: track last N frames for stable detection
class StableDetector:
    def __init__(self, history_size=10, stability_threshold=0.6):
        self.history = deque(maxlen=history_size)
        self.stability_threshold = stability_threshold
        self.locked_motif = ""
        self.lock_frames = 0
        self.LOCK_DURATION = 15  # Frames to hold a motif before allowing change

    def update(self, current_motif):
        self.history.append(current_motif)

        if len(self.history) < 3:
            return current_motif

        # Count occurrences of each motif in history
        motif_counts = {}
        for m in self.history:
            motif_counts[m] = motif_counts.get(m, 0) + 1

        # Find most common motif
        most_common = max(motif_counts, key=motif_counts.get)
        frequency = motif_counts[most_common] / len(self.history)

        # If we have a locked motif, keep it unless new one is very stable
        if self.locked_motif and self.lock_frames > 0:
            self.lock_frames -= 1
            # Only unlock if new motif is dominant and different
            if frequency >= 0.8 and most_common != self.locked_motif:
                self.locked_motif = most_common
                self.lock_frames = self.LOCK_DURATION
            return self.locked_motif

        # Lock onto stable motif
        if frequency >= self.stability_threshold:
            self.locked_motif = most_common
            self.lock_frames = self.LOCK_DURATION
            return most_common

        # Return locked motif if we have one, else most common
        return self.locked_motif if self.locked_motif else most_common

# Global detector instance
stable_detector = StableDetector()

def show_hsv_values(event, x, y, _flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print("HSV at", (x, y), ":", hsv_frame[y, x])

def is_semicircle(contour, min_area=500):
    """
    Detect if a contour is a semi-circular shape.
    Returns (is_semicircle, score) where score indicates confidence.
    Uses relaxed criteria to detect more semi-circles.
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return False, 0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False, 0

    # Circularity: semi-circles typically have circularity ~0.5-0.8
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    x, y, w, h = cv2.boundingRect(contour)

    # Aspect ratio: semi-circles are typically wider than tall (or vice versa)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

    # Extent: ratio of contour area to bounding rect area
    # Semi-circles fill ~50-65% of their bounding rectangle
    extent = area / (w * h) if w * h > 0 else 0

    # Convex hull analysis
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Fit ellipse if enough points (needs at least 5)
    ellipse_match = 0
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        ellipse_aspect = max(ellipse[1]) / min(ellipse[1]) if min(ellipse[1]) > 0 else 0
        # Semi-circles when fit to ellipse have aspect ratio ~1.2-3.5 (relaxed)
        if 1.2 < ellipse_aspect < 3.5:
            ellipse_match = 1

    # Relaxed semi-circle criteria to detect more shapes:
    # - Circularity between 0.3 and 0.95 (wider range)
    # - Aspect ratio between 1.1 and 3.5 (more elongation allowed)
    # - Extent between 0.35 and 0.85 (wider fill range)
    # - Solidity > 0.75 (slightly more irregular shapes allowed)

    is_semi = (
        0.3 < circularity < 0.95 and
        1.1 < aspect_ratio < 3.5 and
        0.35 < extent < 0.85 and
        solidity > 0.75
    )

    # Calculate confidence score
    score = 0
    if is_semi:
        # Ideal semi-circle values
        circ_score = max(0, 1 - abs(circularity - 0.6) / 0.4)
        aspect_score = max(0, 1 - abs(aspect_ratio - 1.8) / 1.0)
        extent_score = max(0, 1 - abs(extent - 0.55) / 0.3)
        score = area * solidity * (circ_score + aspect_score + extent_score + ellipse_match) / 4

    return is_semi, score


def get_dominant_color(contour, mask_green, mask_purple):
    """
    Determine color by sampling multiple points in the contour region.
    Returns 'G', 'P', or None based on majority voting.
    """
    x, y, w, h = cv2.boundingRect(contour)

    # Create a mask for just this contour
    contour_mask = np.zeros(mask_green.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    # Count pixels that match each color within the contour
    green_pixels = cv2.countNonZero(cv2.bitwise_and(mask_green, contour_mask))
    purple_pixels = cv2.countNonZero(cv2.bitwise_and(mask_purple, contour_mask))

    total = green_pixels + purple_pixels
    if total == 0:
        return None

    # Require at least 30% dominance to classify
    if green_pixels > purple_pixels and green_pixels / total > 0.3:
        return 'G'
    elif purple_pixels > green_pixels and purple_pixels / total > 0.3:
        return 'P'

    return None


def runPipeline(image, use_stabilization=True):
    global stable_detector
    output = image.copy()
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV ranges - tuned for your balls
    lower_green, upper_green = (72, 101, 24), (118, 19, 223)  # Lowered S,V for better green pickup
    lower_purple, upper_purple = (134, 50, 80), (170, 255, 255)

    mask_green_raw = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_purple_raw = cv2.inRange(img_hsv, lower_purple, upper_purple)

    # Process each color mask SEPARATELY with appropriate morphology
    kernel = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    # Green mask cleaning
    mask_green = cv2.morphologyEx(mask_green_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Purple mask cleaning
    mask_purple = cv2.morphologyEx(mask_purple_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Combine for contour detection
    mask_combined = cv2.bitwise_or(mask_green, mask_purple)

    # Additional cleanup on combined
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_large)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-filter by area
    MIN_CONTOUR_AREA = 600
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    detected_shapes = []
    green_count = 0
    purple_count = 0

    for contour in contours:
        is_semi, score = is_semicircle(contour, min_area=MIN_CONTOUR_AREA)

        if is_semi:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2

            # Use robust color detection (multiple pixel sampling)
            color = get_dominant_color(contour, mask_green, mask_purple)

            if color == 'G':
                green_count += 1
                detected_shapes.append((center_x, 'G', contour, x, y, score))
                cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
                cv2.putText(output, f'G{green_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif color == 'P':
                purple_count += 1
                detected_shapes.append((center_x, 'P', contour, x, y, score))
                cv2.drawContours(output, [contour], -1, (255, 0, 255), 3)
                cv2.putText(output, f'P{purple_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    detected_shapes.sort(key=lambda b: b[0])
    raw_motif = ''.join([b[1] for b in detected_shapes])

    # Apply temporal stabilization
    if use_stabilization:
        stable_motif = stable_detector.update(raw_motif)
    else:
        stable_motif = raw_motif

    total_count = len(stable_motif)

    # Show both raw and stable motif for debugging
    cv2.putText(output, f"Stable: {stable_motif} ({total_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(output, f"Raw: {raw_motif}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(output, f"G:{green_count} P:{purple_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return stable_motif, output, mask_clean, mask_green, mask_purple


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Color Output")

    last_printed = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stable_motif, output, mask_combined, mask_green, mask_purple = runPipeline(frame)

        # Only print when motif changes (reduces console spam)
        if stable_motif and stable_motif != last_printed:
            print(f"Motif: {stable_motif}")
            last_printed = stable_motif

        cv2.setMouseCallback("Color Output", show_hsv_values, frame)

        cv2.imshow("Color Output", output)
        cv2.imshow("Mask (B&W)", mask_combined)
        cv2.imshow("Green Mask", mask_green)
        cv2.imshow("Purple Mask", mask_purple)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# %%
# HSV Value Checker - Click on any pixel to see its HSV values
# Run this cell to fine-tune your color thresholds

def hsv_checker():
    cap = cv2.VideoCapture(0)

    hsv_values = []

    def on_click(event, x, y, _flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame, hsv_frame = param
            hsv = hsv_frame[y, x]
            bgr = frame[y, x]
            print(f"\n{'='*40}")
            print(f"Position: ({x}, {y})")
            print(f"HSV: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
            print(f"BGR: B={bgr[0]}, G={bgr[1]}, R={bgr[2]}")
            hsv_values.append(hsv)

            if len(hsv_values) > 1:
                h_vals = [v[0] for v in hsv_values]
                s_vals = [v[1] for v in hsv_values]
                v_vals = [v[2] for v in hsv_values]
                print(f"\n--- Suggested range from {len(hsv_values)} samples ---")
                print(f"Lower: ({min(h_vals)-5}, {max(0, min(s_vals)-20)}, {max(0, min(v_vals)-20)})")
                print(f"Upper: ({max(h_vals)+5}, {min(255, max(s_vals)+20)}, {min(255, max(v_vals)+20)})")

    cv2.namedWindow("HSV Checker")
    print("HSV Checker Started")
    print("- LEFT CLICK on colors to sample HSV values")
    print("- Press 'c' to clear samples")
    print("- Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback("HSV Checker", on_click, (frame, hsv_frame))

        # Draw instructions on frame
        cv2.putText(frame, "Click to sample HSV | 'c'=clear | 'q'=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(hsv_values)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("HSV Checker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            hsv_values.clear()
            print("\nSamples cleared!")

    cap.release()
    cv2.destroyAllWindows()

    if hsv_values:
        print(f"\n{'='*40}")
        print("FINAL SUMMARY")
        print(f"{'='*40}")
        h_vals = [v[0] for v in hsv_values]
        s_vals = [v[1] for v in hsv_values]
        v_vals = [v[2] for v in hsv_values]
        print(f"Collected {len(hsv_values)} samples")
        print(f"H range: {min(h_vals)} - {max(h_vals)}")
        print(f"S range: {min(s_vals)} - {max(s_vals)}")
        print(f"V range: {min(v_vals)} - {max(v_vals)}")
        print(f"\nSuggested threshold:")
        print(f"lower = ({max(0, min(h_vals)-5)}, {max(0, min(s_vals)-20)}, {max(0, min(v_vals)-20)})")
        print(f"upper = ({min(179, max(h_vals)+5)}, {min(255, max(s_vals)+20)}, {min(255, max(v_vals)+20)})")

# Uncomment to run:
hsv_checker()
# %%
