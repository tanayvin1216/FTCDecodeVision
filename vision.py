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
        ycrcb_frame = cv2.cvtColor(param, cv2.COLOR_BGR2YCrCb)
        print(f"HSV: {hsv_frame[y, x]} | YCrCb: {ycrcb_frame[y, x]} at ({x}, {y})")

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


# ===== GREEN THRESHOLDS (TIGHTENED) =====
# HSV - tighter saturation requirement
GREEN_HSV_LOWER = np.array([65, 100, 40])
GREEN_HSV_UPPER = np.array([90, 255, 220])
# YCrCb - keeps green, rejects similar colors
GREEN_YCRCB_LOWER = np.array([50, 35, 60])
GREEN_YCRCB_UPPER = np.array([160, 100, 145])

# ===== PURPLE THRESHOLDS (YCrCb ONLY) =====
PURPLE_YCRCB_LOWER = np.array([80, 135, 130])
PURPLE_YCRCB_UPPER = np.array([170, 175, 180])

# ===== DETECTION SETTINGS =====
MIN_BALL_AREA = 1000  # Minimum contour area (filters small noise)
MAX_BALL_AREA = 15000  # Maximum contour area (filters large blobs)


def separate_touching_balls(mask, min_area=800, expected_ball_area=2000):
    """
    Use watershed + erosion to aggressively separate touching balls.
    Returns a list of separated contours.
    """
    if cv2.countNonZero(mask) == 0:
        return []

    kernel_small = np.ones((3, 3), np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    # First pass: Try erosion-based separation (works well for slightly touching balls)
    # Erode to break connections between balls (3 iterations for better separation)
    eroded = cv2.erode(mask, kernel, iterations=3)

    # Find contours after erosion
    eroded_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If erosion created multiple contours, use those
    if len(eroded_contours) > 1:
        result_contours = []
        for contour in eroded_contours:
            area = cv2.contourArea(contour)
            if area > min_area * 0.5:  # Lower threshold since erosion shrinks
                # Dilate back to approximate original size
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                dilated = cv2.dilate(contour_mask, kernel, iterations=3)
                # Intersect with original mask to stay within bounds
                final_mask = cv2.bitwise_and(dilated, mask)
                final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for fc in final_contours:
                    if cv2.contourArea(fc) > min_area:
                        result_contours.append(fc)
        if result_contours:
            return result_contours

    # Second pass: Distance transform + watershed for heavily merged balls
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    if dist_transform.max() == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > min_area]

    # Very aggressive threshold (0.2) to find more seed points for better separation
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Additional erosion on sure_fg to separate close centers
    sure_fg = cv2.erode(sure_fg, kernel_small, iterations=1)

    # Find sure background
    sure_bg = cv2.dilate(mask, kernel_small, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # If only one label found, try even more aggressive approach
    if num_labels <= 2:  # 1 for background + 1 for foreground
        # Try with even lower threshold
        _, sure_fg2 = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
        sure_fg2 = np.uint8(sure_fg2)
        sure_fg2 = cv2.erode(sure_fg2, kernel, iterations=2)
        num_labels, markers = cv2.connectedComponents(sure_fg2)
        unknown = cv2.subtract(sure_bg, sure_fg2)

    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_color, markers)

    # Extract contours from each watershed region
    separated_contours = []
    unique_labels = np.unique(markers)

    for label in unique_labels:
        if label <= 1:
            continue

        label_mask = np.zeros(mask.shape, dtype=np.uint8)
        label_mask[markers == label] = 255

        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # If still too large, force split with heavy erosion
                if area > expected_ball_area * 2:
                    heavy_eroded = cv2.erode(label_mask, kernel, iterations=3)
                    sub_contours, _ = cv2.findContours(heavy_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(sub_contours) > 1:
                        for sc in sub_contours:
                            if cv2.contourArea(sc) > min_area * 0.3:
                                separated_contours.append(sc)
                    else:
                        separated_contours.append(contour)
                else:
                    separated_contours.append(contour)

    # Fallback: if watershed didn't help, return original contours
    if not separated_contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > min_area]

    return separated_contours


def runPipeline(image, use_stabilization=True):
    global stable_detector
    output = image.copy()

    # Convert to color spaces
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # ===== GREEN DETECTION (HSV + YCrCb dual mask) =====
    mask_green_hsv = cv2.inRange(img_hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    mask_green_ycrcb = cv2.inRange(img_ycrcb, GREEN_YCRCB_LOWER, GREEN_YCRCB_UPPER)
    mask_green_raw = cv2.bitwise_and(mask_green_hsv, mask_green_ycrcb)

    # ===== PURPLE DETECTION (YCrCb ONLY) =====
    mask_purple_raw = cv2.inRange(img_ycrcb, PURPLE_YCRCB_LOWER, PURPLE_YCRCB_UPPER)

    # ===== NO MORPHOLOGY - RAW MASKS =====
    # YCrCb masks are already clean, morphology was causing issues
    mask_green = mask_green_raw.copy()
    mask_purple = mask_purple_raw.copy()

    # ===== SIMPLE CONTOUR DETECTION =====
    green_contours_raw, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours_raw, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area - lowered minimum since balls have holes
    green_contours = [c for c in green_contours_raw if 300 < cv2.contourArea(c) < 20000]
    purple_contours = [c for c in purple_contours_raw if 300 < cv2.contourArea(c) < 20000]

    # Combine masks for visualization
    mask_clean = cv2.bitwise_or(mask_green, mask_purple)

    detected_shapes = []
    green_count = 0
    purple_count = 0

    # Process green contours (already filtered, from green mask = definitely green)
    for contour in green_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        area = cv2.contourArea(contour)

        green_count += 1
        detected_shapes.append((center_x, 'G', contour, x, y, area))
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
        cv2.putText(output, f'G{green_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Process purple contours (already filtered, from purple mask = definitely purple)
    for contour in purple_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        area = cv2.contourArea(contour)

        purple_count += 1
        detected_shapes.append((center_x, 'P', contour, x, y, area))
        cv2.drawContours(output, [contour], -1, (255, 0, 255), 3)
        cv2.putText(output, f'P{purple_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Sort by x position (left to right)
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

    print("="*50)
    print("BALL DETECTION")
    print("Green: HSV + YCrCb dual mask")
    print("Purple: YCrCb only")
    print("="*50)

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
    ycrcb_values = []

    def on_click(event, x, y, _flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame, hsv_frame, ycrcb_frame = param
            hsv = hsv_frame[y, x]
            ycrcb = ycrcb_frame[y, x]
            bgr = frame[y, x]
            print(f"\n{'='*40}")
            print(f"Position: ({x}, {y})")
            print(f"HSV: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
            print(f"YCrCb: Y={ycrcb[0]}, Cr={ycrcb[1]}, Cb={ycrcb[2]}")
            print(f"BGR: B={bgr[0]}, G={bgr[1]}, R={bgr[2]}")
            hsv_values.append(hsv)
            ycrcb_values.append(ycrcb)

            if len(hsv_values) > 1:
                h_vals = [v[0] for v in hsv_values]
                s_vals = [v[1] for v in hsv_values]
                v_vals = [v[2] for v in hsv_values]
                y_vals = [v[0] for v in ycrcb_values]
                cr_vals = [v[1] for v in ycrcb_values]
                cb_vals = [v[2] for v in ycrcb_values]
                print(f"\n--- Suggested HSV range from {len(hsv_values)} samples ---")
                print(f"Lower: ({min(h_vals)-5}, {max(0, min(s_vals)-20)}, {max(0, min(v_vals)-20)})")
                print(f"Upper: ({max(h_vals)+5}, {min(255, max(s_vals)+20)}, {min(255, max(v_vals)+20)})")
                print(f"\n--- Suggested YCrCb range from {len(ycrcb_values)} samples ---")
                print(f"Lower: ({max(0, min(y_vals)-20)}, {max(0, min(cr_vals)-10)}, {max(0, min(cb_vals)-10)})")
                print(f"Upper: ({min(255, max(y_vals)+20)}, {min(255, max(cr_vals)+10)}, {min(255, max(cb_vals)+10)})")

    cv2.namedWindow("HSV Checker")
    print("HSV/YCrCb Checker Started")
    print("- LEFT CLICK on colors to sample values")
    print("- Press 'c' to clear samples")
    print("- Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cv2.setMouseCallback("HSV Checker", on_click, (frame, hsv_frame, ycrcb_frame))

        # Draw instructions on frame
        cv2.putText(frame, "Click to sample HSV/YCrCb | 'c'=clear | 'q'=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(hsv_values)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("HSV Checker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            hsv_values.clear()
            ycrcb_values.clear()
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
        y_vals = [v[0] for v in ycrcb_values]
        cr_vals = [v[1] for v in ycrcb_values]
        cb_vals = [v[2] for v in ycrcb_values]
        print(f"Collected {len(hsv_values)} samples")
        print(f"\nHSV ranges:")
        print(f"H range: {min(h_vals)} - {max(h_vals)}")
        print(f"S range: {min(s_vals)} - {max(s_vals)}")
        print(f"V range: {min(v_vals)} - {max(v_vals)}")
        print(f"\nSuggested HSV threshold:")
        print(f"lower = ({max(0, min(h_vals)-5)}, {max(0, min(s_vals)-20)}, {max(0, min(v_vals)-20)})")
        print(f"upper = ({min(179, max(h_vals)+5)}, {min(255, max(s_vals)+20)}, {min(255, max(v_vals)+20)})")
        print(f"\nYCrCb ranges:")
        print(f"Y range: {min(y_vals)} - {max(y_vals)}")
        print(f"Cr range: {min(cr_vals)} - {max(cr_vals)}")
        print(f"Cb range: {min(cb_vals)} - {max(cb_vals)}")
        print(f"\nSuggested YCrCb threshold:")
        print(f"lower = ({max(0, min(y_vals)-20)}, {max(0, min(cr_vals)-10)}, {max(0, min(cb_vals)-10)})")
        print(f"upper = ({min(255, max(y_vals)+20)}, {min(255, max(cr_vals)+10)}, {min(255, max(cb_vals)+10)})")

# Uncomment to run:
# hsv_checker()

# %%
# GREEN DUAL-MASK TUNER
# Tune both HSV and YCrCb thresholds with real-time visual feedback
import cv2
import numpy as np
from collections import deque
def green_tuner():
    """
    Interactive tuner for green ball detection using dual HSV + YCrCb masks.
    Shows individual masks and the combined AND result.
    Press 's' to save values, 'q' to quit.
    """
    cap = cv2.VideoCapture(0)

    # Create windows
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original + Detection")
    cv2.namedWindow("HSV Mask")
    cv2.namedWindow("YCrCb Mask")
    cv2.namedWindow("Combined (AND)")

    # HSV trackbars (H: 0-179, S: 0-255, V: 0-255)
    cv2.createTrackbar("H_min", "Controls", 35, 179, lambda x: None)
    cv2.createTrackbar("H_max", "Controls", 85, 179, lambda x: None)
    cv2.createTrackbar("S_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("S_max", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("V_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("V_max", "Controls", 255, 255, lambda x: None)

    # YCrCb trackbars (all 0-255)
    cv2.createTrackbar("Y_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("Y_max", "Controls", 220, 255, lambda x: None)
    cv2.createTrackbar("Cr_min", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("Cr_max", "Controls", 115, 255, lambda x: None)
    cv2.createTrackbar("Cb_min", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("Cb_max", "Controls", 135, 255, lambda x: None)

    # Morphology controls
    cv2.createTrackbar("Morph_kernel", "Controls", 5, 15, lambda x: None)
    cv2.createTrackbar("Open_iter", "Controls", 1, 5, lambda x: None)
    cv2.createTrackbar("Close_iter", "Controls", 2, 5, lambda x: None)

    # Step size for keyboard adjustments
    STEP_H = 2
    STEP_SV = 5
    STEP_YCRCB = 5

    print("="*60)
    print("GREEN DUAL-MASK TUNER")
    print("="*60)
    print("Use KEYBOARD to tune (sliders may not work on macOS)")
    print("Goal: Green ball = WHITE, everything else = BLACK")
    print("")
    print("KEYBOARD CONTROLS:")
    print("  HSV Hue:        1/! = H_min -/+     2/@ = H_max -/+")
    print("  HSV Sat:        3/# = S_min -/+     4/$ = S_max -/+")
    print("  HSV Val:        5/% = V_min -/+     6/^ = V_max -/+")
    print("  YCrCb Y:        7/& = Y_min -/+     8/* = Y_max -/+")
    print("  YCrCb Cr:       9/( = Cr_min -/+    0/) = Cr_max -/+")
    print("  YCrCb Cb:       -/_ = Cb_min -/+    =/+ = Cb_max -/+")
    print("")
    print("  s = Save values    r = Reset    q = Quit")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get trackbar values
        h_min = cv2.getTrackbarPos("H_min", "Controls")
        h_max = cv2.getTrackbarPos("H_max", "Controls")
        s_min = cv2.getTrackbarPos("S_min", "Controls")
        s_max = cv2.getTrackbarPos("S_max", "Controls")
        v_min = cv2.getTrackbarPos("V_min", "Controls")
        v_max = cv2.getTrackbarPos("V_max", "Controls")

        y_min = cv2.getTrackbarPos("Y_min", "Controls")
        y_max = cv2.getTrackbarPos("Y_max", "Controls")
        cr_min = cv2.getTrackbarPos("Cr_min", "Controls")
        cr_max = cv2.getTrackbarPos("Cr_max", "Controls")
        cb_min = cv2.getTrackbarPos("Cb_min", "Controls")
        cb_max = cv2.getTrackbarPos("Cb_max", "Controls")

        morph_k = max(1, cv2.getTrackbarPos("Morph_kernel", "Controls"))
        open_iter = max(1, cv2.getTrackbarPos("Open_iter", "Controls"))
        close_iter = max(1, cv2.getTrackbarPos("Close_iter", "Controls"))

        # Convert color spaces
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Create masks
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

        lower_ycrcb = np.array([y_min, cr_min, cb_min])
        upper_ycrcb = np.array([y_max, cr_max, cb_max])
        mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)

        # Combined mask (AND operation)
        mask_combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Apply morphology to combined
        kernel = np.ones((morph_k, morph_k), np.uint8)
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        # Find contours and draw on output
        output = frame.copy()
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        for contour in contours:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cv2.putText(output, f"A:{area}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count pixels for feedback
        hsv_pixels = cv2.countNonZero(mask_hsv)
        ycrcb_pixels = cv2.countNonZero(mask_ycrcb)
        combined_pixels = cv2.countNonZero(mask_clean)

        # Add info text - current values and pixel counts
        cv2.putText(output, f"HSV: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(output, f"YCrCb: Y[{y_min}-{y_max}] Cr[{cr_min}-{cr_max}] Cb[{cb_min}-{cb_max}]",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(output, f"Pixels: HSV={hsv_pixels} YCrCb={ycrcb_pixels} Combined={combined_pixels} | Contours: {len(contours)}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show windows
        cv2.imshow("Original + Detection", output)
        cv2.imshow("HSV Mask", mask_hsv)
        cv2.imshow("YCrCb Mask", mask_ycrcb)
        cv2.imshow("Combined (AND)", mask_clean)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n" + "="*60)
            print("SAVED VALUES - Copy these to vision.py:")
            print("="*60)
            print(f"GREEN_HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"GREEN_HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"GREEN_YCRCB_LOWER = np.array([{y_min}, {cr_min}, {cb_min}])")
            print(f"GREEN_YCRCB_UPPER = np.array([{y_max}, {cr_max}, {cb_max}])")
            print(f"# Morphology: kernel={morph_k}, open_iter={open_iter}, close_iter={close_iter}")
            print("="*60 + "\n")
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos("H_min", "Controls", 35)
            cv2.setTrackbarPos("H_max", "Controls", 85)
            cv2.setTrackbarPos("S_min", "Controls", 40)
            cv2.setTrackbarPos("S_max", "Controls", 255)
            cv2.setTrackbarPos("V_min", "Controls", 40)
            cv2.setTrackbarPos("V_max", "Controls", 255)
            cv2.setTrackbarPos("Y_min", "Controls", 40)
            cv2.setTrackbarPos("Y_max", "Controls", 220)
            cv2.setTrackbarPos("Cr_min", "Controls", 0)
            cv2.setTrackbarPos("Cr_max", "Controls", 115)
            cv2.setTrackbarPos("Cb_min", "Controls", 0)
            cv2.setTrackbarPos("Cb_max", "Controls", 135)
            print("Reset to defaults!")

        # Keyboard controls for tuning (number keys and shift+number)
        # H_min: 1/!
        elif key == ord('1'):
            cv2.setTrackbarPos("H_min", "Controls", max(0, h_min - STEP_H))
        elif key == ord('!'):
            cv2.setTrackbarPos("H_min", "Controls", min(179, h_min + STEP_H))
        # H_max: 2/@
        elif key == ord('2'):
            cv2.setTrackbarPos("H_max", "Controls", max(0, h_max - STEP_H))
        elif key == ord('@'):
            cv2.setTrackbarPos("H_max", "Controls", min(179, h_max + STEP_H))
        # S_min: 3/#
        elif key == ord('3'):
            cv2.setTrackbarPos("S_min", "Controls", max(0, s_min - STEP_SV))
        elif key == ord('#'):
            cv2.setTrackbarPos("S_min", "Controls", min(255, s_min + STEP_SV))
        # S_max: 4/$
        elif key == ord('4'):
            cv2.setTrackbarPos("S_max", "Controls", max(0, s_max - STEP_SV))
        elif key == ord('$'):
            cv2.setTrackbarPos("S_max", "Controls", min(255, s_max + STEP_SV))
        # V_min: 5/%
        elif key == ord('5'):
            cv2.setTrackbarPos("V_min", "Controls", max(0, v_min - STEP_SV))
        elif key == ord('%'):
            cv2.setTrackbarPos("V_min", "Controls", min(255, v_min + STEP_SV))
        # V_max: 6/^
        elif key == ord('6'):
            cv2.setTrackbarPos("V_max", "Controls", max(0, v_max - STEP_SV))
        elif key == ord('^'):
            cv2.setTrackbarPos("V_max", "Controls", min(255, v_max + STEP_SV))
        # Y_min: 7/&
        elif key == ord('7'):
            cv2.setTrackbarPos("Y_min", "Controls", max(0, y_min - STEP_YCRCB))
        elif key == ord('&'):
            cv2.setTrackbarPos("Y_min", "Controls", min(255, y_min + STEP_YCRCB))
        # Y_max: 8/*
        elif key == ord('8'):
            cv2.setTrackbarPos("Y_max", "Controls", max(0, y_max - STEP_YCRCB))
        elif key == ord('*'):
            cv2.setTrackbarPos("Y_max", "Controls", min(255, y_max + STEP_YCRCB))
        # Cr_min: 9/(
        elif key == ord('9'):
            cv2.setTrackbarPos("Cr_min", "Controls", max(0, cr_min - STEP_YCRCB))
        elif key == ord('('):
            cv2.setTrackbarPos("Cr_min", "Controls", min(255, cr_min + STEP_YCRCB))
        # Cr_max: 0/)
        elif key == ord('0'):
            cv2.setTrackbarPos("Cr_max", "Controls", max(0, cr_max - STEP_YCRCB))
        elif key == ord(')'):
            cv2.setTrackbarPos("Cr_max", "Controls", min(255, cr_max + STEP_YCRCB))
        # Cb_min: -/_
        elif key == ord('-'):
            cv2.setTrackbarPos("Cb_min", "Controls", max(0, cb_min - STEP_YCRCB))
        elif key == ord('_'):
            cv2.setTrackbarPos("Cb_min", "Controls", min(255, cb_min + STEP_YCRCB))
        # Cb_max: =/+
        elif key == ord('='):
            cv2.setTrackbarPos("Cb_max", "Controls", max(0, cb_max - STEP_YCRCB))
        elif key == ord('+'):
            cv2.setTrackbarPos("Cb_max", "Controls", min(255, cb_max + STEP_YCRCB))

    cap.release()
    cv2.destroyAllWindows()


# Run the tuner:
# green_tuner()
# %%
# PURPLE DUAL-MASK TUNER
# Tune both HSV and YCrCb thresholds for purple/pink ball detection
import cv2
import numpy as np
from collections import deque
def purple_tuner():
    """
    Interactive tuner for purple/pink ball detection using dual HSV + YCrCb masks.
    Shows individual masks and the combined AND result.
    Press 's' to save values, 'q' to quit.
    """
    cap = cv2.VideoCapture(0)

    # Create windows
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original + Detection")
    cv2.namedWindow("HSV Mask")
    cv2.namedWindow("YCrCb Mask")
    cv2.namedWindow("Combined (AND)")

    # HSV trackbars for purple (H: 0-179, hue ~130-170 for purple/magenta)
    cv2.createTrackbar("H_min", "Controls", 130, 179, lambda x: None)
    cv2.createTrackbar("H_max", "Controls", 175, 179, lambda x: None)
    cv2.createTrackbar("S_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("S_max", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("V_min", "Controls", 60, 255, lambda x: None)
    cv2.createTrackbar("V_max", "Controls", 255, 255, lambda x: None)

    # YCrCb trackbars (Cr > 128 means more red, good for pink/purple)
    cv2.createTrackbar("Y_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("Y_max", "Controls", 220, 255, lambda x: None)
    cv2.createTrackbar("Cr_min", "Controls", 130, 255, lambda x: None)
    cv2.createTrackbar("Cr_max", "Controls", 200, 255, lambda x: None)
    cv2.createTrackbar("Cb_min", "Controls", 100, 255, lambda x: None)
    cv2.createTrackbar("Cb_max", "Controls", 180, 255, lambda x: None)

    # Morphology controls
    cv2.createTrackbar("Morph_kernel", "Controls", 5, 15, lambda x: None)
    cv2.createTrackbar("Open_iter", "Controls", 1, 5, lambda x: None)
    cv2.createTrackbar("Close_iter", "Controls", 2, 5, lambda x: None)

    # Step size for keyboard adjustments
    STEP_H = 2
    STEP_SV = 5
    STEP_YCRCB = 5

    print("="*60)
    print("PURPLE DUAL-MASK TUNER")
    print("="*60)
    print("Use KEYBOARD to tune (sliders may not work on macOS)")
    print("Goal: Purple/Pink ball = WHITE, everything else = BLACK")
    print("")
    print("KEYBOARD CONTROLS:")
    print("  HSV Hue:        1/! = H_min -/+     2/@ = H_max -/+")
    print("  HSV Sat:        3/# = S_min -/+     4/$ = S_max -/+")
    print("  HSV Val:        5/% = V_min -/+     6/^ = V_max -/+")
    print("  YCrCb Y:        7/& = Y_min -/+     8/* = Y_max -/+")
    print("  YCrCb Cr:       9/( = Cr_min -/+    0/) = Cr_max -/+")
    print("  YCrCb Cb:       -/_ = Cb_min -/+    =/+ = Cb_max -/+")
    print("")
    print("  s = Save values    r = Reset    q = Quit")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get trackbar values
        h_min = cv2.getTrackbarPos("H_min", "Controls")
        h_max = cv2.getTrackbarPos("H_max", "Controls")
        s_min = cv2.getTrackbarPos("S_min", "Controls")
        s_max = cv2.getTrackbarPos("S_max", "Controls")
        v_min = cv2.getTrackbarPos("V_min", "Controls")
        v_max = cv2.getTrackbarPos("V_max", "Controls")

        y_min = cv2.getTrackbarPos("Y_min", "Controls")
        y_max = cv2.getTrackbarPos("Y_max", "Controls")
        cr_min = cv2.getTrackbarPos("Cr_min", "Controls")
        cr_max = cv2.getTrackbarPos("Cr_max", "Controls")
        cb_min = cv2.getTrackbarPos("Cb_min", "Controls")
        cb_max = cv2.getTrackbarPos("Cb_max", "Controls")

        morph_k = max(1, cv2.getTrackbarPos("Morph_kernel", "Controls"))
        open_iter = max(1, cv2.getTrackbarPos("Open_iter", "Controls"))
        close_iter = max(1, cv2.getTrackbarPos("Close_iter", "Controls"))

        # Convert color spaces
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Create masks
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

        lower_ycrcb = np.array([y_min, cr_min, cb_min])
        upper_ycrcb = np.array([y_max, cr_max, cb_max])
        mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)

        # Combined mask (AND operation for strict matching)
        mask_combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Apply morphology
        kernel = np.ones((morph_k, morph_k), np.uint8)
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        # Find contours and draw
        output = frame.copy()
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        for contour in contours:
            cv2.drawContours(output, [contour], -1, (255, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cv2.putText(output, f"A:{int(area)}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Count pixels for feedback
        hsv_pixels = cv2.countNonZero(mask_hsv)
        ycrcb_pixels = cv2.countNonZero(mask_ycrcb)
        combined_pixels = cv2.countNonZero(mask_clean)

        # Add info text
        cv2.putText(output, f"HSV: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(output, f"YCrCb: Y[{y_min}-{y_max}] Cr[{cr_min}-{cr_max}] Cb[{cb_min}-{cb_max}]",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(output, f"Pixels: HSV={hsv_pixels} YCrCb={ycrcb_pixels} Combined={combined_pixels} | Contours: {len(contours)}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show windows
        cv2.imshow("Original + Detection", output)
        cv2.imshow("HSV Mask", mask_hsv)
        cv2.imshow("YCrCb Mask", mask_ycrcb)
        cv2.imshow("Combined (AND)", mask_clean)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n" + "="*60)
            print("SAVED VALUES - Copy these to vision.py:")
            print("="*60)
            print(f"PURPLE_HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"PURPLE_HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"PURPLE_YCRCB_LOWER = np.array([{y_min}, {cr_min}, {cb_min}])")
            print(f"PURPLE_YCRCB_UPPER = np.array([{y_max}, {cr_max}, {cb_max}])")
            print(f"# Morphology: kernel={morph_k}, open_iter={open_iter}, close_iter={close_iter}")
            print("="*60 + "\n")
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos("H_min", "Controls", 130)
            cv2.setTrackbarPos("H_max", "Controls", 175)
            cv2.setTrackbarPos("S_min", "Controls", 40)
            cv2.setTrackbarPos("S_max", "Controls", 255)
            cv2.setTrackbarPos("V_min", "Controls", 60)
            cv2.setTrackbarPos("V_max", "Controls", 255)
            cv2.setTrackbarPos("Y_min", "Controls", 40)
            cv2.setTrackbarPos("Y_max", "Controls", 220)
            cv2.setTrackbarPos("Cr_min", "Controls", 130)
            cv2.setTrackbarPos("Cr_max", "Controls", 200)
            cv2.setTrackbarPos("Cb_min", "Controls", 100)
            cv2.setTrackbarPos("Cb_max", "Controls", 180)
            print("Reset to defaults!")

        # Keyboard controls
        elif key == ord('1'):
            cv2.setTrackbarPos("H_min", "Controls", max(0, h_min - STEP_H))
        elif key == ord('!'):
            cv2.setTrackbarPos("H_min", "Controls", min(179, h_min + STEP_H))
        elif key == ord('2'):
            cv2.setTrackbarPos("H_max", "Controls", max(0, h_max - STEP_H))
        elif key == ord('@'):
            cv2.setTrackbarPos("H_max", "Controls", min(179, h_max + STEP_H))
        elif key == ord('3'):
            cv2.setTrackbarPos("S_min", "Controls", max(0, s_min - STEP_SV))
        elif key == ord('#'):
            cv2.setTrackbarPos("S_min", "Controls", min(255, s_min + STEP_SV))
        elif key == ord('4'):
            cv2.setTrackbarPos("S_max", "Controls", max(0, s_max - STEP_SV))
        elif key == ord('$'):
            cv2.setTrackbarPos("S_max", "Controls", min(255, s_max + STEP_SV))
        elif key == ord('5'):
            cv2.setTrackbarPos("V_min", "Controls", max(0, v_min - STEP_SV))
        elif key == ord('%'):
            cv2.setTrackbarPos("V_min", "Controls", min(255, v_min + STEP_SV))
        elif key == ord('6'):
            cv2.setTrackbarPos("V_max", "Controls", max(0, v_max - STEP_SV))
        elif key == ord('^'):
            cv2.setTrackbarPos("V_max", "Controls", min(255, v_max + STEP_SV))
        elif key == ord('7'):
            cv2.setTrackbarPos("Y_min", "Controls", max(0, y_min - STEP_YCRCB))
        elif key == ord('&'):
            cv2.setTrackbarPos("Y_min", "Controls", min(255, y_min + STEP_YCRCB))
        elif key == ord('8'):
            cv2.setTrackbarPos("Y_max", "Controls", max(0, y_max - STEP_YCRCB))
        elif key == ord('*'):
            cv2.setTrackbarPos("Y_max", "Controls", min(255, y_max + STEP_YCRCB))
        elif key == ord('9'):
            cv2.setTrackbarPos("Cr_min", "Controls", max(0, cr_min - STEP_YCRCB))
        elif key == ord('('):
            cv2.setTrackbarPos("Cr_min", "Controls", min(255, cr_min + STEP_YCRCB))
        elif key == ord('0'):
            cv2.setTrackbarPos("Cr_max", "Controls", max(0, cr_max - STEP_YCRCB))
        elif key == ord(')'):
            cv2.setTrackbarPos("Cr_max", "Controls", min(255, cr_max + STEP_YCRCB))
        elif key == ord('-'):
            cv2.setTrackbarPos("Cb_min", "Controls", max(0, cb_min - STEP_YCRCB))
        elif key == ord('_'):
            cv2.setTrackbarPos("Cb_min", "Controls", min(255, cb_min + STEP_YCRCB))
        elif key == ord('='):
            cv2.setTrackbarPos("Cb_max", "Controls", max(0, cb_max - STEP_YCRCB))
        elif key == ord('+'):
            cv2.setTrackbarPos("Cb_max", "Controls", min(255, cb_max + STEP_YCRCB))

    cap.release()
    cv2.destroyAllWindows()


# Run purple tuner:
purple_tuner()

# %%
# RAMP-BASED BALL DETECTION
# Detect balls above the ramp edge using ramp color + circle detection

# Ramp color thresholds (tune these for your specific ramp)
BLUE_RAMP_HSV_LOWER = np.array([100, 100, 50])
BLUE_RAMP_HSV_UPPER = np.array([130, 255, 255])

RED_RAMP_HSV_LOWER = np.array([0, 100, 100])
RED_RAMP_HSV_UPPER = np.array([15, 255, 255])
RED_RAMP_HSV_LOWER2 = np.array([165, 100, 100])  # Red wraps around
RED_RAMP_HSV_UPPER2 = np.array([179, 255, 255])


def detect_ramp_edge(image, ramp_color='blue'):
    """
    Detect the top edge of the ramp to define the ball detection region.
    Returns a mask of the region ABOVE the ramp where balls should be.
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if ramp_color == 'blue':
        mask_ramp = cv2.inRange(img_hsv, BLUE_RAMP_HSV_LOWER, BLUE_RAMP_HSV_UPPER)
    elif ramp_color == 'red':
        mask1 = cv2.inRange(img_hsv, RED_RAMP_HSV_LOWER, RED_RAMP_HSV_UPPER)
        mask2 = cv2.inRange(img_hsv, RED_RAMP_HSV_LOWER2, RED_RAMP_HSV_UPPER2)
        mask_ramp = cv2.bitwise_or(mask1, mask2)
    else:
        return None

    # Clean up the ramp mask
    kernel = np.ones((10, 10), np.uint8)
    mask_ramp = cv2.morphologyEx(mask_ramp, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_ramp = cv2.morphologyEx(mask_ramp, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the top edge of the ramp
    # For each column, find the topmost ramp pixel
    height, width = mask_ramp.shape
    top_edge = np.full(width, height, dtype=np.int32)

    for x in range(width):
        col = mask_ramp[:, x]
        ramp_pixels = np.where(col > 0)[0]
        if len(ramp_pixels) > 0:
            top_edge[x] = ramp_pixels[0]

    # Smooth the edge
    top_edge_smooth = cv2.GaussianBlur(top_edge.astype(np.float32).reshape(1, -1), (51, 1), 0).flatten().astype(np.int32)

    # Create mask for region above the ramp (where balls are)
    ball_region_mask = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        if top_edge_smooth[x] < height:
            ball_region_mask[0:top_edge_smooth[x], x] = 255

    return ball_region_mask, top_edge_smooth


def detect_balls_with_circles(image, ball_region_mask=None, min_radius=15, max_radius=50):
    """
    Use Hough Circle detection to find individual balls.
    This works better for separating touching balls than contour-based methods.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # If we have a ball region mask, apply it
    if ball_region_mask is not None:
        gray_blur = cv2.bitwise_and(gray_blur, ball_region_mask)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 2,  # Minimum distance between circle centers
        param1=50,  # Edge detection threshold
        param2=30,  # Circle detection threshold (lower = more circles)
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            detected_circles.append((int(x), int(y), int(r)))

    return detected_circles


def classify_circle_color(image, center_x, center_y, radius, mask_green, mask_purple):
    """
    Classify a detected circle as green or purple based on color masks.
    """
    height, width = mask_green.shape

    # Create a circular mask for this ball
    circle_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)

    # Count green and purple pixels within the circle
    green_pixels = cv2.countNonZero(cv2.bitwise_and(mask_green, circle_mask))
    purple_pixels = cv2.countNonZero(cv2.bitwise_and(mask_purple, circle_mask))

    total = green_pixels + purple_pixels
    if total == 0:
        return None

    if green_pixels > purple_pixels and green_pixels / total > 0.3:
        return 'G'
    elif purple_pixels > green_pixels and purple_pixels / total > 0.3:
        return 'P'

    return None


def runPipelineWithCircles(image, use_stabilization=True, ramp_color='blue'):
    """
    Enhanced pipeline using circle detection for better ball separation.
    """
    global stable_detector
    output = image.copy()

    # Detect ramp and get ball region
    ball_region_mask, ramp_edge = detect_ramp_edge(image, ramp_color)

    # Draw ramp edge for visualization
    if ramp_edge is not None:
        for x in range(0, len(ramp_edge), 5):
            y = ramp_edge[x]
            if 0 < y < image.shape[0]:
                cv2.circle(output, (x, y), 2, (0, 255, 255), -1)

    # Convert to color spaces for classification
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Create color masks (same as before)
    mask_green_hsv = cv2.inRange(img_hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    mask_green_ycrcb = cv2.inRange(img_ycrcb, GREEN_YCRCB_LOWER, GREEN_YCRCB_UPPER)
    mask_green_strict = cv2.bitwise_and(mask_green_hsv, mask_green_ycrcb)
    mask_green_hsv_ext = cv2.inRange(img_hsv, GREEN_HSV_LOWER_EXT, GREEN_HSV_UPPER_EXT)
    mask_green_lab = cv2.inRange(img_lab, GREEN_LAB_LOWER, GREEN_LAB_UPPER)
    mask_green_extended = cv2.bitwise_and(mask_green_hsv_ext, mask_green_lab)
    mask_green = cv2.bitwise_or(mask_green_strict, mask_green_extended)

    mask_purple = cv2.inRange(img_hsv, PURPLE_HSV_LOWER, PURPLE_HSV_UPPER)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detect circles (balls) in the ball region
    circles = detect_balls_with_circles(image, ball_region_mask, min_radius=20, max_radius=60)

    detected_shapes = []
    green_count = 0
    purple_count = 0

    for (cx, cy, r) in circles:
        # Classify the circle color
        color = classify_circle_color(image, cx, cy, r, mask_green, mask_purple)

        if color == 'G':
            green_count += 1
            detected_shapes.append((cx, 'G', cx, cy, r))
            cv2.circle(output, (cx, cy), r, (0, 255, 0), 3)
            cv2.putText(output, f'G{green_count}', (cx - 10, cy - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif color == 'P':
            purple_count += 1
            detected_shapes.append((cx, 'P', cx, cy, r))
            cv2.circle(output, (cx, cy), r, (255, 0, 255), 3)
            cv2.putText(output, f'P{purple_count}', (cx - 10, cy - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            # Unknown color - draw in gray
            cv2.circle(output, (cx, cy), r, (128, 128, 128), 2)

    # Sort by x position
    detected_shapes.sort(key=lambda b: b[0])
    raw_motif = ''.join([b[1] for b in detected_shapes])

    # Apply temporal stabilization
    if use_stabilization:
        stable_motif = stable_detector.update(raw_motif)
    else:
        stable_motif = raw_motif

    total_count = len(stable_motif)

    # Show info
    cv2.putText(output, f"Stable: {stable_motif} ({total_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(output, f"Raw: {raw_motif} | Circles: {len(circles)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(output, f"G:{green_count} P:{purple_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    mask_combined = cv2.bitwise_or(mask_green, mask_purple)
    return stable_motif, output, mask_combined, mask_green, mask_purple


def ramp_tuner():
    """
    Tune the ramp color detection to find the best thresholds.
    Press 'b' for blue ramp mode, 'r' for red ramp mode.
    """
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Ramp Detection")
    cv2.namedWindow("Ball Region")

    # HSV trackbars
    cv2.createTrackbar("H_min", "Controls", 100, 179, lambda x: None)
    cv2.createTrackbar("H_max", "Controls", 130, 179, lambda x: None)
    cv2.createTrackbar("S_min", "Controls", 100, 255, lambda x: None)
    cv2.createTrackbar("S_max", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("V_min", "Controls", 50, 255, lambda x: None)
    cv2.createTrackbar("V_max", "Controls", 255, 255, lambda x: None)

    print("="*60)
    print("RAMP COLOR TUNER")
    print("="*60)
    print("Press 'b' for blue ramp preset, 'r' for red ramp preset")
    print("Press 's' to save, 'q' to quit")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_min = cv2.getTrackbarPos("H_min", "Controls")
        h_max = cv2.getTrackbarPos("H_max", "Controls")
        s_min = cv2.getTrackbarPos("S_min", "Controls")
        s_max = cv2.getTrackbarPos("S_max", "Controls")
        v_min = cv2.getTrackbarPos("V_min", "Controls")
        v_max = cv2.getTrackbarPos("V_max", "Controls")

        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask_ramp = cv2.inRange(img_hsv, lower, upper)

        # Clean up
        kernel = np.ones((10, 10), np.uint8)
        mask_ramp = cv2.morphologyEx(mask_ramp, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find top edge
        height, width = mask_ramp.shape
        top_edge = np.full(width, height, dtype=np.int32)
        for x in range(width):
            col = mask_ramp[:, x]
            ramp_pixels = np.where(col > 0)[0]
            if len(ramp_pixels) > 0:
                top_edge[x] = ramp_pixels[0]

        # Smooth edge
        top_edge_smooth = cv2.GaussianBlur(top_edge.astype(np.float32).reshape(1, -1), (51, 1), 0).flatten().astype(np.int32)

        # Create ball region mask
        ball_region = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            if top_edge_smooth[x] < height:
                ball_region[0:top_edge_smooth[x], x] = 255

        # Draw visualization
        output = frame.copy()
        # Draw ramp edge
        for x in range(0, width, 3):
            y = top_edge_smooth[x]
            if 0 < y < height:
                cv2.circle(output, (x, y), 2, (0, 255, 255), -1)

        cv2.putText(output, f"HSV: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Ramp Detection", output)
        cv2.imshow("Ball Region", ball_region)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            cv2.setTrackbarPos("H_min", "Controls", 100)
            cv2.setTrackbarPos("H_max", "Controls", 130)
            cv2.setTrackbarPos("S_min", "Controls", 100)
            cv2.setTrackbarPos("V_min", "Controls", 50)
            print("Blue ramp preset loaded")
        elif key == ord('r'):
            cv2.setTrackbarPos("H_min", "Controls", 0)
            cv2.setTrackbarPos("H_max", "Controls", 15)
            cv2.setTrackbarPos("S_min", "Controls", 100)
            cv2.setTrackbarPos("V_min", "Controls", 100)
            print("Red ramp preset loaded")
        elif key == ord('s'):
            print(f"\nRAMP_HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"RAMP_HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])")

    cap.release()
    cv2.destroyAllWindows()


# Uncomment to run:
# ramp_tuner()
# %%