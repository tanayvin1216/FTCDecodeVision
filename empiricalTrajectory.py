#%%
"""
Empirical Trajectory Recorder for FTC Shooter
Records ball position over time for different RPM + hood angle combos.
Calculates viX, viY, launch angle, max height, and range per trial.
All trials append to one CSV file.

Workflow:
  1. Set HOOD_ANGLE, RPM, and PIXELS_PER_METER below
  2. Run the cell — camera opens
  3. SPACE to start recording (tracks ball center each frame)
  4. SPACE to stop recording — calculates kinematics and auto-saves to CSV
  5. Press 'q' to quit, change values, run again for next config
"""

import cv2
import numpy as np
import csv
import os
from datetime import datetime
from ballDetect import detect_balls

# ============================================================================
# SET THESE BEFORE EACH RUN
# ============================================================================
HOOD_ANGLE = 4583       # degrees
RPM = 0               # flywheel RPM
PIXELS_PER_METER = 500.0   # calibrate this! (see README in old commits)
CSV_FILE = "trajectory_experiments.csv"
# ============================================================================

CSV_HEADER = [
    'Timestamp', 'Hood_Angle', 'RPM', 'Trial',
    'Frame', 'Time_s',
    'X_px', 'Y_px', 'X_m', 'Y_m',
    'Vx_mps', 'Vy_mps',
    # These repeat per row for easy filtering but are the same for the whole trial
    'Vi_x_mps', 'Vi_y_mps', 'V_initial_mps', 'Launch_Angle_deg',
    'Max_Height_m', 'Range_m'
]


def ensure_csv_header(filepath):
    """Create CSV with header if it doesn't exist yet."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)


def count_existing_trials(filepath):
    """Count how many trials already exist in the CSV."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        trials = set()
        for row in reader:
            if len(row) >= 4:
                trials.add((row[1], row[2], row[3]))
        return len(trials)


def compute_kinematics(points_px, fps, ppm):
    """
    Compute full kinematics from pixel positions.

    Args:
        points_px: list of (x, y) pixel positions
        fps: camera frames per second
        ppm: pixels per meter

    Returns:
        dict with positions in meters, per-frame velocities,
        initial velocity, launch angle, max height, range
    """
    dt = 1.0 / fps

    # Convert pixels to meters (Y flipped so up is positive)
    # Use first point as origin
    ox, oy = points_px[0]
    points_m = []
    for (px, py) in points_px:
        x_m = (px - ox) / ppm
        y_m = (oy - py) / ppm  # flip Y: up is positive
        points_m.append((x_m, y_m))

    # Per-frame velocities via finite differences
    vx_list = []
    vy_list = []
    for i in range(len(points_m)):
        if i == 0:
            dx = points_m[1][0] - points_m[0][0]
            dy = points_m[1][1] - points_m[0][1]
        elif i == len(points_m) - 1:
            dx = points_m[-1][0] - points_m[-2][0]
            dy = points_m[-1][1] - points_m[-2][1]
        else:
            dx = (points_m[i+1][0] - points_m[i-1][0]) / 2
            dy = (points_m[i+1][1] - points_m[i-1][1]) / 2
        vx_list.append(dx / dt)
        vy_list.append(dy / dt)

    # Initial velocity = average of first 3 frames for stability
    n = min(3, len(points_m))
    vix = np.mean(vx_list[:n])
    viy = np.mean(vy_list[:n])
    v0 = np.sqrt(vix**2 + viy**2)
    angle = np.degrees(np.arctan2(viy, vix)) if vix != 0 else 0.0

    y_vals = [p[1] for p in points_m]
    x_vals = [p[0] for p in points_m]
    max_height = max(y_vals)
    total_range = max(x_vals) - min(x_vals)

    return {
        'points_m': points_m,
        'vx_list': vx_list,
        'vy_list': vy_list,
        'vix': vix,
        'viy': viy,
        'v0': v0,
        'angle': angle,
        'max_height': max_height,
        'range': total_range,
    }


def save_trial(filepath, hood_angle, rpm, trial_num, points_px, fps, k):
    """Append one trial's data to the CSV."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(points_px)):
            writer.writerow([
                timestamp, hood_angle, rpm, trial_num,
                i, f"{i / fps:.4f}",
                points_px[i][0], points_px[i][1],
                f"{k['points_m'][i][0]:.6f}", f"{k['points_m'][i][1]:.6f}",
                f"{k['vx_list'][i]:.4f}", f"{k['vy_list'][i]:.4f}",
                f"{k['vix']:.4f}", f"{k['viy']:.4f}",
                f"{k['v0']:.4f}", f"{k['angle']:.2f}",
                f"{k['max_height']:.6f}", f"{k['range']:.6f}"
            ])


# ============================================================================
# MAIN
# ============================================================================

ensure_csv_header(CSV_FILE)
trial_num = count_existing_trials(CSV_FILE) + 1

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

# Calibration click handler
cal_points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cal_points.append((x, y))
        print(f"  Calibration point {len(cal_points)}: ({x}, {y})")
        if len(cal_points) == 2:
            px_dist = np.sqrt((cal_points[1][0] - cal_points[0][0])**2 +
                              (cal_points[1][1] - cal_points[0][1])**2)
            real_m = 70.5 * 0.0254  # 70.5 inches to meters
            ppm = px_dist / real_m
            print(f"\n  Pixel distance:   {px_dist:.1f} px")
            print(f"  Real distance:    {real_m:.4f} m (70.5 in)")
            print(f"  PIXELS_PER_METER: {ppm:.1f}")
            print(f"\n  Set PIXELS_PER_METER = {ppm:.1f} at top of file")
            cal_points.clear()

print("=" * 50)
print("TRAJECTORY RECORDER")
print(f"  Hood Angle:      {HOOD_ANGLE} deg")
print(f"  RPM:             {RPM}")
print(f"  Pixels/Meter:    {PIXELS_PER_METER}")
print(f"  Trial #:         {trial_num}")
print(f"  Saving to:       {CSV_FILE}")
print()
print("  SPACE = start/stop recording")
print("  'c'   = calibrate (click 2 points spanning 46 in)")
print("  'q'   = quit")
print("=" * 50)

recording = False
tracked_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()

    # Detect balls — same as ballDetect.py
    balls, green_mask, purple_mask = detect_balls(frame)

    # Draw all detected balls
    for ball in balls:
        x, y, w, h = ball['bbox']
        clr = (0, 255, 0) if ball['color'] == 'G' else (255, 0, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)

    # Track the largest ball's center
    detection = None
    if balls:
        best = max(balls, key=lambda b: b['bbox'][2] * b['bbox'][3])
        x, y, w, h = best['bbox']
        cx, cy = x + w // 2, y + h // 2
        detection = (cx, cy)

        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        if recording:
            tracked_points.append((cx, cy))

    # Draw trajectory trail
    if len(tracked_points) > 1:
        for i in range(1, len(tracked_points)):
            cv2.line(output, tracked_points[i-1], tracked_points[i], (255, 0, 0), 2)
            cv2.circle(output, tracked_points[i], 3, (0, 0, 255), -1)

    # Status bar
    info = f"Hood: {HOOD_ANGLE} | RPM: {RPM} | Trial: {trial_num}"
    if recording:
        cv2.putText(output, f"REC | {info} | Points: {len(tracked_points)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(output, f"READY | {info} | Balls: {len(balls)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Trajectory Recorder", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        cal_points.clear()
        cv2.setMouseCallback("Trajectory Recorder", on_mouse)
        print("\nCALIBRATION: Click two points spanning 46 inches...")
    elif key == ord(' '):
        if not recording:
            tracked_points.clear()
            recording = True
            print("Recording started...")
        else:
            recording = False
            if len(tracked_points) >= 2:
                k = compute_kinematics(tracked_points, fps, PIXELS_PER_METER)
                save_trial(CSV_FILE, HOOD_ANGLE, RPM, trial_num,
                          tracked_points, fps, k)
                print(f"\nTrial {trial_num} saved ({len(tracked_points)} points)")
                print(f"  Vi_x:         {k['vix']:.3f} m/s")
                print(f"  Vi_y:         {k['viy']:.3f} m/s")
                print(f"  V_initial:    {k['v0']:.3f} m/s")
                print(f"  Launch Angle: {k['angle']:.1f} deg")
                print(f"  Max Height:   {k['max_height']:.3f} m")
                print(f"  Range:        {k['range']:.3f} m")
                trial_num += 1
            else:
                print("Not enough points — trial discarded")

cap.release()
cv2.destroyAllWindows()

# %%
