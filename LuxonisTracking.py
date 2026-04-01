"""
3D ball trajectory tracking using a Luxonis OAK-D camera.

Combines stereo depth with color-based ball detection to produce
real-time 3D ball positions and trajectory history.

Requires: depthai, opencv-python, numpy
"""

import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import depthai as dai
import numpy as np

from ballDetect import detect_balls

# --- Configuration -----------------------------------------------------------

HOOD_ANGLE = 0            # degrees
RPM = 0                   # flywheel RPM
CSV_FILE = "trajectory_experiments_3d.csv"

MAX_TRAJECTORY_POINTS = 200
DEPTH_MEDIAN_KERNEL = 7  # kernel size for median depth around detection center
MIN_VALID_DEPTH_MM = 100
MAX_VALID_DEPTH_MM = 10000
FPS = 30
RGB_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

CSV_HEADER = [
    'Timestamp', 'Hood_Angle', 'RPM', 'Trial',
    'Frame', 'Time_s',
    'X_m', 'Y_m', 'Z_m',
    'Vx_mps', 'Vy_mps', 'Vz_mps',
    'Vi_x_mps', 'Vi_y_mps', 'Vi_z_mps',
    'V_initial_mps', 'Launch_Angle_deg',
    'Max_Height_m', 'Range_m',
]


# --- Pipeline ----------------------------------------------------------------

def build_pipeline():
    """Build a DepthAI pipeline: RGB + stereo depth, both synced."""
    pipeline = dai.Pipeline()

    # Color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(RGB_RESOLUTION)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(FPS)
    cam_rgb.setIspScale(2, 3)  # downscale for performance (720p)

    # Stereo pair
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(MONO_RESOLUTION)
    mono_right.setResolution(MONO_RESOLUTION)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setFps(FPS)
    mono_right.setFps(FPS)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB
    stereo.setOutputSize(
        cam_rgb.getIspWidth(), cam_rgb.getIspHeight()
    )
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.isp.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


# --- 3D projection ----------------------------------------------------------

def pixel_to_3d(x_px, y_px, depth_mm, intrinsics):
    """Convert a 2D pixel + depth into a 3D point (meters) using camera intrinsics."""
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    z = depth_mm / 1000.0
    x = (x_px - cx) * z / fx
    y = (y_px - cy) * z / fy
    return np.array([x, y, z])


def get_median_depth(depth_frame, cx, cy, kernel=DEPTH_MEDIAN_KERNEL):
    """Sample a median depth around (cx, cy) to reduce noise."""
    h, w = depth_frame.shape
    half = kernel // 2
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)

    roi = depth_frame[y0:y1, x0:x1]
    valid = roi[roi > 0]
    if len(valid) == 0:
        return 0
    return int(np.median(valid))


# --- CSV / Kinematics --------------------------------------------------------

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


def compute_kinematics_3d(points_3d, timestamps):
    """
    Compute 3D kinematics from (x, y, z) positions in meters.

    Y is flipped so that up is positive (camera Y points down).

    Returns dict with per-frame velocities, initial velocity,
    launch angle, max height, and range.
    """
    # Use first point as origin, flip Y so up is positive
    ox, oy, oz = points_3d[0]
    pts = [(x - ox, -(y - oy), z - oz) for x, y, z in points_3d]

    # Per-frame velocities via finite differences
    vx_list, vy_list, vz_list = [], [], []
    for i in range(len(pts)):
        if i == 0:
            dt = timestamps[1] - timestamps[0]
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            dz = pts[1][2] - pts[0][2]
        elif i == len(pts) - 1:
            dt = timestamps[-1] - timestamps[-2]
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
            dz = pts[-1][2] - pts[-2][2]
        else:
            dt = (timestamps[i + 1] - timestamps[i - 1]) / 2
            dx = (pts[i + 1][0] - pts[i - 1][0]) / 2
            dy = (pts[i + 1][1] - pts[i - 1][1]) / 2
            dz = (pts[i + 1][2] - pts[i - 1][2]) / 2
        dt = max(dt, 1e-6)
        vx_list.append(dx / dt)
        vy_list.append(dy / dt)
        vz_list.append(dz / dt)

    # Initial velocity = average of first 3 frames
    n = min(3, len(pts))
    vix = np.mean(vx_list[:n])
    viy = np.mean(vy_list[:n])
    viz = np.mean(vz_list[:n])
    v0 = np.sqrt(vix**2 + viy**2 + viz**2)

    # Launch angle in the vertical plane (angle from horizontal)
    horizontal_speed = np.sqrt(vix**2 + viz**2)
    angle = np.degrees(np.arctan2(viy, horizontal_speed)) if horizontal_speed > 0 else 0.0

    y_vals = [p[1] for p in pts]
    max_height = max(y_vals)

    # Range = horizontal distance (XZ plane) from start to end
    total_range = np.sqrt(pts[-1][0]**2 + pts[-1][2]**2)

    return {
        'points': pts,
        'vx_list': vx_list,
        'vy_list': vy_list,
        'vz_list': vz_list,
        'vix': vix,
        'viy': viy,
        'viz': viz,
        'v0': v0,
        'angle': angle,
        'max_height': max_height,
        'range': total_range,
    }


def save_trial_3d(filepath, hood_angle, rpm, trial_num, timestamps, k):
    """Append one trial's 3D data to the CSV."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(k['points'])):
            writer.writerow([
                timestamp, hood_angle, rpm, trial_num,
                i, f"{timestamps[i] - timestamps[0]:.4f}",
                f"{k['points'][i][0]:.6f}",
                f"{k['points'][i][1]:.6f}",
                f"{k['points'][i][2]:.6f}",
                f"{k['vx_list'][i]:.4f}",
                f"{k['vy_list'][i]:.4f}",
                f"{k['vz_list'][i]:.4f}",
                f"{k['vix']:.4f}", f"{k['viy']:.4f}", f"{k['viz']:.4f}",
                f"{k['v0']:.4f}", f"{k['angle']:.2f}",
                f"{k['max_height']:.6f}", f"{k['range']:.6f}",
            ])


# --- Visualization -----------------------------------------------------------

def draw_trajectory_2d(frame, trajectory, color_bgr):
    """Draw the 2D projection of a trajectory on the frame."""
    pts = [(int(p["px"]), int(p["py"])) for p in trajectory]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        thick = max(1, int(alpha * 3))
        cv2.line(frame, pts[i - 1], pts[i], color_bgr, thick)


def draw_3d_info(frame, ball_3d, label, color_bgr, position):
    """Overlay 3D coordinate text near the detection."""
    x, y, z = ball_3d
    text = f"{label}: ({x:.2f}, {y:.2f}, {z:.2f})m"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)


def draw_top_down_view(trajectories, view_size=300):
    """Render a simple top-down (X-Z plane) view of all trajectories."""
    view = np.zeros((view_size, view_size, 3), dtype=np.uint8)

    # axes
    mid = view_size // 2
    cv2.line(view, (mid, 0), (mid, view_size), (40, 40, 40), 1)
    cv2.line(view, (0, view_size), (view_size, view_size), (40, 40, 40), 1)
    cv2.putText(view, "Top-down (XZ)", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    scale = 100  # pixels per meter

    for color_key, traj in trajectories.items():
        color_bgr = (0, 255, 0) if color_key == "G" else (255, 0, 255)
        pts = []
        for p in traj:
            px = int(mid + p["x"] * scale)
            py = int(view_size - p["z"] * scale)
            pts.append((px, py))

        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thick = max(1, int(alpha * 2))
            cv2.line(view, pts[i - 1], pts[i], color_bgr, thick)

        if pts:
            cv2.circle(view, pts[-1], 4, color_bgr, -1)

    return view


def draw_side_view(trajectories, view_size=300):
    """Render a side view (Z-Y plane) of all trajectories."""
    view = np.zeros((view_size, view_size, 3), dtype=np.uint8)
    cv2.putText(view, "Side (ZY)", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    scale = 100
    mid_y = view_size // 2

    for color_key, traj in trajectories.items():
        color_bgr = (0, 255, 0) if color_key == "G" else (255, 0, 255)
        pts = []
        for p in traj:
            px = int(p["z"] * scale)
            py = int(mid_y - p["y"] * scale)
            pts.append((px, py))

        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thick = max(1, int(alpha * 2))
            cv2.line(view, pts[i - 1], pts[i], color_bgr, thick)

        if pts:
            cv2.circle(view, pts[-1], 4, color_bgr, -1)

    return view


# --- Main loop ---------------------------------------------------------------

def main():
    pipeline = build_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # Get RGB camera intrinsics for 3D projection
        calib = device.readCalibration()
        rgb_width = q_rgb.get().getCvFrame().shape[1]
        rgb_height = q_rgb.get().getCvFrame().shape[0]
        intrinsics = np.array(
            calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, rgb_width, rgb_height)
        )

        trajectories = {
            "G": deque(maxlen=MAX_TRAJECTORY_POINTS),
            "P": deque(maxlen=MAX_TRAJECTORY_POINTS),
        }

        # --- CSV recording state ---
        ensure_csv_header(CSV_FILE)
        trial_num = count_existing_trials(CSV_FILE) + 1
        recording = False
        recorded_points_3d = []   # list of (x, y, z) in meters
        recorded_timestamps = []  # list of time.time() values

        print("=" * 50)
        print("LUXONIS 3D BALL TRACKING")
        print(f"  Hood Angle:  {HOOD_ANGLE} deg")
        print(f"  RPM:         {RPM}")
        print(f"  Trial #:     {trial_num}")
        print(f"  Saving to:   {CSV_FILE}")
        print()
        print("  SPACE = start/stop recording")
        print("  'c'   = clear trajectories")
        print("  'q'   = quit")
        print("=" * 50)

        prev_time = time.time()

        while True:
            in_rgb = q_rgb.tryGet()
            in_depth = q_depth.tryGet()
            if in_rgb is None or in_depth is None:
                continue

            frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()

            balls, green_mask, purple_mask = detect_balls(frame)

            output = frame.copy()

            for ball in balls:
                x, y, w, h = ball["bbox"]
                cx = x + w // 2
                cy = y + h // 2

                depth_mm = get_median_depth(depth_frame, cx, cy)

                if not (MIN_VALID_DEPTH_MM < depth_mm < MAX_VALID_DEPTH_MM):
                    continue

                point_3d = pixel_to_3d(cx, cy, depth_mm, intrinsics)

                color_bgr = (0, 255, 0) if ball["color"] == "G" else (255, 0, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.circle(output, (cx, cy), 4, color_bgr, -1)

                draw_3d_info(
                    output, point_3d, ball["color"], color_bgr, (x, y - 10)
                )

                trajectories[ball["color"]].append({
                    "x": point_3d[0],
                    "y": point_3d[1],
                    "z": point_3d[2],
                    "px": cx,
                    "py": cy,
                    "t": time.time(),
                })

            # Record the largest ball's 3D position when recording
            if recording and balls:
                best = max(balls, key=lambda b: b["bbox"][2] * b["bbox"][3])
                bx, by, bw, bh = best["bbox"]
                bcx = bx + bw // 2
                bcy = by + bh // 2
                d_mm = get_median_depth(depth_frame, bcx, bcy)
                if MIN_VALID_DEPTH_MM < d_mm < MAX_VALID_DEPTH_MM:
                    pt = pixel_to_3d(bcx, bcy, d_mm, intrinsics)
                    recorded_points_3d.append((pt[0], pt[1], pt[2]))
                    recorded_timestamps.append(time.time())

            # Draw 2D trajectories on the RGB view
            for color_key, traj in trajectories.items():
                color_bgr = (0, 255, 0) if color_key == "G" else (255, 0, 255)
                draw_trajectory_2d(output, traj, color_bgr)

            # FPS counter
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(output, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Recording status
            info = f"Hood: {HOOD_ANGLE} | RPM: {RPM} | Trial: {trial_num}"
            if recording:
                cv2.putText(output, f"REC | {info} | Pts: {len(recorded_points_3d)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(output, f"READY | {info}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Depth colormap
            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # 3D views
            top_view = draw_top_down_view(trajectories)
            side_view = draw_side_view(trajectories)
            views_combined = np.vstack([top_view, side_view])

            cv2.imshow("3D Ball Tracking - RGB", output)
            cv2.imshow("Depth", depth_vis)
            cv2.imshow("3D Views", views_combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                for traj in trajectories.values():
                    traj.clear()
                print("Trajectories cleared.")
            elif key == ord(" "):
                if not recording:
                    recorded_points_3d.clear()
                    recorded_timestamps.clear()
                    recording = True
                    print("Recording started...")
                else:
                    recording = False
                    if len(recorded_points_3d) >= 2:
                        k = compute_kinematics_3d(recorded_points_3d, recorded_timestamps)
                        save_trial_3d(CSV_FILE, HOOD_ANGLE, RPM, trial_num,
                                      recorded_timestamps, k)
                        print(f"\nTrial {trial_num} saved ({len(recorded_points_3d)} points)")
                        print(f"  Vi_x:         {k['vix']:.3f} m/s")
                        print(f"  Vi_y:         {k['viy']:.3f} m/s")
                        print(f"  Vi_z:         {k['viz']:.3f} m/s")
                        print(f"  V_initial:    {k['v0']:.3f} m/s")
                        print(f"  Launch Angle: {k['angle']:.1f} deg")
                        print(f"  Max Height:   {k['max_height']:.3f} m")
                        print(f"  Range:        {k['range']:.3f} m")
                        trial_num += 1
                    else:
                        print("Not enough points — trial discarded")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
