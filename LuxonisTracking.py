"""
3D ball trajectory tracking using a Luxonis OAK-D camera.

Combines stereo depth with color-based ball detection to produce
real-time 3D ball positions and trajectory history.

Requires: depthai, opencv-python, numpy
"""

import time
from collections import deque

import cv2
import depthai as dai
import numpy as np

from ballDetect import detect_balls

# --- Configuration -----------------------------------------------------------

MAX_TRAJECTORY_POINTS = 200
DEPTH_MEDIAN_KERNEL = 7  # kernel size for median depth around detection center
MIN_VALID_DEPTH_MM = 100
MAX_VALID_DEPTH_MM = 10000
FPS = 30
RGB_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P


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

        print("=" * 50)
        print("LUXONIS 3D BALL TRACKING")
        print("  'c' = clear trajectories")
        print("  'q' = quit")
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

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
