#%%
"""
Empirical Trajectory Analysis for FTC Shooter
Tracks ball trajectory from video and calculates kinematics data.
Outputs CSV data organized by RPM and arc length for graphing.
"""

import cv2
import numpy as np
import csv
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ball color thresholds - TUNE THESE FOR YOUR SPECIFIC BALL
# Using HSV for initial detection (adjust after testing)
BALL_HSV_LOWER = np.array([15, 100, 100])   # Yellow ball default
BALL_HSV_UPPER = np.array([35, 255, 255])

# Alternative: Use YCrCb if HSV isn't working well
USE_YCRCB = False
BALL_YCRCB_LOWER = np.array([100, 130, 0])
BALL_YCRCB_UPPER = np.array([255, 180, 120])

# Detection settings
MIN_BALL_AREA = 100
MAX_BALL_AREA = 50000
MIN_CIRCULARITY = 0.5  # Filter for circular shapes

# -----------------------------------------------------------------------
# PIXELS_PER_METER — HOW TO CALIBRATE
#
# This converts pixel distances to real-world meters. If it's wrong,
# all velocity (m/s), height, and range values will be wrong.
#
# Steps:
#   1. Place an object of known length (e.g. a 1-meter stick, or a
#      field tile which is 24 inches / 0.6096m) in the camera's view
#      at the same distance the ball will travel.
#   2. Run the calibration tool (option 3 in main menu) or take a
#      screenshot from the camera feed.
#   3. Measure how many pixels that known object spans in the image.
#      You can use any image editor or OpenCV mouse callback to get
#      two pixel coordinates and compute the distance:
#        pixel_dist = sqrt((x2-x1)^2 + (y2-y1)^2)
#   4. Divide by the real-world length in meters:
#        PIXELS_PER_METER = pixel_dist / real_length_in_meters
#
# Example: A 0.6096m field tile spans 305 pixels in your frame.
#          PIXELS_PER_METER = 305 / 0.6096 = 500.3
# -----------------------------------------------------------------------
PIXELS_PER_METER = 500.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrajectoryPoint:
    """Single point in the trajectory"""
    frame_num: int
    time_sec: float
    x_pixels: float
    y_pixels: float
    x_meters: float
    y_meters: float
    vx_mps: float = 0.0  # velocity in m/s
    vy_mps: float = 0.0


@dataclass
class TrajectoryData:
    """Complete trajectory for one trial"""
    rpm: float
    arc_length: float
    trial_num: int
    fps: float
    points: List[TrajectoryPoint] = field(default_factory=list)

    # Computed values (filled after trajectory is complete)
    vxi_mps: float = 0.0  # Initial velocity X (m/s)
    vyi_mps: float = 0.0  # Initial velocity Y (m/s)
    vzi_mps: float = 0.0  # Initial velocity Z (estimated from arc, if 3D)
    v_initial_mps: float = 0.0  # Total initial velocity magnitude
    launch_angle_deg: float = 0.0  # Launch angle in degrees
    max_height_m: float = 0.0  # Maximum height reached
    range_m: float = 0.0  # Horizontal distance traveled


@dataclass
class ExperimentSession:
    """Collection of all trials in a session"""
    session_name: str
    trajectories: List[TrajectoryData] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# BALL DETECTOR
# ============================================================================

class BallTracker:
    """Detects and tracks the ball through video frames"""

    def __init__(self, use_ycrcb: bool = False):
        self.use_ycrcb = use_ycrcb

        # HSV thresholds
        self.hsv_lower = BALL_HSV_LOWER.copy()
        self.hsv_upper = BALL_HSV_UPPER.copy()

        # YCrCb thresholds
        self.ycrcb_lower = BALL_YCRCB_LOWER.copy()
        self.ycrcb_upper = BALL_YCRCB_UPPER.copy()

        # Tracking state
        self.tracking = False
        self.trajectory_points = []

    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball in frame.
        Returns: (center_x, center_y, radius) or None if not found
        """
        if self.use_ycrcb:
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(converted, self.ycrcb_lower, self.ycrcb_upper)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_detection = None
        best_circularity = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_BALL_AREA < area < MAX_BALL_AREA:
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)

                    if circularity > MIN_CIRCULARITY and circularity > best_circularity:
                        # Get minimum enclosing circle
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        best_detection = (cx, cy, radius)
                        best_circularity = circularity

        return best_detection

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get the color mask for visualization"""
        if self.use_ycrcb:
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(converted, self.ycrcb_lower, self.ycrcb_upper)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask


# ============================================================================
# TRAJECTORY ANALYZER
# ============================================================================

class TrajectoryAnalyzer:
    """Main class for analyzing ball trajectories from video"""

    def __init__(self, pixels_per_meter: float = PIXELS_PER_METER):
        self.pixels_per_meter = pixels_per_meter
        self.tracker = BallTracker(use_ycrcb=USE_YCRCB)
        self.session = ExperimentSession(
            session_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Origin point for coordinate system (set during calibration)
        self.origin_x = 0
        self.origin_y = 0
        self.y_up = True  # If True, positive Y is up (flip from image coords)

    def set_origin(self, x: int, y: int):
        """Set the origin point for the coordinate system"""
        self.origin_x = x
        self.origin_y = y

    def pixels_to_meters(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Convert pixel coordinates to meters relative to origin"""
        x_m = (x_px - self.origin_x) / self.pixels_per_meter
        if self.y_up:
            # Flip Y so positive is up
            y_m = (self.origin_y - y_px) / self.pixels_per_meter
        else:
            y_m = (y_px - self.origin_y) / self.pixels_per_meter
        return x_m, y_m

    def calculate_velocities(self, trajectory: TrajectoryData):
        """Calculate velocities for all points in trajectory"""
        points = trajectory.points
        if len(points) < 2:
            return

        dt = 1.0 / trajectory.fps

        # Calculate velocities using central difference where possible
        for i in range(len(points)):
            if i == 0:
                # Forward difference for first point
                dx = points[1].x_meters - points[0].x_meters
                dy = points[1].y_meters - points[0].y_meters
            elif i == len(points) - 1:
                # Backward difference for last point
                dx = points[-1].x_meters - points[-2].x_meters
                dy = points[-1].y_meters - points[-2].y_meters
            else:
                # Central difference
                dx = (points[i+1].x_meters - points[i-1].x_meters) / 2
                dy = (points[i+1].y_meters - points[i-1].y_meters) / 2

            points[i].vx_mps = dx / dt
            points[i].vy_mps = dy / dt

        # Calculate initial velocities (average of first few frames for stability)
        num_init_frames = min(3, len(points))
        trajectory.vxi_mps = np.mean([p.vx_mps for p in points[:num_init_frames]])
        trajectory.vyi_mps = np.mean([p.vy_mps for p in points[:num_init_frames]])

        # Total initial velocity
        trajectory.v_initial_mps = np.sqrt(
            trajectory.vxi_mps**2 + trajectory.vyi_mps**2
        )

        # Launch angle
        if trajectory.vxi_mps != 0:
            trajectory.launch_angle_deg = np.degrees(
                np.arctan2(trajectory.vyi_mps, trajectory.vxi_mps)
            )

        # Max height and range
        y_values = [p.y_meters for p in points]
        x_values = [p.x_meters for p in points]
        trajectory.max_height_m = max(y_values) if y_values else 0
        trajectory.range_m = max(x_values) - min(x_values) if x_values else 0

    def analyze_video(self, video_path: str, rpm: float, arc_length: float,
                      trial_num: int = 1, show_preview: bool = True) -> Optional[TrajectoryData]:
        """
        Analyze a video file to extract trajectory data.

        Args:
            video_path: Path to video file
            rpm: Shooter RPM for this trial
            arc_length: Hood arc length for this trial
            trial_num: Trial number for this configuration
            show_preview: Whether to show visualization during processing

        Returns:
            TrajectoryData object with all measured points and computed values
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"Analyzing: {video_path}")
        print(f"RPM: {rpm}, Arc Length: {arc_length}, Trial: {trial_num}")
        print(f"FPS: {fps}, Total Frames: {total_frames}")
        print(f"{'='*60}")

        trajectory = TrajectoryData(
            rpm=rpm,
            arc_length=arc_length,
            trial_num=trial_num,
            fps=fps
        )

        frame_num = 0
        tracking_started = False
        frames_without_ball = 0
        max_frames_without_ball = 10  # Stop tracking after this many misses

        # For drawing trajectory
        trajectory_points_px = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.tracker.detect_ball(frame)

            if detection is not None:
                cx, cy, radius = detection
                tracking_started = True
                frames_without_ball = 0

                # Convert to meters
                x_m, y_m = self.pixels_to_meters(cx, cy)
                time_sec = frame_num / fps

                point = TrajectoryPoint(
                    frame_num=frame_num,
                    time_sec=time_sec,
                    x_pixels=cx,
                    y_pixels=cy,
                    x_meters=x_m,
                    y_meters=y_m
                )
                trajectory.points.append(point)
                trajectory_points_px.append((int(cx), int(cy)))

                if show_preview:
                    # Draw current detection
                    cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            elif tracking_started:
                frames_without_ball += 1
                if frames_without_ball > max_frames_without_ball:
                    print(f"Ball lost after frame {frame_num}, ending trajectory")
                    break

            if show_preview:
                # Draw trajectory path
                if len(trajectory_points_px) > 1:
                    for i in range(1, len(trajectory_points_px)):
                        cv2.line(frame, trajectory_points_px[i-1],
                                trajectory_points_px[i], (255, 0, 0), 2)

                # Draw origin if set
                if self.origin_x > 0 or self.origin_y > 0:
                    cv2.drawMarker(frame, (self.origin_x, self.origin_y),
                                  (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

                # Status text
                status = f"Frame: {frame_num}/{total_frames} | Points: {len(trajectory.points)}"
                cv2.putText(frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show mask in corner
                mask = self.tracker.get_mask(frame)
                mask_small = cv2.resize(mask, (frame.shape[1]//4, frame.shape[0]//4))
                mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[0:mask_small.shape[0], 0:mask_small.shape[1]] = mask_color

                cv2.imshow("Trajectory Analysis", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    # Set origin to current mouse position
                    print("Click to set origin...")

            frame_num += 1

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Calculate velocities
        if len(trajectory.points) >= 2:
            self.calculate_velocities(trajectory)
            self.session.trajectories.append(trajectory)
            print(f"\nTrajectory captured with {len(trajectory.points)} points")
            print(f"Initial Vx: {trajectory.vxi_mps:.3f} m/s")
            print(f"Initial Vy: {trajectory.vyi_mps:.3f} m/s")
            print(f"Initial V: {trajectory.v_initial_mps:.3f} m/s")
            print(f"Launch Angle: {trajectory.launch_angle_deg:.1f} degrees")
            return trajectory
        else:
            print("Not enough points captured for analysis")
            return None

    def analyze_live(self, rpm: float, arc_length: float, trial_num: int = 1,
                     camera_index: int = 0) -> Optional[TrajectoryData]:
        """
        Analyze live camera feed. Press SPACE to start/stop recording trajectory.

        Args:
            rpm: Shooter RPM for this trial
            arc_length: Hood arc length for this trial
            trial_num: Trial number
            camera_index: Camera device index

        Returns:
            TrajectoryData object
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default assumption

        print(f"\n{'='*60}")
        print(f"Live Analysis Mode")
        print(f"RPM: {rpm}, Arc Length: {arc_length}, Trial: {trial_num}")
        print(f"FPS: {fps}")
        print(f"Controls:")
        print(f"  SPACE - Start/Stop recording")
        print(f"  'o' - Set origin at current ball position")
        print(f"  'r' - Reset trajectory")
        print(f"  'q' - Quit and save")
        print(f"{'='*60}")

        trajectory = TrajectoryData(
            rpm=rpm,
            arc_length=arc_length,
            trial_num=trial_num,
            fps=fps
        )

        recording = False
        frame_num = 0
        trajectory_points_px = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.tracker.detect_ball(frame)

            if detection is not None:
                cx, cy, radius = detection

                # Draw current detection
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                if recording:
                    x_m, y_m = self.pixels_to_meters(cx, cy)
                    time_sec = frame_num / fps

                    point = TrajectoryPoint(
                        frame_num=frame_num,
                        time_sec=time_sec,
                        x_pixels=cx,
                        y_pixels=cy,
                        x_meters=x_m,
                        y_meters=y_m
                    )
                    trajectory.points.append(point)
                    trajectory_points_px.append((int(cx), int(cy)))
                    frame_num += 1

            # Draw trajectory path
            if len(trajectory_points_px) > 1:
                for i in range(1, len(trajectory_points_px)):
                    cv2.line(frame, trajectory_points_px[i-1],
                            trajectory_points_px[i], (255, 0, 0), 2)

            # Draw origin
            if self.origin_x > 0 or self.origin_y > 0:
                cv2.drawMarker(frame, (self.origin_x, self.origin_y),
                              (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

            # Status
            rec_status = "RECORDING" if recording else "STANDBY"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, rec_status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Points: {len(trajectory.points)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show mask
            mask = self.tracker.get_mask(frame)
            mask_small = cv2.resize(mask, (frame.shape[1]//4, frame.shape[0]//4))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[0:mask_small.shape[0], 0:mask_small.shape[1]] = mask_color

            cv2.imshow("Live Trajectory Analysis", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                recording = not recording
                if recording:
                    print("Recording started...")
                else:
                    print(f"Recording stopped. {len(trajectory.points)} points captured.")
            elif key == ord('o') and detection is not None:
                self.set_origin(int(detection[0]), int(detection[1]))
                print(f"Origin set to ({self.origin_x}, {self.origin_y})")
            elif key == ord('r'):
                trajectory.points.clear()
                trajectory_points_px.clear()
                frame_num = 0
                print("Trajectory reset")

        cap.release()
        cv2.destroyAllWindows()

        if len(trajectory.points) >= 2:
            self.calculate_velocities(trajectory)
            self.session.trajectories.append(trajectory)
            return trajectory
        return None

    def export_to_csv(self, filepath: str = None):
        """
        Export all trajectory data to CSV file.
        Format: RPM, Arc_Length, Trial, Time, X_m, Y_m, Vx_mps, Vy_mps, ...
        """
        if filepath is None:
            filepath = f"trajectory_data_{self.session.session_name}.csv"

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header row
            writer.writerow([
                'RPM', 'Arc_Length', 'Trial', 'Frame', 'Time_s',
                'X_pixels', 'Y_pixels', 'X_m', 'Y_m',
                'Vx_mps', 'Vy_mps',
                'Vxi_mps', 'Vyi_mps', 'V_initial_mps', 'Launch_Angle_deg',
                'Max_Height_m', 'Range_m'
            ])

            # Data rows
            for traj in self.session.trajectories:
                for point in traj.points:
                    writer.writerow([
                        traj.rpm,
                        traj.arc_length,
                        traj.trial_num,
                        point.frame_num,
                        f"{point.time_sec:.6f}",
                        f"{point.x_pixels:.2f}",
                        f"{point.y_pixels:.2f}",
                        f"{point.x_meters:.6f}",
                        f"{point.y_meters:.6f}",
                        f"{point.vx_mps:.6f}",
                        f"{point.vy_mps:.6f}",
                        f"{traj.vxi_mps:.6f}",
                        f"{traj.vyi_mps:.6f}",
                        f"{traj.v_initial_mps:.6f}",
                        f"{traj.launch_angle_deg:.2f}",
                        f"{traj.max_height_m:.6f}",
                        f"{traj.range_m:.6f}"
                    ])

        print(f"\nData exported to: {filepath}")
        print(f"Total trajectories: {len(self.session.trajectories)}")
        return filepath

    def export_summary_csv(self, filepath: str = None):
        """
        Export summary table with one row per trial.
        Ideal for quick comparison across RPM and arc length combinations.
        """
        if filepath is None:
            filepath = f"trajectory_summary_{self.session.session_name}.csv"

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                'RPM', 'Arc_Length', 'Trial', 'Num_Points',
                'Vxi_mps', 'Vyi_mps', 'Vzi_mps', 'V_initial_mps',
                'Launch_Angle_deg', 'Max_Height_m', 'Range_m',
                'Duration_s'
            ])

            for traj in self.session.trajectories:
                duration = traj.points[-1].time_sec if traj.points else 0
                writer.writerow([
                    traj.rpm,
                    traj.arc_length,
                    traj.trial_num,
                    len(traj.points),
                    f"{traj.vxi_mps:.6f}",
                    f"{traj.vyi_mps:.6f}",
                    f"{traj.vzi_mps:.6f}",
                    f"{traj.v_initial_mps:.6f}",
                    f"{traj.launch_angle_deg:.2f}",
                    f"{traj.max_height_m:.6f}",
                    f"{traj.range_m:.6f}",
                    f"{duration:.4f}"
                ])

        print(f"\nSummary exported to: {filepath}")
        return filepath

    def print_summary(self):
        """Print summary of all captured trajectories"""
        print(f"\n{'='*80}")
        print(f"TRAJECTORY SUMMARY - {self.session.session_name}")
        print(f"{'='*80}")
        print(f"{'RPM':>8} {'Arc':>8} {'Trial':>6} {'Points':>8} "
              f"{'Vxi':>10} {'Vyi':>10} {'V0':>10} {'Angle':>8}")
        print(f"{'':>8} {'':>8} {'':>6} {'':>8} "
              f"{'(m/s)':>10} {'(m/s)':>10} {'(m/s)':>10} {'(deg)':>8}")
        print('-'*80)

        for traj in self.session.trajectories:
            print(f"{traj.rpm:>8.0f} {traj.arc_length:>8.2f} {traj.trial_num:>6} "
                  f"{len(traj.points):>8} {traj.vxi_mps:>10.3f} {traj.vyi_mps:>10.3f} "
                  f"{traj.v_initial_mps:>10.3f} {traj.launch_angle_deg:>8.1f}")

        print('='*80)


# ============================================================================
# CALIBRATION TOOL
# ============================================================================

def run_calibration(camera_index: int = 0):
    """
    Interactive calibration tool to set color thresholds and pixel scale.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera")
        return

    cv2.namedWindow("Calibration")
    cv2.namedWindow("Controls")

    # HSV trackbars
    cv2.createTrackbar("H_Low", "Controls", BALL_HSV_LOWER[0], 179, lambda x: None)
    cv2.createTrackbar("H_High", "Controls", BALL_HSV_UPPER[0], 179, lambda x: None)
    cv2.createTrackbar("S_Low", "Controls", BALL_HSV_LOWER[1], 255, lambda x: None)
    cv2.createTrackbar("S_High", "Controls", BALL_HSV_UPPER[1], 255, lambda x: None)
    cv2.createTrackbar("V_Low", "Controls", BALL_HSV_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("V_High", "Controls", BALL_HSV_UPPER[2], 255, lambda x: None)

    print("="*60)
    print("CALIBRATION MODE")
    print("Adjust sliders to isolate the ball color")
    print("Press 'p' to print current values")
    print("Press 'q' to quit")
    print("="*60)

    tracker = BallTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update thresholds from trackbars
        tracker.hsv_lower[0] = cv2.getTrackbarPos("H_Low", "Controls")
        tracker.hsv_upper[0] = cv2.getTrackbarPos("H_High", "Controls")
        tracker.hsv_lower[1] = cv2.getTrackbarPos("S_Low", "Controls")
        tracker.hsv_upper[1] = cv2.getTrackbarPos("S_High", "Controls")
        tracker.hsv_lower[2] = cv2.getTrackbarPos("V_Low", "Controls")
        tracker.hsv_upper[2] = cv2.getTrackbarPos("V_High", "Controls")

        # Detect ball
        detection = tracker.detect_ball(frame)

        if detection is not None:
            cx, cy, radius = detection
            cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"Ball: ({int(cx)}, {int(cy)})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show mask
        mask = tracker.get_mask(frame)

        # Display values
        vals = f"H:[{tracker.hsv_lower[0]}-{tracker.hsv_upper[0]}] " \
               f"S:[{tracker.hsv_lower[1]}-{tracker.hsv_upper[1]}] " \
               f"V:[{tracker.hsv_lower[2]}-{tracker.hsv_upper[2]}]"
        cv2.putText(frame, vals, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\n# Ball HSV thresholds:")
            print(f"BALL_HSV_LOWER = np.array([{tracker.hsv_lower[0]}, {tracker.hsv_lower[1]}, {tracker.hsv_lower[2]}])")
            print(f"BALL_HSV_UPPER = np.array([{tracker.hsv_upper[0]}, {tracker.hsv_upper[1]}, {tracker.hsv_upper[2]}])")

    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# MAIN - INTERACTIVE SESSION
# ============================================================================

def main():
    """Interactive session for collecting trajectory data"""
    print("="*60)
    print("EMPIRICAL TRAJECTORY ANALYZER")
    print("FTC Shooter Kinematics Data Collection")
    print("="*60)

    analyzer = TrajectoryAnalyzer()

    while True:
        print("\nOptions:")
        print("  1. Analyze video file")
        print("  2. Live camera analysis")
        print("  3. Calibrate color detection")
        print("  4. Set pixel scale (pixels per meter)")
        print("  5. Set origin point")
        print("  6. Print summary")
        print("  7. Export to CSV")
        print("  8. Quit")

        choice = input("\nChoice: ").strip()

        if choice == '1':
            video_path = input("Video file path: ").strip()
            if not os.path.exists(video_path):
                print(f"File not found: {video_path}")
                continue
            rpm = float(input("RPM: "))
            arc_length = float(input("Arc length: "))
            trial = int(input("Trial number (default 1): ") or "1")
            analyzer.analyze_video(video_path, rpm, arc_length, trial)

        elif choice == '2':
            rpm = float(input("RPM: "))
            arc_length = float(input("Arc length: "))
            trial = int(input("Trial number (default 1): ") or "1")
            camera = int(input("Camera index (default 0): ") or "0")
            analyzer.analyze_live(rpm, arc_length, trial, camera)

        elif choice == '3':
            camera = int(input("Camera index (default 0): ") or "0")
            run_calibration(camera)

        elif choice == '4':
            ppm = float(input(f"Pixels per meter (current: {analyzer.pixels_per_meter}): "))
            analyzer.pixels_per_meter = ppm
            print(f"Set to {ppm} pixels/meter")

        elif choice == '5':
            x = int(input("Origin X (pixels): "))
            y = int(input("Origin Y (pixels): "))
            analyzer.set_origin(x, y)
            print(f"Origin set to ({x}, {y})")

        elif choice == '6':
            analyzer.print_summary()

        elif choice == '7':
            analyzer.export_to_csv()
            analyzer.export_summary_csv()

        elif choice == '8':
            if analyzer.session.trajectories:
                save = input("Export data before quitting? (y/n): ").strip().lower()
                if save == 'y':
                    analyzer.export_to_csv()
                    analyzer.export_summary_csv()
            print("Goodbye!")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()

# %%
