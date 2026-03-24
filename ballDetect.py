#%%
"""
Ball detection + tracking using OpenCV tracker.
"""

import cv2
import numpy as np

# GREEN THRESHOLDS (HSV + YCrCb AND)
GREEN_HSV_LOWER = np.array([65, 100, 40])
GREEN_HSV_UPPER = np.array([90, 255, 220])
GREEN_YCRCB_LOWER = np.array([50, 35, 60])
GREEN_YCRCB_UPPER = np.array([160, 100, 145])

# PURPLE THRESHOLDS (YCrCb only)
PURPLE_YCRCB_LOWER = np.array([80, 135, 130])
PURPLE_YCRCB_UPPER = np.array([170, 175, 180])

MIN_BALL_AREA = 300
MAX_BALL_AREA = 20000


def detect_balls(frame):
    """Detect green and purple balls."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # GREEN
    green_hsv = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    green_ycrcb = cv2.inRange(ycrcb, GREEN_YCRCB_LOWER, GREEN_YCRCB_UPPER)
    green_mask = cv2.bitwise_and(green_hsv, green_ycrcb)

    # PURPLE
    purple_mask = cv2.inRange(ycrcb, PURPLE_YCRCB_LOWER, PURPLE_YCRCB_UPPER)

    balls = []

    for mask, color in [(green_mask, 'G'), (purple_mask, 'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_BALL_AREA < area < MAX_BALL_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                balls.append({
                    'bbox': (x, y, w, h),
                    'color': color,
                    'contour': cnt
                })

    # Sort left-to-right by x position
    balls.sort(key=lambda b: b['bbox'][0])
    return balls, green_mask, purple_mask


def get_ball_colors(frame):
    """Detect balls and return their colors as a string, left-to-right.
    e.g. 'GPG' means Green, Purple, Green from left to right."""
    balls, _, _ = detect_balls(frame)
    return ''.join(b['color'] for b in balls)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    trackers = []  # list of (tracker, color)
    tracking = False

    print("=" * 40)
    print("BALL TRACKING")
    print("'s' = start tracking detected balls")
    print("'r' = reset trackers")
    print("'q' = quit")
    print("=" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()

        if tracking and trackers:
            # Update trackers
            for i, (tracker, color) in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    clr = (0, 255, 0) if color == 'G' else (255, 0, 255)
                    cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)
                    cv2.putText(output, f"{color}{i}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

            cv2.putText(output, f"TRACKING {len(trackers)} balls", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Detection mode
            balls, green_mask, purple_mask = detect_balls(frame)

            for ball in balls:
                x, y, w, h = ball['bbox']
                clr = (0, 255, 0) if ball['color'] == 'G' else (255, 0, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)

            cv2.putText(output, f"DETECT: {len(balls)} balls - press 's' to track",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Ball Tracking", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Start tracking current detections
            balls, _, _ = detect_balls(frame)
            trackers = []
            for ball in balls:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, ball['bbox'])
                trackers.append((tracker, ball['color']))
            tracking = True
            print(f"Tracking {len(trackers)} balls")
        elif key == ord('r'):
            trackers = []
            tracking = False
            print("Reset")

    cap.release()
    cv2.destroyAllWindows()

# %%
