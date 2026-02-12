#%%
import cv2
import numpy as np

def show_hsv_values(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        hsv_frame = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print("HSV at", (x, y), ":", hsv_frame[y, x])

def is_semicircle(contour, min_area=500):
    """
    Detect if a contour is a semi-circular shape.
    Returns (is_semicircle, score) where score indicates confidence.
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
        # Semi-circles when fit to ellipse have aspect ratio ~1.5-2.5
        if 1.3 < ellipse_aspect < 3.0:
            ellipse_match = 1

    # Semi-circle criteria:
    # - Circularity between 0.4 and 0.85 (lower than full circle)
    # - Aspect ratio between 1.3 and 2.5 (elongated in one direction)
    # - Extent between 0.45 and 0.75 (fills about half the bounding rect)
    # - High solidity (convex shape)

    is_semi = (
        0.4 < circularity < 0.85 and
        1.3 < aspect_ratio < 2.5 and
        0.45 < extent < 0.75 and
        solidity > 0.85
    )

    # Calculate confidence score
    score = 0
    if is_semi:
        # Ideal semi-circle values
        circ_score = 1 - abs(circularity - 0.6) / 0.3
        aspect_score = 1 - abs(aspect_ratio - 1.8) / 0.7
        extent_score = 1 - abs(extent - 0.55) / 0.2
        score = area * solidity * (circ_score + aspect_score + extent_score + ellipse_match) / 4

    return is_semi, score


def runPipeline(image):
    output = image.copy()
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green, upper_green = (35, 80, 80), (85, 255, 255)
    lower_purple, upper_purple = (115, 40, 40), (170, 255, 255)

    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_purple = cv2.inRange(img_hsv, lower_purple, upper_purple)
    mask_combined = cv2.bitwise_or(mask_green, mask_purple)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes = []
    green_candidates = []
    purple_candidates = []

    for contour in contours:
        # Check if contour is a semi-circle
        is_semi, score = is_semicircle(contour)

        if is_semi:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if mask_green[center_y, center_x] > 0:
                green_candidates.append((contour, x, y, w, h, center_x, center_y, score))
            elif mask_purple[center_y, center_x] > 0:
                purple_candidates.append((contour, x, y, w, h, center_x, center_y, score))

    if green_candidates:
        best_green = max(green_candidates, key=lambda c: c[7])
        contour, x, y, w, h, center_x, center_y, _ = best_green
        detected_shapes.append((center_x, 'G'))
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        cv2.putText(output, 'G-Semi', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if purple_candidates:
        best_purple = max(purple_candidates, key=lambda c: c[7])
        contour, x, y, w, h, center_x, center_y, _ = best_purple
        detected_shapes.append((center_x, 'P'))
        cv2.drawContours(output, [contour], -1, (255, 0, 255), 2)
        cv2.putText(output, 'P-Semi', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    detected_shapes.sort(key=lambda b: b[0])
    shape_order = ''.join([b[1] for b in detected_shapes])

    cv2.putText(output, f"Semi-circles: {shape_order}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return shape_order, output, mask_clean, mask_green, mask_purple


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Color Output")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        shape_order, output, mask_combined, mask_green, mask_purple = runPipeline(frame)

        if shape_order:
            print(f"Semi-circle order (left to right): {shape_order}")

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
