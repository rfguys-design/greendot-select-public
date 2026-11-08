"""Visual check: run detection and cropping on a saved frame. OWNER: James

    python tools/show_crop.py dataset/captures/0001.jpg

You will use this constantly in Week 3 when diagnosing enclosure failures.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

import cv2

from greendot.detect import find_green_dots
from greendot.config import CONFIG
from greendot.imaging import crop_between_dots


def main(path):
    # The script changes into tools/, so resolve relative paths from the project root.
    image_path = path if os.path.isabs(path) else os.path.join("..", path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"could not load image: {path}")

    dots, reason = find_green_dots(frame)
    annotated = frame.copy()
    if dots is None:
        print(f"detection failed: {reason}")
        cv2.imshow("original", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Draw the detected markers before drawing the crop boundary for inspection.
    for x, y, radius in dots:
        cv2.circle(annotated, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    crop = crop_between_dots(frame, dots)
    if crop is None:
        print("crop failed: region is too small")
        cv2.imshow("original", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    height, width = frame.shape[:2]
    left_dot, right_dot = sorted(dots, key=lambda dot: dot[0])
    top_dot, bottom_dot = sorted(dots, key=lambda dot: dot[1])
    inset = CONFIG["inset_margin"]
    x1 = max(0, min(int(left_dot[0] + left_dot[2] + inset), width))
    x2 = max(0, min(int(right_dot[0] - right_dot[2] - inset), width))
    y1 = max(0, min(int(top_dot[1] + top_dot[2] + inset), height))
    y2 = max(0, min(int(bottom_dot[1] - bottom_dot[2] - inset), height))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print(f"crop dimensions: {crop.shape[1]} x {crop.shape[0]}")
    cv2.imshow("original", annotated)
    cv2.imshow("crop", crop)
    cv2.moveWindow("crop", frame.shape[1] + 20, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tools/show_crop.py <image path>")
    main(sys.argv[1])
