"""Live check: camera -> green dot detection -> crop, no voice or AI needed.

    python tools/live_detect.py

Runs while trigger.py and voice.py are still stubs, so it is the quickest way
to see Harper's camera + detection and James's crop working together. The crop
is straightened to the page edge first (greendot/align.py).

Keys in the window:
    Q quit | E/D exposure | A/S green threshold (a_max) | C save current crop
"""

import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

import cv2

from greendot.camera import CameraStream
from greendot.config import CONFIG, ensure_dirs
from greendot.detect import find_green_dots
from greendot.align import crop_aligned

# OpenCV colours are Blue-Green-Red.
GREEN, ORANGE, RED = (0, 255, 0), (0, 165, 255), (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    ensure_dirs()
    camera = CameraStream()
    print(f"Camera {CONFIG['camera_index']} open.")
    print("Keys: Q quit | E/D exposure | A/S green threshold | C save crop")

    crop = None
    try:
        while True:
            frame = camera.read()
            if frame is None:
                # First frame has not arrived yet; keep the window responsive.
                cv2.waitKey(10)
                continue

            # Run detection on the clean frame, draw on a separate copy.
            dots, reason = find_green_dots(frame)
            annotated = frame.copy()

            if dots is None:
                crop = None
                cv2.putText(annotated, reason, (20, 40), FONT, 1.0, ORANGE, 2)
            else:
                for x, y, radius in dots:
                    cv2.circle(annotated, (int(x), int(y)), int(radius), GREEN, 2)
                crop, angle, align_reason = crop_aligned(frame, dots)
                # Show the measured page angle, or why alignment fell back.
                align_text = f"page {angle:+.1f} deg" if align_reason == "ok" else align_reason
                cv2.putText(annotated, align_text, (20, 80), FONT, 0.8, GREEN if align_reason == "ok" else ORANGE, 2)
                if crop is None:
                    cv2.putText(annotated, "region too small", (20, 40), FONT, 1.0, RED, 2)
                else:
                    h, w = crop.shape[:2]
                    cv2.putText(annotated, f"crop {w} x {h}", (20, 40), FONT, 1.0, GREEN, 2)
                    cv2.imshow("crop", crop)

            settings = f"a_max {CONFIG['a_max']}  exposure {CONFIG['manual_exposure']}"
            cv2.putText(annotated, settings, (20, annotated.shape[0] - 20), FONT, 0.7, GREEN, 2)
            cv2.imshow("live detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("e"), ord("d")):
                camera.set_exposure(CONFIG["manual_exposure"] + (1 if key == ord("e") else -1))
                print(f"[tune] manual_exposure = {CONFIG['manual_exposure']}")
            elif key in (ord("a"), ord("s")):
                # Same 1..127 range that validate() accepts; 128 is neutral.
                step = 1 if key == ord("a") else -1
                CONFIG["a_max"] = min(127, max(1, CONFIG["a_max"] + step))
                print(f"[tune] a_max = {CONFIG['a_max']}")
            elif key == ord("c"):
                if crop is None:
                    print("[save] no crop to save")
                else:
                    path = os.path.join(CONFIG["crop_dir"], time.strftime("live_%Y%m%d_%H%M%S.jpg"))
                    cv2.imwrite(path, crop)
                    print(f"[save] {path}")
    finally:
        # Release the camera even on Ctrl+C, or it can stay locked.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
