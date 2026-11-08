"""Green dot detection. OWNER: Harper

Detection uses the LAB a* channel, NOT HSV hue: hue shifts when the camera
re-balances white, a* does not. Do not switch back without a team decision.

Detection FAILS LOUDLY. If exactly two plausible dots are not present, return
(None, reason). Never "take the largest two" -- a silently wrong crop is worse
than no crop at all.
"""

import math

import cv2
import numpy as np

from greendot.config import CONFIG


def find_green_dots(frame):
    """Find the two green markers.

    Returns:
        (dots, reason) where dots is [(x, y, radius), (x, y, radius)] or None.
        reason is "ok" on success, otherwise a short string that goes straight
        into the capture log -- agree the exact strings with Markus so
        evaluate.py can count failure types.
    """
    # Blur first so camera noise doesn't split one dot into many tiny blobs.
    k = CONFIG["blur_kernel"]
    blurred = cv2.GaussianBlur(frame, (k, k), 0)

    # Why LAB a* and not HSV hue:
    # When lighting changes (desk lamp vs. window), the camera re-balances white
    # and tints the whole picture, so the hue of "green" drifts and a hue range
    # tuned at a desk stops matching near a window. In LAB, the a* channel
    # measures green-versus-red directly (128 is neutral, lower is greener) and
    # barely moves when the overall tint changes, so one threshold keeps working.
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Green enough AND not too dark (very dark pixels have unreliable colour).
    mask = ((A < CONFIG["a_max"]) & (L > CONFIG["l_min"])).astype(np.uint8) * 255

    # "Open" removes isolated specks left over after thresholding.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in contours:
        area = cv2.contourArea(c)
        # Size filter: rejects specks and big green objects (a plant, a folder).
        if not CONFIG["min_dot_area"] <= area <= CONFIG["max_dot_area"]:
            continue

        # Shape filter: a circle scores 1.0, a long highlighter stroke scores low.
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < CONFIG["min_circularity"]:
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        radius = math.sqrt(area / math.pi)
        dots.append((x, y, radius))

    # Exactly two, or a clear failure. Never keep "the two biggest": if a third
    # green thing is in view we cannot know which two are the real markers, and
    # a wrong crop is worse than no crop because the user never finds out.
    n = len(dots)
    if n < CONFIG["required_dots"]:
        return None, f"found {n} dots, need 2"
    if n > CONFIG["required_dots"]:
        return None, f"found {n} dots, need exactly 2"
    return dots, "ok"
