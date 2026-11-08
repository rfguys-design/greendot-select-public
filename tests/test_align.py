"""Page alignment tests. Synthetic images, no camera needed.

A dark desk, a white page turned by a known angle, a dark text block on the
page, and green dots just outside two opposite corners of the text block.
"""

import math

import cv2
import numpy as np
import pytest

from greendot.align import crop_aligned, find_page_angle

TEXT_W, TEXT_H = 360, 90


def turned(points, angle, center):
    """Turn points by angle degrees (positive = clockwise on screen) about center."""
    a = math.radians(angle)
    cx, cy = center
    return [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
             cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)) for x, y in points]


def scene(angle, page_center=(640, 360), page_size=(700, 500)):
    img = np.full((720, 1280, 3), 40, np.uint8)                  # dark desk mat
    cx, cy = page_center
    pw, ph = page_size
    page = [(cx - pw / 2, cy - ph / 2), (cx + pw / 2, cy - ph / 2),
            (cx + pw / 2, cy + ph / 2), (cx - pw / 2, cy + ph / 2)]
    cv2.fillPoly(img, [np.int32(turned(page, angle, page_center))], (245, 245, 245))

    # Text block centred where the dots will be, turned with the page.
    tx, ty = 640, 360
    text = [(tx - TEXT_W / 2, ty - TEXT_H / 2), (tx + TEXT_W / 2, ty - TEXT_H / 2),
            (tx + TEXT_W / 2, ty + TEXT_H / 2), (tx - TEXT_W / 2, ty + TEXT_H / 2)]
    cv2.fillPoly(img, [np.int32(turned(text, angle, page_center))], (30, 30, 30))

    # Dots sit diagonally just outside the text block's top-left and
    # bottom-right corners, the way a user places stickers in the margin.
    r, gap = 14, 20
    corners = [(tx - TEXT_W / 2 - gap, ty - TEXT_H / 2 - gap),
               (tx + TEXT_W / 2 + gap, ty + TEXT_H / 2 + gap)]
    dots = []
    for x, y in turned(corners, angle, page_center):
        cv2.circle(img, (int(round(x)), int(round(y))), r, (60, 210, 60), -1)
        dots.append((x, y, float(r)))
    return img, dots


def green_pixels(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return int((lab[:, :, 1] < 115).sum())


@pytest.mark.parametrize("angle", [0, 8, -12, 25, -30])
def test_angle_is_measured(angle):
    img, dots = scene(angle)
    found, reason = find_page_angle(img, dots)
    assert reason == "ok"
    assert abs(found - angle) < 1.5


@pytest.mark.parametrize("angle", [8, -12, 25])
def test_turned_page_gives_straight_full_crop(angle):
    img, dots = scene(angle)
    crop, used, reason = crop_aligned(img, dots)
    assert reason == "ok"
    assert crop is not None
    h, w = crop.shape[:2]
    # The crop should hug the text block plus the margin gap, not a skewed sliver.
    assert abs(w - TEXT_W) < 40 and abs(h - TEXT_H) < 40
    assert green_pixels(crop) == 0


def test_partial_page_off_frame_still_aligns():
    # Page centre near the right border: most of the sheet is outside the frame,
    # so the frame border cuts the outline. Only the paper's real edges count.
    img, dots = scene(15, page_center=(900, 360), page_size=(900, 700))
    found, reason = find_page_angle(img, dots)
    assert reason == "ok"
    assert abs(found - 15) < 1.5


def test_no_paper_falls_back_to_plain_crop():
    img, dots = scene(0)
    img[:] = 245                                   # all white: no page edge anywhere
    for x, y, r in dots:
        cv2.circle(img, (int(x), int(y)), int(r), (60, 210, 60), -1)
    crop, used, reason = crop_aligned(img, dots)
    assert used == 0.0
    assert reason != "ok"
    assert crop is not None
