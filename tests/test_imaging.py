"""Crop tests. OWNER: James

The zero-green-pixels test is the one that matters. Write it first.
"""

import cv2
import numpy as np
import pytest

from greendot.imaging import crop_between_dots, downscale

TEXT_BOX = (420, 300, 860, 420)   # x1, y1, x2, y2


def page(dots):
    img = np.full((720, 1280, 3), 245, np.uint8)
    x1, y1, x2, y2 = TEXT_BOX
    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
    for (x, y, r) in dots:
        cv2.circle(img, (x, y), int(r), (60, 210, 60), -1)
    return img


def test_crop_matches_text_block():
    dots = [(400, 280, 14.0), (880, 440, 14.0)]
    crop = crop_between_dots(page(dots), dots)
    h, w = crop.shape[:2]
    assert abs(w - (TEXT_BOX[2] - TEXT_BOX[0])) < 20
    assert abs(h - (TEXT_BOX[3] - TEXT_BOX[1])) < 20


def test_crop_contains_no_green():
    """THE test. If this fails, the crop is insetting outward, not inward."""
    dots = [(400, 280, 14.0), (880, 440, 14.0)]
    crop = crop_between_dots(page(dots), dots)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    assert int((lab[:, :, 1] < 115).sum()) == 0


def test_opposite_diagonal_gives_same_crop():
    a = [(400, 440, 14.0), (880, 280, 14.0)]     # bottom-left / top-right
    b = [(400, 280, 14.0), (880, 440, 14.0)]     # top-left / bottom-right
    ca, cb = crop_between_dots(page(a), a), crop_between_dots(page(b), b)
    assert ca.shape == cb.shape


def test_dots_too_close_returns_none():
    dots = [(600, 360, 14.0), (612, 368, 14.0)]
    assert crop_between_dots(page(dots), dots) is None


def test_downscale_caps_long_edge():
    img = np.zeros((720, 1280, 3), np.uint8)
    out = downscale(img)
    assert max(out.shape[:2]) <= 1024
