"""Detection tests. OWNER: Harper

These build synthetic images with NumPy and OpenCV -- no camera needed, so they
run anywhere and in CI.
"""

import cv2
import numpy as np
import pytest

from greendot.detect import find_green_dots


def page(dots=(), smear=False, speck=False):
    """White page with a dark text block and optional green markers."""
    img = np.full((720, 1280, 3), 245, np.uint8)
    cv2.rectangle(img, (420, 300), (860, 420), (30, 30, 30), -1)
    for (x, y) in dots:
        cv2.circle(img, (x, y), 14, (60, 210, 60), -1)
    if smear:
        cv2.rectangle(img, (150, 600), (320, 612), (60, 210, 60), -1)
    if speck:
        cv2.circle(img, (1000, 100), 2, (60, 210, 60), -1)
    return img


def test_two_dots_found():
    dots, reason = find_green_dots(page([(400, 280), (880, 440)]))
    assert reason == "ok"
    assert len(dots) == 2


def test_three_dots_fails_loudly():
    dots, reason = find_green_dots(page([(400, 280), (880, 440), (200, 600)]))
    assert dots is None
    assert "2" in reason


def test_one_dot_fails():
    dots, _ = find_green_dots(page([(400, 280)]))
    assert dots is None


def test_smear_rejected_by_circularity():
    dots, reason = find_green_dots(page([(400, 280), (880, 440)], smear=True))
    assert reason == "ok", "a long green smear must not count as a dot"
    assert len(dots) == 2


def test_speck_rejected_by_area():
    dots, reason = find_green_dots(page([(400, 280), (880, 440)], speck=True))
    assert reason == "ok", "a tiny green speck must not count as a dot"
    assert len(dots) == 2
