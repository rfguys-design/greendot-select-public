"""Page alignment: straighten the crop using the edge of the paper.

Two dots give opposite corners but not rotation. If the page is turned, the
rectangle between the dots no longer lines up with the text, and the crop cuts
words off. A sheet of paper has straight edges, so one visible piece of edge is
enough to measure how far the page is turned. We rotate the frame to cancel
that angle, then crop exactly as before with crop_between_dots().

The whole page does not need to be in view -- only part of one edge.
"""

import math

import cv2
import numpy as np

from greendot.config import CONFIG
from greendot.imaging import crop_between_dots


def find_page_angle(frame, dots):
    """Return (angle_degrees, reason). angle is None if no paper edge was found.

    The angle is how far the paper's edges are turned from the frame axes,
    folded into -45..45 degrees: a page turned 90 degrees looks square again.
    """
    # Paper is much brighter than a desk mat. Otsu picks the cut-off between
    # the two brightness groups automatically, so it adapts to room lighting.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Closing fills the dark text and the green dots, so the sheet is one blob.
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # The paper is the bright blob UNDER the dots. Other bright things in view
    # (a white desk, a keyboard) are ignored even if they are bigger.
    mx = (dots[0][0] + dots[1][0]) / 2
    my = (dots[0][1] + dots[1][1]) / 2
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    paper = None
    for c in contours:
        if cv2.pointPolygonTest(c, (float(mx), float(my)), False) >= 0:
            paper = c
            break
    if paper is None:
        return None, "no paper under the dots"

    # Draw the paper's outline, then erase a strip along the frame border.
    # Where the page runs off-screen its "edge" is really the camera's edge,
    # which is always perfectly straight -- it would always vote for 0 degrees.
    h, w = gray.shape
    outline = np.zeros_like(gray)
    cv2.drawContours(outline, [paper], -1, 255, 2)
    m = CONFIG["page_border_margin"]
    outline[:m, :] = 0
    outline[-m:, :] = 0
    outline[:, :m] = 0
    outline[:, -m:] = 0

    min_len = int(CONFIG["page_edge_min_frac"] * max(h, w))
    lines = cv2.HoughLinesP(outline, 1, np.pi / 180, threshold=min_len // 2,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return None, "no straight paper edge in view"

    # Each segment votes with its length, so one long true edge beats a few
    # short wobbly bits of a curled corner.
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45          # fold so horizontal and vertical edges agree
        angles.append(a)
        weights.append(math.hypot(x2 - x1, y2 - y1))

    # Weighted median: robust to a stray diagonal segment at a torn corner.
    order = np.argsort(angles)
    cum = np.cumsum(np.array(weights)[order])
    angle = float(np.array(angles)[order][np.searchsorted(cum, cum[-1] / 2)])
    return angle, "ok"


def rotate_frame_and_dots(frame, dots, angle):
    """Rotate the frame by angle around the dots' midpoint; move the dots too."""
    mx = (dots[0][0] + dots[1][0]) / 2
    my = (dots[0][1] + dots[1][1]) / 2
    # A positive angle in getRotationMatrix2D turns the picture anticlockwise,
    # which levels an edge that slopes down to the right by that many degrees.
    M = cv2.getRotationMatrix2D((mx, my), angle, 1.0)
    h, w = frame.shape[:2]
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    new_dots = []
    for x, y, r in dots:
        nx, ny = M @ np.array([x, y, 1.0])
        new_dots.append((float(nx), float(ny), r))
    return rotated, new_dots


def crop_outline(frame, dots):
    """Corners of the region crop_aligned() would cut, in the ORIGINAL frame.

    Returns (corners, angle, reason). corners is a 4x2 int array for
    cv2.polylines. The preview draws it so you can see the box tilt with the
    page -- if it sits square on the text, the alignment worked.
    """
    angle, reason = (0.0, "alignment off")
    if CONFIG["align_to_page"]:
        angle, reason = find_page_angle(frame, dots)
        if angle is None:
            angle = 0.0
    mx = (dots[0][0] + dots[1][0]) / 2
    my = (dots[0][1] + dots[1][1]) / 2
    M = cv2.getRotationMatrix2D((mx, my), angle, 1.0)
    # In the straightened frame the crop is an upright box between the dots...
    (x1, y1), (x2, y2) = [M @ np.array([x, y, 1.0]) for x, y, _ in dots]
    box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    # ...so turning its corners back by the same angle shows it on the camera image.
    back = cv2.invertAffineTransform(M)
    corners = box @ back[:, :2].T + back[:, 2]
    return corners.astype(np.int32), angle, reason


def crop_aligned(frame, dots):
    """Like crop_between_dots(), but straightened to the page first.

    Returns (crop, angle, reason). angle is 0.0 when alignment is off or no
    edge was found; the crop then falls back to the unrotated behaviour.
    """
    if not CONFIG["align_to_page"]:
        return crop_between_dots(frame, dots), 0.0, "alignment off"
    angle, reason = find_page_angle(frame, dots)
    if angle is None:
        # Falling back is safe: it is exactly what we did before alignment
        # existed, and the reason is reported so the log shows how often.
        return crop_between_dots(frame, dots), 0.0, reason
    rotated, new_dots = rotate_frame_and_dots(frame, dots, angle)
    return crop_between_dots(rotated, new_dots), angle, "ok"
