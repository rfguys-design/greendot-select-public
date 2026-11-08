"""Crop geometry and image preparation. OWNER: James

The crop insets INWARD past each dot's radius. The first prototype padded
OUTWARD by 100px, so every capture pulled in the neighbouring problem. There is
a unit test asserting zero green pixels inside the crop -- never weaken it.
"""

import os

import cv2
import numpy as np

from greendot.config import CONFIG


def crop_between_dots(frame, dots):
    """Return the region between two dots, inset past each dot's radius.

    The dots may be in either diagonal arrangement, so work out which radius
    applies to which edge rather than assuming an order. Clamp to the frame.

    Returns the cropped image, or None if the region is smaller than
    CONFIG["min_crop_dimension"] in either direction.
    """
    # A clean failure is safer than guessing when detection returns bad input.
    if len(dots) != 2:
        return None

    inset = CONFIG["inset_margin"]
    # Sorting each axis independently handles both possible diagonal layouts.
    left_dot, right_dot = sorted(dots, key=lambda dot: dot[0])
    top_dot, bottom_dot = sorted(dots, key=lambda dot: dot[1])

    # Inset inward so the green stickers and nearby content stay outside the crop.
    x1 = int(np.ceil(left_dot[0] + left_dot[2] + inset))
    x2 = int(np.floor(right_dot[0] - right_dot[2] - inset))
    y1 = int(np.ceil(top_dot[1] + top_dot[2] + inset))
    y2 = int(np.floor(bottom_dot[1] - bottom_dot[2] - inset))

    # Keep the slice inside the image even when a marker touches an edge.
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if (x2 - x1) < CONFIG["min_crop_dimension"] or (y2 - y1) < CONFIG["min_crop_dimension"]:
        return None

    return frame[y1:y2, x1:x2]


def downscale(img):
    """Resize so the long edge is at most CONFIG['max_long_edge'] (INTER_AREA)."""
    height, width = img.shape[:2]
    long_edge = max(height, width)
    max_long_edge = CONFIG["max_long_edge"]
    if long_edge <= max_long_edge:
        return img

    scale = max_long_edge / long_edge
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # INTER_AREA is designed to reduce aliasing when shrinking an image.
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def enhance_for_display(crop):
    """A scan-like copy for people to read: white paper, dark text, colour kept.

    For the dashboard only. The model and OCR get the original crop, because
    this step makes its own choices about what is "paper" and what is "ink".
    """
    # 1. Estimate the paper's brightness everywhere: a big closing erases the
    #    thin dark strokes of text, leaving just the lamp's light and shadow.
    #    Light changes slowly across a page, so this is done on a 1/8 copy and
    #    scaled back up -- same result, ~50x faster than at full size.
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    k = max(5, (min(small.shape[:2]) // 12) | 1)
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 3)
    background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    # 2. Dividing by it flattens shadows and turns grey-blue paper white.
    flat = crop.astype(np.float32) / np.maximum(background.astype(np.float32), 1) * 255
    # 3. Stretch levels: the darkest 1% becomes black, and a gamma > 1 makes
    #    faint grey strokes noticeably darker without touching the white paper.
    low = np.percentile(flat, 1)
    out = np.clip((flat - low) / (255 - low), 0, 1) ** 1.6 * 255
    return out.astype(np.uint8)


def preprocess_for_ocr(crop):
    """Grayscale + adaptive threshold, upscaled toward CONFIG['ocr_target_dpi'].

    OCR accuracy on marginal images is dominated by preprocessing, not by the
    choice of OCR engine.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # The physical size of the region is unknown, so scale by letter height
    # instead: Tesseract reads best when capital letters are ~30 px tall.
    # A crop narrower than ~1800 px usually holds text smaller than that.
    scale = min(3.0, max(1.0, CONFIG["ocr_min_width"] / gray.shape[1]))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold copes with uneven lamp light across the page, which a
    # single global threshold does not.
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        CONFIG["ocr_threshold_block"], CONFIG["ocr_threshold_c"],
    )


def _find_tesseract():
    """pytesseract needs the tesseract program; the Windows installer does not add it to PATH."""
    import shutil
    import pytesseract
    if shutil.which("tesseract") is None:
        for path in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    return pytesseract


def _tesseract_words(img, pytesseract):
    """Run tesseract once. Returns (text with line/paragraph breaks, mean word confidence)."""
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    lines, confs = {}, []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    if not confs:
        return "", 0.0

    out, previous = [], None
    for key in sorted(lines):
        # A new block or paragraph gets a blank line; a new line just a newline.
        if previous is not None and key[:2] != previous[:2]:
            out.append("")
        out.append(" ".join(lines[key]))
        previous = key
    return "\n".join(out), sum(confs) / len(confs)


def local_ocr(crop):
    """Verbatim transcription with confidence. Returns (text, mean_confidence).

    Used ONLY for the literal-copy path. A language model transcribing an image
    generates text -- it silently fixes typos and normalises punctuation, which
    is wrong when the user asked to copy exactly. OCR fails visibly instead.

    Returns (None, 0.0) if pytesseract is unavailable.
    """
    try:
        pytesseract = _find_tesseract()
        pytesseract.get_tesseract_version()
    except Exception:
        return None, 0.0

    # Thresholding helps on shadowed pages and hurts on clean ones, so try both
    # and keep whichever tesseract is more confident about.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = min(3.0, max(1.0, CONFIG["ocr_min_width"] / gray.shape[1]))
    plain = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    best = ("", 0.0)
    for img in (preprocess_for_ocr(crop), plain):
        text, conf = _tesseract_words(img, pytesseract)
        if conf > best[1]:
            best = (text, conf)
    return best
