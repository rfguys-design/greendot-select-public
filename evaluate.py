"""Replay the labeled dataset and print the four team metrics. OWNER: Markus

    python evaluate.py          detection + crop only (free, no API calls)
    python evaluate.py --api    also ask the model, to measure task routing

This is the spine of the project. From Week 3 on, every pull request that claims
an improvement reports the before and after numbers from this script.
"""
# evaluate.py runs every photo in our labeled test set through the system and reports what percentage worked at each step, and, more importantly, counts exactly why each failure happened, so we always know what to fix next.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from greendot.config import CONFIG, chdir_to_project_root
from greendot.detect import find_green_dots
from greendot.imaging import crop_between_dots, downscale

# Same folder no matter where you run this from, so dataset paths always work.
chdir_to_project_root()

# The failure names from docs/METRICS.md, in the same order as the weekly
# template, so the breakdown can be copied straight into docs/weekly/.
TAXONOMY = ["no_dots", "extra_dots", "crop_truncated", "crop_intrusion",
            "ocr_error", "routing_error", "api_error"]


def crop_box(frame, crop):
    """Return (x, y, w, h): where in the frame the crop came from.

    crop_between_dots returns frame[y1:y2, x1:x2]. NumPy doesn't copy for that --
    the crop is a window pointing into the frame's memory. So the distance
    between the crop's first pixel and the frame's first pixel tells us exactly
    where the window starts. This keeps evaluate.py in sync with James's crop
    math without copying it here.
    """
    if not np.shares_memory(frame, crop):
        raise ValueError("crop is a copy, not a window into the frame -- "
                         "crop_box needs updating")
    offset = crop.ctypes.data - frame.ctypes.data
    y, rest = divmod(offset, frame.strides[0])  # strides[0] = bytes per row
    x = rest // frame.strides[1]                # strides[1] = bytes per pixel
    h, w = crop.shape[:2]
    return x, y, w, h


def encloses(box, region):
    """True if the crop box fully contains the region a human labeled."""
    x, y, w, h = box
    return (x <= region["x"] and y <= region["y"]
            and x + w >= region["x"] + region["w"]
            and y + h >= region["y"] + region["h"])


def detection_failure(reason):
    """Turn Harper's reason string into a METRICS.md failure name.

    detect.py currently says "found 1 dots, need 2" or "found 3 dots, need
    exactly 2". If detect.py ever returns "no_dots"/"extra_dots" directly
    (its docstring asks Harper and Markus to agree the strings), this still works.
    An unknown string is kept as-is instead of guessed, so nothing gets
    counted in the wrong bucket by accident.
    """
    if reason in ("no_dots", "extra_dots"):
        return reason
    if "need exactly" in reason:
        return "extra_dots"
    if "need 2" in reason:
        return "no_dots"
    return f"unclassified: {reason}"


def evaluate_one(img_path, label, use_api):
    """Run one capture through the pipeline. Returns a result dict.

    "failure" is the METRICS.md name of the FIRST thing that went wrong, or None
    on full success. Stopping at the first failure means every capture lands in
    exactly one bucket, so the bucket counts add up to the number of failures.
    "detail" says more, e.g. which task the model picked instead.
    """
    result = {"id": label.get("id", img_path.stem), "lighting": label.get("lighting"),
              "detected": False, "enclosed": False, "routed": False,
              "failure": None, "detail": ""}
    start = time.time()
    try:
        frame = cv2.imread(str(img_path))
        if frame is None:
            # Not in the taxonomy: it's a broken dataset file, not a system failure.
            result["failure"] = "bad_image_file"
            return result

        dots, reason = find_green_dots(frame)
        if dots is None:
            result["failure"] = detection_failure(reason)
            result["detail"] = reason
            return result
        result["detected"] = True

        crop = crop_between_dots(frame, dots)
        if crop is None:
            # No crop at all means the whole target was cut off.
            result["failure"] = "crop_truncated"
            result["detail"] = "crop smaller than min_crop_dimension"
            return result
        if not encloses(crop_box(frame, crop), label["intended_region"]):
            result["failure"] = "crop_truncated"
            result["detail"] = "missed part of intended region"
            return result
        # TODO (team decision): crop_intrusion -- how many pixels past the
        # target counts as "adjacent content"? Agree with James, put it in config.
        result["enclosed"] = True

        if use_api:
            # Imported here so the free mode never needs an API key.
            from greendot.brain import ask_model
            # Its own try, so a network problem is counted as api_error and
            # not confused with a bug in our own code.
            try:
                reply = ask_model(downscale(crop), label.get("transcript", ""))
            except Exception as e:
                result["failure"] = "api_error"
                result["detail"] = type(e).__name__
                return result
            result["routed"] = reply["task"] == label["intended_task"]
            if not result["routed"]:
                result["failure"] = "routing_error"
                result["detail"] = f"chose {reply['task']}, wanted {label['intended_task']}"
                return result

    except Exception as e:
        # One broken capture must not stop the whole evaluation. Not in the
        # taxonomy on purpose: this means a bug in OUR code, so fix it first.
        result["failure"] = "code_error"
        result["detail"] = f"{type(e).__name__}: {e}"
    finally:
        result["latency_s"] = time.time() - start
    return result


def percent(hits, total):
    """'80.0%  (20/25)' -- always show the counts, so a % from 3 images
    can't pass for one from 300."""
    if total == 0:
        return "n/a"
    return f"{100 * hits / total:5.1f}%  ({hits}/{total})"


def main():
    use_api = "--api" in sys.argv
    captures_dir = Path(CONFIG["dataset_dir"]) / "captures"

    # 1. Load every image that has a label. An unlabeled image can't be scored.
    pairs = []
    for img_path in sorted(captures_dir.glob("*.jpg")):
        label_path = img_path.with_suffix(".json")
        if label_path.exists():
            pairs.append((img_path, json.loads(label_path.read_text(encoding="utf-8"))))
        else:
            print(f"[skip] {img_path.name} has no .json label")

    if not pairs:
        print(f"No labeled captures found in {captures_dir}")
        return

    # API calls cost money, so say how many and ask first. This is the most
    # it could be: captures that fail detection or cropping never reach the model.
    if use_api:
        print(f"--api will make up to {len(pairs)} API calls (one per capture).")
        if input("Continue? [y/N] ").strip().lower() != "y":
            print("Cancelled. Run without --api for the free metrics.")
            return

    # 2. Run the pipeline on each capture.
    results = []
    for img_path, label in pairs:
        r = evaluate_one(img_path, label, use_api)
        print(f"  {r['id']:>6}  {'OK' if r['failure'] is None else r['failure']}")
        results.append(r)

    # 3. Headline numbers, using the exact definitions in docs/METRICS.md.
    # Enclosure is divided by crops attempted (= captures where dots were
    # found), so James's number isn't dragged down by detection misses.
    total = len(results)
    detected = [r for r in results if r["detected"]]
    enclosed = [r for r in detected if r["enclosed"]]
    successes = [r for r in results if r["failure"] is None]

    print()
    print("=" * 60)
    print(f"  RESULTS  ({total} captures, {'with' if use_api else 'without'} API)")
    print("=" * 60)
    print(f"  Detection rate      {percent(len(detected), total)}")
    print(f"  Enclosure rate      {percent(len(enclosed), len(detected))}")
    if use_api:
        # METRICS.md divides by ALL captures: a capture that never reached
        # the model didn't get the right task either.
        routed = [r for r in results if r["routed"]]
        print(f"  Routing accuracy    {percent(len(routed), total)}")
    else:
        print("  Routing accuracy    not measured (run with --api)")
    print(f"  End-to-end success  {percent(len(successes), total)}")
    # Without --api this is only detection + crop time, not the real tap-to-answer time.
    print(f"  Median latency      {statistics.median(r['latency_s'] for r in results):.2f} s")

    # 4. The failure breakdown -- the most useful part.
    failures = Counter(r["failure"] for r in results if r["failure"])
    examples = defaultdict(list)
    for r in results:
        if r["failure"]:
            examples[r["failure"]].append(f"{r['id']} ({r['detail']})" if r["detail"] else r["id"])

    # METRICS.md: "latency" = correct but slower than the target. It's a
    # warning, not a failure -- the answer was still right, so it still counts
    # as an end-to-end success. Only meaningful with --api (the model call is
    # the slow part).
    slow = [r["id"] for r in successes if use_api and r["latency_s"] > CONFIG["latency_target_s"]]

    print()
    print("#" * 60)
    print(f"  FAILURE BREAKDOWN  ({sum(failures.values())} of {total} failed)")
    print("#" * 60)
    # Every taxonomy name is printed, even at 0, so it matches the weekly
    # template row for row. Biggest first: that's what to fix next.
    names = sorted(TAXONOMY, key=lambda name: -failures[name])
    names += [name for name, _ in failures.most_common() if name not in TAXONOMY]
    for name in names:
        if name == "ocr_error":
            print(f"    -  {name:<16} not measured (no OCR path in evaluate.py yet)")
        elif name == "routing_error" and not use_api:
            print(f"    -  {name:<16} not measured (run with --api)")
        else:
            print(f"  {failures[name]:3d}  {name}")
        # The ids tell you which images to open and look at.
        if examples[name]:
            print(f"       {', '.join(examples[name])}")
    if use_api:
        print(f"  {len(slow):3d}  latency (correct, but over {CONFIG['latency_target_s']} s)")
        if slow:
            print(f"       {', '.join(slow)}")

    # Lighting is the most common cause of detection problems, so show it too.
    print()
    print("  Failures by lighting:")
    by_lighting = Counter(r["lighting"] for r in results)
    failed_lighting = Counter(r["lighting"] for r in results if r["failure"])
    for lighting, n in sorted(by_lighting.items(), key=lambda kv: str(kv[0])):
        print(f"    {str(lighting):<12} {failed_lighting[lighting]} of {n} failed")


if __name__ == "__main__":
    main()
