"""
Detection rate on the labeled dataset. OWNER: Harper

    python3 tools/check_detection.py

Runs find_green_dots() on every dataset/captures/*.jpg and prints how often it
found exactly two dots, what went wrong when it didn't, and which files failed.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

import cv2

from greendot.config import CONFIG
from greendot.detect import find_green_dots

captures = Path(CONFIG["dataset_dir"]) / "captures"
images = sorted(captures.glob("*.jpg"))
if not images:
    sys.exit(f"No .jpg files in {captures} -- build the dataset first (Week 2).")

failures = []  # (filename, reason, lighting)
for img_path in images:
    label_path = img_path.with_suffix(".json")
    label = json.loads(label_path.read_text()) if label_path.exists() else {}
    if not label:
        print(f"  warning: no label for {img_path.name}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        failures.append((img_path.name, "could not read image", "?"))
        continue

    _, reason = find_green_dots(frame)
    if reason != "ok":
        failures.append((img_path.name, reason, label.get("lighting", "?")))

total = len(images)
ok = total - len(failures)
print(f"\nDetection rate: {ok}/{total} = {100 * ok / total:.1f}%   (a_max = {CONFIG['a_max']})")

if failures:
    print("\nFailures by reason:")
    for reason, count in Counter(r for _, r, _ in failures).most_common():
        print(f"  {count:3d}  {reason}")

    # Lighting is printed alongside each file because failures often share one
    # lighting condition -- that pattern is what Task 2.2 looks for.
    print("\nFailed files:")
    for name, reason, lighting in failures:
        print(f"  {name:12s} {lighting:12s} {reason}")
