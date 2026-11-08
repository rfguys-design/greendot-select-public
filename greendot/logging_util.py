"""Capture logging. OWNER: Markus

One CSV row per capture -- successes AND failures, written in a finally block.
Every number in the final writeup comes from this file. Without it you cannot
tell whether a change helped.
"""

import csv
import os
import threading

from greendot.config import CONFIG

_write_lock = threading.Lock()

# Every time someone uses GreenDot, whether it works or fails, this function adds one line to a spreadsheet recording what happened, so we can measure our real success rate instead of guessing
def log_capture(row):
    """Append one row, writing the header if the file is new. Thread-safe."""
    path = CONFIG["log_file"]
    # The lock covers the check AND the write. Otherwise two threads could both
    # see "no file yet" and both write a header, or mix their rows together.
    with _write_lock:
        # A missing or empty file needs a header, so the CSV is readable on its own.
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        # newline="" stops Windows adding blank lines between rows; utf-8 keeps
        # accents and symbols from the transcript intact in Excel.
        with open(path, "a", encoding="utf-8", newline="") as f:
            # restval="" fills any missing field with an empty string. A key
            # that is NOT in log_fields raises ValueError, so a typo like
            # "latency" instead of "latency_s" fails loudly instead of vanishing.
            writer = csv.DictWriter(f, fieldnames=CONFIG["log_fields"], restval="")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
