"""Camera access. OWNER: Harper

RULE: this module is the ONLY place in the project that calls cap.read().
A single reader thread removes the race that forced the prototype to grab and
discard five frames before every capture.
"""

import threading
import time

import cv2
import numpy as np

from greendot.config import CONFIG


class CameraStream:
    """Owns the camera. One background thread reads; everyone else calls read()."""

    # Why one thread owns the camera:
    # A webcam keeps a small queue of frames. If the display loop and the capture
    # code each call cap.read() themselves, they take turns pulling from that
    # queue, so whoever asks gets a frame that may be old, and the two parts of
    # the program slow each other down. The prototype worked around this by
    # throwing away five frames before every capture. Here, one thread reads
    # nonstop and keeps only the newest frame, so read() is always instant and
    # always current, no matter how many parts of the program call it.

    def __init__(self, index=None):
        if index is None:
            index = CONFIG["camera_index"]

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera {index} did not open. Try another index in "
                "greendot/config_local.py, e.g. LOCAL = {'camera_index': 1}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

        if CONFIG["lock_exposure"]:
            # Auto modes retune whenever a hand enters the frame, which changes
            # the dot colours mid-capture. The "manual" value for
            # CAP_PROP_AUTO_EXPOSURE differs by platform: 0.25 works on many
            # backends, Linux V4L2 uses 1. Some cameras ignore these entirely.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, CONFIG["manual_exposure"])

        self._rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(CONFIG["camera_rotate"])

        self.frame = None
        self._lock = threading.Lock()
        self._running = True

        # daemon=True: if the program crashes without calling release(), this
        # thread will not keep Python running forever.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Background thread: keep self.frame at the most recent frame."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                # Camera hiccup. Wait briefly instead of spinning at 100% CPU.
                time.sleep(0.01)
                continue
            # A camera mounted sideways delivers sideways text. Turning it here,
            # once, means preview, crops, OCR and the dashboard all see it upright.
            if self._rotate is not None:
                frame = cv2.rotate(frame, self._rotate)
            with self._lock:
                self.frame = frame

    def read(self):
        """Return a copy of the latest frame, or None if none has arrived yet."""
        with self._lock:
            if self.frame is None:
                return None
            # A copy, so a caller drawing on its frame (e.g. circles in the
            # tuning window) cannot change the frame another caller receives.
            return self.frame.copy()

    def set_exposure(self, value):
        """Live tuning from the E/D keys in the monitor window."""
        CONFIG["manual_exposure"] = value
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)

    def auto_expose(self, low=170, high=230, settle=1.0):
        """Step manual exposure until the paper is in low..high, then leave it locked.

        The camera's own auto exposure re-adjusts whenever a hand moves in, so
        brightness wanders and dot colours shift. Measuring once and locking
        gives the convenience of auto with a steady image. The paper level is
        the 90th-percentile brightness, so the dark desk is ignored. Each step
        roughly doubles or halves brightness; a new value takes ~1 s to settle.
        """
        exposure = CONFIG["manual_exposure"]
        tried = {}
        for _ in range(10):
            self.set_exposure(exposure)
            time.sleep(settle)
            frame = self.read()
            if frame is None:
                continue
            level = float(np.percentile(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 90))
            tried[exposure] = level
            step = 1 if level < low else -1 if level > high else 0
            if step == 0:
                break
            if exposure + step in tried or not -13 <= exposure + step <= 0:
                # Target lies between two steps: keep the closest one tried.
                exposure = min(tried, key=lambda e: abs(tried[e] - (low + high) / 2))
                self.set_exposure(exposure)
                break
            exposure += step
        print(f"[camera] exposure locked at {exposure} (paper level {tried.get(exposure, 0):.0f})")

    def release(self):
        """Stop the thread and release the device."""
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
