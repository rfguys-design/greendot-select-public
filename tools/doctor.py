"""
Setup checker. Run this FIRST, before any task, and any time something breaks.

    python tools/doctor.py

It tests every package, the API key, every camera index, the microphone, and audio
playback, then tells you exactly what to fix. This file is already written -- you
do not need to build it.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

PASS = "  [ OK ]"
FAIL = "  [FAIL]"
WARN = "  [warn]"

problems = []
warnings = []


def head(title):
    print(f"\n{title}")
    print("-" * len(title))


# --------------------------------------------------------------------------
print("=" * 58)
print("  GreenDot Select -- setup checker")
print("=" * 58)

head("1. Python version")
v = sys.version_info
if v >= (3, 9):
    print(f"{PASS} Python {v.major}.{v.minor}.{v.micro}")
else:
    print(f"{FAIL} Python {v.major}.{v.minor} -- need 3.9 or newer")
    problems.append("Install Python 3.9+")


# --------------------------------------------------------------------------
head("2. Required packages")

required = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pygame": "pygame",
    "pynput": "pynput",
    "openai": "openai",
}
optional = {
    "sounddevice": "sounddevice   (speech input)",
    "pyperclip": "pyperclip     (clipboard)",
    "pytesseract": "pytesseract   (local OCR)",
    "pyttsx3": "pyttsx3       (offline speech fallback)",
}

for mod, pkg in required.items():
    try:
        __import__(mod)
        print(f"{PASS} {pkg}")
    except Exception as e:
        # Broad except on purpose: some packages raise OSError (not ImportError)
        # when a system library they need is missing.
        print(f"{FAIL} {pkg} is missing or broken -- {type(e).__name__}: {e}")
        problems.append(f"pip install {pkg}")

for mod, label in optional.items():
    try:
        __import__(mod)
        print(f"{PASS} {label}")
    except Exception as e:
        print(f"{WARN} {label} -- unavailable ({type(e).__name__}), that feature "
              f"will be skipped")
        warnings.append(f"pip install {label.split()[0]}")


# --------------------------------------------------------------------------
head("3. API key")

key = os.environ.get("OPENAI_API_KEY", "")
if not key:
    print(f"{FAIL} OPENAI_API_KEY is not set")
    problems.append('Mac/Linux: export OPENAI_API_KEY="sk-..."   '
                    'Windows: setx OPENAI_API_KEY "sk-..." then reopen the terminal')
elif not key.startswith("sk-"):
    print(f"{FAIL} OPENAI_API_KEY is set but does not look like a key")
    problems.append("Check you copied the whole key")
else:
    print(f"{PASS} OPENAI_API_KEY is set (starts {key[:6]}..., {len(key)} chars)")


# --------------------------------------------------------------------------
head("4. Project config")

try:
    from greendot.config import CONFIG, PROJECT_ROOT, validate
    print(f"{PASS} config loaded, {len(CONFIG)} settings")
    print(f"       project root: {PROJECT_ROOT}")
    for issue in validate():
        if "OPENAI_API_KEY" not in issue:
            print(f"{FAIL} {issue}")
            problems.append(issue)
except Exception as e:
    print(f"{FAIL} could not load greendot/config.py: {e}")
    problems.append("Run this from the project folder: python tools/doctor.py")
    CONFIG = {}


# --------------------------------------------------------------------------
head("5. Cameras")

working_cameras = []
try:
    import cv2
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                working_cameras.append(idx)
                print(f"{PASS} camera index {idx} works  ({w}x{h})")
        cap.release()

    if not working_cameras:
        print(f"{FAIL} no working camera found at indexes 0-3")
        problems.append("Plug in a webcam, or close other apps using it (Zoom, Teams)")
    else:
        configured = CONFIG.get("camera_index", 0)
        if configured in working_cameras:
            print(f"{PASS} config uses index {configured}, which works")
        else:
            print(f"{FAIL} config uses index {configured}, which does NOT work")
            problems.append(
                f"Create greendot/config_local.py containing:\n"
                f'       LOCAL = {{"camera_index": {working_cameras[0]}}}'
            )
except Exception as e:
    print(f"{FAIL} camera check failed: {e}")


# --------------------------------------------------------------------------
head("6. Microphone")

try:
    import sounddevice as sd
    ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    if ins:
        print(f"{PASS} {len(ins)} input device(s); default: {sd.query_devices(kind='input')['name']}")
    else:
        print(f"{WARN} no microphone found -- voice input will be skipped")
        warnings.append("Connect a microphone if you want spoken requests")
except Exception as e:
    print(f"{WARN} microphone unavailable ({type(e).__name__}) -- voice input "
          f"will be skipped")


# --------------------------------------------------------------------------
head("7. Audio output")

try:
    import numpy as np
    import pygame
    pygame.mixer.init()
    sr = 22050
    t = np.linspace(0, 0.4, int(sr * 0.4), False)
    tone = (np.sin(440 * 2 * np.pi * t) * 12000).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([tone, tone]))
    pygame.sndarray.make_sound(stereo).play()
    pygame.time.wait(600)
    pygame.mixer.quit()
    print(f"{PASS} played a test tone -- did you hear a short beep?")
    print("       If not: check your system output device (headphones vs speakers)")
except Exception as e:
    print(f"{FAIL} audio playback failed: {e}")
    problems.append("Check your audio output device")


# --------------------------------------------------------------------------
head("8. Folders")

try:
    from greendot.config import ensure_dirs
    ensure_dirs()
    print(f"{PASS} audio/, crops/, dataset/, docs/weekly/ exist")
except Exception as e:
    print(f"{WARN} could not create folders: {e}")


# --------------------------------------------------------------------------
print("\n" + "=" * 58)
if problems:
    print(f"  {len(problems)} PROBLEM(S) TO FIX:\n")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")
    print("\n  Fix these, then run this script again.")
else:
    print("  ALL CHECKS PASSED -- you are ready to start.")
    if working_cameras:
        print(f"  Working camera indexes: {working_cameras}")
if warnings:
    print(f"\n  {len(warnings)} optional thing(s) not installed:")
    for w in warnings:
        print(f"    - {w}")
print("=" * 58)
