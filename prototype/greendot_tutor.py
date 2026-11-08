"""
Green Dot Tutor -- consolidated single-file version.

Place two green dots at opposite corners of a region on paper, tap the
headphone button (or press T), optionally say what you want, and the program
captures the region, decides what to do, and speaks the result.

Replaces: MainCode_V1_Universal_Subject.py, MainCodemath_StepbyStep_V7.py,
          Maincode_version_6_math_greendot.py, Audio_book_main_Version2.py

SETUP
  export OPENAI_API_KEY="sk-..."        (Windows: setx OPENAI_API_KEY "sk-...")
  pip install opencv-python numpy pygame pynput openai
  optional: pip install sounddevice pyperclip pytesseract

NEVER hardcode the API key in this file.
"""

import os
import sys

# Standing rule: run correctly no matter where the script is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import base64
import csv
import glob
import json
import math
import threading
import time
import wave

import cv2
import numpy as np
import pygame
from openai import OpenAI
from pynput import keyboard

# --------------------------------------------------------------------------
# CONFIG -- everything tunable lives here, nowhere else
# --------------------------------------------------------------------------

CONFIG = {
    # Models. These move fast; check current docs before a demo.
    "vision_model": "gpt-4o",
    "stt_model": "whisper-1",
    "tts_model": "tts-1",
    "tts_voice": "nova",

    # Camera
    "camera_index": 0,
    "frame_width": 1280,
    "frame_height": 720,
    "lock_exposure": True,      # auto modes retune when a hand enters frame
    "manual_exposure": -6,      # platform dependent; tune with 'e'/'d' keys

    # Green dot detection (LAB a* channel -- far more stable than HSV hue)
    "a_max": 115,               # green is low a*; 128 is neutral
    "l_min": 40,                # ignore very dark pixels
    "min_dot_area": 60,
    "max_dot_area": 8000,
    "min_circularity": 0.55,

    # Crop geometry
    "inset_margin": 6,          # extra px inset past the dot radius
    "max_long_edge": 1024,      # resize before upload: faster + cheaper

    # Voice
    "voice_enabled": True,
    "record_seconds": 4.0,
    "sample_rate": 16000,
    "default_intent": "Decide from the content what would help most.",

    # Files
    "log_file": "capture_log.csv",
    "crop_dir": "crops",
    "audio_dir": "audio",
    "keep_audio_files": 20,
}

# The skills table. Adding a capability = adding an entry here.
# 'delivery' is how the result reaches the user, not what the model does.
SKILLS = {
    "read_aloud": "Transcribe the text in the region exactly, word for word. "
                  "Do not summarize, correct, or paraphrase.",
    "solve_math": "Work the problem in the region. Explain the steps briefly "
                  "in plain spoken sentences, then give the final answer.",
    "hint":       "The region contains a problem. Give ONE hint toward the "
                  "next step. Do not solve it and do not reveal the answer.",
    "translate":  "Translate the text in the region into English. Output only "
                  "the translation.",
    "explain":    "Explain the content of the region simply, under 120 words.",
    "copy_text":  "Transcribe the text in the region exactly for copying.",
    "unclear":    "Say briefly that the region could not be understood.",
}

SYSTEM_PROMPT = (
    "You help a student working with a physical page. Two green dots mark "
    "opposite corners of a region; the image you receive is that region, "
    "already cropped.\n\n"
    "Decide which task the user wants from what they said and what is in the "
    "image, then produce spoken_text for that task.\n\n"
    "Tasks:\n"
    + "\n".join(f"- {k}: {v}" for k, v in SKILLS.items())
    + "\n\nspoken_text will be read aloud by a speech synthesizer. Write plain "
    "sentences. No bullet points, asterisks, markdown, or LaTeX. Spell out "
    "symbols as words where needed. Be warm and concise."
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tutor_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "enum": list(SKILLS.keys())},
                "spoken_text": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["task", "spoken_text", "confidence"],
            "additionalProperties": False,
        },
    },
}

# --------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES -- degrade gracefully if missing
# --------------------------------------------------------------------------

try:
    import sounddevice as sd
    HAVE_AUDIO_IN = True
except Exception:
    HAVE_AUDIO_IN = False

try:
    import pyperclip
    HAVE_CLIPBOARD = True
except Exception:
    HAVE_CLIPBOARD = False

try:
    import pytesseract
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

# --------------------------------------------------------------------------
# CLIENT / STATE
# --------------------------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY is not set. Export it and rerun.")
client = OpenAI(api_key=api_key)

is_processing = False
action_lock = threading.Lock()


# --------------------------------------------------------------------------
# CAMERA -- one reader thread, everyone else reads the latest frame
# --------------------------------------------------------------------------

class CameraStream:
    """Single owner of cap.read(). Removes the two-threads-one-camera race."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        if CONFIG["lock_exposure"]:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, CONFIG["manual_exposure"])
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if not self.cap.isOpened():
            sys.exit(f"Camera {index} did not open.")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(0.5)  # let a few frames land

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def set_exposure(self, value):
        CONFIG["manual_exposure"] = value
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# --------------------------------------------------------------------------
# DOT DETECTION
# --------------------------------------------------------------------------

def find_green_dots(frame):
    """Return (dots, reason). dots is a list of (x, y, radius) or None.

    Fails loudly: if there are not exactly two plausible dots, we say so
    rather than guessing. A clean failure is recoverable; a wrong crop is not.
    """
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, _ = cv2.split(lab)

    mask = ((A < CONFIG["a_max"]) & (L > CONFIG["l_min"])).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (CONFIG["min_dot_area"] < area < CONFIG["max_dot_area"]):
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4 * math.pi * area / (perim * perim)
        if circularity < CONFIG["min_circularity"]:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        candidates.append((cx, cy, math.sqrt(area / math.pi), area))

    if len(candidates) < 2:
        return None, f"found {len(candidates)} dots, need 2"
    if len(candidates) > 2:
        # Something green and round is in frame that should not be.
        return None, f"found {len(candidates)} dots, need exactly 2"

    dots = [(c[0], c[1], c[2]) for c in candidates]
    return dots, "ok"


def crop_between_dots(frame, dots):
    """Inset INWARD past the dot radius so the stickers are not in the crop."""
    (x1, y1, r1), (x2, y2, r2) = dots
    m = CONFIG["inset_margin"]
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])

    left += int(r1 + m) if x1 < x2 else int(r2 + m)
    right -= int(r2 + m) if x2 > x1 else int(r1 + m)
    top += int(r1 + m) if y1 < y2 else int(r2 + m)
    bottom -= int(r2 + m) if y2 > y1 else int(r1 + m)

    h, w = frame.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)

    if right - left < 20 or bottom - top < 20:
        return None
    return frame[top:bottom, left:right]


def downscale(img):
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= CONFIG["max_long_edge"]:
        return img
    s = CONFIG["max_long_edge"] / long_edge
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# AUDIO
# --------------------------------------------------------------------------

def play_audio(path):
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[audio] playback failed: {e}")


def speak(text, tag="reply"):
    os.makedirs(CONFIG["audio_dir"], exist_ok=True)
    path = os.path.join(CONFIG["audio_dir"], f"{tag}_{int(time.time())}.mp3")
    out = client.audio.speech.create(
        model=CONFIG["tts_model"], voice=CONFIG["tts_voice"], input=text
    )
    out.write_to_file(path)
    time.sleep(0.2)
    play_audio(path)
    return path


def record_and_transcribe():
    """Record a short utterance and return the transcript, or '' if unavailable."""
    if not (CONFIG["voice_enabled"] and HAVE_AUDIO_IN):
        return ""
    fs = CONFIG["sample_rate"]
    n = int(CONFIG["record_seconds"] * fs)
    try:
        print("[voice] listening...")
        audio = sd.rec(n, samplerate=fs, channels=1, dtype="int16")
        sd.wait()
        path = "utterance.wav"
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        with open(path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=CONFIG["stt_model"], file=f
            )
        text = tr.text.strip()
        print(f"[voice] heard: {text!r}")
        return text
    except Exception as e:
        print(f"[voice] failed, continuing without speech: {e}")
        return ""


# --------------------------------------------------------------------------
# LOCAL OCR PATH (verbatim copy -- no model, no network)
# --------------------------------------------------------------------------

def local_ocr(crop):
    """Returns (text, mean_confidence) or (None, 0) if OCR is unavailable."""
    if not HAVE_OCR:
        return None, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for txt, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if txt.strip() and c >= 0:
            words.append(txt)
            confs.append(c)
    if not words:
        return None, 0.0
    return " ".join(words), sum(confs) / len(confs)


# --------------------------------------------------------------------------
# VISION CALL
# --------------------------------------------------------------------------

def ask_model(crop, transcript):
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode("utf-8")
    user_text = transcript or CONFIG["default_intent"]

    resp = client.chat.completions.create(
        model=CONFIG["vision_model"],
        response_format=RESPONSE_SCHEMA,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": "high"}},
            ]},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# --------------------------------------------------------------------------
# LOGGING -- this is what turns "it works" into a number
# --------------------------------------------------------------------------

LOG_FIELDS = ["timestamp", "dots_found", "detect_reason", "crop_w", "crop_h",
              "transcript", "task", "confidence", "ocr_conf", "path_used",
              "latency_s", "reply_chars", "error"]


def log_capture(row):
    new = not os.path.exists(CONFIG["log_file"])
    with open(CONFIG["log_file"], "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


# --------------------------------------------------------------------------
# MAIN ACTION
# --------------------------------------------------------------------------

def handle_capture():
    global is_processing
    with action_lock:
        if is_processing:
            print("[system] busy, ignoring trigger")
            return
        is_processing = True

    t0 = time.time()
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        frame = camera.read()
        if frame is None:
            raise RuntimeError("no frame from camera")

        dots, reason = find_green_dots(frame)
        row["detect_reason"] = reason
        row["dots_found"] = 0 if dots is None else len(dots)
        if dots is None:
            print(f"[detect] {reason}")
            speak("I could not see two green markers.", tag="err")
            return

        crop = crop_between_dots(frame, dots)
        if crop is None:
            print("[detect] region too small")
            speak("That region is too small.", tag="err")
            return

        os.makedirs(CONFIG["crop_dir"], exist_ok=True)
        stamp = int(time.time())
        cv2.imwrite(os.path.join(CONFIG["crop_dir"], f"crop_{stamp}.jpg"), crop)
        row["crop_h"], row["crop_w"] = crop.shape[:2]

        transcript = record_and_transcribe()
        row["transcript"] = transcript

        # Local verbatim path: only when the user explicitly asks to copy.
        wants_copy = any(k in transcript.lower() for k in ("copy", "clipboard"))
        if wants_copy:
            text, conf = local_ocr(crop)
            row["ocr_conf"] = round(conf, 1)
            if text and conf >= 70:
                if HAVE_CLIPBOARD:
                    pyperclip.copy(text)
                row["task"], row["path_used"] = "copy_text", "local_ocr"
                row["reply_chars"] = len(text)
                print(f"[copy] {text[:120]}")
                speak("Copied.", tag="copy")
                return
            print("[copy] OCR confidence low, falling back to the model")

        result = ask_model(downscale(crop), transcript)
        row["task"] = result["task"]
        row["confidence"] = result["confidence"]
        row["path_used"] = "vision_model"
        reply = result["spoken_text"].strip()
        row["reply_chars"] = len(reply)

        print(f"[task] {result['task']}  (confidence {result['confidence']})")
        print(f"[reply] {reply}")

        if result["task"] == "copy_text" and HAVE_CLIPBOARD:
            pyperclip.copy(reply)
        speak(reply)

    except Exception as e:
        row["error"] = str(e)
        print(f"[error] {e}")
        try:
            speak("Something went wrong.", tag="err")
        except Exception:
            pass
    finally:
        row["latency_s"] = round(time.time() - t0, 2)
        log_capture(row)
        is_processing = False


# --------------------------------------------------------------------------
# HOUSEKEEPING
# --------------------------------------------------------------------------

def prune_audio():
    files = sorted(glob.glob(os.path.join(CONFIG["audio_dir"], "*.mp3")),
                   key=os.path.getmtime)
    for p in files[:-CONFIG["keep_audio_files"]]:
        try:
            os.remove(p)
        except OSError:
            pass


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def on_press(key):
    trigger = (key == keyboard.Key.media_play_pause)
    if not trigger:
        try:
            trigger = (key.char == "t")
        except AttributeError:
            trigger = False
    if trigger:
        threading.Thread(target=handle_capture, daemon=True).start()


if __name__ == "__main__":
    pygame.mixer.init()
    os.makedirs(CONFIG["audio_dir"], exist_ok=True)
    os.makedirs(CONFIG["crop_dir"], exist_ok=True)
    prune_audio()

    camera = CameraStream(CONFIG["camera_index"])
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("-" * 58)
    print("GREEN DOT TUTOR")
    print("  Place two green dots at opposite corners of a region.")
    print("  Tap headphone button (or press T) and say what you want.")
    print(f"  voice input: {'on' if HAVE_AUDIO_IN else 'OFF (no sounddevice)'}"
          f" | local OCR: {'on' if HAVE_OCR else 'OFF'}"
          f" | clipboard: {'on' if HAVE_CLIPBOARD else 'OFF'}")
    print("  Window keys: Q quit | E/D exposure | A/S green threshold")
    print("-" * 58)

    while True:
        img = camera.read()
        if img is None:
            time.sleep(0.02)
            continue

        dots, reason = find_green_dots(img)
        if dots:
            (x1, y1, r1), (x2, y2, r2) = dots
            cv2.rectangle(img, (min(x1, x2), min(y1, y2)),
                          (max(x1, x2), max(y1, y2)), (0, 255, 0), 2)
            for (cx, cy, r) in dots:
                cv2.circle(img, (cx, cy), int(r) + 6, (0, 255, 0), 2)
            label, color = "REGION LOCKED", (0, 255, 0)
        else:
            label, color = reason, (0, 165, 255)

        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)
        cv2.putText(img, f"a_max={CONFIG['a_max']} exp={CONFIG['manual_exposure']}",
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if is_processing:
            cv2.putText(img, "WORKING", (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("Green Dot Tutor  (Q to quit)", img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("e"):
            camera.set_exposure(CONFIG["manual_exposure"] + 1)
        elif k == ord("d"):
            camera.set_exposure(CONFIG["manual_exposure"] - 1)
        elif k == ord("a"):
            CONFIG["a_max"] = min(127, CONFIG["a_max"] + 2)
        elif k == ord("s"):
            CONFIG["a_max"] = max(80, CONFIG["a_max"] - 2)

    listener.stop()
    camera.release()
    cv2.destroyAllWindows()
