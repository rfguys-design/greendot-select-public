"""
GreenDot Select -- shared configuration.

ONE source of truth for every tunable value in the project.

HOW THIS FILE IS ORGANISED
    Each student owns one section below. The sections are separate dicts that get
    merged into CONFIG at the bottom. Two people editing different sections
    produce a clean git merge; two people editing the same dict do not.

    Edit only your own section. If you need a value that belongs to someone
    else's section, ask them -- do not add a duplicate key to yours.

RULES
    1. Any value you discover by hand-tuning gets committed HERE. A threshold
       that lives only in your local copy is a bug waiting for demo day.
    2. Machine-specific values (camera index, exposure) go in config_local.py,
       which is gitignored. See the bottom of this file.
    3. Never put the API key here. It comes from the OPENAI_API_KEY environment
       variable, read in brain.py and nowhere else.

USAGE
    from greendot.config import CONFIG, PROJECT_ROOT
    a_max = CONFIG["a_max"]
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# PATHS -- owner: Markus
# Resolved from this file's location so the project runs from any working
# directory. Entry-point scripts additionally call chdir_to_project_root().
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def chdir_to_project_root():
    """Call this at the top of every runnable script (main.py, tools/*, evaluate.py)."""
    os.chdir(PROJECT_ROOT)


PATHS = {
    "project_root": str(PROJECT_ROOT),
    "audio_dir": str(PROJECT_ROOT / "audio"),
    "crop_dir": str(PROJECT_ROOT / "crops"),
    "dataset_dir": str(PROJECT_ROOT / "dataset"),
    "weekly_dir": str(PROJECT_ROOT / "docs" / "weekly"),
    "log_file": str(PROJECT_ROOT / "capture_log.csv"),
    "keep_audio_files": 20,          # prune older mp3s on startup
}


# --------------------------------------------------------------------------
# CAMERA AND DETECTION -- owner: Harper
# --------------------------------------------------------------------------

CAMERA_AND_DETECTION = {
    # Camera. If index 0 is the wrong device on your machine, override it in
    # config_local.py rather than changing it here.
    "camera_index": 0,
    "frame_width": 1280,
    "frame_height": 720,
    "lock_exposure": True,           # auto modes retune when a hand enters frame
    "manual_exposure": -6,           # platform dependent; tune with E/D keys
    "auto_calibrate_exposure": True, # at startup, step exposure until paper is bright, then lock
    "camera_rotate": 0,              # 0/90/180/270 clockwise, for a camera mounted sideways
    "detect_width": 1280,            # detection runs on a copy this wide; crops come from full res

    # Green dot detection, LAB a* channel.
    # a* is the green-to-red axis; 128 is neutral, lower is greener.
    # Hue in HSV shifts with white balance -- a* does not. Do not switch back.
    "a_max": 105,                    # pixel is green if A < a_max
    "l_min": 40,                     # ignore very dark pixels
    "blur_kernel": 5,                # odd number, Gaussian blur before threshold

    # Shape filtering. A dot must be the right size AND round enough.
    # Without these, a green highlighter or a plant becomes a corner.
    "min_dot_area": 60,              # px^2, rejects speckle
    "max_dot_area": 8000,            # px^2, rejects large green objects
    "min_circularity": 0.55,         # 4*pi*A/P^2; a perfect circle is 1.0

    # Detection must find EXACTLY this many dots or report a failure.
    # Never "take the largest two" -- a silently wrong crop is worse than none.
    "required_dots": 2,
}


# --------------------------------------------------------------------------
# CROP AND IMAGING -- owner: James
# --------------------------------------------------------------------------

CROP_AND_IMAGING = {
    # The crop insets INWARD past each dot's radius, so the stickers themselves
    # never appear in the image sent for reading. The first prototype padded
    # outward by 100px and pulled in the neighbouring problem every time.
    "inset_margin": 6,               # extra px inset beyond the dot radius
    "min_crop_dimension": 20,        # below this, report failure instead of cropping

    # Resize before upload: cheaper and faster, no accuracy cost at page scale.
    "max_long_edge": 1024,
    "jpeg_quality": 90,

    # OCR preprocessing (local verbatim copy path).
    "ocr_target_dpi": 300,           # upscale small crops toward this before OCR
    "ocr_threshold_block": 31,       # adaptive threshold block size, odd
    "ocr_threshold_c": 10,           # adaptive threshold constant
    "ocr_min_confidence": 70.0,      # below this, fall back to the vision model
    "ocr_min_width": 1800,           # px; narrower crops are upscaled (max 3x) before OCR

    # Page alignment. Two dots give corners but not rotation, so the crop is
    # straightened using any visible straight edge of the paper. Only part of
    # one edge needs to be in view.
    "align_to_page": True,
    "page_edge_min_frac": 0.12,      # shortest usable edge, fraction of frame long edge
    "page_border_margin": 4,         # px; outline this close to the frame border is ignored
}


# --------------------------------------------------------------------------
# VOICE, TRIGGER AND SKILLS -- owner: Joanna
# --------------------------------------------------------------------------

VOICE = {
    # Trigger. Headphone buttons often send two events per physical tap.
    "trigger_debounce_s": 1.0,
    "keyboard_fallback_key": "t",

    # Speech input.
    "voice_enabled": True,
    "record_seconds": 4.0,           # give up if no speech starts within this
    "record_max_s": 12.0,            # hard limit on one question
    "record_silence_s": 1.0,         # this much quiet after speech ends the recording
    "speech_level_min": 300,         # int16 RMS; quieter than this is never speech
    "sample_rate": 16000,
    "input_device": "",              # part of a mic's name to always record from; "" = Windows default
    "default_intent": "Decide from the content what would help most.",

    # Speech output.
    "tts_voice": "nova",
    "use_local_tts_fallback": True,  # offline voice when the network is down

    # Spoken feedback strings, so wording changes don't require code changes.
    "msg_no_dots": "I could not see two green markers.",
    "msg_region_small": "That region is too small.",
    "msg_error": "Something went wrong.",
    "msg_copied": "Copied.",
    "msg_image_ready": "Your picture is ready. It's on the screen and copied.",
}


# --------------------------------------------------------------------------
# MODELS -- owner: Markus
# These names change; confirm against current API docs before any demo.
# --------------------------------------------------------------------------

MODELS = {
    "vision_model": "gpt-4o",
    "stt_model": "whisper-1",
    "tts_model": "gpt-4o-mini-tts",  # takes voice instructions (accent, language)
    "image_detail": "high",
    # Image editing (edit_image skill): the crop plus the user's words in, a new picture out.
    "image_model": "gpt-image-2.5-flare",   # "gpt-image-2.5-sunburst" = slower, more faithful
    "image_quality": "high",                # low / medium / high / xhigh / max / auto
    "image_size": "auto",                   # auto keeps the drawing's proportions
    "image_timeout_s": 180,                 # complex edits can take up to ~2 minutes
    "default_image_instruction": "Make a polished, high resolution digital version of this drawing.",
    "request_timeout_s": 30,
    "max_retries": 2,
}


# --------------------------------------------------------------------------
# LOGGING AND EVALUATION -- owner: Markus
# --------------------------------------------------------------------------

EVALUATION = {
    # One row per capture, written in a finally block -- successes AND failures.
    # Every number in the final writeup comes from this file.
    "log_fields": [
        "timestamp", "dots_found", "detect_reason", "crop_w", "crop_h",
        "transcript", "task", "confidence", "ocr_conf", "path_used",
        "latency_s", "reply_chars", "error",
    ],
    "save_crops": True,              # keep crop images for failure diagnosis
    "latency_target_s": 3.0,         # end-to-end target, tap to first audio

    # Dashboard: a local web page showing each result (hint text, plot, copied text).
    "dashboard_port": 8765,
    "plot_dir": str(PROJECT_ROOT / "crops" / "plots"),
    "image_dir": str(PROJECT_ROOT / "crops" / "images"),
}


# --------------------------------------------------------------------------
# MERGE -- do not edit below this line without telling the team
# --------------------------------------------------------------------------

CONFIG = {}
for _section in (PATHS, CAMERA_AND_DETECTION, CROP_AND_IMAGING,
                 VOICE, MODELS, EVALUATION):
    _overlap = set(_section) & set(CONFIG)
    if _overlap:
        raise RuntimeError(
            f"Duplicate config keys across sections: {sorted(_overlap)}. "
            "Two students have defined the same key -- agree on one owner."
        )
    CONFIG.update(_section)


# Machine-specific overrides. Create greendot/config_local.py (gitignored) with:
#     LOCAL = {"camera_index": 1, "manual_exposure": -5}
# Use it for anything true of YOUR laptop only. Never for team-wide tuning.
try:
    from greendot.config_local import LOCAL  # type: ignore
    CONFIG.update(LOCAL)
except ImportError:
    pass


def validate():
    """Cheap sanity checks. Called by main.py at startup."""
    problems = []

    if not os.environ.get("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set in the environment.")
    if not 0 < CONFIG["a_max"] < 128:
        problems.append("a_max must be between 0 and 128 (128 is neutral green-red).")
    if CONFIG["min_dot_area"] >= CONFIG["max_dot_area"]:
        problems.append("min_dot_area must be smaller than max_dot_area.")
    if not 0 < CONFIG["min_circularity"] <= 1.0:
        problems.append("min_circularity must be between 0 and 1.")
    if CONFIG["blur_kernel"] % 2 == 0:
        problems.append("blur_kernel must be odd.")
    if CONFIG["ocr_threshold_block"] % 2 == 0:
        problems.append("ocr_threshold_block must be odd.")
    if CONFIG["required_dots"] != 2:
        problems.append("required_dots is 2 for this design; changing it needs a team decision.")

    return problems


def ensure_dirs():
    """Create the output directories if they do not exist."""
    for key in ("audio_dir", "crop_dir", "dataset_dir", "weekly_dir", "plot_dir", "image_dir"):
        Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    chdir_to_project_root()
    ensure_dirs()
    issues = validate()
    print(f"project root: {PROJECT_ROOT}")
    print(f"{len(CONFIG)} config keys loaded")
    for k in sorted(CONFIG):
        print(f"  {k:24s} {CONFIG[k]}")
    if issues:
        print("\nPROBLEMS:")
        for p in issues:
            print(f"  - {p}")
    else:
        print("\nconfig OK")
