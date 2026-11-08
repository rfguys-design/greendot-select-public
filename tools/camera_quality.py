"""Image quality check: live preview + full-resolution captures, no green dots needed.

    python tools/camera_quality.py                    # camera_index from config, 1280x720
    python tools/camera_quality.py --index 2 --res 4k # auto-calibrates exposure, then locks it
    python tools/camera_quality.py --res 1080 --auto  # camera's own auto exposure (drifts)
    python tools/camera_quality.py --index 2 --res 4k --exposure -3  # fixed, no calibration

Shows numbers on the preview so cameras and settings can be compared:
    sharp   variance of the Laplacian (higher = sharper; compare same scene only)
    bright  mean brightness 0-255 (paper under good light ~170-220)
    clip    % of pixels blown to white / crushed to black
    noise   estimated sensor noise (lower is better)

Keys in the window:
    Q quit | SPACE/C save full-res frame | X auto-expose again (then locks)
    E/D exposure | F autofocus on/off
    Z zoom 1:1 centre (check text detail at real pixel size)
Saved frames go to crops/quality/ with their numbers in quality_log.csv.
"""

import argparse
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

import cv2
import numpy as np

from greendot.config import CONFIG, ensure_dirs

RESOLUTIONS = {"720": (1280, 720), "1080": (1920, 1080), "4k": (3840, 2160)}
GREEN, ORANGE = (0, 255, 0), (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PREVIEW_WIDTH = 1280


def measure(frame):
    """Quality numbers for one BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Noise: median absolute deviation of a fine-detail filter (Immerkaer method).
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    noise = np.median(np.abs(cv2.filter2D(gray.astype(np.float64), -1, kernel))) * 1.4826 / 6
    return {
        "sharp": sharp,
        "bright": gray.mean(),
        "clip_white": (gray >= 250).mean() * 100,
        "clip_black": (gray <= 5).mean() * 100,
        "noise": noise,
    }


def paper_level(frame):
    """Brightness of the paper: 90th percentile, so the dark desk around it is ignored."""
    return np.percentile(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 90)


def wait_frame(camera, seconds):
    """Let a new exposure settle (about 1 s on the IPEVO), keeping the window alive."""
    end = time.time() + seconds
    while time.time() < end:
        cv2.waitKey(20)
    return camera.read()


def calibrate_exposure(camera, low=170, high=230, settle=1.0):
    """Step manual exposure until the paper sits in low..high, then leave it locked.

    The camera's own auto exposure re-adjusts whenever a hand or page moves, so
    brightness wanders. Measuring once and locking gives auto convenience with
    a steady image. Each exposure step roughly doubles or halves brightness.
    """
    CONFIG["lock_exposure"] = True
    exposure = CONFIG["manual_exposure"]
    camera.set_exposure(exposure)
    tried = {}
    for _ in range(10):
        frame = wait_frame(camera, settle)
        if frame is None:
            continue
        level = paper_level(frame)
        tried[exposure] = level
        print(f"[auto] exposure {exposure}: paper {level:.0f}")
        if level < low:
            step = 1
        elif level > high:
            step = -1
        else:
            break
        if exposure + step in tried or not -13 <= exposure + step <= 0:
            # Target falls between two steps (or at the end of the range):
            # keep whichever tried value is closest to the middle of the band.
            target = (low + high) / 2
            exposure = min(tried, key=lambda e: abs(tried[e] - target))
            camera.set_exposure(exposure)
            break
        exposure += step
        camera.set_exposure(exposure)
    print(f"[auto] exposure locked at {CONFIG['manual_exposure']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="camera index (default: config)")
    parser.add_argument("--res", choices=RESOLUTIONS, default="720")
    parser.add_argument("--auto", action="store_true", help="keep auto exposure / white balance")
    parser.add_argument("--exposure", type=int, default=None, help="manual exposure (default: config)")
    args = parser.parse_args()

    CONFIG["frame_width"], CONFIG["frame_height"] = RESOLUTIONS[args.res]
    if args.exposure is not None:
        CONFIG["manual_exposure"] = args.exposure
    if args.auto:
        CONFIG["lock_exposure"] = False

    # Imported after CONFIG is adjusted, since CameraStream reads it on open.
    from greendot.camera import CameraStream

    ensure_dirs()
    out_dir = os.path.join(CONFIG["crop_dir"], "quality")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "quality_log.csv")

    camera = CameraStream(args.index)
    index = CONFIG["camera_index"] if args.index is None else args.index
    print(f"Camera {index} open, asked for {args.res}.")
    print("Keys: Q quit | SPACE/C save | X auto-expose | E/D exposure | F autofocus | Z 1:1 zoom")

    if not args.auto and args.exposure is None:
        calibrate_exposure(camera)

    autofocus = True
    zoom = False
    reported = False
    try:
        while True:
            frame = camera.read()
            if frame is None:
                cv2.waitKey(10)
                continue

            h, w = frame.shape[:2]
            if not reported:
                print(f"Camera actually delivers {w} x {h}.")
                reported = True

            m = measure(frame)

            if zoom:
                # Real pixels from the centre, no scaling: shows true text detail.
                ch, cw = min(h, 720), min(w, PREVIEW_WIDTH)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                view = frame[y0:y0 + ch, x0:x0 + cw].copy()
            else:
                scale = PREVIEW_WIDTH / w
                view = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale))) if scale < 1 else frame.copy()

            lines = [
                f"{w}x{h}  {'1:1 centre' if zoom else 'fit'}",
                f"sharp {m['sharp']:.0f}",
                f"bright {m['bright']:.0f}",
                f"clip white {m['clip_white']:.1f}%  black {m['clip_black']:.1f}%",
                f"noise {m['noise']:.2f}",
            ]
            exposure = "auto" if not CONFIG["lock_exposure"] else CONFIG["manual_exposure"]
            lines.append(f"exposure {exposure}  autofocus {'on' if autofocus else 'off'}")
            for i, text in enumerate(lines):
                # Dark outline so the text reads on white paper.
                cv2.putText(view, text, (20, 40 + 32 * i), FONT, 0.8, (0, 0, 0), 4)
                colour = ORANGE if i == 2 and not 120 <= m["bright"] <= 230 else GREEN
                cv2.putText(view, text, (20, 40 + 32 * i), FONT, 0.8, colour, 2)
            cv2.imshow("camera quality", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord(" "), ord("c")):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                name = f"cam{index}_{w}x{h}_{stamp}.png"  # PNG: no JPEG artefacts in the check
                cv2.imwrite(os.path.join(out_dir, name), frame)
                new_log = not os.path.exists(log_path)
                with open(log_path, "a") as log:
                    if new_log:
                        log.write("file,width,height,exposure,autofocus,sharp,bright,clip_white,clip_black,noise\n")
                    log.write(
                        f"{name},{w},{h},{exposure},{int(autofocus)},{m['sharp']:.1f},{m['bright']:.1f},"
                        f"{m['clip_white']:.2f},{m['clip_black']:.2f},{m['noise']:.2f}\n"
                    )
                print(f"[save] {name}  sharp {m['sharp']:.0f}  bright {m['bright']:.0f}  noise {m['noise']:.2f}")
            elif key in (ord("e"), ord("d")):
                CONFIG["lock_exposure"] = True
                camera.set_exposure(CONFIG["manual_exposure"] + (1 if key == ord("e") else -1))
                print(f"[tune] manual_exposure = {CONFIG['manual_exposure']}")
            elif key == ord("x"):
                calibrate_exposure(camera)
            elif key == ord("f"):
                autofocus = not autofocus
                camera.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
                print(f"[tune] autofocus {'on' if autofocus else 'off'}")
            elif key == ord("z"):
                zoom = not zoom
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
