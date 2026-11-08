"""Orchestration and the live monitor window. OWNER: Markus

Run with:  python -m greendot.main
Keys in the window: Q quit | X re-measure exposure | E/D exposure | A/S green threshold
Capture: tap the headphone button (single or double), press T anywhere,
or double-click the camera window or the dashboard page.
Results also appear on the dashboard at http://localhost:8765.
"""
#main.py shows a live preview of what the camera sees, and every time you tap, it runs one complete capture from photo to spoken answer and logs the result, even when something goes wrong.
import os
import re
import sys
import threading
import time

import cv2

from greendot import clipboard, dashboard
from greendot.brain import ask_model, edit_image
from greendot.camera import CameraStream
from greendot.config import CONFIG, chdir_to_project_root, ensure_dirs, validate
from greendot.detect import find_green_dots
from greendot.align import crop_aligned, crop_outline
from greendot.imaging import downscale, enhance_for_display, local_ocr
from greendot.logging_util import log_capture
from greendot.plots import render_chart
from greendot.trigger import start_listener
from greendot.voice import record_and_transcribe, speak

is_processing = False
action_lock = threading.Lock()
camera = None

# "copy the text for me" goes to local OCR, not the model:
# a model re-types text and silently "fixes" it; OCR copies what is there.
COPY_REQUEST = re.compile(r"\b(copy|clipboard)\b", re.IGNORECASE)
# "copy the image / picture / diagram" copies the cleaned-up crop itself, no reading.
COPY_IMAGE_REQUEST = re.compile(
    r"\b(copy|clipboard)\b.*\b(image|picture|photo|figure|diagram|drawing|graph|chart)s?\b"
    r"|\b(image|picture|photo|figure|diagram|drawing|graph|chart)s?\b.*\b(copy|clipboard)\b",
    re.IGNORECASE)


def detect(frame):
    """Find the dots on a smaller copy of the frame, return them in full-size pixels.

    The dot size limits in config are tuned for ~1280 px wide frames, and
    detection is much faster there. Cropping still uses the full 4K frame, so
    small text stays sharp for OCR and the model.
    """
    # Long edge, so a portrait (rotated) frame is scaled the same way.
    scale = CONFIG["detect_width"] / max(frame.shape[:2])
    if scale >= 1.0:
        dots, reason = find_green_dots(frame)
        return dots, reason, frame, 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dots, reason = find_green_dots(small)
    if dots is not None:
        dots = [(x / scale, y / scale, r / scale) for x, y, r in dots]
    return dots, reason, small, scale


def handle_image_edit(crop, crop_path, transcript, result, row, start):
    """Make a new picture from the drawing in the crop, then show, copy and announce it."""
    instruction = transcript or CONFIG["default_image_instruction"]
    done = {}

    def draw():
        try:
            done["png"] = edit_image(downscale(crop), instruction)
        except Exception as e:
            done["error"] = e

    worker = threading.Thread(target=draw, daemon=True)
    worker.start()
    dashboard.set_status("thinking", "drawing your picture (about 20 s)")
    speak(result["spoken_text"], voice=result["voice"], instructions=result["voice_instructions"])
    dashboard.set_status("thinking", "drawing your picture (about 20 s)")
    worker.join()
    if "error" in done:
        raise done["error"]   # handle_capture's except reports and logs it

    path = os.path.join(CONFIG["image_dir"], time.strftime("image_%Y%m%d_%H%M%S.png"))
    with open(path, "wb") as f:
        f.write(done["png"])
    image = cv2.imread(path)
    clipboard.copy_image(image)
    h, w = image.shape[:2]
    row["path_used"] = "image_model"
    row["reply_chars"] = len(result["spoken_text"])
    dashboard.add_result(task="edit_image", transcript=transcript, crop=crop_path, image=path,
                         note=f"New picture {w} x {h} · copied to clipboard · {CONFIG['image_model']}",
                         path="image model", latency=round(time.time() - start, 1))
    dashboard.set_status("speaking")
    speak(CONFIG["msg_image_ready"], tag="status")


def handle_capture():
    """One full capture: detect -> crop -> listen -> ask -> show -> speak -> log.

    Guarded by a lock so a second trigger during processing is ignored rather
    than queued. EVERY path -- success, detection failure, exception -- writes a
    log row in the finally block.
    """
    global is_processing

    # Step 1: check and set the flag while holding the lock, so two taps that
    # arrive at the same instant can't both see "not busy" and both start.
    # We ignore extra taps instead of queueing them: a queued tap would answer
    # a question about a page the user has already moved.
    with action_lock:
        if is_processing:
            print("[capture] Still working on the last one -- tap ignored.")
            return
        is_processing = True

    # Step 2: start the clock now so latency covers the whole capture.
    start = time.time()
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    dashboard.set_status("captured")

    try:
        # Step 3: no frame means the camera is broken -- that's an error, not
        # a normal "no dots" result, so raise and let except record it.
        frame = camera.read()
        if frame is None:
            raise RuntimeError("camera returned no frame")

        # Step 4: log the reason even on success, so evaluate.py can count
        # every kind of detection result.
        dots, reason, _, _ = detect(frame)
        row["detect_reason"] = reason
        row["dots_found"] = len(dots) if dots else 0
        if dots is None:
            dashboard.add_result(task="no_dots", note=f"{CONFIG['msg_no_dots']} ({reason})", warn=True)
            speak(CONFIG["msg_no_dots"], tag="status")
            return  # finally still runs, so this failure is logged too

        # Step 5: straighten to the page edge first, so a turned page still crops cleanly.
        crop, angle, align_reason = crop_aligned(frame, dots)
        print(f"[capture] page angle {angle:+.1f} deg ({align_reason})")
        if crop is None:
            speak(CONFIG["msg_region_small"], tag="status")
            return
        row["crop_h"], row["crop_w"] = crop.shape[:2]

        # Step 6: keep the exact image that was read, so a wrong answer can be
        # traced to a bad crop or a bad model reply. The dashboard shows it too.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(CONFIG["crop_dir"], stamp + ".jpg"), crop)
        # The dashboard shows a cleaned-up copy that is easier to read.
        enhanced = enhance_for_display(crop)
        crop_path = os.path.join(CONFIG["crop_dir"], stamp + "_view.jpg")
        cv2.imwrite(crop_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Step 7: record_and_transcribe never raises; it returns "" on failure.
        dashboard.set_status("listening", "ask your question")
        transcript = record_and_transcribe()
        row["transcript"] = transcript

        # Step 8a: "copy the image" puts the cleaned-up crop on the clipboard.
        if COPY_IMAGE_REQUEST.search(transcript):
            clipboard.copy_image(enhanced)
            row["task"], row["path_used"] = "copy_image", "local"
            h, w = enhanced.shape[:2]
            dashboard.add_result(task="copy_image", transcript=transcript, crop=crop_path,
                                 note=f"Image copied to clipboard ({w} x {h}, contrast enhanced) · paste with Ctrl+V",
                                 path="local", latency=round(time.time() - start, 1))
            dashboard.set_status("speaking")
            speak(CONFIG["msg_copied"], tag="status")
            return

        # Step 8b: copy requests try local OCR first.
        if COPY_REQUEST.search(transcript):
            dashboard.set_status("thinking", "reading text")
            text, conf = local_ocr(crop)
            row["ocr_conf"] = round(conf, 1)
            if text and conf >= CONFIG["ocr_min_confidence"]:
                clipboard.copy_text(text)
                row["task"], row["path_used"] = "copy_text", "ocr"
                row["reply_chars"] = len(text)
                dashboard.add_result(task="copy_text", transcript=transcript, text=text, crop=crop_path,
                                     note=f"Copied to clipboard · OCR confidence {conf:.0f}%",
                                     path="local OCR", latency=round(time.time() - start, 1))
                dashboard.set_status("speaking")
                speak(CONFIG["msg_copied"], tag="status")
                return
            # Handwriting, math, a blurry page: OCR is unsure, so let the model read it.
            print(f"[capture] OCR confidence {conf:.0f} too low, using the model")

        # Step 8b: downscale first -- smaller images are faster and cheaper to send.
        dashboard.set_status("thinking")
        result = ask_model(downscale(crop), transcript)
        row["task"] = result["task"]
        row["confidence"] = result["confidence"]
        row["path_used"] = row.get("path_used") or "vision"

        # Step 8c: "make it more hairy" -- the crop and the user's own words go
        # to the image model. Drawing takes ~20 s, so the confirmation is
        # spoken WHILE it draws instead of leaving the user in silence.
        if result["task"] == "edit_image":
            handle_image_edit(crop, crop_path, transcript, result, row, start)
            return

        # Step 9: show the result on the dashboard BEFORE speaking, so a hint
        # is on screen while it is being read.
        note, plot_path = "", None
        if result["task"] == "plot_table":
            plot_path = render_chart(result["chart"])
            note = "" if plot_path else "No numbers found in the table."
        if result["task"] == "copy_text":
            clipboard.copy_text(result["display_text"])
            note = "Copied to clipboard (read by the vision model; OCR was unsure)"
        if result["voice_instructions"]:
            note = (note + " · " if note else "") + f"Voice: {result['voice']}, {result['voice_instructions']}"
        dashboard.add_result(task=result["task"], transcript=transcript,
                             text=result["display_text"] or result["spoken_text"],
                             crop=crop_path, plot=plot_path, note=note, path=row["path_used"],
                             latency=round(time.time() - start, 1))

        # Step 10
        dashboard.set_status("speaking")
        spoken = CONFIG["msg_copied"] if result["task"] == "copy_text" else result["spoken_text"]
        speak(spoken, voice=result["voice"], instructions=result["voice_instructions"])
        row["reply_chars"] = len(result["spoken_text"])

    except Exception as e:
        # Step 11: one failed capture must never kill the program -- the user
        # should be able to fix the page and tap again.
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"[capture] {row['error']}")
        dashboard.add_result(task="unclear", note=row["error"], warn=True)
        try:
            speak(CONFIG["msg_error"], tag="status")
        except Exception:
            pass  # speech may be the thing that broke; the log still has it

    finally:
        # Step 12: runs after success, after an early return, AND after an
        # error -- so every single attempt ends up in the log.
        row["latency_s"] = round(time.time() - start, 2)
        try:
            log_capture(row)
        finally:
            # Always clear the flag, or one crash would block every future tap.
            is_processing = False
            dashboard.set_status("ready")


# main starts the camera and the button listener, then shows a live window that circles the green dots in real time, so you can see exactly what the program will read before you tap.

def main():
    global camera

    # Chinese and other non-ASCII text must not crash print() on a GBK console.
    sys.stdout.reconfigure(errors="replace")

    # Step 1: run from the project root so every relative path in CONFIG works,
    # and show config problems now instead of as a confusing crash later.
    chdir_to_project_root()
    ensure_dirs()
    for problem in validate():
        print(f"[config] {problem}")

    # The dashboard's Capture button and double-click are a backup trigger for
    # when another app (Snipping Tool, a media tab) takes the headphone keys.
    dashboard.start(on_capture=handle_capture)

    # Step 2: store the camera in the module variable, because handle_capture
    # runs on the trigger's thread and reads frames from this same object.
    camera = CameraStream()
    if CONFIG["lock_exposure"] and CONFIG["auto_calibrate_exposure"]:
        dashboard.set_status("starting", "measuring light")
        while camera.read() is None:
            time.sleep(0.05)
        camera.auto_expose()

    # Step 3: the listener calls handle_capture on its own thread, so the
    # preview window keeps updating while the AI is thinking.
    listener = start_listener(handle_capture)
    dashboard.set_status("ready")

    # Step 4
    print("GreenDot Select is running.")
    print("Put two green dots on opposite corners of what you want, then tap.")
    print("Window keys: Q quit | X re-measure exposure | E/D exposure | A/S green threshold")

    # Double-clicking the camera window also starts a capture. On its own
    # thread, like the headphone trigger, so the preview keeps running.
    def on_mouse(event, *_):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("[trigger] double-click")
            threading.Thread(target=handle_capture, daemon=True).start()
    cv2.namedWindow("GreenDot Select")
    cv2.setMouseCallback("GreenDot Select", on_mouse)

    # OpenCV colours are Blue-Green-Red, not Red-Green-Blue.
    green, orange, red = (0, 255, 0), (0, 165, 255), (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # try/finally so the camera is released even if the loop crashes or the
    # user presses Ctrl+C -- otherwise the camera can stay locked until reboot.
    try:
        while True:
            # Step 5: the preview runs the same detection as a real capture,
            # so what you see circled is exactly what a tap would use.
            frame = camera.read()
            if frame is None:
                # The camera thread hasn't delivered its first frame yet.
                # waitKey still runs so the window stays responsive.
                cv2.waitKey(10)
                continue

            dots, reason, view, scale = detect(frame)
            if dots is not None:
                small_dots = [tuple(v * scale for v in d) for d in dots]
                for x, y, r in small_dots:
                    cv2.circle(view, (int(x), int(y)), int(r), green, 2)
                # The box tilts with the page: it is the region a tap would crop.
                # Green = straightened to a paper edge; orange = no edge found,
                # so the crop would be an unrotated rectangle.
                corners, angle, align_reason = crop_outline(view, small_dots)
                aligned = align_reason == "ok"
                cv2.polylines(view, [corners], True, green if aligned else orange, 2)
                label = f"page {angle:+.1f} deg" if aligned else align_reason
                cv2.putText(view, label, (20, 40), font, 0.9, green if aligned else orange, 2)
            else:
                # Showing WHY detection failed tells you what to fix: lighting,
                # a dot hidden by a hand, or a green object in view.
                cv2.putText(view, reason, (20, 40), font, 1.0, orange, 2)

            settings = f"a_max {CONFIG['a_max']}  exposure {CONFIG['manual_exposure']}"
            cv2.putText(view, settings, (20, view.shape[0] - 20), font, 0.7, green, 2)

            if is_processing:
                cv2.putText(view, "WORKING", (view.shape[1] - 200, 40), font, 1.0, red, 3)

            # Step 6: keep the window on a laptop screen, even for a portrait frame.
            if view.shape[0] > 900:
                view = cv2.resize(view, None, fx=900 / view.shape[0], fy=900 / view.shape[0])
            cv2.imshow("GreenDot Select", view)
            key = cv2.waitKey(1) & 0xFF  # & 0xFF keeps just the key code

            if key == ord("q"):
                print("[main] Q pressed in the camera window -- quitting")
                break
            elif key == ord("x"):
                # Lighting changed: measure again (the preview pauses ~1 s per step).
                camera.auto_expose()
            elif key in (ord("e"), ord("d")):
                # Tuning live beats editing config and restarting 20 times.
                camera.set_exposure(CONFIG["manual_exposure"] + (1 if key == ord("e") else -1))
                print(f"[tune] manual_exposure = {CONFIG['manual_exposure']}")
            elif key in (ord("a"), ord("s")):
                # Changing CONFIG in place works because find_green_dots reads
                # CONFIG["a_max"] on every call. Keep it inside the range that
                # validate() accepts (1 to 127); 128 means neutral, not green.
                step = 1 if key == ord("a") else -1
                CONFIG["a_max"] = min(127, max(1, CONFIG["a_max"] + step))
                # Write the value down: this only changes it until you quit.
                print(f"[tune] a_max = {CONFIG['a_max']}")

    finally:
        # Step 7: shut down in reverse order of starting up.
        listener.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
