"""Capture trigger. OWNER: Joanna

Fires on the headphone media key or the keyboard fallback key. Debounced,
because one physical tap usually produces two events.

Media-key behaviour differs by OS and some systems need accessibility
permissions. Record what each machine needed in docs/SETUP.md.
"""

import threading
import time

from pynput import keyboard

from greendot.config import CONFIG

# A single headphone tap sends play/pause; a double tap usually sends "next
# track" and a triple tap "previous track". Any of them starts a capture.
MEDIA_KEYS = {
    keyboard.Key.media_play_pause,
    keyboard.Key.media_next,
    keyboard.Key.media_previous,
}


def start_listener(callback):
    """Start a global key listener that calls callback() on each debounced tap.

    callback runs on its own thread so the listener never blocks.
    Returns the listener object so main.py can stop it at shutdown.
    """
    fallback = CONFIG["keyboard_fallback_key"]
    last_fire = [0.0]  # a list so the nested function can update it

    def on_press(key):
        is_media = key in MEDIA_KEYS
        is_fallback = getattr(key, "char", None) == fallback
        if not (is_media or is_fallback):
            return

        # A double tap arrives as two events well inside the debounce window,
        # so it starts one capture, not two.
        now = time.time()
        if now - last_fire[0] < CONFIG["trigger_debounce_s"]:
            return
        last_fire[0] = now

        print(f"[trigger] {key}")
        # A capture takes seconds; running it here would freeze the listener.
        threading.Thread(target=callback, daemon=True).start()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener
