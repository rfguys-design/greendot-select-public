
import time
from greendot.trigger import start_listener
from greendot.voice import speak


def handle_trigger():
    print("Trigger activated!")
    speak("Listening")


if __name__ == "__main__":
    print("Press T to test voice feedback integration. Ctrl+C to stop.")
    start_listener(handle_trigger)
    while True:
        time.sleep(1)