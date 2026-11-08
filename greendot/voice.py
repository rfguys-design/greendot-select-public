"""Speech in and out. OWNER: Joanna

In:  record from the default microphone until the user stops talking, then
     transcribe with the OpenAI speech-to-text model.
Out: stream OpenAI text-to-speech straight to the speakers so the first words
     play while the rest is still being generated. The model takes a voice and
     free-text instructions, which is how "a British man" or "in Chinese" works.
     Falls back to the offline Windows/macOS voice when the network fails.
"""

import glob
import os
import threading
import time
import wave

import numpy as np
import sounddevice as sd

from greendot.config import CONFIG

TTS_RATE = 24000          # OpenAI "pcm" output: 24 kHz, 16-bit, mono
_speak_lock = threading.Lock()


def _client():
    # brain.py is the only place that reads the API key.
    from greendot.brain import get_client
    return get_client()


def _save_wav(path, samples, rate):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.tobytes())


def _refresh_devices():
    """Re-scan audio devices. PortAudio keeps the list from startup, so a
    Bluetooth headset that reconnects or switches mode looks 'unavailable'."""
    sd._terminate()
    sd._initialize()


def _input_device():
    """The microphone to record from: the first input whose name contains
    CONFIG['input_device'] (e.g. 'IPEVO'), else the Windows default.

    Pinning it matters: if Windows' default mic is a Bluetooth headset, using
    it flips the headset into phone-call mode, which kills music-quality
    output AND turns its tap controls into call buttons."""
    wanted = CONFIG.get("input_device")
    if wanted:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and wanted.lower() in d["name"].lower():
                return i
        print(f"[voice] no microphone matching {wanted!r}; using the default")
    return None


def beep(freq=880, seconds=0.12):
    """Short tone so the user knows the microphone is now listening.

    Never fatal: a missing beep must not throw away the user's question."""
    t = np.arange(int(TTS_RATE * seconds)) / TTS_RATE
    tone = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    for attempt in range(2):
        try:
            sd.play(tone, TTS_RATE, blocking=True)
            return
        except sd.PortAudioError as e:
            if attempt == 0:
                _refresh_devices()
            else:
                print(f"[voice] beep failed ({e}); listening anyway")


def play_audio(path):
    """Play a saved wav file and wait for it to finish."""
    with wave.open(path, "rb") as f:
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    sd.play(samples, rate, blocking=True)


def speak(text, tag="reply", voice=None, instructions=""):
    """Say text aloud. voice/instructions pick the speaker and accent."""
    if not text:
        return
    voice = voice or CONFIG["tts_voice"]
    # One sentence at a time: two captures must not talk over each other.
    with _speak_lock:
        try:
            _speak_openai(text, tag, voice, instructions)
            return "openai"
        except Exception as e:
            print(f"[voice] online speech failed ({type(e).__name__}: {e}); using offline voice")
        if CONFIG["use_local_tts_fallback"]:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                return "offline"
            except Exception as e:
                print(f"[voice] offline speech failed too: {e}")
        print(f"[voice] (could not speak) {text}")
        return "none"


def _speak_openai(text, tag, voice, instructions):
    kwargs = {"model": CONFIG["tts_model"], "voice": voice, "input": text,
              "response_format": "pcm"}
    if instructions:
        kwargs["instructions"] = instructions
    # Pick up a headset that reconnected since the last sentence.
    _refresh_devices()

    chunks = []
    leftover = b""
    with _client().audio.speech.with_streaming_response.create(**kwargs) as response:
        with sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16") as out:
            for chunk in response.iter_bytes(4096):
                # int16 samples are 2 bytes; keep an odd trailing byte for next time.
                chunk = leftover + chunk
                cut = len(chunk) - (len(chunk) % 2)
                chunk, leftover = chunk[:cut], chunk[cut:]
                if chunk:
                    out.write(chunk)
                    chunks.append(chunk)

    # Keep what was said, so a bad reading can be replayed and diagnosed.
    if chunks:
        name = time.strftime(f"%Y%m%d_%H%M%S_{tag}.wav")
        samples = np.frombuffer(b"".join(chunks), dtype=np.int16)
        _save_wav(os.path.join(CONFIG["audio_dir"], name), samples, TTS_RATE)
        prune_audio()


def _record_until_quiet():
    """Record mono audio until ~1 s of silence after speech, or the time limit.

    Returns int16 samples, or None if nobody spoke.
    """
    rate = CONFIG["sample_rate"]
    block = int(rate * 0.05)                     # 50 ms blocks
    max_blocks = int(CONFIG["record_max_s"] / 0.05)
    quiet_needed = int(CONFIG["record_silence_s"] / 0.05)
    give_up = int(CONFIG["record_seconds"] / 0.05)  # no speech by then -> stop

    audio, levels = [], []
    started, quiet = False, 0
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=block,
                        device=_input_device()) as stream:
        for i in range(max_blocks):
            data, _ = stream.read(block)
            audio.append(data.copy())
            level = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            levels.append(level)

            # The first 0.2 s measures the room's background noise.
            if i < 4:
                continue
            noise = np.median(levels[:4])
            loud = level > max(noise * 3.0, CONFIG["speech_level_min"])
            if loud:
                started, quiet = True, 0
            elif started:
                quiet += 1
                if quiet >= quiet_needed:
                    break
            elif i >= give_up:
                break
    # Printed so speech_level_min can be tuned for a new microphone.
    print(f"[voice] loudest {max(levels):.0f}, room noise {np.median(levels[:4]):.0f}, "
          f"speech threshold {max(np.median(levels[:4]) * 3.0, CONFIG['speech_level_min']):.0f}")
    return np.concatenate(audio)[:, 0] if started else None


def record_and_transcribe():
    """Record a short utterance and return the transcript, or '' if unavailable.

    Never raises: if the mic or the network is missing, return '' and let the
    model decide the task from the image alone.
    """
    if not CONFIG["voice_enabled"]:
        return ""
    try:
        beep()
        samples = _record_until_quiet()
        if samples is None:
            print("[voice] heard nothing")
            return ""
        path = os.path.join(CONFIG["audio_dir"], time.strftime("%Y%m%d_%H%M%S_question.wav"))
        _save_wav(path, samples, CONFIG["sample_rate"])
        with open(path, "rb") as f:
            result = _client().audio.transcriptions.create(model=CONFIG["stt_model"], file=f)
        text = result.text.strip()
        print(f"[voice] heard: {text}")
        return text
    except Exception as e:
        print(f"[voice] transcription failed: {type(e).__name__}: {e}")
        return ""


def prune_audio():
    """Delete all but the newest CONFIG['keep_audio_files'] audio files."""
    files = sorted(glob.glob(os.path.join(CONFIG["audio_dir"], "*.wav")), key=os.path.getmtime)
    for old in files[:-CONFIG["keep_audio_files"]]:
        try:
            os.remove(old)
        except OSError:
            pass
