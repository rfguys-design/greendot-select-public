# GreenDot Select

**A physical select button for paper.**

Mark any region of a printed page with two green dots, tap your headphones, and
ask out loud. GreenDot Select crops exactly that region and reads it aloud,
gives a hint, solves it, translates it, explains it, copies its text or picture
to your computer, turns a table into a chart, or colors in a drawing. Spoken
answers need no screen.

Only the marked region is sent to the AI. The rest of the page never leaves the
laptop.

## How it works

```
tap → latest camera frame → find the 2 green dots → straighten to the page edge
    → crop inward past the dots → hear the request → route it → answer → log the row
```

| Step | What runs | Where |
|---|---|---|
| Trigger | Headphone media button, `T` key, or double-click (`pynput`, 1 s debounce) | laptop |
| Camera | One reader thread keeps the newest frame; exposure is stepped until the paper is bright but not clipped, then locked | laptop |
| Dot detection | Blur, convert to LAB, threshold the a* (green–red) channel, filter contours by area and circularity; exactly two dots or it says so | laptop, OpenCV |
| Alignment | Otsu threshold, find the page blob, Hough line segments on its outline, length-weighted median angle; rotate around the dots' midpoint | laptop, OpenCV |
| Crop | Inset inward past each dot from the full-resolution frame, downscale to 1024 px | laptop, OpenCV |
| Speech in | Whisper transcription of the spoken request | OpenAI API |
| Copy text | Tesseract OCR first (verbatim); the model only when OCR confidence is low | laptop |
| Everything else | One gpt-4o vision call with a strict JSON schema; the reply names the skill | OpenAI API |
| Output | Speech (gpt-4o-mini-tts), clipboard, matplotlib chart, generated picture, local dashboard | laptop + API |
| Log | One CSV row per capture, including failures | laptop |

### Skills

All skills live in one table in [`greendot/skills.py`](greendot/skills.py):
`read_aloud`, `solve_math`, `hint`, `translate`, `explain`, `copy_text`,
`plot_table`, `edit_image`, and `unclear`. The table builds both the system
prompt and the list of allowed tasks in the JSON schema, so a new text skill is
one new table entry. A skill with a new kind of output also needs a handler in
`main.py`.

## Requirements

- Python 3.10 or newer
- An OpenAI API key in the `OPENAI_API_KEY` environment variable
- A camera looking down at the desk (a document camera works best), a
  microphone, and headphones with a play/pause button
- Two green dots or stickers
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on
  `PATH` for verbatim text copying

Hardware details, including camera placement and dot size, are in
[docs/HARDWARE.md](docs/HARDWARE.md). Platform notes are in
[docs/SETUP.md](docs/SETUP.md).

## Quick start

```bash
git clone https://github.com/rfguys-design/greendot-select-public.git
cd greendot-select-public
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."        # Windows PowerShell: $env:OPENAI_API_KEY = "sk-..."

python tools/doctor.py                # checks packages, key, camera, mic, audio
python -m greendot.main               # run it
```

Put the dots at two opposite corners of what you want, then tap the headphone
button (or press `T`) and say your request, such as "give me a hint" or
"copy this picture". Results also appear on a local dashboard at
http://localhost:8765.

The key is read only from the environment. Never put it in a source file.

## Tests and tools

```bash
python -m pytest tests/ -v            # detection, alignment and crop tests
python tools/live_detect.py           # live view of dots, page angle and crop, no API calls
python evaluate.py                    # scores labeled captures in dataset/
```

## Repository layout

| Path | What it is |
|---|---|
| `greendot/` | The application, one module per area |
| `tests/` | Unit tests for detection, alignment and cropping |
| `tools/` | Diagnostic scripts (setup check, live detection, camera quality) |
| `dataset/` | Label format for the evaluation set |
| `evaluate.py` | Scores the pipeline on labeled captures |
| `docs/` | Hardware, setup and metrics notes |
| `prototype/` | The original single-file version, kept for reference |

## Team

| Student | Area |
|---|---|
| Harper | Camera and dot detection |
| James | Cropping, imaging, OCR and charts |
| Joanna | Trigger, voice and skills |
| Markus | Orchestration, API calls, logging and evaluation |

Built with mentor support and AI coding assistance.

## License

[MIT](LICENSE)
