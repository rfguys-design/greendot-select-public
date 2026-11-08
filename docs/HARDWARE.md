# 🔧 Hardware Setup — read this before writing any code

Everything physical: what to buy, how to set it up, how to check it works, and what
"normal" looks like so you know whether something is broken.

---

## 🛒 Shopping list

| Item | What to get | Notes |
|---|---|---|
| **Green dot stickers** | Fluorescent/neon green round labels, **¾ inch (19 mm)** | The single most important purchase. Office-supply colour-coding dots. Get neon, not dark green. |
| **Webcam** | Any 1080p USB webcam | A laptop's built-in camera works for early testing but points at your face, not the desk. |
| **Camera mount** | Gooseneck phone/webcam stand, or a small tripod | Must hold the camera **pointing down at the desk**, steady, about 40–50 cm above the page. |
| **Desk lamp** | Any adjustable lamp | For controlled lighting tests. |
| **Headphones** | Bluetooth earbuds with a tap/play button | You already confirmed this works. |
| **Backup trigger** | Bluetooth camera shutter remote (~$8) | Pairs as a keyboard and sends a plain key. Saves you if media keys misbehave on someone's laptop. |
| **Test pages** | Printed worksheets: math problems, paragraphs, a table, a diagram | Also handwrite a few. You'll need variety for the dataset. |

**Why ¾ inch:** at 45 cm from a 1080p camera, a 19 mm dot is roughly 40 pixels
across — about 1200 px² of area, comfortably above the 60 px² minimum and well
below the 8000 px² maximum in `config.py`. Much smaller and detection gets
unreliable; much bigger and the dots cover your text.

**Buy two packs.** They get lost, and you'll want spares in different sizes if the
first choice doesn't work.

---

## 📐 Physical setup

```
        [camera on stand, pointing straight down]
                        |
                        | 40-50 cm
                        |
    ____________________v____________________
   |  ●                                      |   ● = green dot
   |        The math problem you want        |
   |        to select goes here              |
   |                                  ●      |
   |_________________________________________|
              page flat on the desk
```

**Rules:**

1. **Camera points down at the page**, not at your face. Roughly straight down is
   easiest to start with; tilted angles come later in the dataset.
2. **40–50 cm above the desk.** Close enough that the dots are big, far enough that
   a whole letter page fits in view.
3. **Page flat.** Tape the corners if it curls. A curved page breaks the crop
   geometry.
4. **Nothing else green in the frame.** No green pens, highlighters, notebooks,
   water bottles, or plants. Our detector reports an error when it sees three green
   things — that's correct behaviour, but it's annoying while you're building.
5. **Avoid your own shadow.** Put the lamp on the opposite side from where your
   hand reaches in.

---

## 📍 Where to put the dots

```
   ●───────────────────────┐
   │   3 x + 7 = 22        │     ← dot at each OPPOSITE corner
   │   x = ?               │        (top-left and bottom-right,
   └───────────────────────●         or top-right and bottom-left)
```

- **Two dots, at opposite corners** of what you want selected
- **Just outside the text**, not on top of it — the crop cuts inward past each dot,
  so a dot sitting on a word means that word gets cut off
- **At least 3 cm apart**, or the crop comes out too small
- Either diagonal works: top-left + bottom-right, or top-right + bottom-left

---

## ✅ Setup checklist — do this before your first task

Run the checker first:

```bash
python tools/doctor.py
```

It tests everything below automatically and tells you exactly what's wrong. Fix
anything it flags before you start coding.

Then check by eye:

- [ ] Camera is mounted, pointing down, page fills most of the frame
- [ ] Two green dots on the page, both clearly visible in the preview
- [ ] No other green objects anywhere in the frame
- [ ] Headphones connected and audio plays through them
- [ ] `OPENAI_API_KEY` is set in your environment

---

## 📊 What "normal" looks like

Use this to tell the difference between "still broken" and "working, needs tuning."

| Thing | Broken | Normal | Great |
|---|---|---|---|
| Green dot detection, good light | never finds 2 | finds 2 most of the time | finds 2 every time |
| Detection when you move the page | loses them constantly | occasional miss | steady |
| Detection rate on the dataset (week 2) | under 50% | **60–85% on the first run** | over 90% after tuning |
| Crop contents | includes a dot, or the next problem | just your region, small margin | exactly your region |
| Tap → first sound | nothing happens | **under 1 second** ("thinking" sound) | instant |
| Tap → spoken answer | over 15 s, or hangs | **3–8 seconds** | under 3 s |
| Speech transcription | gibberish | mostly right | exact |

**A 60–85% detection rate on your first dataset run is normal and fine.** That is
the whole point of week 2 — you measure it, find the pattern in the failures, and
push the number up. Do not panic and rewrite everything.

---

## 🩺 Common problems

**"Camera didn't open" or a black window**
Your webcam isn't index 0. Run `python tools/doctor.py` — it tries every index and
tells you which ones work. Then create `greendot/config_local.py`:
```python
LOCAL = {"camera_index": 1}
```
That file is gitignored, so it only affects your machine.

**Dots found in one room but not another**
Expected — this is exactly the problem the LAB colour space is meant to reduce.
Adjust `a_max` with the A/S keys in the preview window until it works in both, then
commit that value.

**"found 3 dots, need exactly 2"**
Something else green is in the frame. Look at the preview — it draws a circle on
everything it thinks is a dot. Usually a highlighter or a book spine.

**The dots are found but the crop is empty or tiny**
Your dots are too close together, or they're on the same side instead of opposite
corners.

**No sound**
Check the system output device is your headphones, not the laptop speakers.
`python tools/doctor.py` plays a test tone.

**Tap does nothing (macOS)**
System Settings → Privacy & Security → Accessibility → add your terminal or IDE,
then **restart the terminal**. Write down what you did in `docs/SETUP.md`.

**Blurry pictures**
Most webcams autofocus. Give it a second after moving the page. If it never
focuses, the page may be closer than the camera's minimum focus distance — move
the camera up.
