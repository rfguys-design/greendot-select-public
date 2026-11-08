# Setup notes (per platform)

Fill this in as you go. Media-key and microphone behaviour differs between
operating systems, and the person who solves it should write it down here rather
than in a chat message that scrolls away.

## macOS
- Media key listener requires Accessibility permission:
  System Settings → Privacy & Security → Accessibility → add your terminal/IDE.
- Microphone permission is requested on first use.
- Camera index: _record what worked_

## Windows
- Media key listener: _record whether it worked out of the box_
- `pygame` mixer keeps a file handle open — always stop and unload before
  loading a new file.
- Camera index: _record what worked_

### Windows camera notes (test laptop, Logitech C920, 2026-09-30)
- **Find your camera number** with `python tools/doctor.py`. Virtual cameras
  and built-in cameras take the low numbers: here 0 was "Poly Camera Pro"
  (grey placeholder), 1 the built-in HP camera (black, privacy shutter), and
  the C920 was **2**. Put it in `greendot/config_local.py`.
- **Grey screen, or the camera takes a minute or more / never delivers a
  frame:** another program is holding it. Check for camera software running
  in the background (Poly Camera Pro and its `PolyCamProService` service,
  Logitech tools, Teams, Zoom, OBS) and turn the camera off in Discord.
  Stopping a service needs an admin terminal:
  `Stop-Service PolyCamProService`.
- **Slow camera start (60–76 s):** OpenCV's Media Foundation backend can take
  over a minute to open the C920. Disabling its hardware transforms brought
  it down to about 1.5 s:
  ```powershell
  setx OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS 0
  ```
  then restart the terminal / VS Code. (DirectShow, `cv2.CAP_DSHOW`, was
  tried and returned blank frames.)
- **Exposure looks wrong right after start-up:** the exposure set while the
  camera opens can be ignored. Pressing E then D in the preview window
  re-applies it. Exposure −7 to −8 worked under a desk lamp.
- **Resolution:** the C920 delivers 1920×1080 at 30 fps, but with autofocus it
  came out blurrier than 720p, and the AI read it worse. Stay at 720p until
  fixed focus is added to `camera.py`.
- **Camera height matters more than resolution:** lowering the camera so the
  page filled more of the frame took the AI from 0/3 to 3/3 correct on a
  math problem.
- **Live check:** `python tools/live_detect.py` shows the detected dots, the
  page angle and the crop window.

## Linux
- _record what was needed_

## Tesseract (optional, for local OCR)
The `pytesseract` package needs the tesseract binary installed separately.
- macOS: `brew install tesseract`
- Windows: installer from the tesseract project; add it to PATH
- Linux: `sudo apt install tesseract-ocr`

## Per-machine config
Create `greendot/config_local.py` (gitignored):

```python
LOCAL = {"camera_index": 1, "manual_exposure": -5}
```
