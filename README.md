# Vision Assistant

A Python desktop app for real-time object detection with spoken announcements and offline voice commands. Built with TensorFlow Hub's SSD MobileNet v2, OpenCV, Tkinter, Vosk, and pyttsx3.

## Features

- Live bounding boxes, confidence scores, object counts, and a stats overlay.
- Watch lists to highlight objects, filter detections, and prioritize announcements.
- Spoken scene descriptions, object counting, and relative object locations.
- Camera selection, resolution controls, mirroring, and detection pause/resume.
- Snapshots, video recording, and optional pixelation of detected people.
- Saved preferences in `settings.json`.

The app keeps these 15 COCO categories: **person, bicycle, car, motorcycle, bus, truck, cat, dog, backpack, bottle, cup, chair, couch, laptop, and cell phone**. Other model categories are filtered out.

## Setup

The instructions below target **Windows with 64-bit Python 3.10**, matching this project's TensorFlow 2.10 dependencies. Run commands from the repository root.

```powershell
py -3.10 -m venv .
.\Scripts\python.exe -m pip install -r requirements.txt
```

The virtual environment intentionally lives at the repository root: `run.bat` and the Python launcher look for `Scripts/python.exe` there. Its files are excluded from Git.

The detection model downloads automatically on first use and is cached in `models/tfhub_cache/`. Internet access is needed for installation and the first model download; inference then runs locally.

### Optional voice commands

Download and extract **`vosk-model-en-in-0.5`** from the [official Vosk model list](https://alphacephei.com/vosk/models). Place the extracted directory beside `object_detection.py`:

```text
Object-Detection/
├── object_detection.py
├── vosk-model-en-in-0.5/
│   ├── am/
│   ├── conf/
│   └── ...
└── vision_assistant/
```

Enable **Listen for voice commands** in the Voice tab. Without the Vosk model, use `--no-voice`. Spoken announcements use the system's text-to-speech voices and do not require Vosk.

## Test with images — no camera required

These two sample images were newly AI-generated for this repository. They use no webcam captures or pre-existing local photos. They are smoke-test inputs, not a labeled accuracy benchmark; detection scores and missed objects can vary.

| Desk | Dog and bicycle |
| --- | --- |
| ![Sample desk with a laptop, cell phone, cup, bottle, and chair](samples/desk.png) | ![Sample dog sitting beside a bicycle in a park](samples/dog-bicycle.png) |
| Visible categories: laptop, cell phone, cup, bottle, chair | Visible categories: dog, bicycle |

From the repository root, start the project interpreter:

```powershell
.\Scripts\python.exe
```

Paste this code. It calls the existing detector directly, prints predictions, and saves annotated copies under `snapshots/`. It never starts the GUI, camera discovery, microphone, or speech output. Running it again replaces only these two named output files.

```python
from pathlib import Path
import cv2
from vision_assistant.detector import ObjectDetector
from vision_assistant.recorder import save_image
from vision_assistant.render import draw_detections

detector = ObjectDetector()
detector.load()
output = Path("snapshots")
output.mkdir(exist_ok=True)

for name in ("desk", "dog-bicycle"):
    source = Path("samples") / f"{name}.png"
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"Could not read {source}")
    detections = detector.detect(frame, threshold=0.5)
    print(name, [(d.label, round(d.score, 3)) for d in detections])
    draw_detections(frame, detections)
    save_image(output / f"sample_{name}_detections.png", frame)

```

Press Enter on the final blank line to finish the loop; use `exit()` to leave Python. Open the saved PNGs to inspect the results. The desktop GUI currently accepts live camera input only.

A local smoke test at threshold `0.5` detected the dog and bicycle. On the desk image it detected the laptop, chair, cell phone, and cup, missed the bottle, and returned a second laptop box. These samples help expose missed and duplicate detections as well as confirm that inference runs.

## Run the desktop app

**These commands open a camera.** For testing without camera access, use the image example above.

```powershell
.\run.bat
```

Or start with voice commands and automatic announcements disabled:

```powershell
.\Scripts\python.exe object_detection.py --no-voice --mute
```

The committed `settings.json` currently disables both voice commands and automatic announcements. Enable them in the Voice tab when needed. Changes to settings, including command-line overrides, are saved when the app closes.

| Option | Effect |
| --- | --- |
| `--camera 0` | Use a specific camera index; automatic selection tries index 1, then 0 |
| `--confidence 0.6` | Set the confidence threshold; clamped to 0.05–0.99 |
| `--no-voice` | Disable microphone voice commands |
| `--mute` | Disable automatic spoken announcements at startup |
| `--no-autostart` | Skip starting the continuous camera preview |
| `--help` | Show options without starting the app |

**`--no-autostart` does not prevent camera access:** the GUI still probes camera devices during startup. The camera refresh button also probes devices.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| Space | Start / stop camera |
| S | Save snapshot |
| R | Start / stop recording |
| D | Describe the scene |
| P | Pause / resume detection |
| M | Mute / unmute automatic announcements |
| F11 / Esc | Toggle / leave full screen |
| Ctrl+Q | Quit |

### Voice examples

With voice commands enabled and the speech model installed, say:

- “start camera” / “stop camera”
- “what do you see”
- “where is my phone”
- “how many people”
- “take a picture”
- “start recording” / “stop recording”
- “pause” / “resume”
- “mute” / “unmute”
- “goodbye”

## Project layout

```text
object_detection.py     Entry point and command-line options
run.bat                 Windows launcher
requirements.txt        Pinned Python dependencies
settings.json           Persisted preferences
samples/                New sample images and generation prompts
vision_assistant/
  app.py, panels.py      Tkinter interface and controls
  controller.py         Camera, detector, speech, and recording coordination
  detector.py           Model loading, inference, filtering, and duplicate suppression
  camera.py             Threaded camera capture and device discovery
  scene.py              Stable object counts and scene descriptions
  voice.py, speech.py    Voice command recognition and text-to-speech
  render.py             Detection overlays and privacy pixelation
  recorder.py           Snapshot and video output
```

`models/`, `vosk-model-*/`, `snapshots/`, `recordings/`, and the virtual environment are ignored by Git. Recordings use MP4 when supported, with an AVI fallback. Review local changes to `settings.json` before committing them.

## Troubleshooting

- **TensorFlow installation fails:** use 64-bit Python 3.10 and the pinned requirements; newer Python versions may lack compatible wheels.
- **Model loading fails:** allow the initial download, check network access, and ensure `models/` is writable.
- **Camera unavailable:** close other apps using it, check Windows camera permissions, or select another camera index.
- **Voice commands unavailable:** check the exact Vosk directory name, microphone permissions, and the selected input device.
- **No detections:** check the supported categories and confidence threshold. Small, obscured, or poorly lit objects may be missed.

Privacy pixelation applies to detected person boxes; missed detections remain visible.
