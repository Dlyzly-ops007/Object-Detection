"""Vision Assistant — real-time object detection with a GUI, spoken announcements and voice commands.

Run:  python object_detection.py            (see --help for options)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _use_project_python() -> None:
    """Re-launch under this folder's virtual environment if started with another Python.

    The packages (TensorFlow 2.10, TF-Hub, Vosk...) live in this venv's Python 3.10. A plain
    `python object_detection.py` may pick a different install that lacks them.
    """
    here = Path(__file__).resolve().parent
    venv_python = here / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if venv_python.is_file() and Path(sys.prefix).resolve() != here:
        sys.exit(subprocess.call([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]))


_use_project_python()

from vision_assistant.config import TFHUB_CACHE_DIR  # noqa: E402

# Must be set before TensorFlow / TF-Hub are imported: keeps the model on disk so it is only downloaded once.
os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection assistant")
    parser.add_argument("--camera", type=int, help="camera index to use (default: external first, then built-in)")
    parser.add_argument("--confidence", type=float, help="detection threshold between 0 and 1")
    parser.add_argument("--no-voice", action="store_true", help="disable voice commands")
    parser.add_argument("--mute", action="store_true", help="start with spoken announcements off")
    parser.add_argument("--no-autostart", action="store_true", help="don't open the camera on launch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sys.platform == "win32":
        try:  # crisp text on scaled (125%/150%) displays instead of blurry bitmap scaling
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    from vision_assistant.app import main as run_app
    from vision_assistant.config import Settings

    settings = Settings.load()
    if args.camera is not None:
        settings.camera_index = args.camera
    if args.confidence is not None:
        settings.confidence = min(0.99, max(0.05, args.confidence))
    if args.no_voice:
        settings.voice_commands = False
    if args.mute:
        settings.tts_enabled = False
    if args.no_autostart:
        settings.auto_start_camera = False
    run_app(settings)


if __name__ == "__main__":
    main()
