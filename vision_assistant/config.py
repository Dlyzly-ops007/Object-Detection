"""Paths, constants, COCO vocabulary and persisted user settings."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
TFHUB_CACHE_DIR = ROOT_DIR / "models" / "tfhub_cache"
VOSK_MODEL_PATH = ROOT_DIR / "vosk-model-en-in-0.5"
SETTINGS_FILE = ROOT_DIR / "settings.json"
SNAPSHOT_DIR = ROOT_DIR / "snapshots"
RECORDING_DIR = ROOT_DIR / "recordings"

AUDIO_SAMPLE_RATE = 16000
AUTO_CAMERA = -1  # "external webcam first, then the built-in one"

RESOLUTIONS = {
    "640 x 480": (640, 480),
    "800 x 600": (800, 600),
    "1280 x 720": (1280, 720),
    "1920 x 1080": (1920, 1080),
}

# Full COCO label map used by the TF-Hub SSD MobileNet v2 model.
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}
ALL_LABELS = sorted(COCO_LABELS.values())

_IRREGULAR_PLURALS = {
    "person": "people", "mouse": "mice", "knife": "knives", "skis": "skis",
    "scissors": "scissors", "sheep": "sheep", "bus": "buses", "couch": "couches",
    "sandwich": "sandwiches", "wine glass": "wine glasses", "bench": "benches",
    "toothbrush": "toothbrushes", "hair drier": "hair driers", "tv": "TVs",
}

# Spoken words that should resolve to a COCO label ("where is my phone?").
LABEL_ALIASES = {
    "people": "person", "persons": "person", "man": "person", "woman": "person",
    "someone": "person", "somebody": "person", "human": "person",
    "phone": "cell phone", "mobile": "cell phone", "cellphone": "cell phone",
    "television": "tv", "monitor": "tv", "screen": "tv",
    "sofa": "couch", "table": "dining table", "bike": "bicycle",
    "motorbike": "motorcycle", "plane": "airplane", "ball": "sports ball",
    "plant": "potted plant", "teddy": "teddy bear", "computer": "laptop",
    "mug": "cup", "glass": "wine glass", "fridge": "refrigerator",
    "puppy": "dog", "kitten": "cat", "bag": "backpack", "remote control": "remote",
}

DEFAULT_WATCHLIST = ("person", "dog", "cat", "cell phone", "laptop")


def pluralize(label: str) -> str:
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith(("s", "sh", "ch", "x")):
        return label + "es"
    return label + "s"


def with_article(label: str) -> str:
    return ("an " if label[0] in "aeiou" else "a ") + label


def quantity(label: str, count: int) -> str:
    """'a laptop', '2 people'."""
    return with_article(label) if count == 1 else f"{count} {pluralize(label)}"


@dataclass
class Settings:
    """User-tweakable settings, persisted to settings.json between runs."""

    # Camera
    camera_index: int = AUTO_CAMERA
    resolution: str = "640 x 480"
    mirror: bool = False
    auto_start_camera: bool = True
    # Detection
    confidence: float = 0.5
    max_detections: int = 20
    watchlist: list = field(default_factory=lambda: list(DEFAULT_WATCHLIST))
    show_only_watchlist: bool = False
    announce_only_watchlist: bool = False
    alert_on_watchlist: bool = True
    # Display
    show_labels: bool = True
    show_scores: bool = True
    show_hud: bool = True
    privacy_blur: bool = False
    box_thickness: int = 2
    # Speech output
    tts_enabled: bool = True
    tts_rate: int = 170
    tts_volume: float = 1.0
    tts_voice: str = ""
    announce_cooldown: float = 3.0
    announce_departures: bool = False
    # Voice commands
    voice_commands: bool = True
    mic_device: str = ""

    _RANGES = {
        "confidence": (0.05, 0.99), "max_detections": (1, 100), "box_thickness": (1, 6),
        "tts_rate": (80, 300), "tts_volume": (0.0, 1.0), "announce_cooldown": (0.5, 30.0),
    }

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        settings = cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return settings
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable settings file %s: %s", path, exc)
            return settings
        if not isinstance(raw, dict):
            return settings

        for f in fields(cls):
            if f.name not in raw:
                continue
            default, value = getattr(settings, f.name), raw[f.name]
            try:
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, (int, float)):
                    value = type(default)(value)
                    lo, hi = cls._RANGES.get(f.name, (value, value))
                    value = type(default)(min(max(value, lo), hi))
                elif isinstance(default, list):
                    value = [v for v in value if v in COCO_LABELS.values()]
                else:
                    value = str(value)
            except (TypeError, ValueError):
                continue
            setattr(settings, f.name, value)
        if settings.resolution not in RESOLUTIONS:
            settings.resolution = cls.resolution
        return settings

    def save(self, path: Path = SETTINGS_FILE) -> None:
        try:
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not save settings: %s", exc)

    @property
    def frame_size(self) -> tuple:
        return RESOLUTIONS[self.resolution]
