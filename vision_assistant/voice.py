"""Offline voice commands with Vosk + sounddevice."""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from .config import ALL_LABELS, AUDIO_SAMPLE_RATE, LABEL_ALIASES, VOSK_MODEL_PATH, pluralize

log = logging.getLogger(__name__)
SetLogLevel(-1)


@dataclass(frozen=True)
class VoiceCommand:
    action: str
    text: str
    label: str | None = None


# action -> trigger phrases. The longest matching phrase wins, so "stop recording" beats "stop".
COMMANDS = {
    "camera_start": ("start", "hello", "start camera", "camera on", "open camera"),
    "camera_stop": ("stop", "stop camera", "camera off", "close camera"),
    "describe": ("what do you see", "what can you see", "describe", "look around",
                 "what is in front of me", "what's in front of me"),
    "snapshot": ("snapshot", "take a picture", "take a photo", "take picture", "take photo", "capture"),
    "record_start": ("start recording", "begin recording", "record video"),
    "record_stop": ("stop recording", "end recording"),
    "pause": ("pause", "pause detection"),
    "resume": ("resume", "resume detection", "continue"),
    "mute": ("mute", "be quiet", "silence", "stop talking", "quiet"),
    "unmute": ("unmute", "speak up", "start talking"),
    "help": ("help", "what can i say"),
    "exit": ("exit application", "close application", "quit application", "goodbye"),
}
QUERIES = {
    "count": ("how many",),
    "locate": ("where is", "where are", "where's", "find", "is there", "are there",
               "do you see", "can you see"),
}

_PHRASES = sorted(((tuple(p.split()), a) for a, ps in COMMANDS.items() for p in ps),
                  key=lambda item: -len(item[0]))

_LABEL_WORDS = {}
for _label in ALL_LABELS:
    _LABEL_WORDS[_label] = _label
    _LABEL_WORDS[pluralize(_label).lower()] = _label
for _alias, _label in LABEL_ALIASES.items():
    _LABEL_WORDS[_alias] = _label
    _LABEL_WORDS.setdefault(pluralize(_alias), _label)
_LABEL_PATTERN = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _LABEL_WORDS), key=len, reverse=True)) + r")\b")


def _contains(tokens: list, phrase: tuple) -> bool:
    n = len(phrase)
    return any(tuple(tokens[i:i + n]) == phrase for i in range(len(tokens) - n + 1))


def find_label(text: str) -> str | None:
    match = _LABEL_PATTERN.search(text)
    return _LABEL_WORDS[match.group(1)] if match else None


def parse_command(text: str) -> VoiceCommand | None:
    text = text.lower().strip()
    if not text:
        return None
    label = find_label(text)
    if label:
        for action, phrases in QUERIES.items():
            if any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases):
                return VoiceCommand(action, text, label)
    tokens = text.split()
    for phrase, action in _PHRASES:
        if _contains(tokens, phrase):
            return VoiceCommand(action, text)
    return None


def list_input_devices() -> list:
    """[(index, name)] of input devices on the default host API (avoids listing every mic 4 times)."""
    try:
        default_api = sd.default.hostapi
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("Could not list audio devices: %s", exc)
        return []
    return [(i, dev["name"]) for i, dev in enumerate(devices)
            if dev["max_input_channels"] > 0 and (default_api < 0 or dev["hostapi"] == default_api)]


def resolve_device(name: str):
    """Device index for a saved device name, or None (system default) if it is gone."""
    if not name:
        return None
    return next((i for i, dev_name in list_input_devices() if dev_name == name), None)


class VoiceListener:
    """Streams the microphone into Vosk on a background thread.

    Callbacks run on the listener thread: on_command(VoiceCommand), on_heard(text, final),
    on_level(0..1), on_status(state, message). `is_muted()` lets the app ignore the mic
    while it is talking, so it never obeys its own voice.
    """

    def __init__(self, on_command, on_heard=None, on_level=None, on_status=None, is_muted=None):
        self._on_command = on_command
        self._on_heard = on_heard or (lambda text, final: None)
        self._on_level = on_level or (lambda level: None)
        self._on_status = on_status or (lambda state, msg: None)
        self._is_muted = is_muted or (lambda: False)
        self._model = None
        self._audio = queue.Queue(maxsize=50)
        self._running = threading.Event()
        self._thread = None

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self, device: str = "") -> None:
        if self.running:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, args=(resolve_device(device),), name="voice", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            self._audio.put_nowait(bytes(indata))
        except queue.Full:
            pass  # recogniser fell behind; dropping audio beats growing latency

    def _open_stream(self, device):
        """16 kHz is ideal for Vosk; fall back to the device's native rate if it refuses."""
        try:
            return sd.RawInputStream(samplerate=AUDIO_SAMPLE_RATE, blocksize=4000, device=device,
                                     dtype="int16", channels=1, callback=self._callback), AUDIO_SAMPLE_RATE
        except Exception:
            rate = int(sd.query_devices(device, "input")["default_samplerate"])
            return sd.RawInputStream(samplerate=rate, blocksize=rate // 4, device=device,
                                     dtype="int16", channels=1, callback=self._callback), rate

    def _run(self, device) -> None:
        try:
            if self._model is None:
                self._on_status("loading", "Loading speech model…")
                if not VOSK_MODEL_PATH.is_dir():
                    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
                self._model = Model(str(VOSK_MODEL_PATH))
            stream, rate = self._open_stream(device)
        except Exception as exc:
            log.error("Voice commands unavailable: %s", exc)
            self._on_status("error", str(exc))
            self._running.clear()
            return

        recognizer = KaldiRecognizer(self._model, rate)
        was_muted, last_level, last_partial = False, 0.0, ""
        with stream:
            self._on_status("listening", "Listening")
            while self._running.is_set():
                try:
                    data = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue

                now = time.time()
                if now - last_level > 0.06:
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
                    dbfs = 20 * np.log10(rms / 32768 + 1e-9)
                    self._on_level(min(1.0, max(0.0, (dbfs + 55) / 50)))  # -55 dBFS (silence) .. -5 dBFS
                    last_level = now

                if self._is_muted():
                    if not was_muted:
                        recognizer.Reset()
                        was_muted = True
                    continue
                was_muted = False

                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                    last_partial = ""
                    if text:
                        self._on_heard(text, True)
                        command = parse_command(text)
                        if command:
                            self._on_command(command)
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial and partial != last_partial:
                        last_partial = partial
                        self._on_heard(partial, False)
        self._on_level(0.0)
        self._on_status("off", "Off")
