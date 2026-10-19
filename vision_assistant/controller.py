"""Application core: wires camera, detector, tracker, speech, voice and recorder together.

No Tk code lives here. Background threads report to the UI through `emit(kind, *args)`,
which the UI drains on its own thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from . import scene
from .camera import CameraError, CameraStream, probe_cameras
from .config import RECORDING_DIR, SNAPSHOT_DIR, Settings
from .detector import ObjectDetector
from .recorder import VideoRecorder, save_image, timestamped
from .render import draw_detections, draw_hud
from .speech import Speaker
from .voice import VoiceListener

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    detections: list = field(default_factory=list)
    visible: dict = field(default_factory=dict)      # stable label -> count
    frame_size: tuple = (0, 0)
    timestamp: float = 0.0


class Controller:
    def __init__(self, settings: Settings, emit):
        self.settings = settings
        self.emit = emit
        self.detector = ObjectDetector()
        self.tracker = scene.SceneTracker()
        self.speaker = Speaker()
        self.voice = VoiceListener(
            on_command=lambda cmd: emit("command", cmd),
            on_heard=lambda text, final: emit("heard", text, final),
            on_level=lambda level: emit("level", level),
            on_status=lambda state, msg: emit("mic", state, msg),
            is_muted=self.speaker.recently_spoke,
        )
        self.camera = None
        self.recorder = None
        self.ai_paused = False
        self.ai_fps = 0.0
        self.inference_ms = 0.0
        self.last_frame = None  # most recent annotated frame (what the user sees)

        self._result = DetectionResult()
        self._lock = threading.Lock()
        self._camera_busy = False
        self._worker = None
        self._worker_stop = threading.Event()
        self._pending_appeared = {}
        self._pending_departed = []
        self._last_announce = 0.0
        self._last_seq = -1
        self.apply_speech_settings()

    # ------------------------------------------------------------------ lifecycle
    def load_model_async(self) -> None:
        def work():
            self.emit("model", "busy", "Loading…")
            try:
                seconds = self.detector.load()
            except Exception as exc:
                log.exception("Model load failed")
                self.emit("model", "error", f"Failed: {exc}")
                return
            self.emit("model", "ok", f"Ready ({seconds:.1f}s)")
        threading.Thread(target=work, name="model-loader", daemon=True).start()

    def probe_cameras_async(self) -> None:
        def work():
            in_use = (self.camera.index,) if self.camera else ()
            self.emit("cameras", probe_cameras(skip=in_use))
        threading.Thread(target=work, name="camera-probe", daemon=True).start()

    def shutdown(self) -> None:
        self.stop_recording()
        self.voice.stop()
        self.stop_camera(quiet=True)
        self.speaker.shutdown()

    # ------------------------------------------------------------------ camera
    @property
    def camera_on(self) -> bool:
        return self.camera is not None and self.camera.alive

    def start_camera(self) -> None:
        if self._camera_busy or self.camera_on:
            return
        self._camera_busy = True
        s = self.settings
        self.emit("camera", "busy", "Opening…")

        def work():
            try:
                cam = CameraStream(s.camera_index, s.frame_size, s.mirror).start()
            except CameraError as exc:
                self.emit("camera", "error", str(exc))
                self.say("Camera not working")
                return
            finally:
                self._camera_busy = False
            self.camera = cam
            self._worker_stop = threading.Event()
            self._worker = threading.Thread(target=self._detect_loop, args=(cam, self._worker_stop),
                                             name="detector", daemon=True)
            self._worker.start()
            self.emit("camera", "ok", f"Camera {cam.index} · {cam.size[0]}×{cam.size[1]}")
            self.say("Camera activated")
        threading.Thread(target=work, name="camera-open", daemon=True).start()

    def stop_camera(self, quiet: bool = False) -> None:
        self.stop_recording()
        self._worker_stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
            self._worker = None
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        with self._lock:
            self.tracker.reset()
            self._result = DetectionResult()
        self._pending_appeared.clear()
        self._pending_departed.clear()
        self.last_frame = None
        if not quiet:
            self.emit("camera", "off", "Off")
            self.say("Camera stopped")

    def restart_camera(self) -> None:
        if self.camera_on:
            self.stop_camera(quiet=True)
            self.start_camera()

    def set_ai_paused(self, paused: bool) -> None:
        self.ai_paused = paused
        self.ai_fps = 0.0
        with self._lock:
            self.tracker.reset()  # otherwise everything "leaves" the moment detection resumes
            self._result = DetectionResult(frame_size=self._result.frame_size)
        self._pending_appeared.clear()
        self._pending_departed.clear()

    def set_mirror(self, mirror: bool) -> None:
        self.settings.mirror = mirror
        if self.camera:
            self.camera.mirror = mirror

    # ------------------------------------------------------------------ detection
    def _detect_loop(self, cam: CameraStream, stop: threading.Event) -> None:
        seq, error_logged, last_done = 0, False, time.perf_counter()
        while not stop.is_set() and cam.alive:
            seq, frame = cam.wait_for_frame(seq, timeout=0.5)
            if frame is None:
                continue
            if self.ai_paused or not self.detector.ready:
                with self._lock:
                    self._result = DetectionResult(frame_size=(frame.shape[1], frame.shape[0]))
                continue

            s = self.settings
            watch = frozenset(s.watchlist)
            start = time.perf_counter()
            try:
                detections = self.detector.detect(frame, s.confidence, s.max_detections,
                                                  watch if s.show_only_watchlist else None)
            except Exception as exc:
                if not error_logged:
                    log.exception("Inference failed")
                    self.emit("log", f"Detection error: {exc}", "error")
                    error_logged = True
                continue
            done = time.perf_counter()
            ms, rate = 1000 * (done - start), 1 / max(1e-3, done - last_done)
            self.inference_ms = 0.9 * self.inference_ms + 0.1 * ms if self.inference_ms else ms
            self.ai_fps = 0.9 * self.ai_fps + 0.1 * rate if self.ai_fps else rate
            last_done = done

            now = time.time()
            with self._lock:
                update = self.tracker.update(detections, now)
                self._result = DetectionResult(detections, self.tracker.visible,
                                               (frame.shape[1], frame.shape[0]), now)
            if update:
                self._on_scene_update(update, watch)
            self._flush_announcements()

        if not stop.is_set():  # the camera died underneath us
            self.emit("camera", "error", "Camera disconnected")

    @property
    def result(self) -> DetectionResult:
        with self._lock:
            return self._result

    def _on_scene_update(self, update: scene.SceneUpdate, watch: frozenset) -> None:
        s = self.settings
        self.emit("scene", update.appeared, update.departed)
        alerts = [label for label in update.appeared if label in watch]
        if alerts and s.alert_on_watchlist:
            self.emit("alert", alerts)
        for label, n in update.appeared.items():
            if not s.announce_only_watchlist or label in watch:
                self._pending_appeared[label] = n
        if s.announce_departures:
            self._pending_departed.extend(label for label in update.departed
                                          if not s.announce_only_watchlist or label in watch)

    def _flush_announcements(self) -> None:
        """Speak what changed, at most once per cooldown. Unlike before, nothing is lost during the
        cooldown: changes accumulate and are spoken together when it expires."""
        s = self.settings
        if not (self._pending_appeared or self._pending_departed):
            return
        if not s.tts_enabled:
            self._pending_appeared.clear()
            self._pending_departed.clear()
            return
        if time.time() - self._last_announce < s.announce_cooldown or self.speaker.speaking.is_set():
            return

        visible = self.result.visible
        appeared = {k: visible.get(k, n) for k, n in self._pending_appeared.items() if k in visible}
        departed = [k for k in dict.fromkeys(self._pending_departed) if k not in visible]
        self._pending_appeared.clear()
        self._pending_departed.clear()

        watch = s.watchlist
        order = sorted(appeared, key=lambda k: (k not in watch, -appeared[k]))
        parts = []
        if appeared:
            parts.append("I see " + scene.summarize(appeared, order, limit=3) + ".")
        if departed:
            gone = scene.join_words([f"the {k}" for k in departed[:3]])
            parts.append(gone[0].upper() + gone[1:] + " left.")
        if parts:
            self._last_announce = time.time()
            self.say(" ".join(parts))

    # ------------------------------------------------------------------ frames
    def compose_frame(self):
        """Annotated copy of the newest camera frame, or None if there is nothing new."""
        cam = self.camera
        if cam is None:
            return None
        seq, frame = cam.latest()
        if frame is None or seq == self._last_seq:
            return None
        self._last_seq = seq
        s, result = self.settings, self.result
        out = frame.copy()
        detections = [] if self.ai_paused else result.detections
        draw_detections(out, detections, show_labels=s.show_labels, show_scores=s.show_scores,
                        thickness=s.box_thickness, highlight=frozenset(s.watchlist), privacy_blur=s.privacy_blur)
        if s.show_hud:
            lines = [f"CAM {cam.index}  {out.shape[1]}x{out.shape[0]}  {cam.fps:4.1f} fps"]
            if self.detector.ready and not self.ai_paused:
                lines.append(f"AI  {self.ai_fps:4.1f} fps  {self.inference_ms:4.0f} ms")
                lines.append(f"{len(detections)} object{'s' if len(detections) != 1 else ''}")
            warning = ("AI PAUSED" if self.ai_paused else
                       "" if self.detector.ready else "Loading AI model...")
            draw_hud(out, lines, recording_seconds=self.recorder.elapsed if self.recorder else None,
                     warning=warning)
        if self.recorder:
            self.recorder.write(out)
        self.last_frame = out
        return out

    # ------------------------------------------------------------------ actions
    def say(self, text: str, force: bool = False) -> None:
        """Speak and log. `force` speaks even with announcements muted (direct answers)."""
        if self.settings.tts_enabled or force:
            self.speaker.say(text, force=True)
            self.emit("speech", text)

    def apply_speech_settings(self) -> None:
        s = self.settings
        self.speaker.enabled = s.tts_enabled
        self.speaker.rate = s.tts_rate
        self.speaker.volume = s.tts_volume
        self.speaker.voice_id = s.tts_voice

    def set_voice_commands(self, enabled: bool) -> None:
        self.settings.voice_commands = enabled
        if enabled:
            self.voice.start(self.settings.mic_device)
        else:
            self.voice.stop()

    def restart_voice(self) -> None:
        if self.voice.running:
            self.voice.stop()
            self.voice.start(self.settings.mic_device)

    def describe(self) -> str:
        if not self.camera_on:
            return "The camera is off."
        if not self.detector.ready:
            return "I'm still loading my vision model."
        r = self.result
        return scene.describe_scene(r.detections, *r.frame_size)

    def answer(self, action: str, label: str) -> str:
        if not self.camera_on:
            return "The camera is off."
        r = self.result
        if action == "count":
            return scene.count(label, r.detections)
        return scene.locate(label, r.detections, *r.frame_size)

    def snapshot(self):
        if self.last_frame is None:
            raise RuntimeError("No camera image to save")
        path = timestamped(SNAPSHOT_DIR, "snapshot", ".png")
        save_image(path, self.last_frame)
        return path

    def start_recording(self):
        if self.recorder or not self.camera_on:
            return None
        fps = min(60.0, max(5.0, round(self.camera.fps) or 20.0))
        self.recorder = VideoRecorder(timestamped(RECORDING_DIR, "recording", ".mp4"), self.camera.size, fps)
        return self.recorder.path

    def stop_recording(self):
        recorder, self.recorder = self.recorder, None
        return recorder.stop() if recorder else None
