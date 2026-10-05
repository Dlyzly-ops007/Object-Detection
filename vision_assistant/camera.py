"""Threaded webcam capture that always hands out the newest frame."""

from __future__ import annotations

import logging
import sys
import threading
import time

import cv2

from .config import AUTO_CAMERA

log = logging.getLogger(__name__)

# DirectShow opens webcams far faster than MSMF on Windows.
_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

try:  # silence "can't capture by index" spam while probing
    cv2.setLogLevel(2)
except Exception:
    pass


# DirectShow misbehaves (duplicate frames, native crashes) when two threads open/release devices at once.
_DEVICE_LOCK = threading.Lock()


class CameraError(RuntimeError):
    pass


def _open(index: int, size: tuple | None = None):
    cap = cv2.VideoCapture(index, _BACKEND)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    if size:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def probe_cameras(max_index: int = 4, skip: tuple = ()) -> list:
    """Indices of cameras that open and deliver a frame. Indices in `skip` (already in use) count as present."""
    found = []
    for i in range(max_index):
        if i in skip:
            found.append(i)
            continue
        with _DEVICE_LOCK:
            try:
                cap = _open(i)
                if cap is not None:
                    ok, _ = cap.read()
                    cap.release()
                    if ok:
                        found.append(i)
            except cv2.error as exc:
                log.debug("Probing camera %d failed: %s", i, exc)
    return found


class CameraStream:
    def __init__(self, index: int = AUTO_CAMERA, size: tuple = (640, 480), mirror: bool = False):
        self.requested_index = index
        self.index = None
        self.size = size
        self.mirror = mirror
        self.fps = 0.0
        self._cap = None
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = threading.Event()
        self._thread = None

    def start(self) -> "CameraStream":
        """Open the device (blocking, can take a second or two) and start the capture thread."""
        # Auto mode mirrors the original behaviour: prefer an external webcam, fall back to the laptop one.
        candidates = [1, 0] if self.requested_index == AUTO_CAMERA else [self.requested_index]
        for idx in candidates:
            with _DEVICE_LOCK:
                cap = _open(idx, self.size)
                if cap is None:
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._cap, self.index = cap, idx
                    self.size = (frame.shape[1], frame.shape[0])
                    break
                cap.release()
        if self._cap is None:
            raise CameraError("No working camera found" if self.requested_index == AUTO_CAMERA
                              else f"Camera {self.requested_index} could not be opened")

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        failures, t_last = 0, time.perf_counter()
        while self._running.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 50:  # ~device unplugged
                    log.error("Camera %s stopped delivering frames", self.index)
                    break
                time.sleep(0.02)
                continue
            failures = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            dt, t_last = now - t_last, now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()
        self._running.clear()
        with self._cond:
            self._cond.notify_all()

    @property
    def alive(self) -> bool:
        return self._running.is_set()

    def latest(self):
        """(seq, frame) of the newest frame without waiting. frame is shared: copy before drawing."""
        with self._cond:
            return self._seq, self._frame

    def wait_for_frame(self, after_seq: int, timeout: float = 0.5):
        """Block until a frame newer than `after_seq` arrives. Returns (seq, frame) or (after_seq, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or not self._running.is_set(), timeout)
            if self._seq > after_seq:
                return self._seq, self._frame
            return after_seq, None

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            with _DEVICE_LOCK:
                self._cap.release()
            self._cap = None
