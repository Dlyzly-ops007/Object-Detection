"""Background video recorder and snapshot saving."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def timestamped(folder: Path, prefix: str, ext: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}{ext}"


def save_image(path: Path, frame) -> None:
    # imencode + write_bytes also works for non-ASCII paths, unlike cv2.imwrite on Windows.
    ok, buf = cv2.imencode(path.suffix, frame)
    if not ok:
        raise OSError(f"Could not encode {path.name}")
    path.write_bytes(buf.tobytes())


class VideoRecorder:
    """Writes frames on its own thread so a slow disk never stalls the preview."""

    def __init__(self, path: Path, size: tuple, fps: float):
        self.size = size
        self.started_at = time.time()
        self.frames = 0
        self._queue = queue.Queue(maxsize=64)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            path = path.with_suffix(".avi")
            self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
        if not self._writer.isOpened():
            raise OSError("No working video codec found")
        self.path = path
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def write(self, frame) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            log.debug("Recorder queue full, dropping a frame")

    def _loop(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)
            self.frames += 1
        self._writer.release()

    def stop(self) -> Path:
        self._queue.put(None)
        self._thread.join(timeout=5)
        return self.path
