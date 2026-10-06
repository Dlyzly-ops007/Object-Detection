"""Drawing helpers: per-class colours, bounding boxes, privacy blur and the HUD overlay."""

from __future__ import annotations

import zlib
from functools import lru_cache

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


@lru_cache(maxsize=None)
def class_color(label: str) -> tuple:
    """Stable, bright BGR colour per class so the same object always gets the same colour."""
    hue = zlib.crc32(label.encode()) % 180
    hsv = np.uint8([[[hue, 190, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def class_color_hex(label: str) -> str:
    b, g, r = class_color(label)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, alpha: float) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def blur_region(frame: np.ndarray, box: tuple) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = frame[y1:y2, x1:x2]
    # Pixelate: much stronger than a Gaussian blur and cheaper for large boxes.
    small = cv2.resize(roi, (max(1, (x2 - x1) // 16), max(1, (y2 - y1) // 16)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def draw_detections(frame: np.ndarray, detections, *, show_labels=True, show_scores=True,
                    thickness=2, highlight=frozenset(), privacy_blur=False) -> None:
    """Draw detections in place. Highlighted (watch-list) classes get a heavier frame."""
    h, w = frame.shape[:2]
    scale = max(0.45, min(w, h) / 900)
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = class_color(det.label)
        if privacy_blur and det.label == "person":
            blur_region(frame, det.box)

        t = thickness + (1 if det.label in highlight else 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        # Corner accents make boxes readable on busy backgrounds.
        corner = max(8, min(x2 - x1, y2 - y1) // 6)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, t + 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, t + 2, cv2.LINE_AA)

        if not (show_labels or show_scores):
            continue
        parts = []
        if show_labels:
            parts.append(det.label)
        if show_scores:
            parts.append(f"{det.score:.0%}")
        text = " ".join(parts)
        (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
        ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 5), (x1 + tw + 8, ty + base), color, -1)
        cv2.putText(frame, text, (x1 + 4, ty - 2), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines, *, recording_seconds: float | None = None, warning: str = "") -> None:
    """Semi-transparent stats panel (top-left), REC badge (top-right) and warning banner (bottom)."""
    h, w = frame.shape[:2]
    scale = max(0.42, min(w, h) / 1000)
    pad, line_h = 8, int(26 * scale / 0.5)

    if lines:
        width = max(cv2.getTextSize(line, FONT, scale, 1)[0][0] for line in lines) + 2 * pad
        _blend_rect(frame, 8, 8, 8 + width, 8 + pad + line_h * len(lines), (15, 15, 15), 0.55)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8 + pad, 8 + (i + 1) * line_h - 4), FONT, scale,
                        (235, 235, 235), 1, cv2.LINE_AA)

    if recording_seconds is not None:
        m, s = divmod(int(recording_seconds), 60)
        text = f"REC {m:02d}:{s:02d}"
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
        x = w - tw - 40
        _blend_rect(frame, x - 8, 8, w - 8, 16 + th + 8, (15, 15, 15), 0.55)
        if int(recording_seconds * 2) % 2 == 0:  # blink
            cv2.circle(frame, (x + 4, 12 + th // 2 + 4), max(4, th // 3), (40, 40, 235), -1, cv2.LINE_AA)
        cv2.putText(frame, text, (x + 16, 12 + th + 2), FONT, scale, (235, 235, 235), 1, cv2.LINE_AA)

    if warning:
        (tw, th), _ = cv2.getTextSize(warning, FONT, scale * 1.1, 2)
        x, y = (w - tw) // 2, h - 20
        _blend_rect(frame, x - 12, y - th - 12, x + tw + 12, y + 10, (20, 20, 120), 0.7)
        cv2.putText(frame, warning, (x, y), FONT, scale * 1.1, (255, 255, 255), 2, cv2.LINE_AA)
