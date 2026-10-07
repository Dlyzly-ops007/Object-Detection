"""Turns noisy per-frame detections into stable scene events and natural sentences."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .config import pluralize, quantity, with_article


@dataclass
class _Track:
    history: deque = field(default_factory=lambda: deque(maxlen=6))  # recent per-frame counts
    streak: int = 0
    last_seen: float = 0.0
    stable: bool = False
    count: int = 0  # smoothed count while stable
    peak: int = 0  # highest count already reported; only exceeding it is news
    peak_seen: float = 0.0


@dataclass
class SceneUpdate:
    appeared: dict = field(default_factory=dict)   # label -> count (new, or count went up)
    departed: list = field(default_factory=list)   # labels that left the scene

    def __bool__(self) -> bool:
        return bool(self.appeared or self.departed)


class SceneTracker:
    """An object 'appears' after `appear_frames` consecutive sightings and 'departs' after
    being unseen for `vanish_seconds`. This kills the flicker that made the old version
    miss or repeat objects, and lets an object be announced again once it comes back."""

    def __init__(self, appear_frames: int = 5, vanish_seconds: float = 4.0):
        self.appear_frames = appear_frames
        self.vanish_seconds = vanish_seconds
        self._tracks = {}
        self.session_counts = Counter()  # how many times each class entered the scene

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def visible(self) -> dict:
        return {label: t.count for label, t in self._tracks.items() if t.stable}

    def update(self, detections, now: float) -> SceneUpdate:
        counts = Counter(d.label for d in detections)
        update = SceneUpdate()

        for label, n in counts.items():
            track = self._tracks.setdefault(label, _Track())
            track.history.append(n)
            track.streak += 1
            track.last_seen = now
            # Median of recent frames: one noisy frame should not change "2 people" to "3".
            smoothed = max(1, sorted(track.history)[len(track.history) // 2])
            if not track.stable:
                if track.streak < self.appear_frames:
                    continue
                track.stable = True
                self.session_counts[label] += 1
            track.count = smoothed
            # A box that flickers makes the count bounce 1-2-1-2; without the peak memory every
            # bounce would be announced as a new arrival.
            if smoothed > track.peak:
                update.appeared[label] = smoothed
            if smoothed >= track.peak or now - track.peak_seen > self.vanish_seconds:
                track.peak, track.peak_seen = smoothed, now

        for label in list(self._tracks):
            track = self._tracks[label]
            if label in counts:
                continue
            track.streak = 0
            track.history.append(0)
            if now - track.last_seen > self.vanish_seconds:
                if track.stable:
                    update.departed.append(label)
                del self._tracks[label]
        return update


def join_words(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def summarize(counts: dict, order: list | None = None, limit: int = 4) -> str:
    """{'person': 2, 'laptop': 1} -> '2 people and a laptop'."""
    labels = order if order is not None else sorted(counts, key=lambda k: -counts[k])
    parts = [quantity(label, counts[label]) for label in labels[:limit]]
    extra = len(labels) - limit
    if extra > 0:
        parts.append(f"{extra} more {'thing' if extra == 1 else 'things'}")
    return join_words(parts)


def position_of(det, width: int, height: int) -> str:
    cx, _ = det.center
    if cx < width / 3:
        where = "on the left"
    elif cx > 2 * width / 3:
        where = "on the right"
    else:
        where = "in the center"
    if det.area > 0.25 * width * height:
        where = "close up " + where
    return where


def describe_scene(detections, width: int, height: int, limit: int = 5) -> str:
    if not detections:
        return "I don't see anything I recognise right now."
    groups = {}
    for det in sorted(detections, key=lambda d: -d.score):
        groups.setdefault((det.label, position_of(det, width, height)), []).append(det)
    phrases = [f"{quantity(label, len(dets))} {where}" for (label, where), dets in list(groups.items())[:limit]]
    if len(groups) > limit:
        phrases.append("a few other things")
    return "I see " + join_words(phrases) + "."


def locate(label: str, detections, width: int, height: int) -> str:
    matches = [d for d in detections if d.label == label]
    if not matches:
        return f"I don't see {with_article(label) if label != 'person' else 'anyone'} right now."
    if len(matches) == 1:
        return f"Yes, there is {with_article(label)} {position_of(matches[0], width, height)}."
    places = Counter(position_of(d, width, height) for d in matches)
    return f"I see {len(matches)} {pluralize(label)}: " + join_words(
        [f"{n} {where}" for where, n in places.most_common()]) + "."


def count(label: str, detections) -> str:
    n = sum(1 for d in detections if d.label == label)
    if n == 0:
        return f"I don't see any {pluralize(label)}."
    return f"I count {quantity(label, n)}." if n > 1 else f"I see just one {label}."
