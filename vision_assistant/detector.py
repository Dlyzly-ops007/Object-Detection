"""TF-Hub SSD MobileNet v2 object detector."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import COCO_LABELS, MODEL_URL, TFHUB_CACHE_DIR

log = logging.getLogger(__name__)
NMS_IOU = 0.45  # boxes of the same class overlapping more than this are the same object


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple  # (x1, y1, x2, y2) in pixels

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class ObjectDetector:
    """Loads the model lazily (TensorFlow import alone takes seconds) and runs inference."""

    def __init__(self, model_url: str = MODEL_URL):
        self.model_url = model_url
        self._model = None
        self._tf = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> float:
        """Import TensorFlow, load (or download once) the model and warm it up. Returns seconds taken."""
        start = time.perf_counter()
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE_DIR))
        TFHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        import tensorflow as tf
        import tensorflow_hub as hub

        tf.get_logger().setLevel(logging.ERROR)
        model = hub.load(self.model_url)
        # The first call builds the graph and takes ~2s; do it now rather than on the first frame.
        model(tf.zeros((1, 320, 320, 3), dtype=tf.uint8))
        self._tf, self._model = tf, model
        return time.perf_counter() - start

    def detect(self, frame_bgr: np.ndarray, threshold: float = 0.5, max_detections: int = 20,
               allowed: frozenset | None = None) -> list:
        if self._model is None:
            return []
        tf = self._tf
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = self._model(tf.convert_to_tensor(rgb[np.newaxis, ...], dtype=tf.uint8))

        boxes = outputs["detection_boxes"][0].numpy()
        classes = outputs["detection_classes"][0].numpy().astype(int)
        scores = outputs["detection_scores"][0].numpy()

        h, w = frame_bgr.shape[:2]
        candidates = {}
        # Outputs are already sorted by score, so we can stop at the first one under the threshold.
        for (y1, x1, y2, x2), cls, score in zip(boxes, classes, scores):
            if score < threshold:
                break
            label = COCO_LABELS.get(cls)
            if label is None or (allowed is not None and label not in allowed):
                continue
            box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            candidates.setdefault(label, []).append(Detection(label, float(score), box))

        detections = []
        for group in candidates.values():
            detections.extend(self._suppress_duplicates(group))
        detections.sort(key=lambda d: -d.score)
        return detections[:max_detections]

    @staticmethod
    def _suppress_duplicates(group: list) -> list:
        """The model's built-in NMS is loose (IoU 0.6) and often leaves two boxes on one object,
        which makes counts wrong ("3 people" for 2). A stricter per-class pass fixes that."""
        if len(group) < 2:
            return group
        rects = [[d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]] for d in group]
        keep = cv2.dnn.NMSBoxes(rects, [d.score for d in group], 0.0, NMS_IOU)
        return [group[i] for i in np.array(keep).flatten()]
