"""Non-blocking text-to-speech built on pyttsx3."""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time

import pyttsx3

log = logging.getLogger(__name__)


class Speaker:
    """Speaks on a background thread so the video never freezes while talking.

    pyttsx3 2.99 on Windows silently skips every utterance after the first when an
    engine is reused, so a fresh `pyttsx3.Engine` is built for each utterance.
    """

    MAX_PENDING = 2  # older announcements are dropped: stale news is worse than none

    def __init__(self):
        self.enabled = True
        self.rate = 170
        self.volume = 1.0
        self.voice_id = ""
        self.voices = []  # [(id, name)], filled once the worker starts
        self.speaking = threading.Event()
        self.last_finished = 0.0
        self._queue = queue.Queue()
        self._interrupt = threading.Event()
        self._voices_ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="speaker", daemon=True)
        self._thread.start()

    def say(self, text: str, *, urgent: bool = False, force: bool = False) -> None:
        """Queue `text`. `urgent` drops anything pending; `force` speaks even when muted."""
        if not text or not (self.enabled or force):
            return
        if urgent:
            self._drain()
        while self._queue.qsize() >= self.MAX_PENDING:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def interrupt(self) -> None:
        """Stop the current sentence and forget anything queued."""
        self._drain()
        if self.speaking.is_set():
            self._interrupt.set()

    def recently_spoke(self, within: float = 0.6) -> bool:
        return self.speaking.is_set() or time.time() - self.last_finished < within

    def wait_for_voices(self, timeout: float = 5.0) -> list:
        self._voices_ready.wait(timeout)
        return self.voices

    def shutdown(self) -> None:
        self.interrupt()
        self._queue.put(None)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _loop(self) -> None:
        try:  # SAPI is COM-based; a non-main thread must initialise COM itself.
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass

        try:
            self._on_this_thread(self._list_voices)
        except Exception as exc:
            log.error("Text-to-speech unavailable: %s", exc)
        self._voices_ready.set()

        while True:
            text = self._queue.get()
            if text is None:
                break
            self._interrupt.clear()
            self.speaking.set()
            try:
                self._on_this_thread(self._speak, text)
            except Exception as exc:
                log.error("Speech failed: %s", exc)
            finally:
                self.speaking.clear()
                self.last_finished = time.time()

    @staticmethod
    def _on_this_thread(fn, *args) -> None:
        """Run `fn` (which returns the engine it built), then destroy that engine on this thread.

        An unreferenced engine is cyclic garbage. If Python's garbage collector happens to free
        it on another thread, COM blocks that thread until this one pumps messages again: that
        froze the model loader for 30+ seconds. So the engine is kept alive until automatic GC
        is paused, then released and collected right here.
        """
        engine = error = None
        try:
            engine = fn(*args)
        except Exception as exc:
            error = exc
        gc.disable()
        try:
            if error is not None:
                error = error.with_traceback(None)  # the traceback's frames still hold the engine
            del engine
            gc.collect()
        finally:
            gc.enable()
        if error is not None:
            raise error

    def _list_voices(self):
        engine = pyttsx3.Engine()
        self.voices = [(v.id, v.name) for v in engine.getProperty("voices")]
        return engine

    def _speak(self, text: str):
        engine = pyttsx3.Engine()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        if self.voice_id:
            engine.setProperty("voice", self.voice_id)

        def on_word(name, location, length):
            if self._interrupt.is_set():
                engine.stop()

        engine.connect("started-word", on_word)
        engine.say(text)
        engine.runAndWait()
        return engine
