"""Main window."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

from . import theme
from .config import AUTO_CAMERA, RECORDING_DIR, SNAPSHOT_DIR, Settings
from .controller import Controller
from .panels import build_camera_tab, build_detection_tab, build_voice_tab, show_help_window
from .render import class_color_hex
from .voice import list_input_devices
from .widgets import LevelMeter, LogView, StatusChip, VideoCanvas

log = logging.getLogger(__name__)

class App:
    TICK_MS = 15

    def __init__(self, root: tk.Tk, settings: Settings):
        self.root = root
        self.settings = settings
        self.events = queue.Queue()
        self.controller = Controller(settings, lambda *event: self.events.put(event))
        self.vars = {}
        self._last_panel_update = 0.0
        self._shown_rows = None
        self._fullscreen = False

        theme.apply_theme(root)
        self.scale = theme.ui_scale(root)
        root.title("Vision Assistant")
        w, h = min(self.px(1360), int(root.winfo_screenwidth() * 0.92)),             min(self.px(860), int(root.winfo_screenheight() * 0.88))
        root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{max(0, (root.winfo_screenheight() - h) // 3)}")
        root.minsize(min(w, self.px(1100)), min(h, self.px(700)))
        self._build()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._startup()
        self._tick()

    def px(self, n: float) -> int:
        return int(n * getattr(self, "scale", 1.0))

    # ================================================================== layout
    def _build(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, padding=(18, 12, 18, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Vision Assistant", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Real-time object detection · offline voice control",
                  style="Subtitle.TLabel").pack(anchor="w")
        self.chips = {}
        for key in ("rec", "mic", "camera", "model"):
            chip = StatusChip(header, {"rec": "Recording"}.get(key, key))
            chip.pack(side="right", padx=(8, 0))
            self.chips[key] = chip
        self.chips["rec"].set("off", "Off")

        body = ttk.Frame(root, padding=(18, 0, 18, 16))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_main_column(body)
        self._build_sidebar(body)

    def _build_main_column(self, body) -> None:
        col = ttk.Frame(body)
        col.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        col.columnconfigure(0, weight=1)
        col.rowconfigure(0, weight=1)

        self.video = VideoCanvas(col)
        self.video.grid(row=0, column=0, sticky="nsew")

        bar = ttk.Frame(col, padding=(0, 10))
        bar.grid(row=1, column=0, sticky="ew")
        self.btn_camera = ttk.Button(bar, text="▶  Start camera", style="Accent.TButton", width=16,
                                     command=self.toggle_camera)
        self.btn_pause = ttk.Button(bar, text="❚❚  Pause AI", command=self.toggle_pause, width=13)
        self.btn_snap = ttk.Button(bar, text="◉  Snapshot", command=self.snapshot)
        self.btn_record = ttk.Button(bar, text="●  Record", command=self.toggle_recording, width=15)
        self.btn_describe = ttk.Button(bar, text="»  Describe", command=self.describe)
        self.btn_mute = ttk.Button(bar, text="Mute", command=self.toggle_mute, width=9)
        for btn in (self.btn_camera, self.btn_pause, self.btn_snap, self.btn_record, self.btn_describe,
                    self.btn_mute):
            btn.pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="?", width=3, command=self.show_help).pack(side="right")

        self.log_view = LogView(col)
        self.log_view.grid(row=2, column=0, sticky="ew")

    def _build_sidebar(self, body) -> None:
        side = ttk.Frame(body, width=self.px(370))
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)

        notebook = ttk.Notebook(side)
        notebook.grid(row=0, column=0, sticky="ew")
        for title, builder in (("Detection", build_detection_tab), ("Voice", build_voice_tab),
                               ("Camera", build_camera_tab)):
            tab = ttk.Frame(notebook, style="Card.TFrame", padding=14)
            builder(self, tab)
            notebook.add(tab, text=title)

        # ---- "In view" table
        view = ttk.Frame(side, style="Card.TFrame", padding=(14, 12))
        view.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        view.columnconfigure(0, weight=1)
        view.rowconfigure(1, weight=1)
        ttk.Label(view, text="IN VIEW", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.session_label = ttk.Label(view, text="", style="Muted.TLabel")
        self.session_label.grid(row=0, column=1, sticky="e")
        self.tree = ttk.Treeview(view, columns=("object", "count", "conf"), show="headings", height=5,
                                 selectmode="none")
        for col, text, width, anchor in (("object", "Object", 170, "w"), ("count", "Count", 60, "center"),
                                         ("conf", "Confidence", 90, "e")):
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=self.px(width), anchor=anchor, stretch=col == "object")
        self.tree.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        # ---- microphone card
        mic = ttk.Frame(side, style="Card.TFrame", padding=(14, 12))
        mic.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        mic.columnconfigure(0, weight=1)
        ttk.Label(mic, text="HEARD", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.meter = LevelMeter(mic, width=self.px(120), height=self.px(10))
        self.meter.grid(row=0, column=1, sticky="e")
        self.heard_label = ttk.Label(mic, text="Say “what do you see?”", style="Muted.TLabel",
                                     wraplength=self.px(330))
        self.heard_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------ small builders used by panels
    def check(self, parent, text: str, attr: str, on_change=None) -> ttk.Checkbutton:
        var = tk.BooleanVar(value=getattr(self.settings, attr))
        self.vars[attr] = var

        def changed():
            setattr(self.settings, attr, var.get())
            if on_change:
                on_change(var.get())
        box = ttk.Checkbutton(parent, text=text, variable=var, command=changed, style="Card.TCheckbutton")
        box.pack(anchor="w", fill="x")
        return box

    def set_var(self, attr: str, value) -> None:
        setattr(self.settings, attr, value)
        if attr in self.vars:
            self.vars[attr].set(value)

    # ================================================================== startup / loop
    def _startup(self) -> None:
        c, s = self.controller, self.settings
        self.log("Welcome! Press Space or say “start” to begin. Click ? for all commands.", "muted")
        self.video.show_placeholder("Camera is off", "Press Space, click Start camera, or say “start”")
        c.load_model_async()
        if not s.auto_start_camera:  # otherwise probed once the camera is open
            c.probe_cameras_async()
        threading.Thread(target=lambda: self.events.put(("voices", c.speaker.wait_for_voices())),
                         daemon=True).start()
        self._populate_mics()
        if s.voice_commands:
            c.voice.start(s.mic_device)
        else:
            self.chips["mic"].set("off", "Off")
        if s.auto_start_camera:
            self.start_camera()
        self._refresh_buttons()

    def _tick(self) -> None:
        try:
            self._drain_events()
            frame = self.controller.compose_frame()
            if frame is not None:
                self.video.show_frame(frame)
            now = time.monotonic()
            if now - self._last_panel_update > 0.25:
                self._last_panel_update = now
                self._update_panels()
        except Exception:
            log.exception("UI update failed")
        self.root.after(self.TICK_MS, self._tick)

    def _update_panels(self) -> None:
        c = self.controller
        result = c.result
        best = {}
        for det in result.detections:
            best[det.label] = max(best.get(det.label, 0.0), det.score)
        rows = sorted(((label, n, best.get(label)) for label, n in result.visible.items()),
                      key=lambda r: (-r[1], r[0]))
        if rows != self._shown_rows:
            self._shown_rows = rows
            self.tree.delete(*self.tree.get_children())
            for label, n, conf in rows:
                tag = f"c_{label}"
                self.tree.tag_configure(tag, foreground=class_color_hex(label))
                self.tree.insert("", "end", values=(f"■  {label}", n, f"{conf:.0%}" if conf else "—"),
                                 tags=(tag,))
        seen = c.tracker.session_counts
        self.session_label.configure(text=f"{sum(seen.values())} sightings · {len(seen)} kinds" if seen else "")
        if c.recorder:
            m, sec = divmod(int(c.recorder.elapsed), 60)
            self.btn_record.configure(text=f"■  Stop  {m:02d}:{sec:02d}")
            self.chips["rec"].set("rec", f"{m:02d}:{sec:02d}")

    # ================================================================== events from worker threads
    def _drain_events(self) -> None:
        for _ in range(200):
            try:
                kind, *args = self.events.get_nowait()
            except queue.Empty:
                return
            handler = getattr(self, f"_on_{kind}", None)
            if handler:
                handler(*args)

    def _on_model(self, state: str, text: str) -> None:
        self.chips["model"].set(state, text)
        if state == "ok":
            self.log(f"AI model {text.lower()}.", "muted")
        elif state == "error":
            self.log(f"AI model {text}", "error")

    def _on_camera(self, state: str, text: str) -> None:
        self.chips["camera"].set(state, text if state != "error" else "Error")
        if state == "busy":
            self.video.show_placeholder("Opening camera…", "This can take a moment")
        elif state == "error":
            self.controller.stop_camera(quiet=True)
            self.video.show_placeholder("Camera unavailable", text)
            self.log(text, "error")
        elif state == "off":
            self.video.show_placeholder("Camera is off", "Press Space, click Start camera, or say “start”")
            self.log("Camera stopped.", "muted")
        elif state == "ok":
            self.log(f"{text} started.", "muted")
            self.controller.probe_cameras_async()
        self._refresh_buttons()

    def _on_cameras(self, indices: list) -> None:
        self.camera_choices = [("Auto (external first)", AUTO_CAMERA)] + [(f"Camera {i}", i) for i in indices]
        self.camera_combo.configure(values=[name for name, _ in self.camera_choices])
        current = next((n for n, i in self.camera_choices if i == self.settings.camera_index), None)
        self.camera_combo.set(current or f"Camera {self.settings.camera_index}")

    def _on_voices(self, voices: list) -> None:
        self.voice_choices = voices
        self.voice_combo.configure(values=[name for _, name in voices])
        name = next((n for vid, n in voices if vid == self.settings.tts_voice), voices[0][1] if voices else "")
        self.voice_combo.set(name)

    def _on_mic(self, state: str, message: str) -> None:
        mapping = {"loading": "busy", "listening": "ok", "error": "error", "off": "off"}
        self.chips["mic"].set(mapping.get(state, "off"), {"error": "Error"}.get(state, message))
        if state == "error":
            self.log(f"Voice commands unavailable: {message}", "error")
        elif state == "listening":
            self.log("Voice commands ready.", "muted")

    def _on_heard(self, text: str, final: bool) -> None:
        self.heard_label.configure(text=f"“{text}”" if final else f"{text}…",
                                   style="Heard.TLabel" if final else "Muted.TLabel")

    def _on_level(self, level: float) -> None:
        self.meter.set(level)

    def _on_speech(self, text: str) -> None:
        self.log(f"» {text}", "speech")

    def _on_log(self, text: str, tag: str = "info") -> None:
        self.log(text, tag)

    def _on_scene(self, appeared: dict, departed: list) -> None:
        parts = [f"+{n} {label}" for label, n in appeared.items()] + [f"−{label}" for label in departed]
        if parts:
            self.log("  ".join(parts), "muted")

    def _on_alert(self, labels: list) -> None:
        self.video.flash(theme.ORANGE)
        self.log("Watch list: " + ", ".join(labels), "alert")

    def _on_command(self, cmd) -> None:
        self.log(f"Voice command: {cmd.action.replace('_', ' ')}" + (f" ({cmd.label})" if cmd.label else ""),
                 "command")
        c = self.controller
        actions = {
            "camera_start": self.start_camera, "camera_stop": self.stop_camera, "describe": self.describe,
            "snapshot": self.snapshot, "record_start": lambda: self.toggle_recording(True),
            "record_stop": lambda: self.toggle_recording(False), "pause": lambda: self.toggle_pause(True),
            "resume": lambda: self.toggle_pause(False), "mute": lambda: self.toggle_mute(True),
            "unmute": lambda: self.toggle_mute(False),
            "help": lambda: c.say("You can say: what do you see, where is my phone, how many people, "
                                  "take a picture, start recording, mute, or stop.", force=True),
            "exit": lambda: (c.say("Goodbye", force=True), self.root.after(1500, self.quit)),
        }
        if cmd.action in ("locate", "count"):
            c.say(c.answer(cmd.action, cmd.label), force=True)
        elif cmd.action in actions:
            actions[cmd.action]()

    # ================================================================== actions
    def log(self, message: str, tag: str = "info") -> None:
        self.log_view.append(message, tag)

    def start_camera(self) -> None:
        self.controller.start_camera()
        self._refresh_buttons()

    def stop_camera(self) -> None:
        if self.controller.recorder:
            self.toggle_recording(False)
        self.controller.stop_camera()
        self._refresh_buttons()

    def toggle_camera(self) -> None:
        self.stop_camera() if self.controller.camera_on else self.start_camera()

    def toggle_pause(self, paused: bool | None = None) -> None:
        paused = not self.controller.ai_paused if paused is None else paused
        self.controller.set_ai_paused(paused)
        self.log("Detection paused." if paused else "Detection resumed.", "muted")
        self._refresh_buttons()

    def toggle_mute(self, muted: bool | None = None) -> None:
        muted = self.settings.tts_enabled if muted is None else muted
        self.set_var("tts_enabled", not muted)
        self.controller.apply_speech_settings()
        if muted:
            self.controller.speaker.interrupt()
            self.log("Announcements muted.", "muted")
        else:
            self.controller.say("Announcements on")
        self._refresh_buttons()

    def describe(self) -> None:
        self.controller.say(self.controller.describe(), force=True)

    def snapshot(self) -> None:
        try:
            path = self.controller.snapshot()
        except Exception as exc:
            self.log(f"Snapshot failed: {exc}", "error")
            return
        self.video.flash("white", 250)
        self.log(f"Snapshot saved: {path.name}", "info")
        self.controller.say("Snapshot saved")

    def toggle_recording(self, start: bool | None = None) -> None:
        c = self.controller
        start = c.recorder is None if start is None else start
        if start and c.recorder is None:
            if not c.camera_on:
                self.log("Start the camera before recording.", "error")
                return
            try:
                path = c.start_recording()
            except Exception as exc:
                self.log(f"Recording failed: {exc}", "error")
                return
            self.log(f"Recording to {path.name}", "info")
            c.say("Recording started")
        elif not start and c.recorder is not None:
            path = c.stop_recording()
            self.log(f"Recording saved: {path.name}", "info")
            c.say("Recording saved")
            self.chips["rec"].set("off", "Off")
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        c = self.controller
        on = c.camera_on
        if c._camera_busy:
            self.btn_camera.configure(text="…  Opening", style="TButton", state="disabled")
        elif on:
            self.btn_camera.configure(text="■  Stop camera", style="Danger.TButton", state="normal")
        else:
            self.btn_camera.configure(text="▶  Start camera", style="Accent.TButton", state="normal")
        for btn in (self.btn_snap, self.btn_record, self.btn_describe, self.btn_pause):
            btn.configure(state="normal" if on else "disabled")
        self.btn_pause.configure(text="▶  Resume AI" if c.ai_paused else "❚❚  Pause AI")
        if not c.recorder:
            self.btn_record.configure(text="●  Record")
        self.btn_mute.configure(text="Unmute" if not self.settings.tts_enabled else "Mute")

    # ================================================================== settings callbacks (used by panels)
    def on_camera_selected(self, _event=None) -> None:
        name = self.camera_combo.get()
        self.settings.camera_index = dict(self.camera_choices).get(name, AUTO_CAMERA)
        self.controller.restart_camera()
        self._refresh_buttons()

    def on_resolution_selected(self, _event=None) -> None:
        self.settings.resolution = self.resolution_combo.get()
        self.controller.restart_camera()
        self._refresh_buttons()

    def on_voice_selected(self, _event=None) -> None:
        name = self.voice_combo.get()
        self.settings.tts_voice = next((vid for vid, n in self.voice_choices if n == name), "")
        self.controller.apply_speech_settings()

    def _populate_mics(self) -> None:
        names = ["System default"] + [name for _, name in list_input_devices()]
        self.mic_combo.configure(values=names)
        self.mic_combo.set(self.settings.mic_device if self.settings.mic_device in names else names[0])

    def on_mic_selected(self, _event=None) -> None:
        name = self.mic_combo.get()
        self.settings.mic_device = "" if name == "System default" else name
        self.controller.restart_voice()

    def on_watchlist_changed(self, _event=None) -> None:
        self.settings.watchlist = [self.watch_listbox.get(i) for i in self.watch_listbox.curselection()]

    def test_voice(self) -> None:
        self.controller.say("Hello! This is how I will sound.", force=True)

    @staticmethod
    def open_folder(path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def open_snapshots(self) -> None:
        self.open_folder(SNAPSHOT_DIR)

    def open_recordings(self) -> None:
        self.open_folder(RECORDING_DIR)

    # ================================================================== keys, help, quit
    def _bind_keys(self) -> None:
        keys = {"<space>": self.toggle_camera, "s": self.snapshot, "r": self.toggle_recording,
                "d": self.describe, "p": self.toggle_pause, "m": self.toggle_mute,
                "<F11>": self.toggle_fullscreen, "<Escape>": lambda: self.toggle_fullscreen(False),
                "<Control-q>": self.quit}
        for key, fn in keys.items():
            self.root.bind(key, lambda e, fn=fn: self._shortcut(e, fn))

    def _shortcut(self, event, fn):
        cls = event.widget.winfo_class()
        if cls in ("TCombobox", "TSpinbox", "TEntry", "Entry"):
            return None
        if event.keysym == "space" and cls in ("TButton", "TCheckbutton", "Listbox"):
            return None  # let the focused widget handle its own Space
        if fn in (self.snapshot, self.toggle_recording, self.describe, self.toggle_pause) \
                and not self.controller.camera_on:
            return "break"
        fn()
        return "break"

    def toggle_fullscreen(self, value: bool | None = None) -> None:
        self._fullscreen = not self._fullscreen if value is None else value
        self.root.attributes("-fullscreen", self._fullscreen)

    def show_help(self) -> None:
        show_help_window(self.root)

    def quit(self) -> None:
        self.settings.save()
        try:
            self.controller.shutdown()
        finally:
            self.root.destroy()


def main(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = tk.Tk()
    App(root, settings or Settings.load())
    root.mainloop()
