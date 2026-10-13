"""Sidebar settings tabs. Each builder receives the App and the tab frame to fill."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from . import theme
from .config import ALL_LABELS, AUTO_CAMERA, RESOLUTIONS
from .widgets import LabeledScale

SHORTCUTS = (
    ("Space", "Start / stop camera"), ("S", "Take snapshot"), ("R", "Start / stop recording"),
    ("D", "Describe the scene"), ("P", "Pause / resume detection"), ("M", "Mute / unmute"),
    ("F11", "Toggle full screen"), ("Ctrl+Q", "Quit"),
)
VOICE_HELP = (
    ("“start” / “stop”", "Camera on / off"), ("“what do you see”", "Describe the scene"),
    ("“where is my phone”", "Find an object"), ("“how many people”", "Count objects"),
    ("“take a picture”", "Snapshot"), ("“start / stop recording”", "Record video"),
    ("“pause” / “resume”", "Detection on / off"), ("“mute” / “unmute”", "Announcements"),
    ("“goodbye”", "Close the app"),
)


def _section(parent, text: str, hint: str = "") -> None:
    ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(12, 2))
    if hint:
        wrap = int(320 * theme.ui_scale(parent))
        ttk.Label(parent, text=hint, style="Muted.TLabel", wraplength=wrap).pack(anchor="w", pady=(0, 4))


def _scale(parent, text, from_, to, attr, app, *, step=1, fmt=str, scale=1, on_change=None) -> None:
    """Slider bound to a settings attribute. `scale` maps slider units to setting units (e.g. % -> 0..1)."""
    def changed(value):
        setattr(app.settings, attr, value / scale if scale != 1 else value)
        if on_change:
            on_change()
    LabeledScale(parent, text, from_, to, getattr(app.settings, attr) * scale, step=step, fmt=fmt,
                 on_change=changed).pack(fill="x", pady=(4, 6))


def _combo(parent, width=34) -> ttk.Combobox:
    combo = ttk.Combobox(parent, state="readonly", width=width)
    combo.pack(anchor="w", fill="x", pady=(2, 6))
    return combo


def build_detection_tab(app, tab) -> None:
    _scale(tab, "Confidence threshold", 20, 95, "confidence", app, step=5, scale=100,
           fmt=lambda v: f"{v}%")
    _scale(tab, "Max objects per frame", 1, 50, "max_detections", app)

    _section(tab, "WATCH LIST", "Highlighted on screen, announced first, and can trigger alerts.")
    box = ttk.Frame(tab, style="Card.TFrame")
    box.pack(fill="x")
    lb = tk.Listbox(box, selectmode="multiple", height=7, exportselection=False, activestyle="none",
                    bg=theme.SURFACE_2, fg=theme.TEXT, selectbackground=theme.ACCENT, selectforeground="white",
                    highlightthickness=0, bd=0, font=("TkDefaultFont", 10))
    bar = ttk.Scrollbar(box, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=bar.set)
    lb.pack(side="left", fill="x", expand=True)
    bar.pack(side="right", fill="y")
    for i, label in enumerate(ALL_LABELS):
        lb.insert("end", label)
        if label in app.settings.watchlist:
            lb.selection_set(i)
    lb.bind("<<ListboxSelect>>", app.on_watchlist_changed)
    app.watch_listbox = lb

    ttk.Frame(tab, style="Card.TFrame", height=8).pack()
    app.check(tab, "Only show watch-list objects", "show_only_watchlist")
    app.check(tab, "Only announce watch-list objects", "announce_only_watchlist")
    app.check(tab, "Flash when a watch-list object appears", "alert_on_watchlist")


def build_voice_tab(app, tab) -> None:
    speech = app.controller.apply_speech_settings

    _section(tab, "SPOKEN FEEDBACK")
    app.check(tab, "Announce new objects", "tts_enabled", lambda on: app.toggle_mute(not on))
    app.check(tab, "Announce when objects leave", "announce_departures")
    _scale(tab, "Minimum gap between announcements", 1, 15, "announce_cooldown", app, step=0.5,
           fmt=lambda v: f"{v:g} s")

    ttk.Label(tab, text="Voice", style="Card.TLabel").pack(anchor="w", pady=(4, 0))
    app.voice_choices = []
    app.voice_combo = _combo(tab)
    app.voice_combo.bind("<<ComboboxSelected>>", app.on_voice_selected)
    _scale(tab, "Speaking rate", 100, 260, "tts_rate", app, step=5, fmt=lambda v: f"{v} wpm", on_change=speech)
    _scale(tab, "Volume", 0, 100, "tts_volume", app, step=5, scale=100, fmt=lambda v: f"{v}%",
           on_change=speech)
    ttk.Button(tab, text="Test voice", style="Small.TButton", command=app.test_voice).pack(anchor="w")

    _section(tab, "VOICE COMMANDS")
    app.check(tab, "Listen for voice commands", "voice_commands", app.controller.set_voice_commands)
    ttk.Label(tab, text="Microphone", style="Card.TLabel").pack(anchor="w", pady=(4, 0))
    app.mic_combo = _combo(tab)
    app.mic_combo.bind("<<ComboboxSelected>>", app.on_mic_selected)


def build_camera_tab(app, tab) -> None:
    ttk.Label(tab, text="Camera", style="Card.TLabel").pack(anchor="w")
    row = ttk.Frame(tab, style="Card.TFrame")
    row.pack(fill="x", pady=(2, 6))
    app.camera_choices = [("Auto (external first)", AUTO_CAMERA)]
    app.camera_combo = ttk.Combobox(row, state="readonly", values=["Auto (external first)"], width=26)
    app.camera_combo.set("Auto (external first)" if app.settings.camera_index == AUTO_CAMERA
                         else f"Camera {app.settings.camera_index}")
    app.camera_combo.pack(side="left", fill="x", expand=True)
    app.camera_combo.bind("<<ComboboxSelected>>", app.on_camera_selected)
    ttk.Button(row, text="↻", width=3, style="Small.TButton",
               command=app.controller.probe_cameras_async).pack(side="left", padx=(6, 0))

    ttk.Label(tab, text="Resolution", style="Card.TLabel").pack(anchor="w")
    app.resolution_combo = _combo(tab)
    app.resolution_combo.configure(values=list(RESOLUTIONS))
    app.resolution_combo.set(app.settings.resolution)
    app.resolution_combo.bind("<<ComboboxSelected>>", app.on_resolution_selected)
    app.check(tab, "Mirror image", "mirror", app.controller.set_mirror)
    app.check(tab, "Start camera when the app opens", "auto_start_camera")

    _section(tab, "OVERLAY")
    app.check(tab, "Show labels", "show_labels")
    app.check(tab, "Show confidence", "show_scores")
    app.check(tab, "Show stats overlay", "show_hud")
    app.check(tab, "Privacy mode (pixelate people)", "privacy_blur")
    _scale(tab, "Box thickness", 1, 5, "box_thickness", app)

    _section(tab, "SAVED MEDIA")
    row = ttk.Frame(tab, style="Card.TFrame")
    row.pack(fill="x")
    ttk.Button(row, text="Open snapshots", style="Small.TButton", command=app.open_snapshots).pack(side="left")
    ttk.Button(row, text="Open recordings", style="Small.TButton",
               command=app.open_recordings).pack(side="left", padx=(8, 0))


def show_help_window(root) -> None:
    win = tk.Toplevel(root, bg=theme.SURFACE, padx=22, pady=18)
    win.title("Help")
    win.transient(root)
    win.resizable(False, False)
    for col, (title, rows) in enumerate((("KEYBOARD", SHORTCUTS), ("VOICE COMMANDS", VOICE_HELP))):
        frame = ttk.Frame(win, style="Card.TFrame")
        frame.grid(row=0, column=col, sticky="n", padx=(0, 30) if col == 0 else 0)
        ttk.Label(frame, text=title, style="Section.TLabel").grid(row=0, column=0, columnspan=2,
                                                                 sticky="w", pady=(0, 8))
        for r, (key, desc) in enumerate(rows, start=1):
            ttk.Label(frame, text=key, style="Value.TLabel").grid(row=r, column=0, sticky="w", pady=2)
            ttk.Label(frame, text=desc, style="Card.TLabel").grid(row=r, column=1, sticky="w", padx=(14, 0))
    ttk.Button(win, text="Close", command=win.destroy).grid(row=1, column=1, sticky="e", pady=(16, 0))
    win.bind("<Escape>", lambda e: win.destroy())
    win.update_idletasks()
    x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
    y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
    win.geometry(f"+{max(0, x)}+{max(0, y)}")
    win.focus_set()
