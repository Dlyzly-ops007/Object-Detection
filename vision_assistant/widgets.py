"""Custom Tk widgets: video canvas, level meter, log view, status chip, labelled slider."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import cv2

from . import theme


class VideoCanvas(tk.Canvas):
    """Shows BGR frames scaled to fit (aspect preserved) using Tk's native PPM loader (no PIL needed)."""

    def __init__(self, master, **kw):
        super().__init__(master, bg="black", highlightthickness=3, highlightbackground=theme.BG, bd=0, **kw)
        self._photo = None
        self._image_id = self.create_image(0, 0, anchor="center")
        self._title_id = self.create_text(0, 0, fill=theme.TEXT, font=("TkDefaultFont", 16, "bold"))
        self._sub_id = self.create_text(0, 0, fill=theme.MUTED, font=("TkDefaultFont", 10))
        self._flash_job = None
        self.bind("<Configure>", self._layout)

    def _layout(self, _event=None) -> None:
        w, h = self.winfo_width(), self.winfo_height()
        self.coords(self._image_id, w // 2, h // 2)
        self.coords(self._title_id, w // 2, h // 2 - 12)
        self.coords(self._sub_id, w // 2, h // 2 + 18)

    def show_frame(self, frame) -> None:
        cw, ch = max(1, self.winfo_width() - 6), max(1, self.winfo_height() - 6)
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        size = (max(1, int(fw * scale)), max(1, int(fh * scale)))
        if size != (fw, fh):
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        header = f"P6 {size[0]} {size[1]} 255 ".encode()
        self._photo = tk.PhotoImage(width=size[0], height=size[1], data=header + rgb.tobytes(), format="PPM")
        self.itemconfigure(self._image_id, image=self._photo, state="normal")
        self.itemconfigure(self._title_id, state="hidden")
        self.itemconfigure(self._sub_id, state="hidden")

    def show_placeholder(self, title: str, subtitle: str = "") -> None:
        self._photo = None
        self.itemconfigure(self._image_id, image="", state="hidden")
        self.itemconfigure(self._title_id, text=title, state="normal")
        self.itemconfigure(self._sub_id, text=subtitle, state="normal")
        self._layout()

    def flash(self, color: str = theme.ORANGE, duration_ms: int = 900) -> None:
        self.configure(highlightbackground=color)
        if self._flash_job:
            self.after_cancel(self._flash_job)
        self._flash_job = self.after(duration_ms, lambda: self.configure(highlightbackground=theme.BG))


class LevelMeter(tk.Canvas):
    SEGMENTS = 20

    def __init__(self, master, width=160, height=10, **kw):
        super().__init__(master, width=width, height=height, bg=theme.SURFACE, highlightthickness=0, **kw)
        seg_w = width / self.SEGMENTS
        self._bars = []
        for i in range(self.SEGMENTS):
            x = i * seg_w
            self._bars.append(self.create_rectangle(x + 1, 1, x + seg_w - 1, height - 1,
                                                    fill=theme.SURFACE_2, outline=""))
        self._level = 0.0

    def set(self, level: float) -> None:
        # Fast attack, slow release reads much better than raw RMS.
        self._level = level if level > self._level else self._level * 0.8 + level * 0.2
        lit = int(self._level * self.SEGMENTS + 0.5)
        for i, bar in enumerate(self._bars):
            if i >= lit:
                color = theme.SURFACE_2
            elif i >= self.SEGMENTS * 0.85:
                color = theme.RED
            elif i >= self.SEGMENTS * 0.65:
                color = theme.ORANGE
            else:
                color = theme.GREEN
            self.itemconfigure(bar, fill=color)


class LogView(ttk.Frame):
    MAX_LINES = 400
    TAGS = {"info": theme.TEXT, "muted": theme.MUTED, "speech": theme.ACCENT, "command": theme.GREEN,
            "alert": theme.ORANGE, "error": theme.RED}

    def __init__(self, master, **kw):
        super().__init__(master, style="Card.TFrame", **kw)
        self.text = tk.Text(self, bg=theme.SURFACE, fg=theme.TEXT, bd=0, highlightthickness=0, wrap="word",
                            height=6, padx=10, pady=6, font=("Consolas", 9), state="disabled",
                            insertbackground=theme.TEXT, selectbackground=theme.ACCENT)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.text.tag_configure("time", foreground=theme.MUTED)
        for tag, color in self.TAGS.items():
            self.text.tag_configure(tag, foreground=color)

    def append(self, message: str, tag: str = "info") -> None:
        at_bottom = self.text.yview()[1] > 0.98
        self.text.configure(state="normal")
        self.text.insert("end", time.strftime("%H:%M:%S  "), "time")
        self.text.insert("end", message + "\n", tag if tag in self.TAGS else "info")
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > self.MAX_LINES:
            self.text.delete("1.0", f"{lines - self.MAX_LINES}.0")
        self.text.configure(state="disabled")
        if at_bottom:
            self.text.see("end")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class StatusChip(tk.Frame):
    """● LABEL  value — a small pill in the header."""

    def __init__(self, master, label: str):
        super().__init__(master, bg=theme.SURFACE, padx=10, pady=5)
        self._dot = tk.Label(self, text="●", bg=theme.SURFACE, fg=theme.MUTED, font=("TkDefaultFont", 9))
        self._name = tk.Label(self, text=label.upper(), bg=theme.SURFACE, fg=theme.MUTED,
                              font=("TkDefaultFont", 8, "bold"))
        self._value = tk.Label(self, text="—", bg=theme.SURFACE, fg=theme.TEXT, font=("TkDefaultFont", 9))
        self._dot.pack(side="left")
        self._name.pack(side="left", padx=(4, 6))
        self._value.pack(side="left")

    def set(self, state: str, text: str) -> None:
        self._dot.configure(fg=theme.STATE_COLORS.get(state, theme.MUTED))
        self._value.configure(text=text)


class LabeledScale(ttk.Frame):
    """Label + live value readout + slider. Calls on_change(value) with the rounded value."""

    def __init__(self, master, text: str, from_: float, to: float, value: float, *, step: float = 1,
                 fmt=str, on_change=None):
        super().__init__(master, style="Card.TFrame")
        self._step, self._fmt, self._on_change = step, fmt, on_change
        value = self._snap(value)
        self.var = tk.DoubleVar(value=value)
        self._last = value
        ttk.Label(self, text=text, style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self._readout = ttk.Label(self, text=fmt(value), style="Value.TLabel")
        self._readout.grid(row=0, column=1, sticky="e")
        ttk.Scale(self, from_=from_, to=to, variable=self.var, command=self._changed).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.columnconfigure(0, weight=1)

    def _snap(self, value: float):
        value = round(value / self._step) * self._step
        return int(value) if float(self._step).is_integer() else round(value, 4)

    def _changed(self, _raw) -> None:
        value = self._snap(self.var.get())
        self._readout.configure(text=self._fmt(value))
        if value != self._last:
            self._last = value
            if self._on_change:
                self._on_change(value)
