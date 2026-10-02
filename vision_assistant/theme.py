"""Dark ttk theme."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = "#0f1116"
SURFACE = "#171a21"
SURFACE_2 = "#20242e"
BORDER = "#2b303c"
TEXT = "#e7e9ef"
MUTED = "#8a92a6"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6b9fff"
GREEN = "#34c46b"
RED = "#ff5c5c"
ORANGE = "#ffb020"

STATE_COLORS = {"ok": GREEN, "busy": ORANGE, "error": RED, "off": MUTED, "rec": RED}


def pick_family(root: tk.Misc) -> str:
    available = set(tkfont.families(root))
    return next((f for f in ("Segoe UI", "Helvetica Neue", "Inter", "DejaVu Sans") if f in available),
                "TkDefaultFont")


def ui_scale(root: tk.Misc) -> float:
    """Screen scaling factor (1.0 at 96 DPI) for sizes given in pixels."""
    return max(1.0, root.winfo_fpixels("1i") / 96)


def apply_theme(root: tk.Tk) -> str:
    family = pick_family(root)
    scale = ui_scale(root)
    root.configure(bg=BG)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(family=family, size=10)

    root.option_add("*TCombobox*Listbox.background", SURFACE_2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.font", (family, 10))

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE_2, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, troughcolor=SURFACE_2, focuscolor=ACCENT,
                    selectbackground=ACCENT, selectforeground="white", insertcolor=TEXT, font=(family, 10))

    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED, font=(family, 9))
    style.configure("Value.TLabel", background=SURFACE, foreground=ACCENT, font=(family, 9, "bold"))
    style.configure("Section.TLabel", background=SURFACE, foreground=MUTED, font=(family, 8, "bold"))
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=(family, 15, "bold"))
    style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=(family, 9))
    style.configure("Heard.TLabel", background=SURFACE, foreground=TEXT, font=(family, 10, "italic"))
    style.configure("Status.TLabel", background=SURFACE, foreground=MUTED, font=(family, 9))

    style.configure("TButton", background=SURFACE_2, foreground=TEXT, borderwidth=1, padding=(12, 7),
                    relief="flat", font=(family, 10))
    style.map("TButton", background=[("disabled", SURFACE), ("pressed", BORDER), ("active", BORDER)],
              foreground=[("disabled", MUTED)])
    style.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                    font=(family, 10, "bold"))
    style.map("Accent.TButton", background=[("disabled", SURFACE_2), ("pressed", ACCENT), ("active", ACCENT_HOVER)])
    style.configure("Danger.TButton", background=RED, foreground="white", bordercolor=RED,
                    font=(family, 10, "bold"))
    style.map("Danger.TButton", background=[("pressed", RED), ("active", "#ff7676")])
    style.configure("Small.TButton", padding=(8, 3), font=(family, 9))

    for name in ("TCheckbutton", "Card.TCheckbutton"):
        style.configure(name, background=SURFACE, foreground=TEXT, indicatorbackground=SURFACE_2,
                        indicatorforeground="white", indicatormargin=(0, 0, 6, 0), padding=(0, 3))
        style.map(name, background=[("active", SURFACE)],
                  indicatorbackground=[("selected", ACCENT), ("active", BORDER)])

    style.configure("TCombobox", fieldbackground=SURFACE_2, background=SURFACE_2, foreground=TEXT,
                    arrowcolor=TEXT, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE_2)], foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", SURFACE_2)], selectforeground=[("readonly", TEXT)])
    style.configure("TSpinbox", fieldbackground=SURFACE_2, foreground=TEXT, arrowcolor=TEXT, padding=3)
    style.configure("Horizontal.TScale", background=ACCENT, troughcolor=SURFACE_2, bordercolor=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT, gripcount=0, sliderlength=int(16 * scale))
    style.map("Horizontal.TScale", background=[("active", ACCENT_HOVER)],
              lightcolor=[("active", ACCENT_HOVER)], darkcolor=[("active", ACCENT_HOVER)])
    style.configure("Vertical.TScrollbar", background=SURFACE_2, troughcolor=SURFACE, arrowcolor=MUTED,
                    bordercolor=SURFACE)
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SURFACE_2, thickness=4)

    style.configure("TNotebook", background=SURFACE, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(14, 7), borderwidth=0,
                    font=(family, 10))
    style.map("TNotebook.Tab", background=[("selected", SURFACE_2)], foreground=[("selected", TEXT)])

    style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT, rowheight=int(24 * scale),
                    borderwidth=0)
    style.map("Treeview", background=[("selected", SURFACE_2)])
    style.configure("Treeview.Heading", background=SURFACE_2, foreground=MUTED, relief="flat",
                    font=(family, 9, "bold"))
    style.map("Treeview.Heading", background=[("active", BORDER)])
    return family
