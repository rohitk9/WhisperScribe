import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from logging.handlers import RotatingFileHandler
from tkinter import filedialog, messagebox

import customtkinter as ctk

import engine

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:  # drag & drop is optional; browsing still works
    HAS_DND = False

# A windowed (no console) build has no stdout/stderr; progress bars from huggingface/tqdm would crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

APP_NAME = "WhisperScribe"
# When frozen by PyInstaller, bundled files live in sys._MEIPASS and logs go next to the .exe.
BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else BUNDLE_DIR
LOG_DIR = os.path.join(APP_DIR, "logs")
SETTINGS_PATH = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP_NAME, "settings.json")
SETTINGS_VERSION = 2

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[RotatingFileHandler(os.path.join(LOG_DIR, "whisperscribe.log"), maxBytes=1_000_000, backupCount=3,
                                  encoding="utf-8")],
)
log = logging.getLogger("whisperscribe")
logging.getLogger("httpx").setLevel(logging.WARNING)  # HuggingFace download chatter

# Suppress the symlink warning from HuggingFace
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
engine.register_cuda_dlls()

# --- Palette: (light, dark) ---
BG = ("#F4F5F7", "#101216")
SIDEBAR = ("#FFFFFF", "#16191F")
CARD = ("#FFFFFF", "#1B1F26")
FIELD = ("#F7F8FA", "#14171C")
BORDER = ("#E2E5EA", "#2A2F38")
TEXT = ("#111827", "#E5E7EB")
MUTED = ("#6B7280", "#8B93A1")
ACCENT = "#10B981"
ACCENT_HOVER = "#0E9F6E"
ACCENT_SOFT = ("#E7F8F1", "#12302A")
DANGER = "#EF4444"
DANGER_HOVER = "#DC2626"
SPEAKER_COLORS = ["#60A5FA", "#F472B6", "#FBBF24", "#A78BFA", "#34D399", "#F87171"]

PROMPT_PRESETS = {
    "Summary": "Summarize this recording in a short paragraph, followed by 3-5 key takeaways as bullet points.",
    "Action items": "List every action item, decision, and owner mentioned. Use bullet points.",
    "Key points": "Extract the main topics discussed as concise bullet points.",
    "Notes": "Write structured meeting notes with sections: Overview, Discussion, Decisions, Action Items.",
}

DEFAULT_SETTINGS = {
    "version": SETTINGS_VERSION,
    "language": "English",
    "model": engine.DEFAULT_MODEL,
    "device": "Auto",
    "speakers": "Off",
    "format": "Plain text (.txt)",
    "summary_model": engine.DEFAULT_SUMMARY_MODEL,
    "open_when_done": True,
    "appearance": "Dark",
    "last_dir": "",
}


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("version", 1) < SETTINGS_VERSION:
            # v1 defaulted to "Base"; benchmarks showed Large v3 Turbo is as fast and far more accurate.
            saved.pop("model", None)
            saved["version"] = SETTINGS_VERSION
        settings.update(saved)
    except FileNotFoundError:
        pass
    except Exception:
        log.warning("Could not read settings", exc_info=True)
    return settings


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def human_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def output_path_for(input_path: str, fmt_name: str) -> str:
    return os.path.splitext(input_path)[0] + engine.OUTPUT_FORMATS[fmt_name]


class TranscriberApp(ctk.CTk):
    def __init__(self):
        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings["appearance"])
        ctk.set_default_color_theme("green")
        super().__init__(fg_color=BG)

        self.title(APP_NAME)
        self.geometry("1240x800")
        self.minsize(1040, 720)
        icon_path = os.path.join(BUNDLE_DIR, "icon.ico")
        if os.path.exists(icon_path):
            # CTk resets the icon shortly after start-up on Windows, so set it again after a delay.
            self.after(250, lambda: self.iconbitmap(icon_path))

        self.engine = engine.Engine()
        self.ui_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.running = False
        self.files = []
        self.output_auto = True
        self.written_files = []
        self._seg_t0 = None
        self._drag_leave_job = None

        self.f_brand = ctk.CTkFont("Segoe UI", 22, "bold")
        self.f_h1 = ctk.CTkFont("Segoe UI", 20, "bold")
        self.f_h2 = ctk.CTkFont("Segoe UI", 14, "bold")
        self.f_body = ctk.CTkFont("Segoe UI", 13)
        self.f_body_bold = ctk.CTkFont("Segoe UI", 13, "bold")
        self.f_small = ctk.CTkFont("Segoe UI", 12)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self._enable_drag_and_drop()
        self._bind_shortcuts()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._drain_queue)
        threading.Thread(target=self._detect_gpu, daemon=True).start()

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=SIDEBAR, border_width=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.pack_propagate(False)

        ctk.CTkLabel(sb, text="🎙  " + APP_NAME, font=self.f_brand, text_color=TEXT).pack(anchor="w", padx=22, pady=(26, 0))
        ctk.CTkLabel(sb, text="Private, on-device transcription", font=self.f_small, text_color=MUTED).pack(
            anchor="w", padx=22, pady=(2, 20))

        self.lang_var = self._sidebar_option(sb, "Language", list(engine.LANGUAGES), "language")
        self.model_var = self._sidebar_option(sb, "Speech model", list(engine.MODELS), "model")
        self.speakers_var = self._sidebar_option(sb, "Speaker labels", engine.SPEAKER_COUNTS, "speakers",
                                                 hint="Off · Auto-detect · or a fixed count")
        self.device_var = self._sidebar_option(sb, "Processing device", engine.DEVICES, "device")
        self.format_var = self._sidebar_option(sb, "Output format", list(engine.OUTPUT_FORMATS), "format",
                                               command=lambda _: self._on_format_change())

        self.open_var = tk.BooleanVar(value=self.settings["open_when_done"])
        ctk.CTkSwitch(sb, text="Open file when finished", variable=self.open_var, font=self.f_body,
                      progress_color=ACCENT, command=self._save_settings).pack(anchor="w", padx=24, pady=(4, 0))

        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=24, pady=22)
        self.device_label = ctk.CTkLabel(bottom, text="● Checking hardware…", font=self.f_small, text_color=MUTED)
        self.device_label.pack(anchor="w", pady=(0, 8))
        self.appearance = ctk.CTkSegmentedButton(bottom, values=["Light", "Dark", "System"], font=self.f_small,
                                                 selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
                                                 command=self._on_appearance)
        self.appearance.set(self.settings["appearance"])
        self.appearance.pack(fill="x")

    def _option_menu(self, parent, values, var, command, **kw):
        return ctk.CTkOptionMenu(parent, values=values, variable=var, font=self.f_body, dropdown_font=self.f_body,
                                 fg_color=FIELD, button_color=BORDER, button_hover_color=MUTED, text_color=TEXT,
                                 corner_radius=8, command=command, **kw)

    def _sidebar_option(self, parent, label, values, key, hint=None, command=None):
        ctk.CTkLabel(parent, text=label.upper(), font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED,
                     height=18).pack(anchor="w", padx=24)
        value = self.settings.get(key)
        var = tk.StringVar(value=value if value in values else DEFAULT_SETTINGS[key])

        def on_change(choice):
            self._save_settings()
            if command:
                command(choice)

        self._option_menu(parent, values, var, on_change, height=32).pack(fill="x", padx=24, pady=(3, 0))
        if hint:
            ctk.CTkLabel(parent, text=hint, font=self.f_small, text_color=MUTED, height=18).pack(anchor="w", padx=24)
        ctk.CTkFrame(parent, height=10, fg_color="transparent").pack()
        return var

    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14, border_width=1, border_color=BORDER, **kw)

    def _ghost_button(self, parent, text, command, **kw):
        opts = dict(font=self.f_body, fg_color="transparent", border_width=1, border_color=BORDER, text_color=TEXT,
                    hover_color=BORDER)
        opts.update(kw)
        return ctk.CTkButton(parent, text=text, command=command, **opts)

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        main.grid_columnconfigure(0, weight=1, uniform="cols")
        main.grid_columnconfigure(1, weight=1, uniform="cols")
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="New transcription", font=self.f_h1, text_color=TEXT).grid(row=0, column=0, sticky="w",
                                                                                        pady=(0, 12))
        self._build_drop_zone(left).grid(row=1, column=0, sticky="nsew")
        self._build_output_row(left).grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self._build_prompt(left).grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self._build_actions(left).grid(row=4, column=0, sticky="ew", pady=(16, 0))
        self._build_progress(left).grid(row=5, column=0, sticky="ew", pady=(14, 0))

        self._build_results(main).grid(row=0, column=1, sticky="nsew")

    def _build_drop_zone(self, parent):
        dz = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=2, border_color=BORDER, height=170)
        self.drop_zone = dz
        inner = ctk.CTkFrame(dz, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        self.dz_icon = ctk.CTkLabel(inner, text="⇪", font=ctk.CTkFont("Segoe UI Symbol", 38), text_color=ACCENT)
        self.dz_icon.pack()
        self.dz_title = ctk.CTkLabel(inner, text="Drop audio or video files here", font=self.f_h2, text_color=TEXT)
        self.dz_title.pack(pady=(4, 0))
        self.dz_sub = ctk.CTkLabel(inner, text=self._dz_hint(), font=self.f_small, text_color=MUTED, wraplength=380)
        self.dz_sub.pack(pady=(2, 0))
        self.dz_clear = self._ghost_button(inner, "Remove", self._clear_input, width=80, height=26,
                                           font=self.f_small, text_color=MUTED)

        for w in (dz, inner, self.dz_icon, self.dz_title, self.dz_sub):
            w.bind("<Button-1>", lambda e: self.browse_input())
            w.bind("<Enter>", lambda e: self._set_drop_highlight(True, hover=True))
            w.bind("<Leave>", lambda e: self._set_drop_highlight(False, hover=True))
            try:
                w.configure(cursor="hand2")
            except (ValueError, tk.TclError):
                pass
        return dz

    @staticmethod
    def _dz_hint():
        return "or click to browse  ·  one file or a whole batch  ·  MP3, M4A, WAV, MP4…"

    def _build_output_row(self, parent):
        card = self._card(parent)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Save to", font=self.f_h2, text_color=TEXT).grid(row=0, column=0, padx=(16, 10),
                                                                              pady=12)
        self.output_var = tk.StringVar()
        self.output_entry = ctk.CTkEntry(card, textvariable=self.output_var, font=self.f_small, fg_color=FIELD,
                                         border_color=BORDER, height=34)
        self.output_entry.grid(row=0, column=1, sticky="ew", pady=12)
        self.output_entry.bind("<Key>", lambda e: setattr(self, "output_auto", False))
        self.output_btn = self._ghost_button(card, "Change…", self.browse_output, width=90, height=34)
        self.output_btn.grid(row=0, column=2, padx=12, pady=12)
        return card

    def _build_prompt(self, parent):
        card = self._card(parent)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(12, 0))
        ctk.CTkLabel(head, text="AI instructions", font=self.f_h2, text_color=TEXT, height=20).pack(side="left")
        value = self.settings.get("summary_model")
        self.summary_model_var = tk.StringVar(
            value=value if value in engine.SUMMARY_MODELS else engine.DEFAULT_SUMMARY_MODEL)
        self._option_menu(head, list(engine.SUMMARY_MODELS), self.summary_model_var,
                          lambda _: self._save_settings(), width=170, height=26).pack(side="right")
        ctk.CTkLabel(card, text="Optional · a local LLM runs your instruction on the transcript", font=self.f_small,
                     text_color=MUTED, height=18).pack(anchor="w", padx=16)

        chips = ctk.CTkFrame(card, fg_color="transparent")
        chips.pack(fill="x", padx=16, pady=(8, 0))
        for name, text in PROMPT_PRESETS.items():
            self._ghost_button(chips, name, lambda t=text: self._set_prompt(t), height=26, width=10, corner_radius=13,
                               font=self.f_small, hover_color=ACCENT_SOFT).pack(side="left", padx=(0, 6))

        self.prompt_box = ctk.CTkTextbox(card, height=64, font=self.f_body, fg_color=FIELD, border_width=1,
                                         border_color=BORDER, corner_radius=8, wrap="word")
        self.prompt_box.pack(fill="x", padx=16, pady=(10, 14))
        return card

    def _build_actions(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid_columnconfigure(0, weight=1)
        self.start_btn = ctk.CTkButton(row, text="Transcribe", height=46, corner_radius=10,
                                       font=ctk.CTkFont("Segoe UI", 15, "bold"), fg_color=ACCENT,
                                       hover_color=ACCENT_HOVER, command=self.start_processing)
        self.start_btn.grid(row=0, column=0, sticky="ew")
        self.cancel_btn = self._ghost_button(row, "Cancel", self.cancel_processing, width=110, height=46,
                                             corner_radius=10, font=ctk.CTkFont("Segoe UI", 15, "bold"),
                                             text_color=MUTED, hover_color=DANGER_HOVER, state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=(10, 0))
        return row

    def _build_progress(self, parent):
        card = self._card(parent)
        card.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(card, text="Ready", font=self.f_h2, text_color=TEXT, anchor="w")
        self.status_label.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 0))
        self.pct_label = ctk.CTkLabel(card, text="", font=self.f_h2, text_color=ACCENT)
        self.pct_label.grid(row=0, column=1, sticky="e", padx=16, pady=(12, 0))
        self.progress = ctk.CTkProgressBar(card, height=8, corner_radius=4, progress_color=ACCENT, fg_color=BORDER)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(8, 0))
        self.progress.set(0)
        self.meta_label = ctk.CTkLabel(card, text="Tip: Ctrl+O opens files, Ctrl+Enter starts.",
                                       font=self.f_small, text_color=MUTED, anchor="w")
        self.meta_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(4, 10))
        return card

    def _build_results(self, parent):
        card = self._card(parent)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(card, fg_color="transparent", segmented_button_selected_color=ACCENT,
                                   segmented_button_selected_hover_color=ACCENT_HOVER, anchor="nw")
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=(4, 0))
        self.tabs._segmented_button.configure(font=self.f_body, height=30)
        self.transcript_box = self._result_box(self.tabs.add("Transcript"),
                                               "The transcript will stream in here as it's recognised.")
        self.summary_box = self._result_box(self.tabs.add("Summary"),
                                            "Add AI instructions to get a summary, action items, or notes.")

        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 14))
        self.result_btns = []
        for text, cmd in (("Copy", self._copy_result), ("Open file", self._open_output),
                          ("Show in folder", self._show_in_folder)):
            b = self._ghost_button(bar, text, cmd, width=10, height=32, state="disabled")
            b.pack(side="left", padx=(0, 8))
            self.result_btns.append(b)
        self.result_btns[0].configure(state="normal")
        return card

    def _result_box(self, tab, placeholder):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        box = ctk.CTkTextbox(tab, font=self.f_body, fg_color=FIELD, border_width=0, corner_radius=10, wrap="word")
        box.grid(row=0, column=0, sticky="nsew")
        box.tag_config("ts", foreground=ACCENT)
        box.tag_config("muted", foreground="#8B93A1")
        box.tag_config("heading", foreground=ACCENT)
        for i, color in enumerate(SPEAKER_COLORS):
            box.tag_config(f"spk{i}", foreground=color)
        # CTkTextbox forbids fonts in tag_config (DPI scaling), so set it on the inner tk.Text.
        box._textbox.tag_config("bold", font=self.f_body_bold)
        box.insert("end", placeholder, "muted")
        box.configure(state="disabled")
        return box

    # ------------------------------------------------------------ drag & drop
    def _enable_drag_and_drop(self):
        if not HAS_DND:
            log.info("tkinterdnd2 not installed; drag & drop disabled")
            self.dz_title.configure(text="Click to choose audio or video files")
            return
        try:
            TkinterDnD.require(self)
        except Exception:
            log.warning("Could not initialise drag & drop", exc_info=True)
            self.dz_title.configure(text="Click to choose audio or video files")
            return
        # Accept drops anywhere in the window, not only on the drop zone.
        self._register_drop_target(self)

    def _register_drop_target(self, widget):
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", self._on_drag_enter)
            widget.dnd_bind("<<DropPosition>>", self._on_drag_enter)
            widget.dnd_bind("<<DropLeave>>", self._on_drag_leave)
            widget.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._register_drop_target(child)

    def _on_drag_enter(self, event):
        if not self.running:
            self._set_drop_highlight(True)
        return event.action

    def _on_drag_leave(self, event):
        self._set_drop_highlight(False)
        return event.action

    def _on_drop(self, event):
        self._set_drop_highlight(False, immediate=True)
        if self.running:
            return event.action
        paths = []
        for p in self.tk.splitlist(event.data):
            if os.path.isdir(p):  # a dropped folder adds every supported file inside it
                paths += sorted(os.path.join(p, f) for f in os.listdir(p))
            else:
                paths.append(p)
        supported = [p for p in paths if os.path.isfile(p) and p.lower().endswith(engine.AUDIO_EXTENSIONS)]
        if not supported:
            self._flash_drop_error("That file type isn't supported — try MP3, M4A, WAV, MP4…")
        else:
            self._set_inputs(supported)
            skipped = len([p for p in paths if os.path.isfile(p)]) - len(supported)
            if skipped:
                self.meta_label.configure(text=f"Skipped {skipped} unsupported file(s).")
        return event.action

    def _set_drop_highlight(self, on, hover=False, immediate=False):
        # Leave/enter fire as the pointer crosses child widgets; debounce "off" so it doesn't flicker.
        if self._drag_leave_job:
            self.after_cancel(self._drag_leave_job)
            self._drag_leave_job = None
        if on:
            if not self.running:
                self._apply_drop_style("hover" if hover else "drag")
        elif immediate:
            self._apply_drop_style("idle")
        else:
            self._drag_leave_job = self.after(60, lambda: self._apply_drop_style("idle"))

    def _apply_drop_style(self, style):
        if getattr(self, "_drop_style", None) == style:
            return  # avoid redrawing on every <<DropPosition>> event
        self._drop_style = style
        border, fill = {"idle": (BORDER, CARD), "hover": (ACCENT, CARD), "drag": (ACCENT, ACCENT_SOFT)}[style]
        self.drop_zone.configure(border_color=border, fg_color=fill)

    def _flash_drop_error(self, msg):
        self._drop_style = "error"
        self.drop_zone.configure(border_color=DANGER, fg_color=CARD)
        self.meta_label.configure(text=msg, text_color=DANGER)
        self.after(1600, lambda: (self._apply_drop_style("idle"), self.meta_label.configure(text_color=MUTED)))

    # --------------------------------------------------------------- inputs
    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self.browse_input())
        self.bind("<Control-Return>", lambda e: self.start_processing())
        self.bind("<Escape>", lambda e: self.cancel_processing() if self.running else None)

    def browse_input(self):
        if self.running:
            return
        exts = " ".join("*" + e for e in engine.AUDIO_EXTENSIONS)
        paths = filedialog.askopenfilenames(title="Select audio or video files",
                                            initialdir=self.settings.get("last_dir") or None,
                                            filetypes=[("Audio / video", exts), ("All files", "*.*")])
        if paths:
            self._set_inputs(list(paths))

    def _set_inputs(self, paths):
        seen, files = set(), []
        for p in paths:
            p = os.path.normpath(p)
            if p.lower() not in seen:
                seen.add(p.lower())
                files.append(p)
        self.files = files
        self.settings["last_dir"] = os.path.dirname(files[0])
        self._save_settings()
        total = sum(os.path.getsize(p) for p in files)
        folder = os.path.basename(os.path.dirname(files[0])) or os.path.dirname(files[0])
        self.dz_icon.configure(text="♫")
        if len(files) == 1:
            self.dz_title.configure(text=os.path.basename(files[0]))
            self.dz_sub.configure(text=f"{human_size(total)}  ·  in {folder}")
        else:
            names = ", ".join(os.path.basename(p) for p in files[:2])
            more = f" + {len(files) - 2} more" if len(files) > 2 else ""
            self.dz_title.configure(text=f"{len(files)} files ready")
            self.dz_sub.configure(text=f"{names}{more}  ·  {human_size(total)}")
        self.dz_clear.configure(text="Remove" if len(files) == 1 else "Clear all")
        self.dz_clear.pack(pady=(10, 0))
        self._refresh_output_row()
        self.status_label.configure(text="Ready to transcribe", text_color=TEXT)
        self.meta_label.configure(text="Press Transcribe (or Ctrl+Enter) to start.", text_color=MUTED)

    def _clear_input(self):
        if self.running:
            return
        self.files = []
        self.dz_icon.configure(text="⇪")
        self.dz_title.configure(text="Drop audio or video files here")
        self.dz_sub.configure(text=self._dz_hint())
        self.dz_clear.pack_forget()
        self.output_auto = True
        self._refresh_output_row()

    def _refresh_output_row(self):
        batch = len(self.files) > 1
        self.output_entry.configure(state="normal")
        if batch:
            self.output_var.set("Next to each audio file")
            self.output_entry.configure(state="disabled")
            self.output_btn.configure(state="disabled")
            return
        self.output_btn.configure(state="normal")
        if not self.files:
            self.output_var.set("")
        elif self.output_auto or not self.output_var.get():
            self.output_auto = True
            self.output_var.set(output_path_for(self.files[0], self.format_var.get()))

    def _on_format_change(self):
        current = self.output_var.get()
        if len(self.files) > 1:
            return
        if self.output_auto and self.files:
            self.output_var.set(output_path_for(self.files[0], self.format_var.get()))
        elif current:
            self.output_var.set(output_path_for(current, self.format_var.get()))

    def browse_output(self):
        ext = engine.OUTPUT_FORMATS[self.format_var.get()]
        current = self.output_var.get()
        path = filedialog.asksaveasfilename(
            title="Save output as", defaultextension=ext,
            initialdir=os.path.dirname(current) if current else (self.settings.get("last_dir") or None),
            initialfile=os.path.basename(current) if current else "",
            filetypes=[("Output", "*" + ext), ("All files", "*.*")])
        if path:
            self.output_auto = False
            self.output_var.set(os.path.normpath(path))

    def _set_prompt(self, text):
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", text)

    # ------------------------------------------------------------ settings
    def _on_appearance(self, mode):
        ctk.set_appearance_mode(mode)
        self._save_settings()

    def _save_settings(self):
        try:
            self.settings.update(
                language=self.lang_var.get(), model=self.model_var.get(), device=self.device_var.get(),
                speakers=self.speakers_var.get(), format=self.format_var.get(),
                summary_model=self.summary_model_var.get(), open_when_done=bool(self.open_var.get()),
                appearance=self.appearance.get() or self.settings["appearance"])
        except AttributeError:
            pass  # called before all widgets exist
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            log.warning("Could not save settings", exc_info=True)

    def _detect_gpu(self):
        try:
            import ctranslate2
            n = ctranslate2.get_cuda_device_count()
            text = "● GPU ready (CUDA)" if n else "● CPU only (no CUDA GPU)"
            color = ACCENT if n else MUTED
        except Exception:
            text, color = "● Hardware status unknown", MUTED
        self.post(self.device_label.configure, text=text, text_color=color)

    # --------------------------------------------------------- thread → UI
    def post(self, fn, *args, **kwargs):
        """Schedule `fn` on the Tk thread. Tk isn't thread-safe, so worker threads must use this."""
        self.ui_queue.put((fn, args, kwargs))

    def _drain_queue(self):
        try:
            while True:
                fn, args, kwargs = self.ui_queue.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
        self.after(50, self._drain_queue)

    # ----------------------------------------------------------- processing
    def start_processing(self):
        if self.running:
            return
        files = [f for f in self.files if os.path.isfile(f)]
        if not files:
            self._flash_drop_error("Choose an audio file first — drop it in or click the box.")
            return
        fmt = self.format_var.get()
        if len(files) == 1:
            output = self.output_var.get().strip() or output_path_for(files[0], fmt)
            self.output_var.set(output)
            if not os.path.isdir(os.path.dirname(os.path.abspath(output))):
                messagebox.showwarning("Invalid location", "The output folder doesn't exist.")
                return
            jobs = [(files[0], output)]
        else:
            jobs = [(f, output_path_for(f, fmt)) for f in files]
        existing = [o for _, o in jobs if os.path.exists(o)]
        if existing:
            names = "\n".join(os.path.basename(o) for o in existing[:5]) + ("\n…" if len(existing) > 5 else "")
            if not messagebox.askyesno("Overwrite files?", f"These files already exist:\n\n{names}\n\nReplace them?"):
                return

        self.running = True
        self.cancel_event.clear()
        self.written_files = []
        self._set_controls_running(True)
        self._reset_box(self.transcript_box)
        self._reset_box(self.summary_box)
        self.tabs.set("Transcript")
        self.progress.set(0)
        self.pct_label.configure(text="0%")
        self.meta_label.configure(text="", text_color=MUTED)

        opts = dict(fmt=fmt, prompt=self.prompt_box.get("1.0", "end-1c").strip(), language=self.lang_var.get(),
                    model=self.model_var.get(), device=self.device_var.get(), speakers=self.speakers_var.get(),
                    summary_model=self.summary_model_var.get())
        log.info("Starting %d job(s): %s", len(jobs), {k: v for k, v in opts.items() if k != "prompt"})
        threading.Thread(target=self._worker, args=(jobs, opts), daemon=True).start()

    def _worker(self, jobs, opts):
        started = time.time()
        failures = []
        n = len(jobs)
        for i, (input_path, output_path) in enumerate(jobs):
            if self.cancel_event.is_set():
                break
            prefix = f"File {i + 1} of {n} · " if n > 1 else ""
            status = lambda text, p=prefix: self.post(self._set_status, p + text)
            self.post(self._begin_file, input_path, i, n)
            try:
                result = self.engine.transcribe(
                    input_path, opts["model"], opts["device"], opts["language"], self.cancel_event,
                    on_status=status,
                    on_segment=lambda seg, p, dur, i=i: self.post(self._on_segment, seg, (i + p) / n, dur),
                    word_timestamps=opts["speakers"] != "Off",
                )
                if opts["speakers"] != "Off" and result.segments:
                    self.engine.label_speakers(input_path, result, opts["speakers"], self.cancel_event, status)
                    self.post(self._show_speaker_transcript, result, n > 1)
                self.post(self._on_transcribed, result)
                if opts["prompt"] and result.text:
                    result.summary = self.engine.summarize(result.speaker_text, opts["prompt"], self.cancel_event,
                                                           status, opts["summary_model"])
                    self.post(self._show_summary, result.summary, os.path.basename(input_path) if n > 1 else None)
                status("Saving…")
                written = engine.save_output(result, output_path, opts["fmt"], opts["prompt"])
                self.post(self._file_done, written, result)
            except engine.Cancelled:
                break
            except Exception as exc:
                log.exception("Processing failed for %s", input_path)
                failures.append((input_path, exc))
                if n == 1:
                    self.post(self._on_failed, exc)
                    return
        if self.cancel_event.is_set():
            log.info("Job cancelled")
            self.post(self._on_cancelled)
        else:
            self.post(self._on_finished, n, failures, time.time() - started)

    def _set_status(self, text):
        self.status_label.configure(text=text, text_color=TEXT)

    def _begin_file(self, path, index, total):
        self._seg_t0 = None
        if total > 1:
            if index:
                self._append(self.transcript_box, "\n")
            self._append(self.transcript_box, f"▍{os.path.basename(path)}\n", "heading")

    def _on_segment(self, seg, progress, duration):
        now = time.time()
        if self._seg_t0 is None:
            self._seg_t0 = (now, progress)
        t0, p0 = self._seg_t0
        self.progress.set(progress)
        self.pct_label.configure(text=f"{progress * 100:.0f}%")
        if progress - p0 > 0.01:
            remaining = (now - t0) / (progress - p0) * (1 - progress)
            self.meta_label.configure(text=f"About {human_duration(remaining)} left")
        self._append(self.transcript_box, f"[{engine.format_timestamp(seg.start)}] ", "ts")
        self._append(self.transcript_box, seg.text.strip() + "\n")

    def _show_speaker_transcript(self, result, batch):
        """Replace the streamed text for this file with the speaker-labelled version."""
        box = self.transcript_box
        if not batch:
            self._reset_box(box)
        else:  # drop everything after this file's heading
            box.configure(state="normal")
            start = box.search("▍", "end", backwards=True)
            if start:
                box.delete(f"{start} lineend", "end")
            box.configure(state="disabled")
            self._append(box, "\n")
        current = None
        for i, s in enumerate(result.segments):
            if s.speaker != current:
                current = s.speaker
                idx = int(s.speaker.split()[-1]) - 1 if s.speaker else 0
                if i:
                    self._append(box, "\n\n")
                self._append(box, f"[{engine.format_timestamp(s.start)}] ", "ts")
                self._append(box, f"{s.speaker}\n", f"spk{idx % len(SPEAKER_COLORS)}")
            self._append(box, s.text.strip() + " ")
        self._append(box, "\n")

    def _on_transcribed(self, result):
        extra = f"  ·  {result.num_speakers} speaker(s)" if result.num_speakers else ""
        self.meta_label.configure(text=self._result_summary(result) + extra)
        if not result.segments:
            self._append(self.transcript_box, "No speech was detected in this file.\n", "muted")

    @staticmethod
    def _result_summary(result):
        return f"{result.language.upper()}  ·  {result.confidence:.0f}% confidence  ·  {result.device.upper()}"

    def _show_summary(self, summary, title=None):
        if title is None:
            self._reset_box(self.summary_box)
        else:
            self._append(self.summary_box, f"▍{title}\n", "heading")
        self._append_markdown(self.summary_box, summary + ("\n\n" if title else ""))
        self.tabs.set("Summary")

    def _append_markdown(self, box, text):
        """Show LLM output without raw markdown: **bold** and '# headings' become bold text."""
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip("#").strip() if line.startswith("#") else None
            if stripped is not None:
                self._append(box, stripped + "\n", "bold")
                continue
            for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", line)):
                self._append(box, part, "bold" if i % 2 else None)

    def _file_done(self, written, result):
        self.written_files += written

    def _on_finished(self, n, failures, elapsed):
        self._finish()
        self.progress.set(1)
        self.pct_label.configure(text="100%")
        if failures and len(failures) == n:
            self.status_label.configure(text="All files failed", text_color=DANGER)
        elif failures:
            self.status_label.configure(text=f"✓ Done · {len(failures)} of {n} failed", text_color=DANGER)
        else:
            self.status_label.configure(text=f"✓ Done in {human_duration(elapsed)}", text_color=ACCENT)
        saved = [f for f in self.written_files if not f.endswith("_summary.txt")]
        if n == 1 and saved:
            self.meta_label.configure(text=f"Saved {os.path.basename(saved[0])}  ·  " + self.meta_label.cget("text"))
        elif n > 1:
            self.meta_label.configure(text=f"Saved {len(saved)} transcript(s) next to the audio files")
        if failures:
            details = "\n".join(f"• {os.path.basename(p)}: {e}" for p, e in failures[:6])
            messagebox.showwarning("Some files failed", f"{details}\n\nSee logs/whisperscribe.log for details.")
        for b in self.result_btns:
            b.configure(state="normal")
        self.bell()
        if self.open_var.get() and n == 1:
            self._open_output()

    def _on_cancelled(self):
        self._finish()
        self.progress.set(0)
        self.pct_label.configure(text="")
        self.status_label.configure(text="Cancelled", text_color=MUTED)
        done = len([f for f in self.written_files if not f.endswith("_summary.txt")])
        self.meta_label.configure(text=f"{done} file(s) were finished before cancelling." if done
                                  else "Nothing was saved.")
        if done:
            for b in self.result_btns:
                b.configure(state="normal")

    def _on_failed(self, exc):
        self._finish()
        self.status_label.configure(text="Something went wrong", text_color=DANGER)
        self.meta_label.configure(text=f"Details are in {os.path.join('logs', 'whisperscribe.log')}")
        hint = ""
        if engine._looks_like_gpu_error(exc):
            hint = "\n\nTip: set Processing device to CPU in the sidebar and try again."
        messagebox.showerror("Processing failed", f"{exc}{hint}")

    def _finish(self):
        self.running = False
        self.cancel_event.clear()
        self._set_controls_running(False)

    def cancel_processing(self):
        if not self.running or self.cancel_event.is_set():
            return
        if messagebox.askyesno("Cancel transcription?", "Stop the current transcription? Progress will be lost."):
            log.info("Cancel requested by user")
            self.cancel_event.set()
            self.status_label.configure(text="Cancelling… (finishing current step)")
            self.cancel_btn.configure(state="disabled")

    def _set_controls_running(self, running):
        self.start_btn.configure(state="disabled" if running else "normal",
                                 text="Working…" if running else "Transcribe")
        self.cancel_btn.configure(state="normal" if running else "disabled",
                                  text_color=DANGER if running else MUTED,
                                  border_color=DANGER if running else BORDER)
        self.dz_clear.configure(state="disabled" if running else "normal")
        for b in self.result_btns[1:]:
            b.configure(state="disabled")

    # ------------------------------------------------------------ results
    def _reset_box(self, box):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.configure(state="disabled")

    def _append(self, box, text, tag=None):
        if not text:
            return
        box.configure(state="normal")
        box.insert("end", text, tag) if tag else box.insert("end", text)
        box.see("end")
        box.configure(state="disabled")

    def _copy_result(self):
        box = self.transcript_box if self.tabs.get() == "Transcript" else self.summary_box
        self.clipboard_clear()
        self.clipboard_append(box.get("1.0", "end-1c"))
        self.meta_label.configure(text=f"{self.tabs.get()} copied to clipboard.", text_color=MUTED)

    def _open_output(self):
        target = next((f for f in reversed(self.written_files) if not f.endswith("_summary.txt")), None)
        if target and os.path.exists(target):
            try:
                os.startfile(target)
            except Exception:
                log.error("Failed to open output file", exc_info=True)

    def _show_in_folder(self):
        if self.written_files:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.written_files[0])])

    def _on_close(self):
        if self.running and not messagebox.askyesno("Quit WhisperScribe?", "A transcription is still running. Quit anyway?"):
            return
        self.cancel_event.set()
        self._save_settings()
        self.destroy()


def selftest(audio_path, speakers="Auto", prompt="List the action items."):
    """Headless end-to-end check, mainly for frozen builds: `WhisperScribe.exe --selftest file.wav`.
    Writes the result next to the audio and a PASS/FAIL line to the log; exit code 0 on success."""
    ev = threading.Event()
    eng = engine.Engine()
    try:
        t = time.time()
        res = eng.transcribe(audio_path, engine.DEFAULT_MODEL, "Auto", "Auto-detect", ev, log.info, lambda *a: None,
                             word_timestamps=speakers != "Off")
        if speakers != "Off":
            eng.label_speakers(audio_path, res, speakers, ev, log.info)
        if prompt:
            res.summary = eng.summarize(res.speaker_text, prompt, ev, log.info)
        out = engine.save_output(res, output_path_for(audio_path, "Plain text (.txt)"), "Plain text (.txt)", prompt)
        log.info("SELFTEST PASS in %.1fs: device=%s speakers=%d segments=%d summary_chars=%d -> %s", time.time() - t,
                 res.device, res.num_speakers, len(res.segments), len(res.summary), out)
        return 0
    except Exception:
        log.exception("SELFTEST FAIL")
        return 1


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    log.info("App started.")
    app = TranscriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
