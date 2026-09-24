import json
import logging
import os
import queue
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

APP_NAME = "WhisperScribe"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "logs")
SETTINGS_PATH = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP_NAME, "settings.json")

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

PROMPT_PRESETS = {
    "Summary": "Summarize this recording in a short paragraph, followed by 3-5 key takeaways as bullet points.",
    "Action items": "List every action item, decision, and owner mentioned. Use bullet points.",
    "Key points": "Extract the main topics discussed as concise bullet points.",
    "Notes": "Write structured meeting notes with sections: Overview, Discussion, Decisions, Action Items.",
}

DEFAULT_SETTINGS = {
    "language": "English",
    "model": "Base",
    "device": "Auto",
    "format": "Plain text (.txt)",
    "open_when_done": True,
    "appearance": "Dark",
    "last_dir": "",
}


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            settings.update(json.load(f))
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


class TranscriberApp(ctk.CTk):
    def __init__(self):
        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings["appearance"])
        ctk.set_default_color_theme("green")
        super().__init__(fg_color=BG)

        self.title(APP_NAME)
        self.geometry("1240x780")
        self.minsize(1040, 680)
        icon_path = os.path.join(APP_DIR, "icon.ico")
        if os.path.exists(icon_path):
            # CTk resets the icon shortly after start-up on Windows, so set it again after a delay.
            self.after(250, lambda: self.iconbitmap(icon_path))

        self.engine = engine.Engine()
        self.ui_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.running = False
        self.input_path = ""
        self.output_auto = True
        self.written_files = []
        self._seg_t0 = None
        self._drag_leave_job = None
        self._hover_leave_job = None

        self.f_brand = ctk.CTkFont("Segoe UI", 22, "bold")
        self.f_h1 = ctk.CTkFont("Segoe UI", 20, "bold")
        self.f_h2 = ctk.CTkFont("Segoe UI", 14, "bold")
        self.f_body = ctk.CTkFont("Segoe UI", 13)
        self.f_small = ctk.CTkFont("Segoe UI", 12)
        self.f_mono = ctk.CTkFont("Cascadia Mono", 12)

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

        ctk.CTkLabel(sb, text="🎙  " + APP_NAME, font=self.f_brand, text_color=TEXT).pack(anchor="w", padx=22, pady=(28, 0))
        ctk.CTkLabel(sb, text="Private, on-device transcription", font=self.f_small, text_color=MUTED).pack(
            anchor="w", padx=22, pady=(2, 24))

        self.lang_var = self._sidebar_option(sb, "Language", list(engine.LANGUAGES), "language")
        self.model_var = self._sidebar_option(sb, "Model", list(engine.MODELS), "model",
                                              hint="Bigger = more accurate, slower")
        self.device_var = self._sidebar_option(sb, "Processing device", engine.DEVICES, "device")
        self.format_var = self._sidebar_option(sb, "Output format", list(engine.OUTPUT_FORMATS), "format",
                                               command=lambda _: self._on_format_change())

        self.open_var = tk.BooleanVar(value=self.settings["open_when_done"])
        ctk.CTkSwitch(sb, text="Open file when finished", variable=self.open_var, font=self.f_body,
                      progress_color=ACCENT, command=self._save_settings).pack(anchor="w", padx=24, pady=(8, 0))

        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=24, pady=24)
        self.device_label = ctk.CTkLabel(bottom, text="● Checking hardware…", font=self.f_small, text_color=MUTED)
        self.device_label.pack(anchor="w", pady=(0, 10))
        self.appearance = ctk.CTkSegmentedButton(bottom, values=["Light", "Dark", "System"], font=self.f_small,
                                                 selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
                                                 command=self._on_appearance)
        self.appearance.set(self.settings["appearance"])
        self.appearance.pack(fill="x")

    def _sidebar_option(self, parent, label, values, key, hint=None, command=None):
        ctk.CTkLabel(parent, text=label.upper(), font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=MUTED).pack(
            anchor="w", padx=24)
        value = self.settings.get(key)
        var = tk.StringVar(value=value if value in values else DEFAULT_SETTINGS[key])

        def on_change(choice):
            self._save_settings()
            if command:
                command(choice)

        ctk.CTkOptionMenu(parent, values=values, variable=var, font=self.f_body, dropdown_font=self.f_body,
                          fg_color=FIELD, button_color=BORDER, button_hover_color=MUTED, text_color=TEXT,
                          height=34, corner_radius=8, command=on_change).pack(fill="x", padx=24, pady=(4, 0))
        if hint:
            ctk.CTkLabel(parent, text=hint, font=self.f_small, text_color=MUTED).pack(anchor="w", padx=24)
        ctk.CTkFrame(parent, height=14, fg_color="transparent").pack()
        return var

    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14, border_width=1, border_color=BORDER, **kw)

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
        dz = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=2, border_color=BORDER, height=190)
        self.drop_zone = dz
        inner = ctk.CTkFrame(dz, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        self.dz_icon = ctk.CTkLabel(inner, text="⇪", font=ctk.CTkFont("Segoe UI Symbol", 40), text_color=ACCENT)
        self.dz_icon.pack()
        self.dz_title = ctk.CTkLabel(inner, text="Drop an audio or video file here", font=self.f_h2,
                                     text_color=TEXT)
        self.dz_title.pack(pady=(4, 0))
        self.dz_sub = ctk.CTkLabel(inner, text="or click to browse  ·  MP3, M4A, WAV, FLAC, OGG, MP4, MKV…",
                                   font=self.f_small, text_color=MUTED, wraplength=380)
        self.dz_sub.pack(pady=(2, 0))
        self.dz_clear = ctk.CTkButton(inner, text="Remove", width=80, height=26, font=self.f_small,
                                      fg_color="transparent", border_width=1, border_color=BORDER, text_color=MUTED,
                                      hover_color=BORDER, command=self._clear_input)

        for w in (dz, inner, self.dz_icon, self.dz_title, self.dz_sub):
            w.bind("<Button-1>", lambda e: self.browse_input())
            w.bind("<Enter>", lambda e: self._set_drop_highlight(True, hover=True))
            w.bind("<Leave>", lambda e: self._set_drop_highlight(False, hover=True))
            try:
                w.configure(cursor="hand2")
            except (ValueError, tk.TclError):
                pass
        return dz

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
        ctk.CTkButton(card, text="Change…", width=90, height=34, font=self.f_body, fg_color="transparent",
                      border_width=1, border_color=BORDER, text_color=TEXT, hover_color=BORDER,
                      command=self.browse_output).grid(row=0, column=2, padx=12, pady=12)
        return card

    def _build_prompt(self, parent):
        card = self._card(parent)
        ctk.CTkLabel(card, text="AI instructions", font=self.f_h2, text_color=TEXT, height=20).pack(
            anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(card, text="Optional · a local LLM runs your instruction on the transcript", font=self.f_small,
                     text_color=MUTED, height=18).pack(anchor="w", padx=16)

        chips = ctk.CTkFrame(card, fg_color="transparent")
        chips.pack(fill="x", padx=16, pady=(8, 0))
        for name, text in PROMPT_PRESETS.items():
            ctk.CTkButton(chips, text=name, height=26, width=10, corner_radius=13, font=self.f_small,
                          fg_color="transparent", border_width=1, border_color=BORDER, text_color=TEXT,
                          hover_color=ACCENT_SOFT, command=lambda t=text: self._set_prompt(t)).pack(side="left",
                                                                                              padx=(0, 6))

        self.prompt_box = ctk.CTkTextbox(card, height=70, font=self.f_body, fg_color=FIELD, border_width=1,
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
        self.cancel_btn = ctk.CTkButton(row, text="Cancel", width=110, height=46, corner_radius=10,
                                        font=ctk.CTkFont("Segoe UI", 15, "bold"), fg_color="transparent",
                                        border_width=1, border_color=BORDER, text_color=MUTED,
                                        hover_color=DANGER_HOVER, state="disabled", command=self.cancel_processing)
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
        self.meta_label = ctk.CTkLabel(card, text="Tip: press Ctrl+O to open a file, Ctrl+Enter to start.",
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
            b = ctk.CTkButton(bar, text=text, width=10, height=32, font=self.f_body, fg_color="transparent",
                              border_width=1, border_color=BORDER, text_color=TEXT, hover_color=BORDER,
                              state="disabled", command=cmd)
            b.pack(side="left", padx=(0, 8))
            self.result_btns.append(b)
        self.result_btns[0].configure(state="normal")
        return card

    def _result_box(self, tab, placeholder):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        box = ctk.CTkTextbox(tab, font=self.f_body, fg_color=FIELD, border_width=0, corner_radius=10, wrap="word")
        box.grid(row=0, column=0, sticky="nsew")
        box.tag_config("ts", foreground="#10B981")
        box.tag_config("placeholder", foreground="#8B93A1")
        box.insert("end", placeholder, "placeholder")
        box.configure(state="disabled")
        return box

    # ------------------------------------------------------------ drag & drop
    def _enable_drag_and_drop(self):
        if not HAS_DND:
            log.info("tkinterdnd2 not installed; drag & drop disabled")
            self.dz_title.configure(text="Click to choose an audio or video file")
            return
        try:
            TkinterDnD.require(self)
        except Exception:
            log.warning("Could not initialise drag & drop", exc_info=True)
            self.dz_title.configure(text="Click to choose an audio or video file")
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
        files = [f for f in self.tk.splitlist(event.data) if os.path.isfile(f)]
        supported = [f for f in files if f.lower().endswith(engine.AUDIO_EXTENSIONS)]
        if not supported:
            self._flash_drop_error("That file type isn't supported — try MP3, M4A, WAV, MP4…")
        else:
            self._set_input(supported[0])
            if len(files) > 1:
                self.meta_label.configure(text=f"{len(files)} files dropped — using the first one.")
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
        path = filedialog.askopenfilename(title="Select audio or video file",
                                          initialdir=self.settings.get("last_dir") or None,
                                          filetypes=[("Audio / video", exts), ("All files", "*.*")])
        if path:
            self._set_input(path)

    def _set_input(self, path):
        self.input_path = os.path.normpath(path)
        self.settings["last_dir"] = os.path.dirname(self.input_path)
        self._save_settings()
        size = os.path.getsize(self.input_path)
        self.dz_icon.configure(text="♫")
        self.dz_title.configure(text=os.path.basename(self.input_path))
        folder = os.path.basename(os.path.dirname(self.input_path)) or os.path.dirname(self.input_path)
        self.dz_sub.configure(text=f"{human_size(size)}  ·  in {folder}")
        self.dz_clear.pack(pady=(10, 0))
        if self.output_auto or not self.output_var.get():
            self.output_auto = True
            self.output_var.set(self._default_output())
        self.status_label.configure(text="Ready to transcribe", text_color=TEXT)
        self.meta_label.configure(text="Press Transcribe (or Ctrl+Enter) to start.", text_color=MUTED)

    def _clear_input(self):
        if self.running:
            return
        self.input_path = ""
        self.dz_icon.configure(text="⇪")
        self.dz_title.configure(text="Drop an audio or video file here")
        self.dz_sub.configure(text="or click to browse  ·  MP3, M4A, WAV, FLAC, OGG, MP4, MKV…")
        self.dz_clear.pack_forget()
        if self.output_auto:
            self.output_var.set("")

    def _default_output(self):
        ext = engine.OUTPUT_FORMATS[self.format_var.get()]
        return os.path.splitext(self.input_path)[0] + ext

    def _on_format_change(self):
        current = self.output_var.get()
        if self.output_auto and self.input_path:
            self.output_var.set(self._default_output())
        elif current:
            self.output_var.set(os.path.splitext(current)[0] + engine.OUTPUT_FORMATS[self.format_var.get()])

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
                format=self.format_var.get(), open_when_done=bool(self.open_var.get()),
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
        if not self.input_path or not os.path.isfile(self.input_path):
            self._flash_drop_error("Choose an audio file first — drop it in or click the box.")
            return
        output = self.output_var.get().strip()
        if not output:
            output = self._default_output()
            self.output_var.set(output)
        if os.path.exists(output) and not messagebox.askyesno(
                "Overwrite file?", f"{os.path.basename(output)} already exists.\n\nReplace it?"):
            return
        if not os.path.isdir(os.path.dirname(os.path.abspath(output))):
            messagebox.showwarning("Invalid location", "The output folder doesn't exist.")
            return

        self.running = True
        self.cancel_event.clear()
        self.written_files = []
        self._seg_t0 = None
        self._set_controls_running(True)
        self._reset_box(self.transcript_box)
        self._reset_box(self.summary_box)
        self.tabs.set("Transcript")
        self.progress.set(0)
        self.pct_label.configure(text="0%")
        self.meta_label.configure(text="", text_color=MUTED)

        job = dict(
            input_path=self.input_path, output_path=output, fmt=self.format_var.get(),
            prompt=self.prompt_box.get("1.0", "end-1c").strip(), language=self.lang_var.get(),
            model=self.model_var.get(), device=self.device_var.get(),
        )
        log.info("Starting job: %s", {k: v for k, v in job.items() if k != "prompt"})
        threading.Thread(target=self._worker, kwargs=job, daemon=True).start()

    def _worker(self, input_path, output_path, fmt, prompt, language, model, device):
        started = time.time()
        status = lambda text: self.post(self._set_status, text)
        try:
            result = self.engine.transcribe(
                input_path, model, device, language, self.cancel_event,
                on_status=status,
                on_segment=lambda seg, p, dur: self.post(self._on_segment, seg, p, dur),
            )
            self.post(self._on_transcribed, result)
            if prompt and result.text:
                result.summary = self.engine.summarize(result.text, prompt, self.cancel_event, status)
                self.post(self._show_summary, result.summary)
            status("Saving…")
            written = engine.save_output(result, output_path, fmt, prompt)
            self.post(self._on_finished, result, written, time.time() - started)
        except engine.Cancelled:
            log.info("Job cancelled")
            self.post(self._on_cancelled)
        except Exception as exc:
            log.exception("Processing failed")
            self.post(self._on_failed, exc)

    def _set_status(self, text):
        self.status_label.configure(text=text, text_color=TEXT)

    def _on_segment(self, seg, progress, duration):
        now = time.time()
        if self._seg_t0 is None:
            self._seg_t0 = (now, progress)
        t0, p0 = self._seg_t0
        self.progress.set(progress)
        self.pct_label.configure(text=f"{progress * 100:.0f}%")
        if progress - p0 > 0.01:
            remaining = (now - t0) / (progress - p0) * (1 - progress)
            self.meta_label.configure(
                text=f"About {human_duration(remaining)} left  ·  {human_duration(seg.end)} of {human_duration(duration)}")
        self._append(self.transcript_box, f"[{engine.format_timestamp(seg.start)}] ", "ts")
        self._append(self.transcript_box, seg.text.strip() + "\n")

    def _on_transcribed(self, result):
        self.progress.set(1)
        self.pct_label.configure(text="100%")
        self.meta_label.configure(text=self._result_summary(result))
        if not result.segments:
            self._append(self.transcript_box, "No speech was detected in this file.", "placeholder")

    @staticmethod
    def _result_summary(result):
        return f"{result.language.upper()}  ·  {result.confidence:.0f}% confidence  ·  {result.device.upper()}"

    def _show_summary(self, summary):
        self._reset_box(self.summary_box)
        self._append(self.summary_box, summary)
        self.tabs.set("Summary")

    def _on_finished(self, result, written, elapsed):
        self.written_files = written
        self._finish()
        self.status_label.configure(text=f"✓ Done in {human_duration(elapsed)}", text_color=ACCENT)
        self.meta_label.configure(text=f"Saved {os.path.basename(written[0])}  ·  {self._result_summary(result)}")
        for b in self.result_btns:
            b.configure(state="normal")
        self.bell()
        if self.open_var.get():
            self._open_output()

    def _on_cancelled(self):
        self._finish()
        self.progress.set(0)
        self.pct_label.configure(text="")
        self.status_label.configure(text="Cancelled", text_color=MUTED)
        self.meta_label.configure(text="Nothing was saved.")

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
        box.configure(state="normal")
        box.insert("end", text, tag) if tag else box.insert("end", text)
        box.see("end")
        box.configure(state="disabled")

    def _copy_result(self):
        box = self.transcript_box if self.tabs.get() == "Transcript" else self.summary_box
        text = box.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.meta_label.configure(text=f"{self.tabs.get()} copied to clipboard.", text_color=MUTED)

    def _open_output(self):
        if self.written_files and os.path.exists(self.written_files[0]):
            try:
                os.startfile(self.written_files[0])
            except Exception:
                log.error("Failed to open output file", exc_info=True)

    def _show_in_folder(self):
        if self.written_files:
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.written_files[0])])

    def _on_close(self):
        if self.running and not messagebox.askyesno("Quit WhisperScribe?", "A transcription is still running. Quit anyway?"):
            return
        self.cancel_event.set()
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    log.info("App started.")
    app = TranscriberApp()
    app.mainloop()
