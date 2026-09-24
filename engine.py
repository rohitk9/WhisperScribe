"""Transcription / summarization engine for WhisperScribe.

Kept free of any UI code so it can be reused from scripts or tested on its own.
"""
import logging
import math
import os
import site
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("whisperscribe.engine")

AUDIO_EXTENSIONS = (
    ".m4a", ".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".opus",
    ".webm", ".mp4", ".mkv", ".mov", ".avi",
)

# Display name -> (whisper language code, initial prompt)
LANGUAGES = {
    "Auto-detect": (None, None),
    "English": ("en", None),
    "Hindi": ("hi", None),
    "Hinglish": (None, "This is a mix of Hindi and English."),
}

# Display name -> faster-whisper model id
MODELS = {
    "Tiny (fastest)": "tiny",
    "Base": "base",
    "Small": "small",
    "Medium": "medium",
    "Large v3 Turbo (best)": "large-v3-turbo",
}

DEVICES = ["Auto", "GPU (CUDA)", "CPU"]

OUTPUT_FORMATS = {
    "Plain text (.txt)": ".txt",
    "Timestamped text (.txt)": ".txt",
    "Subtitles (.srt)": ".srt",
}

SUMMARY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SUMMARY_CHUNK_CHARS = 6000


class Cancelled(Exception):
    pass


def register_cuda_dlls():
    """Make pip-installed NVIDIA DLLs (cuBLAS, cuDNN, ...) discoverable on Windows."""
    if os.name != "nt":
        return
    try:
        packages = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            packages.append(site.getusersitepackages())
        for site_pkg in packages:
            for lib_dir in ["cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"]:
                bin_path = os.path.join(site_pkg, "nvidia", lib_dir, "bin")
                if os.path.exists(bin_path):
                    os.add_dll_directory(bin_path)
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        log.warning("Could not register CUDA DLL directories", exc_info=True)


def _looks_like_gpu_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cuda", "cublas", "cudnn", "gpu", "device"))


def format_timestamp(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    segments: list = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    confidence: float = 0.0
    device: str = ""
    summary: str = ""

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


class Engine:
    """Loads models lazily and caches them between runs."""

    def __init__(self):
        self._whisper = None
        self._whisper_key = None
        self._summarizer = None
        self._lock = threading.Lock()

    # ---------- Whisper ----------
    def _load_whisper(self, model_id: str, device: str):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        key = (model_id, device)
        if self._whisper_key != key:
            log.info("Loading WhisperModel(%s, device=%s, compute_type=%s)", model_id, device, compute_type)
            self._whisper = None  # free old model before loading a new one
            self._whisper = WhisperModel(model_id, device=device, compute_type=compute_type)
            self._whisper_key = key
        return self._whisper

    def transcribe(
        self,
        path: str,
        model_name: str,
        device_choice: str,
        language_name: str,
        cancel: threading.Event,
        on_status: Callable[[str], None],
        on_segment: Callable[[Segment, float, float], None],
    ) -> Result:
        """Transcribe `path`. `on_segment(segment, progress_0_to_1, duration)` is called per segment."""
        model_id = MODELS.get(model_name, "base")
        lang, initial_prompt = LANGUAGES.get(language_name, (None, None))

        if device_choice == "CPU":
            devices = ["cpu"]
        elif device_choice == "GPU (CUDA)":
            devices = ["cuda"]
        else:
            devices = ["cuda", "cpu"]

        last_exc = None
        for device in devices:
            emitted = False
            try:
                with self._lock:
                    on_status(f"Loading {model_id} model on {device.upper()}…")
                    model = self._load_whisper(model_id, device)
                    if cancel.is_set():
                        raise Cancelled()

                    on_status("Transcribing…")
                    segments, info = model.transcribe(
                        path, beam_size=5, vad_filter=True, language=lang, initial_prompt=initial_prompt
                    )
                    result = Result(language=info.language, duration=info.duration, device=device)
                    weighted_conf, total_len = 0.0, 0.0
                    for seg in segments:
                        if cancel.is_set():
                            raise Cancelled()
                        emitted = True
                        s = Segment(seg.start, seg.end, seg.text)
                        result.segments.append(s)
                        seg_len = max(seg.end - seg.start, 0.01)
                        weighted_conf += math.exp(seg.avg_logprob) * seg_len
                        total_len += seg_len
                        progress = min(seg.end / info.duration, 1.0) if info.duration else 0.0
                        on_segment(s, progress, info.duration)
                    result.confidence = (weighted_conf / total_len * 100) if total_len else 0.0
                    return result
            except Cancelled:
                raise
            except Exception as exc:
                last_exc = exc
                # Only fall back to CPU if nothing was produced yet and it smells like a GPU problem.
                if device == "cuda" and "cpu" in devices and not emitted and _looks_like_gpu_error(exc):
                    log.warning("GPU transcription failed, falling back to CPU: %s", exc)
                    on_status("GPU unavailable — falling back to CPU…")
                    self._whisper, self._whisper_key = None, None
                    continue
                raise
        raise last_exc  # pragma: no cover

    # ---------- Summarization ----------
    def _load_summarizer(self):
        if self._summarizer is None:
            from transformers import pipeline

            log.info("Loading summarizer %s", SUMMARY_MODEL)
            self._summarizer = pipeline("text-generation", model=SUMMARY_MODEL, device_map="auto")
        return self._summarizer

    def _ask(self, summarizer, instruction: str, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant running locally. "
                                          "Follow the user's instruction using only the provided transcript."},
            {"role": "user", "content": f"Instruction: {instruction}\n\nTranscript:\n{text}"},
        ]
        out = summarizer(messages, max_new_tokens=512, return_full_text=False)
        return out[0]["generated_text"].strip()

    def summarize(self, text: str, prompt: str, cancel: threading.Event, on_status: Callable[[str], None]) -> str:
        """Run `prompt` over the transcript. Long transcripts are chunked (map → reduce) instead of truncated."""
        on_status("Loading local LLM for summarization…")
        summarizer = self._load_summarizer()

        chunks = [text[i:i + SUMMARY_CHUNK_CHARS] for i in range(0, len(text), SUMMARY_CHUNK_CHARS)] or [""]
        if len(chunks) == 1:
            on_status("Generating summary…")
            return self._ask(summarizer, prompt, chunks[0])

        partials = []
        for i, chunk in enumerate(chunks, 1):
            if cancel.is_set():
                raise Cancelled()
            on_status(f"Summarizing part {i} of {len(chunks)}…")
            partials.append(self._ask(summarizer, prompt, chunk))

        if cancel.is_set():
            raise Cancelled()
        on_status("Combining partial summaries…")
        combined = "\n\n".join(f"Part {i}:\n{p}" for i, p in enumerate(partials, 1))
        return self._ask(
            summarizer,
            f"These are notes from consecutive parts of one recording. Merge them into a single answer "
            f"to this instruction, removing duplicates: {prompt}",
            combined[:SUMMARY_CHUNK_CHARS * 2],
        )


# ---------- Output ----------
def render_output(result: Result, fmt_name: str, prompt: str = "") -> str:
    if fmt_name == "Subtitles (.srt)":
        blocks = []
        for i, s in enumerate(result.segments, 1):
            blocks.append(
                f"{i}\n{format_timestamp(s.start, True)} --> {format_timestamp(s.end, True)}\n{s.text.strip()}\n"
            )
        return "\n".join(blocks)

    parts = []
    if prompt and result.summary:
        parts += ["=== AI Summary ===", f"Prompt: {prompt}", "", result.summary, "", "=" * 50, ""]
    parts.append(f"=== Full Transcription (Language: {result.language}, Confidence: {result.confidence:.1f}%) ===")
    if fmt_name == "Timestamped text (.txt)":
        parts += [f"[{format_timestamp(s.start)}] {s.text.strip()}" for s in result.segments]
    else:
        parts.append(result.text)
    return "\n".join(parts) + "\n"


def save_output(result: Result, output_path: str, fmt_name: str, prompt: str = "") -> list:
    """Write the result; returns the list of files written."""
    written = []
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_output(result, fmt_name, prompt))
    written.append(output_path)

    # Subtitles can't hold a summary, so write it alongside.
    if fmt_name == "Subtitles (.srt)" and prompt and result.summary:
        summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n\n{result.summary}\n")
        written.append(summary_path)
    return written
