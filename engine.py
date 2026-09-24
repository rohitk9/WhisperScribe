"""Transcription / summarization engine for WhisperScribe.

Kept free of any UI code so it can be reused from scripts or tested on its own.
"""
import gc
import logging
import math
import os
import site
import sys
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
# Benchmarked on an RTX 5080 (see docs/BENCHMARKS.md): Turbo is as fast as Base and as accurate as Large v3.
MODELS = {
    "Large v3 Turbo · best": "large-v3-turbo",
    "Large v3": "large-v3",
    "Medium": "medium",
    "Small": "small",
    "Base": "base",
    "Tiny (fastest)": "tiny",
}
DEFAULT_MODEL = "Large v3 Turbo · best"

DEVICES = ["Auto", "GPU (CUDA)", "CPU"]

OUTPUT_FORMATS = {
    "Plain text (.txt)": ".txt",
    "Timestamped text (.txt)": ".txt",
    "Subtitles (.srt)": ".srt",
}

# Display name -> local LLM used for "AI instructions". Chosen from 7 candidates, see docs/BENCHMARKS.md.
# chunk_chars: how much transcript each call sees. quant "4bit" needs an NVIDIA GPU (bitsandbytes).
SUMMARY_MODELS = {
    "Qwen2.5 7B · best": {"id": "Qwen/Qwen2.5-7B-Instruct", "quant": "4bit", "chunk_chars": 24000},
    "Qwen3 1.7B · fast": {"id": "Qwen/Qwen3-1.7B", "chunk_chars": 24000},
    "Qwen2.5 1.5B · light": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "chunk_chars": 16000},
}
DEFAULT_SUMMARY_MODEL = "Qwen2.5 7B · best"
CPU_SUMMARY_FALLBACK = "Qwen3 1.7B · fast"  # 4-bit models can't run without CUDA


def split_chunks(text: str, size: int) -> list:
    """Split text into pieces of at most `size` chars, preferring sentence boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if cut <= start + size // 2:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


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
        if getattr(sys, "frozen", False):  # PyInstaller bundle: nvidia/*/bin is copied next to the app
            packages.append(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
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
    speaker: str = ""
    words: list = field(default_factory=list, repr=False)  # [(start, end, word)] when word timestamps are on


def split_on_pauses(segments, min_gap=0.3, min_len=0.8):
    """Split segments at pauses / sentence ends so a single segment rarely spans two speakers."""
    out = []
    for seg in segments:
        if len(seg.words) < 2:
            out.append(seg)
            continue
        pieces, cur = [], [seg.words[0]]
        for prev, word in zip(seg.words, seg.words[1:]):
            gap = word[0] - prev[1]
            if gap >= min_gap or (prev[2].rstrip().endswith((".", "?", "!")) and gap >= 0.1):
                pieces.append(cur)
                cur = []
            cur.append(word)
        pieces.append(cur)
        # Too-short pieces carry little voice information: glue each to the neighbour it's closest to in time.
        while len(pieces) > 1:
            short = [i for i, p in enumerate(pieces) if p[-1][1] - p[0][0] < min_len]
            if not short:
                break
            i = short[0]
            gap_prev = pieces[i][0][0] - pieces[i - 1][-1][1] if i > 0 else float("inf")
            gap_next = pieces[i + 1][0][0] - pieces[i][-1][1] if i < len(pieces) - 1 else float("inf")
            j = i - 1 if gap_prev <= gap_next else i
            pieces[j:j + 2] = [pieces[j] + pieces[j + 1]]
        for p in pieces:
            out.append(Segment(p[0][0], p[-1][1], "".join(w[2] for w in p).strip(), words=p))
    return out


@dataclass
class Result:
    segments: list = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    confidence: float = 0.0
    device: str = ""
    summary: str = ""
    num_speakers: int = 0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def speaker_text(self) -> str:
        """Transcript with "Speaker N:" prefixes on each change of speaker (plain text if no speakers)."""
        if not any(s.speaker for s in self.segments):
            return self.text
        lines, current = [], None
        for s in self.segments:
            if s.speaker != current:
                lines.append(f"\n{s.speaker}: {s.text.strip()}")
                current = s.speaker
            else:
                lines[-1] += " " + s.text.strip()
        return "\n".join(line.strip() for line in lines).strip()


# ---------- Speaker labels ----------
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
SPEAKER_COUNTS = ["Off", "Auto", "2", "3", "4", "5", "6"]


def _merge_similar_clusters(X, labels, merge_sim):
    """Merge clusters whose mean voices are near-identical (silhouette alone can't detect "one speaker")."""
    import numpy as np

    labels = np.array(labels)
    while True:
        ids = sorted(set(labels.tolist()))
        if len(ids) < 2:
            return labels
        C = np.array([X[labels == i].mean(0) for i in ids])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ C.T
        np.fill_diagonal(S, -1)
        a, b = np.unravel_index(S.argmax(), S.shape)
        if S[a, b] < merge_sim:
            return labels
        labels[labels == ids[b]] = ids[a]


def _absorb_tiny_clusters(X, labels, min_share=0.05):
    """A "speaker" with only one or two odd segments is almost always an outlier of a real speaker."""
    import numpy as np

    labels = np.array(labels)
    min_size = max(2, int(round(len(labels) * min_share)))
    ids, counts = np.unique(labels, return_counts=True)
    big = [i for i, c in zip(ids, counts) if c >= min_size]
    if not big or len(big) == len(ids):
        return labels
    C = np.array([X[labels == i].mean(0) for i in big])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    for i in set(ids) - set(big):
        rows = np.where(labels == i)[0]
        labels[rows] = [big[int((C @ X[r]).argmax())] for r in rows]
    return labels


def cluster_speakers(embeddings, num_speakers=None, max_speakers=6, merge_sim=0.80):
    """Cluster L2-normalised embeddings. Returns a label per row, numbered by first appearance.

    WavLM calibration (docs/BENCHMARKS.md): same-voice segments ~0.93 cosine similarity, different voices ~0.4,
    so in auto mode clusters whose centroids are more similar than `merge_sim` are treated as one speaker.
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X = np.asarray(embeddings, dtype=np.float32)
    n = len(X)
    if n < 2:
        return [0] * n

    def fit(k):
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)

    if num_speakers:
        labels = fit(min(num_speakers, n))
    else:
        best, labels = -1.0, np.zeros(n, dtype=int)
        for k in range(2, min(max_speakers, n - 1) + 1):
            cand = fit(k)
            score = silhouette_score(X, cand, metric="cosine")
            if score > best:
                best, labels = score, cand
        if best < 0.12:  # no clear structure: a single speaker
            labels = np.zeros(n, dtype=int)
        labels = _absorb_tiny_clusters(X, _merge_similar_clusters(X, labels, merge_sim))

    order = {}
    return [order.setdefault(int(label), len(order)) for label in labels]


def smooth_labels(labels, durations, min_dur=1.0):
    """Short segments sandwiched between two turns of the same speaker usually belong to that speaker."""
    labels = list(labels)
    for i in range(1, len(labels) - 1):
        if durations[i] < min_dur and labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]
    return labels


class Engine:
    """Loads models lazily and caches them between runs."""

    def __init__(self):
        self._whisper = None
        self._whisper_key = None
        self._summarizer = None
        self._summarizer_key = None
        self._speaker_model = None
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
        word_timestamps: bool = False,
    ) -> Result:
        """Transcribe `path`. `on_segment(segment, progress_0_to_1, duration)` is called per segment."""
        model_id = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
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
                        path, beam_size=5, vad_filter=True, language=lang, initial_prompt=initial_prompt,
                        word_timestamps=word_timestamps,
                    )
                    result = Result(language=info.language, duration=info.duration, device=device)
                    weighted_conf, total_len = 0.0, 0.0
                    for seg in segments:
                        if cancel.is_set():
                            raise Cancelled()
                        emitted = True
                        s = Segment(seg.start, seg.end, seg.text,
                                    words=[(w.start, w.end, w.word) for w in (seg.words or [])])
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
    def _load_summarizer(self, model_name: str):
        import torch

        spec = SUMMARY_MODELS.get(model_name) or SUMMARY_MODELS[DEFAULT_SUMMARY_MODEL]
        if spec.get("quant") and not torch.cuda.is_available():
            log.info("No CUDA GPU: using %s instead of %s", CPU_SUMMARY_FALLBACK, model_name)
            spec = SUMMARY_MODELS[CPU_SUMMARY_FALLBACK]
        if self._summarizer_key != spec["id"] + str(spec.get("quant")):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._summarizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            kwargs = {"dtype": torch.bfloat16, "device_map": "cuda" if torch.cuda.is_available() else "cpu"}
            if spec.get("quant") == "4bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            if not torch.cuda.is_available():
                kwargs["dtype"] = torch.float32
            log.info("Loading summarizer %s (%s)", spec["id"], spec.get("quant") or "bf16")
            tokenizer = AutoTokenizer.from_pretrained(spec["id"])
            model = AutoModelForCausalLM.from_pretrained(spec["id"], **kwargs)
            model.eval()
            self._summarizer = (tokenizer, model, spec)
            self._summarizer_key = spec["id"] + str(spec.get("quant"))
        return self._summarizer

    def _ask(self, summarizer, instruction: str, text: str, max_new_tokens: int = 900) -> str:
        import torch

        tokenizer, model, spec = summarizer
        messages = [
            {"role": "system", "content": "You are a precise assistant running locally. Follow the user's "
                                          "instruction using only facts stated in the transcript. Never invent "
                                          "names, numbers or dates. Answer in the transcript's language."},
            {"role": "user", "content": f"Instruction: {instruction}\n\nTranscript:\n{text}"},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,  # Qwen3: skip the hidden "thinking" phase; ignored by other templates
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def summarize(self, text: str, prompt: str, cancel: threading.Event, on_status: Callable[[str], None],
                  model_name: str = None) -> str:
        """Run `prompt` over the transcript. Long transcripts are chunked (map → reduce) instead of truncated."""
        model_name = model_name or DEFAULT_SUMMARY_MODEL
        self.unload_whisper()  # give the LLM the whole GPU; Whisper reloads in ~2 s when needed again
        on_status(f"Loading {model_name}…")
        summarizer = self._load_summarizer(model_name)
        chunk_chars = summarizer[2].get("chunk_chars", 24000)

        chunks = split_chunks(text, chunk_chars)
        if len(chunks) == 1:
            on_status("Generating summary…")
            return self._ask(summarizer, prompt, chunks[0])

        partials = []
        for i, chunk in enumerate(chunks, 1):
            if cancel.is_set():
                raise Cancelled()
            on_status(f"Summarizing part {i} of {len(chunks)}…")
            partials.append(self._ask(summarizer, prompt, chunk, max_new_tokens=600))

        if cancel.is_set():
            raise Cancelled()
        on_status("Combining partial summaries…")
        combined = "\n\n".join(f"Part {i}:\n{p}" for i, p in enumerate(partials, 1))
        return self._ask(
            summarizer,
            f"These are notes from consecutive parts of one recording. Merge them into a single answer "
            f"to this instruction, removing duplicates: {prompt}",
            combined[:chunk_chars],
        )

    def label_speakers(self, path: str, result: "Result", speakers: str, cancel: threading.Event,
                       on_status: Callable[[str], None]):
        """Assign "Speaker N" to each segment. `speakers` is "Auto" or a number (as text)."""
        import numpy as np
        import torch
        from faster_whisper import decode_audio
        from transformers import AutoFeatureExtractor, WavLMForXVector

        result.segments = split_on_pauses(result.segments)
        segs = result.segments
        if not segs:
            return
        on_status("Identifying speakers…")
        if self._speaker_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading speaker model %s on %s", SPEAKER_MODEL, device)
            self._speaker_model = (AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL),
                                   WavLMForXVector.from_pretrained(SPEAKER_MODEL).to(device).eval())
        extractor, model = self._speaker_model

        sr = 16000
        audio = decode_audio(path, sampling_rate=sr)
        windows = []
        for s in segs:
            # Pad very short segments to 1.5 s of context; cap long ones at 10 s around the centre.
            mid, half = (s.start + s.end) / 2, max((s.end - s.start) / 2, 0.75)
            half = min(half, 5.0)
            a, b = int(max(mid - half, 0) * sr), int(min(mid + half, len(audio) / sr) * sr)
            windows.append(audio[a:b] if b > a else np.zeros(sr, dtype=np.float32))

        embeddings = []
        # torch's bundled cuDNN clashes with the CUDA-12 cuDNN that faster-whisper loads into the same
        # process (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH), so run the small conv stack without cuDNN.
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
            for i in range(0, len(windows), 16):
                if cancel.is_set():
                    raise Cancelled()
                batch = windows[i:i + 16]
                inputs = extractor(batch, sampling_rate=sr, return_tensors="pt", padding=True,
                                   return_attention_mask=True).to(model.device)
                emb = model(**inputs).embeddings
                embeddings.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
                on_status(f"Identifying speakers… {min(i + 16, len(windows))}/{len(windows)}")
        X = np.concatenate(embeddings)

        labels = cluster_speakers(X, None if speakers == "Auto" else int(speakers))
        labels = smooth_labels(labels, [s.end - s.start for s in segs])
        for s, label in zip(segs, labels):
            s.speaker = f"Speaker {label + 1}"
        result.num_speakers = len(set(labels))

    def unload_whisper(self):
        """Free VRAM held by the speech model (e.g. before loading a large LLM)."""
        self._whisper, self._whisper_key = None, None
        gc.collect()


# ---------- Output ----------
def render_output(result: Result, fmt_name: str, prompt: str = "") -> str:
    if fmt_name == "Subtitles (.srt)":
        blocks = []
        for i, s in enumerate(result.segments, 1):
            who = f"[{s.speaker}] " if s.speaker else ""
            blocks.append(
                f"{i}\n{format_timestamp(s.start, True)} --> {format_timestamp(s.end, True)}\n{who}{s.text.strip()}\n"
            )
        return "\n".join(blocks)

    parts = []
    if prompt and result.summary:
        parts += ["=== AI Summary ===", f"Prompt: {prompt}", "", result.summary, "", "=" * 50, ""]
    header = f"Language: {result.language}, Confidence: {result.confidence:.1f}%"
    if result.num_speakers:
        header += f", Speakers: {result.num_speakers}"
    parts.append(f"=== Full Transcription ({header}) ===")
    if fmt_name == "Timestamped text (.txt)":
        parts += [f"[{format_timestamp(s.start)}] " + (f"{s.speaker}: " if s.speaker else "") + s.text.strip()
                  for s in result.segments]
    else:
        parts.append(result.speaker_text)
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
