"""Compare faster-whisper models on your own hardware.

Usage:
    python benchmarks/bench_whisper.py --real long_recording.m4a --ref-audio clip.wav --ref-text clip.txt

--real       any real recording (only numbers are printed, never its text)
--ref-audio  a clip with a known transcript, used for word error rate (WER)
--ref-text   the known transcript for --ref-audio
"""
import argparse
import gc
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine  # noqa: E402

engine.register_cuda_dlls()
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio  # noqa: E402
import jiwer  # noqa: E402

CANDIDATES = [
    # (label, model id, batched)
    ("tiny", "tiny", False),
    ("base", "base", False),
    ("small", "small", False),
    ("medium", "medium", False),
    ("distil-large-v3", "distil-large-v3", False),
    ("large-v3-turbo", "large-v3-turbo", False),
    ("large-v3-turbo (batched)", "large-v3-turbo", True),
    ("large-v3", "large-v3", False),
    ("large-v3 (batched)", "large-v3", True),
]


try:  # Whisper's own normalizer maps "thirty nine dollars" and "$39" to the same thing
    from whisper.normalizers import EnglishTextNormalizer
    _normalizer = EnglishTextNormalizer()
except ImportError:
    _normalizer = None


def norm(text):
    if _normalizer:
        return _normalizer(text)
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def run(model, audio, batched, language=None):
    t = time.perf_counter()
    if batched:
        segs, info = BatchedInferencePipeline(model).transcribe(audio, batch_size=16, vad_filter=True,
                                                               language=language)
    else:
        segs, info = model.transcribe(audio, beam_size=5, vad_filter=True, language=language)
    segs = list(segs)
    elapsed = time.perf_counter() - t
    w = sum(max(s.end - s.start, 0.01) for s in segs)
    conf = sum(math.exp(s.avg_logprob) * max(s.end - s.start, 0.01) for s in segs) / w * 100 if w else 0
    return " ".join(s.text.strip() for s in segs), elapsed, conf, info.language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--out", default="bench_whisper.json")
    ap.add_argument("--only", nargs="*", help="labels to run")
    args = ap.parse_args()

    real = decode_audio(args.real)
    ref_audio = decode_audio(args.ref_audio)
    ref_text = norm(open(args.ref_text, encoding="utf-8").read())
    real_min = len(real) / 16000 / 60
    print(f"Real clip: {real_min:.1f} min, reference clip: {len(ref_audio) / 16000:.0f} s\n")

    rows, texts = [], {}
    for label, model_id, batched in CANDIDATES:
        if args.only and label not in args.only:
            continue
        t = time.perf_counter()
        model = WhisperModel(model_id, device="cuda", compute_type="float16")
        load = time.perf_counter() - t
        run(model, ref_audio[:16000 * 5], batched)  # warm-up
        hyp, _, ref_conf, _ = run(model, ref_audio, batched, language="en")
        wer = jiwer.wer(ref_text, norm(hyp)) * 100
        text, secs, conf, lang = run(model, real, batched)
        texts[label] = norm(text)
        row = dict(model=label, load_s=round(load, 1), real_s=round(secs, 1),
                   speed_x=round(real_min * 60 / secs, 1), conf_real=round(conf, 1),
                   conf_ref=round(ref_conf, 1), wer_ref=round(wer, 2), lang=lang)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        gc.collect()

    # Agreement with the largest non-batched model (pseudo ground truth for the real clip)
    anchor = "large-v3" if "large-v3" in texts else rows[-1]["model"]
    for row in rows:
        row["diff_vs_" + anchor.replace(" ", "_")] = round(jiwer.wer(texts[anchor], texts[row["model"]]) * 100, 1)
    json.dump(rows, open(args.out, "w"), indent=1)

    print(f"\n{'model':26} {'load':>6} {'time':>7} {'speed':>7} {'conf':>6} {'WER':>6} {'Δ' + anchor:>12}")
    for r in rows:
        print(f"{r['model']:26} {r['load_s']:>5}s {r['real_s']:>6}s {r['speed_x']:>6}x {r['conf_real']:>5}% "
              f"{r['wer_ref']:>5}% {r['diff_vs_' + anchor.replace(' ', '_')]:>11}%")


if __name__ == "__main__":
    main()
