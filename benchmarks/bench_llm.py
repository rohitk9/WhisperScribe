"""Compare local summary LLMs: speed, VRAM and how many known facts they capture.

Usage:
    python benchmarks/bench_llm.py --transcript meeting.txt --facts facts.json [--only "Qwen3 4B Instruct"]

facts.json is a list of fact checks; each check is a list of alternative keywords, e.g.
    [["marcus"], ["friday"], ["39", "thirty nine"]]
A fact counts as captured if any alternative appears in the model's answer.
"""
import argparse
import gc
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine  # noqa: E402
import torch  # noqa: E402

# Everything that was compared (the app menu only keeps the winners). Registered into engine for the run.
CANDIDATES = {
    "Qwen2.5 0.5B": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "chunk_chars": 6000},
    "Qwen2.5 1.5B": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "chunk_chars": 16000},
    "Qwen3 1.7B": {"id": "Qwen/Qwen3-1.7B", "chunk_chars": 24000},
    "Qwen2.5 3B": {"id": "Qwen/Qwen2.5-3B-Instruct", "chunk_chars": 24000},
    "Phi-4 mini (3.8B)": {"id": "microsoft/Phi-4-mini-instruct", "chunk_chars": 24000},
    "Qwen3 4B Instruct": {"id": "Qwen/Qwen3-4B-Instruct-2507", "chunk_chars": 24000},
    "Qwen2.5 7B (4-bit)": {"id": "Qwen/Qwen2.5-7B-Instruct", "quant": "4bit", "chunk_chars": 24000},
}
engine.SUMMARY_MODELS.update(CANDIDATES)

TASKS = {
    "action_items": "List every action item, decision, and owner mentioned. Use bullet points.",
    "notes": "Write structured meeting notes with sections: Overview, Discussion, Decisions, Action Items.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--long-repeat", type=int, default=12, help="repeat transcript N times for a long-input timing")
    ap.add_argument("--out", default="bench_llm.json")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    text = open(args.transcript, encoding="utf-8").read().strip()
    facts = json.load(open(args.facts, encoding="utf-8"))
    long_text = "\n".join(f"Meeting segment {i + 1}. {text}" for i in range(args.long_repeat))
    cancel = threading.Event()
    rows = []

    for name in CANDIDATES:
        if args.only and name not in args.only:
            continue
        eng = engine.Engine()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            summ = eng._load_summarizer(name)
        except Exception as exc:
            print(f"{name}: failed to load: {exc}")
            continue
        load = time.perf_counter() - t
        eng._ask(summ, "Say OK.", "OK", max_new_tokens=4)  # warm-up
        tok = summ[0]

        row = {"model": name, "load_s": round(load, 1), "outputs": {}}
        recall, gen_times, tps = [], [], []
        for task, prompt in TASKS.items():
            t = time.perf_counter()
            out = eng._ask(summ, prompt, text)
            dt = time.perf_counter() - t
            n_tok = len(tok(out)["input_ids"])
            low = out.lower()
            hits = [any(k in low for k in alts) for alts in facts]
            recall.append(sum(hits) / len(hits))
            gen_times.append(dt)
            tps.append(n_tok / dt)
            row["outputs"][task] = out
            row[f"{task}_missed"] = [alts[0] for alts, h in zip(facts, hits) if not h]

        t = time.perf_counter()
        long_out = eng.summarize(long_text, TASKS["action_items"], cancel, lambda s: None, name)
        row["long_s"] = round(time.perf_counter() - t, 1)
        row["long_chars"] = len(long_text)
        row["long_out_tokens"] = len(tok(long_out)["input_ids"])
        row["outputs"]["long"] = long_out
        row.update(fact_recall=round(sum(recall) / len(recall) * 100), gen_s=round(sum(gen_times) / len(gen_times), 1),
                   tok_per_s=round(sum(tps) / len(tps)), vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "outputs"}), flush=True)
        del summ, eng
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n{'model':24} {'facts':>6} {'short':>7} {'long':>7} {'tok/s':>6} {'VRAM':>6} {'load':>6}")
    for r in rows:
        print(f"{r['model']:24} {r['fact_recall']:>5}% {r['gen_s']:>6}s {r['long_s']:>6}s {r['tok_per_s']:>6} "
              f"{r['vram_gb']:>5}G {r['load_s']:>5}s")


if __name__ == "__main__":
    main()
