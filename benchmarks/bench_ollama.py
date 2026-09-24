"""Compare Ollama models for transcript chat (Q&A) and summaries: accuracy, speed, VRAM.

Usage:
    python benchmarks/bench_ollama.py --models phi4:latest gemma4:12b qwen3.5:9b

Uses benchmarks/data/meeting_transcript.txt + meeting_qa.json by default. The chat test puts the whole transcript
in context (what a single recording's thread sees); the long test hides it inside ~25k tokens of other meetings
to check the model still finds details (what a long recording or several retrieved chunks look like).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from integrations import OllamaClient  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
SYSTEM = ("You answer questions about a meeting transcript. Use only facts stated in the transcript. If the "
          "transcript does not contain the answer, say it was not mentioned. Be concise.")
TASKS = {
    "action_items": "List every action item, decision, and owner mentioned. Use bullet points.",
    "notes": "Write structured meeting notes with sections: Overview, Discussion, Decisions, Action Items.",
}


def grade(answer, q, refusal_markers):
    low = answer.lower()
    refused = any(m in low for m in refusal_markers)
    if q.get("unanswerable"):
        return refused
    ok = all(any(k in low for k in group) for group in q["expect"])
    return ok and not any(f in low for f in q.get("forbid", []))


def filler(n_tokens):
    """Neutral 'other meeting' text to bury the real transcript in (roughly 1 token per 0.75 words)."""
    topics = ["the office move", "hiring for the support team", "the quarterly security review",
              "vendor contracts", "the mobile app roadmap", "the customer advisory board"]
    lines, i = [], 0
    while len(" ".join(lines).split()) < n_tokens * 0.75:
        t = topics[i % len(topics)]
        lines.append(f"Speaker {i % 3 + 3}: On {t}, we reviewed the current status, agreed the next update is due in "
                     f"the regular weekly sync, and noted there were no blockers raised this week for {t}.")
        i += 1
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--transcript", default=os.path.join(DATA, "meeting_transcript.txt"))
    ap.add_argument("--qa", default=os.path.join(DATA, "meeting_qa.json"))
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--out", default="bench_ollama.json")
    args = ap.parse_args()

    ollama = OllamaClient()
    transcript = open(args.transcript, encoding="utf-8").read().strip()
    qa = json.load(open(args.qa, encoding="utf-8"))
    half = filler(12000)
    long_transcript = f"{half}\n{transcript}\n{filler(12000)}"
    rows = []

    for model in args.models:
        for m in ollama.loaded():  # start each model from a clean GPU
            ollama.unload(m["name"])
        max_ctx = ollama.context_length(model)
        num_ctx = min(args.num_ctx, max_ctx) if max_ctx else args.num_ctx
        t = time.perf_counter()
        ollama.chat(model, [{"role": "user", "content": "Say OK."}], num_ctx=num_ctx, max_tokens=5)
        load_s = time.perf_counter() - t
        loaded = ollama.loaded()
        vram = sum(m.get("size_vram", 0) for m in loaded) / 1e9
        on_gpu = sum(m.get("size_vram", 0) for m in loaded) / max(sum(m.get("size", 0) for m in loaded), 1)
        row = {"model": model, "max_ctx": max_ctx, "num_ctx": num_ctx, "gpu_share": f"{on_gpu:.0%}",
               "load_s": round(load_s, 1), "vram_gb": round(vram, 1), "answers": {}}

        def ask(context, question):
            r = ollama.chat(model, [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": f"Transcript:\n{context}\n\nQuestion: {question}"}],
                            num_ctx=num_ctx)
            return r["message"]["content"].strip(), r

        # 1. chat Q&A with the transcript in context
        passed, times, tps = 0, [], []
        for q in qa["questions"]:
            t = time.perf_counter()
            ans, r = ask(transcript, q["q"])
            times.append(time.perf_counter() - t)
            tps.append(r.get("eval_count", 0) / max(r.get("eval_duration", 1) / 1e9, 1e-6))
            ok = grade(ans, q, qa["refusal_markers"])
            passed += ok
            row["answers"][q["q"]] = {"ok": ok, "answer": ans}
        row["qa_score"] = f"{passed}/{len(qa['questions'])}"
        row["qa_s"] = round(sum(times) / len(times), 1)
        row["tok_per_s"] = round(sum(tps) / len(tps))

        # 2. the same questions with the meeting buried in ~24k tokens of other material
        passed_long, t = 0, time.perf_counter()
        for q in qa["questions"]:
            ans, r = ask(long_transcript, q["q"])
            ok = grade(ans, q, qa["refusal_markers"])
            passed_long += ok
            row["answers"]["long:" + q["q"]] = {"ok": ok, "answer": ans}
        row["long_qa_score"] = f"{passed_long}/{len(qa['questions'])}"
        row["long_qa_s"] = round((time.perf_counter() - t) / len(qa["questions"]), 1)
        row["prompt_tokens_long"] = r.get("prompt_eval_count")

        # 3. summaries (same fact check as bench_llm.py)
        recall, sum_times = [], []
        for task, prompt in TASKS.items():
            t = time.perf_counter()
            out = ollama.chat(model, [{"role": "system", "content": "Follow the instruction using only facts in "
                                       "the transcript. Never invent names, numbers or dates."},
                                      {"role": "user", "content": f"Instruction: {prompt}\n\nTranscript:\n{transcript}"}],
                              num_ctx=num_ctx)["message"]["content"]
            sum_times.append(time.perf_counter() - t)
            low = out.lower()
            recall.append(sum(any(k in low for k in alts) for alts in qa["facts"]) / len(qa["facts"]))
            row["answers"][f"summary:{task}"] = {"answer": out}
        row["fact_recall"] = round(sum(recall) / len(recall) * 100)
        row["summary_s"] = round(sum(sum_times) / len(sum_times), 1)
        row["vram_gb"] = round(max(vram, sum(m.get("size_vram", 0) for m in ollama.loaded()) / 1e9), 1)

        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "answers"}), flush=True)

    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n{'model':20} {'ctx':>6} {'Q&A':>6} {'long Q&A':>9} {'facts':>6} {'answer':>7} {'summary':>8} "
          f"{'tok/s':>6} {'VRAM':>6} {'on GPU':>7}")
    for r in rows:
        print(f"{r['model']:20} {r['num_ctx']:>6} {r['qa_score']:>6} {r['long_qa_score']:>9} {r['fact_recall']:>5}% "
              f"{r['qa_s']:>6}s {r['summary_s']:>7}s {r['tok_per_s']:>6} {r['vram_gb']:>5}G {r['gpu_share']:>7}")


if __name__ == "__main__":
    main()
