# Model benchmarks

How WhisperScribe's default models were chosen. Everything ran locally on the machine below. To re-run on your own
hardware, see [Reproducing](#reproducing).

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080 (16 GB) |
| CPU | AMD Ryzen 7 9800X3D |
| RAM | 32 GB DDR5 |
| Software | Python 3.14, faster-whisper 1.2 (CTranslate2 4.8, float16), PyTorch 2.14 + CUDA 13, transformers 5.17 |

## Speech-to-text

Two test sets:

- **Real recording:** 10.4 minutes of a real English conversation. Only numbers were recorded, never its text. With no
  human transcript available, "Differs from Large v3" measures how far each model's text is from `large-v3`.
- **Reference clip:** a 137-second, two-voice scripted meeting (Windows text-to-speech with light background noise)
  with a known transcript, used for word error rate (WER). Text is normalised with Whisper's English normalizer, so
  "thirty nine dollars" and "$39" count as the same.

| Model | Time for 10.4 min | Speed | Avg. confidence | WER (reference) | Differs from Large v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 9.6 s | 65× | 71.9 % | 4.43 % | 25.3 % |
| base *(old default)* | 9.6 s | 65× | 78.1 % | 2.22 % | 15.2 % |
| small | 14.0 s | 45× | 81.4 % | 1.58 % | 21.4 % |
| medium | 18.9 s | 33× | 83.5 % | 0.63 % | 9.4 % |
| distil-large-v3 | 8.8 s | 71× | 83.7 % | 2.85 % | 15.6 % |
| **large-v3-turbo** ✅ | **8.2 s** | **76×** | **87.1 %** | **0.32 %** | **7.0 %** |
| large-v3-turbo (batched ×16) | 3.2 s | 197× | 87.4 % | — | 11.3 % |
| large-v3 | 37.2 s | 17× | 87.7 % | 0.63 % | — |
| large-v3 (batched ×16) | 22.5 s | 28× | 87.5 % | — | 9.3 % |

**Pick: Large v3 Turbo.** It's the fastest non-batched model, has the lowest error rate, and is within 0.6 points of
the best confidence. Batched decoding is 2.5× faster again, but its output drifts further from Large v3. The accurate
mode already handles an hour of audio in about a minute, so the app doesn't use batching.

*"Confidence" is the duration-weighted average of `exp(avg_logprob)` per segment, the same number the app shows.*

## Local LLM for AI instructions

Test: the reference meeting transcript, run through the app's **Action items** and **Meeting notes** prompts. The
score is how many of 10 known facts appear in the answer (owners Marcus / Priya, deadlines Friday / Wednesday, $39
price, October 15 launch, caching fix as a blocker, onboarding moved to a follow-up release, and so on). "Long input"
is about 22,000 characters, roughly 30 minutes of speech, summarised in one pass. Each model ran alone on an idle GPU.
Gated models (Llama 3.2, Gemma 3) were skipped because they need a Hugging Face account.

| Model | Facts found | Short answer | Long input | Tokens/s | Peak VRAM | Notes from reading the output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5 0.5B *(old default)* | 50 % | 3.6 s | — | 58 | 1.3 GB | Lists one action item; unusable |
| Qwen2.5 1.5B | 95 % | 4.8 s | 6.4 s | 48 | 4.7 GB | Good, but says the *launch* was pushed back (it wasn't) |
| Qwen3 1.7B | 95 % | 6.1 s | 3.0 s | 49 | 7.6 GB | Accurate and concise |
| Qwen2.5 3B | 90 % | 5.6 s | — | — | 9.9 GB | Assigns owners to the wrong tasks |
| Phi-4 mini 3.8B | 90 % | 5.8 s | 10.2 s | 49 | 13.3 GB | Invents "$39 per *year*" |
| Qwen3 4B Instruct 2507 | 95 % | 9.2 s | 124 s | 34 | 15.7 GB | Accurate, but very slow and nearly out of VRAM on long input |
| **Qwen2.5 7B Instruct (4-bit NF4)** ✅ | **100 %** | **5.8 s** | **6.0 s** | 39 | 12.2 GB | Most complete and accurate; correct owners once speaker labels are on |

The long-input time for the 0.5B model and the 3B's tokens/s came from an earlier run that shared the GPU, so they
aren't shown.

**Pick: Qwen2.5 7B in 4-bit.** It's the only model that caught every fact without inventing any, and it's as quick as
the 1.5B–3.8B models. The app lists three choices: **7B · best** (default), **Qwen3 1.7B · fast**, and
**Qwen2.5 1.5B · light**. Machines without an NVIDIA GPU switch to Qwen3 1.7B automatically, because 4-bit loading
needs CUDA. Before summarising, the app frees the speech model's VRAM and summarises at most 24,000 characters per
pass, so peak VRAM stays well under 16 GB even for long recordings.

## Chat model (Ollama, used through AnythingLLM)

WhisperScribe sends transcripts to AnythingLLM, which answers questions through Ollama. The same model can also write
WhisperScribe's summaries, so only one LLM sits in VRAM. The test uses
[`benchmarks/data/meeting_qa.json`](../benchmarks/data/meeting_qa.json): 10 questions about the reference meeting,
two of which ask about things the meeting never mentions and should be declined. "Long" buries the same meeting
inside about 22,000 tokens of other discussion, like a long recording or several retrieved transcripts. Every model
ran fully on the GPU at its maximum context (capped at 32K).

| Model (Ollama tag) | Context | Q&A | Q&A, long | Summary facts | Answer time | Tokens/s | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phi4:latest *(AnythingLLM's model before)* | 16K max | 10/10 | **5/10** | 95 % | 0.3 s | 93 | 14.4 GB |
| **qwen3.5:9b** ✅ | 256K | **10/10** | **10/10** | 90 % | **0.3 s** | **119** | **9.9 GB** |
| ministral-3:14b | 256K | 10/10 | 10/10 | 100 % | 1.1 s | 101 | 15.4 GB |
| gemma4:12b | — | not tested: needs a newer Ollama than AnythingLLM 1.16.1 bundles | | | | | |

**phi4 made things up on long meetings.** The meeting didn't fit in its 16K window, and instead of declining it
answered "$59", "due Wednesday" and "100 responses", none of which appear in the meeting.

**Pick: qwen3.5 9B.** It's perfect on both Q&A tests, the fastest, and at 9.9 GB it leaves room for Whisper next to
it. ministral-3 14B writes slightly more complete summaries but fills the whole 16 GB card; it's still available in
the summary menu.

### Two fixes needed to use it through AnythingLLM

- **Thinking mode.** qwen3.5 "thinks" before answering, and AnythingLLM can't turn that off (Ollama's `think: false`
  isn't sent for chats). Answers took 8–50 s and the reasoning showed up in replies. WhisperScribe creates
  `qwen3.5-nothink:9b` in Ollama: the same weights (no download, no extra disk) with a chat template that starts
  every answer with an empty `<think></think>`, which is Qwen's official non-thinking mode. Answers now start in
  under a second.
- **Context size.** Ollama reloads a model (about 40 s) whenever the requested context size changes. WhisperScribe
  reads AnythingLLM's Ollama token limit and uses the same value for summaries, so switching between chatting and
  summarizing costs nothing: 1.3 s for a summary, 3.1 s for the next chat.

End to end through the real AnythingLLM API (upload the transcript, embed it, then chat in a thread with the
recommended workspace settings), the same 10 questions scored **10/10** at about 5 s per answer, each with the
recording cited as a source. Attribution questions ("who is writing it up?") were correct 9/9 across repeated
fresh threads. That took one prompt fix: the model has to credit "I'll do it" to the speaker label on that line.

## Speaker labels

The speaker model is `microsoft/wavlm-base-plus-sv` (open, no account needed). Each transcript segment is split at
pauses using word timestamps, turned into a voice fingerprint, and grouped by agglomerative clustering. Auto mode
picks the number of speakers by silhouette score, then merges groups whose average voices are more than 0.80 similar
and folds single stray segments into the nearest speaker.

On the two-voice reference meeting:

| Setup | Segments labelled correctly |
| --- | ---: |
| Whisper segments only | 95 % (some segments contained the end of one turn and the start of the next) |
| + split at pauses (word timestamps) | **100 %** (38 segments, 2 speakers found automatically) |

Measured similarity between voice fingerprints: same speaker about 0.93, different speakers about 0.40. Each voice on
its own was correctly detected as a single speaker.

Known limit: one-word replies ("Agreed.") sometimes stay with the previous speaker, because Whisper's word timings
don't always show the gap before them.

## End to end

The full pipeline on a real **57-minute** recording (Large v3 Turbo with word timestamps, Auto speakers, Qwen2.5 7B
action items):

| Step | Time |
| --- | ---: |
| Transcribe | 62.7 s |
| Speaker labels (2 found) | 16.0 s |
| Summary (2 chunks) | 37.4 s |
| **Total** | **1 min 56 s** (peak VRAM 7.3 GB, confidence 90.0 %) |

## Reproducing

```bash
python benchmarks/bench_whisper.py --real my_recording.m4a --ref-audio clip.wav --ref-text clip.txt
python benchmarks/bench_llm.py --transcript clip.txt --facts facts.json
python benchmarks/bench_ollama.py --models phi4:latest qwen3.5:9b ministral-3:14b   # uses benchmarks/data/
```

`bench_whisper.py` prints only numbers for `--real`, so private recordings are safe to use. `facts.json` is a list of
keyword alternatives, for example `[["marcus"], ["friday"], ["39", "thirty nine"]]`.
