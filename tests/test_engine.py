"""Fast unit tests for engine.py — no models, GPU or audio needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine  # noqa: E402
from engine import Result, Segment  # noqa: E402


def make_result(speakers=False, summary=""):
    segs = [
        Segment(0.0, 2.5, " Hello there.", "Speaker 1" if speakers else ""),
        Segment(2.5, 5.0, " How are you?", "Speaker 1" if speakers else ""),
        Segment(5.2, 7.9, " Fine, thanks.", "Speaker 2" if speakers else ""),
    ]
    return Result(segments=segs, language="en", duration=8.0, confidence=91.23, summary=summary,
                  num_speakers=2 if speakers else 0)


# ---------- timestamps ----------
@pytest.mark.parametrize("seconds, srt, expected", [
    (0, False, "00:00:00"),
    (61.5, False, "00:01:01"),
    (3725.042, True, "01:02:05,042"),
    (59.9996, True, "00:01:00,000"),  # rounding carries into the next second
])
def test_format_timestamp(seconds, srt, expected):
    assert engine.format_timestamp(seconds, srt) == expected


# ---------- output rendering ----------
def test_plain_text_output():
    out = engine.render_output(make_result(), "Plain text (.txt)")
    assert "Confidence: 91.2%" in out
    assert "Hello there. How are you? Fine, thanks." in out
    assert "AI Summary" not in out


def test_summary_is_included_when_prompted():
    out = engine.render_output(make_result(summary="- do the thing"), "Plain text (.txt)", prompt="Action items")
    assert out.index("=== AI Summary ===") < out.index("=== Full Transcription")
    assert "Prompt: Action items" in out and "- do the thing" in out


def test_timestamped_output_with_speakers():
    out = engine.render_output(make_result(speakers=True), "Timestamped text (.txt)")
    assert "[00:00:05] Speaker 2: Fine, thanks." in out
    assert "Speakers: 2" in out


def test_plain_text_groups_consecutive_speaker_turns():
    text = make_result(speakers=True).speaker_text
    assert text.splitlines() == ["Speaker 1: Hello there. How are you?", "Speaker 2: Fine, thanks."]


def test_srt_output():
    out = engine.render_output(make_result(speakers=True), "Subtitles (.srt)")
    blocks = out.strip().split("\n\n")
    assert len(blocks) == 3
    assert blocks[2].splitlines() == ["3", "00:00:05,200 --> 00:00:07,900", "[Speaker 2] Fine, thanks."]


def test_srt_summary_goes_to_side_file(tmp_path):
    path = tmp_path / "talk.srt"
    written = engine.save_output(make_result(summary="Short."), str(path), "Subtitles (.srt)", prompt="Summarize")
    assert written == [str(path), str(tmp_path / "talk_summary.txt")]
    assert "Short." in (tmp_path / "talk_summary.txt").read_text(encoding="utf-8")
    assert "Short." not in path.read_text(encoding="utf-8")


# ---------- chunking ----------
def test_split_chunks_short_text_is_single_chunk():
    assert engine.split_chunks("  hello world  ", 100) == ["hello world"]


def test_split_chunks_respects_size_and_keeps_all_words():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = engine.split_chunks(text, 500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    assert all(c.endswith(".") for c in chunks[:-1])  # cut at sentence boundaries


# ---------- speaker segmentation ----------
def words(*items):
    return [(s, e, w) for s, e, w in items]


def test_split_on_pauses_splits_at_long_gap():
    seg = Segment(0, 6, "", words=words((0, 1, " One"), (1, 2.2, " two."), (3.5, 4.5, " Three"), (4.5, 6, " four.")))
    parts = engine.split_on_pauses([seg])
    assert [p.text for p in parts] == ["One two.", "Three four."]
    assert (parts[1].start, parts[1].end) == (3.5, 6)


def test_split_on_pauses_merges_short_piece_into_nearest_neighbour():
    # "Yes." is too short to stand alone; it sits closer to the following words than the preceding ones.
    seg = Segment(0, 8, "", words=words((0, 1, " Is"), (1, 2, " it?"), (3, 3.4, " Yes."), (3.45, 5, " It"),
                                        (5, 6, " is.")))
    parts = engine.split_on_pauses([seg])
    assert [p.text for p in parts] == ["Is it?", "Yes. It is."]


def test_split_on_pauses_leaves_segments_without_words_alone():
    seg = Segment(0, 3, "No words")
    assert engine.split_on_pauses([seg]) == [seg]


# ---------- clustering ----------
def two_speaker_embeddings(n=10, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=32), rng.normal(size=32)
    rows = [(a if i % 3 else b) + rng.normal(scale=0.05, size=32) for i in range(n)]
    X = np.array(rows)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_cluster_speakers_auto_finds_two():
    labels = engine.cluster_speakers(two_speaker_embeddings())
    assert labels[0] == 0 and labels[1] == 1  # numbered by first appearance
    assert [i for i, x in enumerate(labels) if x == 0] == [0, 3, 6, 9]


def test_cluster_speakers_fixed_count():
    assert len(set(engine.cluster_speakers(two_speaker_embeddings(), num_speakers=3))) == 3


def test_cluster_speakers_single_voice():
    rng = np.random.default_rng(1)
    base = rng.normal(size=32)
    X = np.array([base + rng.normal(scale=0.01, size=32) for _ in range(8)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert set(engine.cluster_speakers(X)) == {0}


def test_cluster_speakers_absorbs_single_outlier():
    rng = np.random.default_rng(2)
    base = rng.normal(size=32)
    X = np.array([base + rng.normal(scale=0.05, size=32) for _ in range(20)])
    X[7] = base + rng.normal(scale=1.0, size=32)  # one noisy segment from the same person
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert set(engine.cluster_speakers(X)) == {0}


def test_cluster_speakers_tiny_inputs():
    assert engine.cluster_speakers(np.ones((1, 4))) == [0]


def test_smooth_labels_fixes_short_blips():
    assert engine.smooth_labels([0, 1, 0, 1], [3, 0.4, 3, 3]) == [0, 0, 0, 1]
    assert engine.smooth_labels([0, 1, 0], [3, 2.0, 3]) == [0, 1, 0]  # long enough to be trusted


# ---------- misc ----------
def test_gpu_error_detection():
    assert engine._looks_like_gpu_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert not engine._looks_like_gpu_error(ValueError("bad file"))


def test_every_model_menu_entry_is_valid():
    assert engine.DEFAULT_MODEL in engine.MODELS
    assert engine.DEFAULT_SUMMARY_MODEL in engine.SUMMARY_MODELS
    assert all("id" in spec for spec in engine.SUMMARY_MODELS.values())
