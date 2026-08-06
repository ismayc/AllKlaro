"""Dump what Whisper actually writes for the real recording, for hand-checking.

Two open items need the same artifact and neither can be closed without it:

  #4 asks why German audio sometimes comes out as English words, and which
     repetition loops survive the filters. Both are claims about raw ASR
     output, so they cannot be settled from the trace files -- those record
     timings and never the text.
  #5 asks what Whisper writes when a Berliner speaks. The dialect lexicon
     currently keys on Hessian spellings only, and nothing can be added to it
     without ground truth.

Segmentation goes through the app's own VadSession, so the utterances here
are the same ones the pipeline would translate, not an arbitrary split.

    uv run python tools/dump_transcripts.py --audio ~/.cache/allklaro/demo4.wav
    uv run python tools/dump_transcripts.py --language de --out forced-de.jsonl

`--language` unset means Whisper auto-detects, which is the condition under
which the EN-for-German failure was seen. Forcing `de` is the comparison that
shows what the same audio yields when detection cannot go wrong.
"""
import argparse
import json
import wave

import numpy as np

import server as srv


def frames(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == srv.SAMPLE_RATE, w.getframerate()
        assert w.getsampwidth() == 2, w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:                      # the recording is one mic; mix if not
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    n = srv.FRAME_SAMPLES
    for i in range(0, len(pcm) - n + 1, n):
        yield pcm[i:i + n]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True)
    p.add_argument("--language", default=None,
                   help="force a language; omit to let Whisper auto-detect")
    p.add_argument("--out", default="transcripts.jsonl")
    a = p.parse_args()

    vad = srv.VadSession()
    rows, idx, t = [], 0, 0.0
    for f in frames(a.audio):
        t += srv.FRAME_MS / 1000
        utt = vad.feed(f)
        if utt is None:
            continue
        idx += 1
        res = srv.transcribe(utt, a.language)
        raw = (res.get("text") or "").strip()
        cleaned = srv.clean_transcript(res)
        segs = res.get("segments") or []
        row = {
            "i": idx,
            "t_end": round(t, 2),
            "dur_sec": round(len(utt) / srv.SAMPLE_RATE, 2),
            "split": vad.split_reason,
            "detected_language": res.get("language"),
            "raw": raw,
            "cleaned": cleaned,
            # Dropped entirely: the filters decided this was an artifact. A
            # wrong drop is as much a bug as a missed loop, so both directions
            # are visible here.
            "dropped": bool(raw) and not cleaned,
            "n_segments": len(segs),
            "max_compression_ratio": max(
                (s.get("compression_ratio", 0.0) for s in segs), default=0.0),
            "min_avg_logprob": min(
                (s.get("avg_logprob", 0.0) for s in segs), default=0.0),
            "max_no_speech_prob": max(
                (s.get("no_speech_prob", 0.0) for s in segs), default=0.0),
        }
        rows.append(row)
        print(f"{idx:3} {row['dur_sec']:5.1f}s {row['detected_language'] or '??':>3} "
              f"{'DROP' if row['dropped'] else '    '} {raw[:88]}", flush=True)

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    langs = {}
    for r in rows:
        langs[r["detected_language"]] = langs.get(r["detected_language"], 0) + 1
    print(f"\n{len(rows)} utterances -> {a.out}")
    print(f"detected languages: {langs}")
    print(f"dropped as artifact: {sum(r['dropped'] for r in rows)}")


if __name__ == "__main__":
    main()
