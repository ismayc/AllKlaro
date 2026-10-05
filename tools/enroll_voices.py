#!/usr/bin/env python3
"""Save the voices in a recording so AllKlaro can say who is talking.

The app's "Names" option labels a paragraph with a saved voice. This builds
the saved voices from a recording of the people you talk to: it cuts the
audio the way the app does, describes every chunk of 3 s or more with the
speaker model, groups the chunks that sound alike, and writes one profile per
group to `~/.cache/allklaro/voices.json`. Needs `uv sync --extra speakers`.

    # 1. Find the voices. They are saved as "Voice 1", "Voice 2", ... (most
    #    speech first), with a few moments of each to listen to.
    uv run python tools/enroll_voices.py call.m4a

    # 2. Name them, once you know who is who. No audio is read again.
    uv run python tools/enroll_voices.py --rename "Voice 1=Anna" "Voice 2=Bert"

    # One person who barely speaks in the call: a recording of only them.
    uv run python tools/enroll_voices.py --name Chris chris-reading.m4a

    uv run python tools/enroll_voices.py --list

The profiles describe real people's voices. They stay in `~/.cache`, are
never committed, and are only ever compared on this machine.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speakers  # noqa: E402

# Chunks closer than this (average-link cosine distance) are one voice. On a
# 67-minute family call 0.55 gave three groups holding 94% of the long chunks,
# each matching one of a transcription service's speakers; at 0.6 two of the
# three merged.
SAME_VOICE_DIST = 0.55
MIN_CHUNKS = 8          # fewer than this is a stray, not a person


def read_audio(path: Path) -> np.ndarray:
    """Any recording as 16 kHz mono int16 (afconvert does the conversion)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                        "-c", "1", str(path), str(wav)], check=True)
        with wave.open(str(wav)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def long_chunks(audio: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(start second, audio) for every chunk the app's own VAD would cut that
    is long enough to describe a voice."""
    import server as srv

    srv.load_silero()
    vad = srv.VadSession(srv.make_scorer())
    out = []
    for i in range(0, len(audio) - srv.FRAME_SAMPLES, srv.FRAME_SAMPLES):
        chunk = vad.feed(audio[i:i + srv.FRAME_SAMPLES])
        if chunk is not None and len(chunk) >= speakers.MIN_SEC * srv.SAMPLE_RATE:
            end = (i + srv.FRAME_SAMPLES) / srv.SAMPLE_RATE
            out.append((end - len(chunk) / srv.SAMPLE_RATE, np.asarray(chunk)))
    return out


def group_voices(vectors: np.ndarray, max_dist: float = SAME_VOICE_DIST,
                 min_chunks: int = MIN_CHUNKS) -> list[np.ndarray]:
    """Indexes of the chunks in each voice, the voice with the most chunks
    first. Average-link clustering on cosine distance; groups smaller than
    `min_chunks` are dropped as strays."""
    if len(vectors) < 2:
        return []
    from scipy.cluster.hierarchy import fcluster, linkage

    labels = fcluster(linkage(vectors, method="average", metric="cosine"),
                      t=max_dist, criterion="distance")
    groups = [np.nonzero(labels == k)[0] for k in np.unique(labels)]
    groups = [g for g in groups if len(g) >= min_chunks]
    return sorted(groups, key=len, reverse=True)


def profile(vectors: np.ndarray) -> list[float]:
    """One unit vector for a voice: the mean of its chunks."""
    mean = vectors.mean(axis=0)
    return (mean / np.linalg.norm(mean)).tolist()


def load_saved(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save(path: Path, voices: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(voices))


def rename(voices: dict, pairs: list[str]) -> dict:
    """Apply "old=new" renames, keeping the order. Unknown names are errors:
    a typo must not silently leave "Voice 2" on screen."""
    mapping = {}
    for pair in pairs:
        old, sep, new = pair.partition("=")
        if not sep or not new.strip() or old.strip() not in voices:
            raise SystemExit(f'cannot rename {pair!r}: use "Saved name=New '
                             f'name"; saved names are {list(voices)}')
        mapping[old.strip()] = new.strip()
    return {mapping.get(name, name): vec for name, vec in voices.items()}


def stamp(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="?", help="a recording to find voices in")
    p.add_argument("--name", help="the recording is one person: save them "
                                  "under this name, beside the voices "
                                  "already saved")
    p.add_argument("--rename", nargs="+", metavar="OLD=NEW",
                   help="rename saved voices; reads no audio")
    p.add_argument("--list", action="store_true", help="show the saved names")
    p.add_argument("--out", default=str(speakers.VOICES_PATH))
    args = p.parse_args(argv)
    out = Path(args.out)

    if args.list:
        print("\n".join(load_saved(out)) or "no saved voices")
        return 0
    if args.rename:
        voices = rename(load_saved(out), args.rename)
        save(out, voices)
        print("saved voices:", ", ".join(voices))
        return 0
    if not args.audio:
        p.error("give a recording, or --rename / --list")
    if speakers.load() is None:
        sys.exit("the speaker model is not installed: uv sync --extra speakers")

    chunks = long_chunks(read_audio(Path(args.audio)))
    vectors = [speakers.embed(c) for _, c in chunks]
    kept = [(start, v) for (start, _), v in zip(chunks, vectors) if v is not None]
    if not kept:
        sys.exit(f"no stretch of speech of {speakers.MIN_SEC:g} s or more in "
                 "that recording")
    starts = np.array([s for s, _ in kept])
    matrix = np.stack([v for _, v in kept])

    if args.name:
        voices = load_saved(out)
        voices[args.name] = {"vec": profile(matrix), "n": len(kept)}
        save(out, voices)
        print(f"saved {args.name!r} from {len(kept)} chunks; "
              f"saved voices: {', '.join(voices)}")
        return 0

    groups = group_voices(matrix)
    if not groups:
        sys.exit("no voice with enough speech to save; for one person who "
                 "says little, record them alone and use --name")
    voices = {}
    print(f"{len(kept)} chunks of {speakers.MIN_SEC:g} s or more, "
          f"{len(groups)} voices:")
    for n, g in enumerate(groups, 1):
        voices[f"Voice {n}"] = {"vec": profile(matrix[g]), "n": len(g)}
        listen = ", ".join(stamp(starts[i]) for i in
                           g[np.linspace(0, len(g) - 1, 4).astype(int)])
        print(f"  Voice {n}: {len(g)} chunks; listen at {listen}")
    save(out, voices)
    print(f'saved to {out}. Name them with --rename "Voice 1=Name" ...')
    return 0


if __name__ == "__main__":
    sys.exit(main())
