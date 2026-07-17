# AllKlaro — live German / Spanish / English conversation translator

*All* (EN) + *Klaro!* (DE) + *claro* (ES): everything understood, in all three
languages. Fully local, near-realtime speech translation for conversations
with German or Spanish speakers. Nothing leaves your machine.

**Pipeline:** browser mic (or BlackHole loopback) → WebSocket → Silero VAD
(energy fallback) → [mlx-whisper](https://github.com/ml-explore/mlx-examples)
`large-v3-turbo` (Apple-GPU) → Ollama (`gemma3:12b` by default), with recent
conversation context → translation streamed back token by token.

---

## What you need

- **An Apple Silicon Mac** (M1 or newer). Speech recognition runs on the GPU
  via Apple's MLX — Intel Macs and other platforms won't work.
- **16 GB+ RAM** recommended (the default 12B translation model needs ~8 GB;
  smaller models work on 8 GB Macs — see [Controls](#controls)).
- **~12 GB free disk**: Whisper weights ~1.6 GB, `gemma3:12b` ~8 GB.
- **[Homebrew](https://brew.sh)** to install the two tools below.

## Setup (one time)

```bash
# 1. Install uv (Python manager) and Ollama (runs the translation model)
brew install uv ollama

# 2. Start Ollama and pull the translation model (~8 GB download)
brew services start ollama        # or run `ollama serve` in its own terminal
ollama pull gemma3:12b

# 3. Get the code and install Python dependencies
git clone https://github.com/<your-username>/AllKlaro.git
cd AllKlaro
uv sync
```

## Run

```bash
uv run uvicorn server:app --host 127.0.0.1 --port 8710
```

then open <http://127.0.0.1:8710> — or double-click
**`Start AllKlaro.command`** in Finder, which starts Ollama if needed, starts
the server, and opens the page. (If macOS refuses to run it, right-click →
Open once, or `chmod +x "Start AllKlaro.command"`.)

**The first launch is slow**: the server downloads the Whisper weights
(~1.6 GB) and the Silero VAD model (~2 MB) from the internet, then loads them.
Watch the terminal for `Whisper ready.` After that, startup takes seconds and
no network is needed beyond your own machine.

Press **Start**, allow microphone access, and talk. Each pause finalizes an
utterance, which is transcribed and translated within the selected language
pair. The green level meter next to the status dot shows whether audio is
arriving — if it never moves, the wrong input device is selected.

## Capturing call audio (optional — Zoom/Teams/Meet/videos)

Out of the box AllKlaro hears your **microphone**. To translate the *other*
side of a call (or any audio your Mac plays), install the free
[BlackHole](https://existential.audio/blackhole/) loopback driver and create
two virtual devices — a one-time setup:

1. `brew install --cask blackhole-2ch`, then log out/in (or reboot) so the
   device appears.
2. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup"), click **+** →
   **Create Multi-Output Device**: check ✅ your speakers/headphones **and**
   ✅ BlackHole 2ch. Set your speakers as **Primary**, enable **Drift
   Correction** on BlackHole only. Rename it **"Meeting + BlackHole"**.
3. Route the call's audio through it: in the conferencing app set **Speaker →
   "Meeting + BlackHole"** (your mic setting stays unchanged). For browser
   audio/videos there is no per-app setting — set the macOS **system output**
   to "Meeting + BlackHole" (⌥-click the 🔊 menu-bar icon → Output).
4. In AllKlaro, set **Input → "You + Them (mic + BlackHole)"** — your voice
   and the call are captured as separate, labeled channels. (Or *BlackHole
   2ch* for the call audio only.)

Gotchas: if the level meter stays flat while the call is playing, the audio
isn't reaching BlackHole — re-check step 3, which is the step everyone
misses. Bluetooth headphones (AirPods) must be **members** of the
Multi-Output device to hear the call, and macOS likes to silently switch
output to plain AirPods when they connect — re-select "Meeting + BlackHole"
after connecting them.

## Controls

- **Input** — which audio device to capture. *You + Them* (call mode, the
  default when BlackHole is installed) captures mic + call separately and
  labels each card You/Them; duplicated speech picked up by both channels is
  dropped automatically. Device names appear after the first mic permission.
- **Direction** — *Auto* pairs (DE↔EN, ES↔EN, ES↔DE) detect each utterance's
  language and translate to the other; forced single directions also pin
  Whisper's language (helps with heavy accents). Multi-target modes
  (*German → English + Spanish*) translate into both at once. Add languages by
  extending `LANG_NAMES` in `server.py` plus `<option>`s in
  `static/index.html`.
- **Model** — any installed Ollama chat model. `gemma3:12b` is the default;
  `gemma3:4b` or `qwen2.5:7b-instruct` are faster / lighter on RAM,
  `qwen2.5:32b-instruct` higher quality but slower.
- **Pause** — how much silence ends an utterance (default 700 ms). German
  puts the meaning-carrying verb last, so cutting early hurts; a fragment
  that ends without sentence-final punctuation is automatically **merged**
  with the speaker's next utterance (within 2 s) and re-translated whole.
- **Speak** — reads translations aloud (macOS voices); capture is muted while
  speaking so the app doesn't translate itself.
- **Summarize** — asks Ollama for an English summary plus a vocabulary list
  for language-learning review.
- **Export** — downloads the conversation (and summary) as Markdown.
- Click any card for a **full-screen big-text view** (to show someone).
- Settings persist across reloads; the WebSocket auto-reconnects and a
  wake-lock keeps the screen on while listening.
- **Glossary** — put names and recurring terms in `glossary.txt` (see the
  comments inside). Terms bias Whisper's recognition; `term = translation`
  lines also pin the translator's wording. Hot-reloaded on save.

## Phone access (in-person conversations)

Run **`Start AllKlaro (Phone).command`** — it generates a self-signed
certificate (mic access requires HTTPS off-localhost) and serves on your LAN,
printing the exact URL. On a phone on the same Wi-Fi, open
`https://<mac-ip>:8710` in the browser, accept the certificate warning, press
Start, allow the microphone, and lay the phone between the speakers. Note:
this exposes the app to everyone on the network — use networks you trust.

## Speed & correctness machinery

- **Speculative transcription** — Whisper starts ~320 ms into a pause instead
  of waiting the full pause length; discarded if speech resumes.
- **Partials never block finals** — live previews are skipped whenever real
  work is pending, and only look at the last ~12 s of audio.
- **Prompt caching** — static system prompt + history as chat turns, so
  Ollama's prefix cache only processes the new sentence. Reverse-direction
  turns are flipped so both sides of the conversation provide context.
- **Continuous-speech splitting** — speech with no real pauses (videos, fast
  talkers) is split at natural micro-pauses after ~8 s instead of stalling.
- **Pair-constrained language detection** — a detection outside the active
  pair (e.g. Dutch for German speech) triggers a re-decode pinned to the
  pair's primary language.
- **Confidence filtering** — segments Whisper marks as likely non-speech
  (music, jingles, background chatter) are dropped, plus a blocklist of
  Whisper's known stock hallucinations ("Thank you.", "Untertitelung des
  ZDF…").
- **VAD hysteresis** — starting speech needs Silero probability > 0.5,
  continuing only > 0.35, so quiet word-endings don't chop sentences.

## Tests

```bash
uv run pytest                                   # fast unit tests (mocked Whisper/Ollama)
RUN_INTEGRATION=1 uv run pytest -m integration  # real Whisper + Ollama, ~25 s
```

Unit tests cover the VAD state machine and scorers, hallucination and
confidence filtering, direction/multi-target resolution, fragment merging,
echo dedupe, speculative transcription, conversation context, speaker
channels, pause configuration, glossary, export/summarize, and the WebSocket
protocol — including error paths (Ollama down, missing model, malformed
control messages, corrupt audio frames). Integration tests synthesize German
and Spanish speech with macOS `say` and run it through the real pipeline.

## Troubleshooting

- **"Cannot reach Ollama"** banner → Ollama isn't running: `brew services
  start ollama` or `ollama serve`.
- **Level meter flat** → wrong Input device, or (for calls) audio isn't
  routed into BlackHole — see the call-audio section above.
- **Cards appear but translations are slow to start** → the model was idle
  and is reloading into memory; it stays warm for 60 min after each use.
- **VAD tuning** — constants at the top of `server.py` (pause length is also
  a UI slider).
