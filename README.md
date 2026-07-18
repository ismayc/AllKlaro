# 🗣️ AllKlaro

<img src="docs/logo.png" width="96" align="right" alt="AllKlaro logo" />

**Live, fully-local speech translation between German, English, and Spanish.**

> ***All*** *(EN)* + ***Klaro!*** *(DE)* + ***claro*** *(ES)* — everything
> understood, in all three languages.

![local](https://img.shields.io/badge/privacy-100%25%20local-3ecf8e)
![platform](https://img.shields.io/badge/platform-Apple%20Silicon-4fa3ff)
![tests](https://img.shields.io/badge/tests-200%2B%20passing-f5a623)

🌐 **Website:** [ismayc.github.io/AllKlaro](https://ismayc.github.io/AllKlaro/)
— also in [Deutsch](https://ismayc.github.io/AllKlaro/?lang=de) and
[Español](https://ismayc.github.io/AllKlaro/?lang=es)

Someone speaks German or Spanish on your call — two seconds later the English
translation is streaming onto your screen, color-coded by speaker and language.
You answer in English; they get German or Spanish back. Nothing ever leaves
your machine: speech
recognition runs on the Mac's GPU ([mlx-whisper](https://github.com/ml-explore/mlx-examples)
`large-v3-turbo`), voice detection is neural ([Silero VAD](https://github.com/snakers4/silero-vad)),
and translation is a local [Ollama](https://ollama.com) model (`gemma3:12b`
by default) that sees the conversation's recent context.

```
🎙️ mic / 🔊 call audio ─► VAD ─► Whisper (Apple GPU) ─► Ollama ─► 🖥️ streaming cards
```

## ✨ What it does

| | |
|---|---|
| 🔄 **Auto direction** | Detects each utterance's language, translates to the other (DE↔EN, ES↔EN, ES↔DE), or force a direction — including *German → English + Spanish* at once |
| 📞 **Call mode** | Captures your mic and the meeting as separate **You / Them** channels; echoes picked up twice are dropped automatically |
| 🗣️ **Tap to hear** | Tap any phrase to hear it spoken in its own language (great for pronunciation) |
| 🧠 **Context-aware** | Translations see the recent conversation, so pronouns and topics resolve correctly |
| 🧩 **German-aware pausing** | Verb-final sentences chopped by a short pause are re-joined and re-translated whole |
| ⚡ **Draft + refine** | A small model streams a translation instantly; the main model re-translates behind the scenes and swaps in its better answer |
| ✏️ **Correctable** | Edit any translation in place — corrections are saved locally and steer future translations of similar sentences (few-shot retrieval) |
| 📖 **Learner tools** | One-click **Summarize** (summary + vocabulary list) and Markdown **Export** for later review |
| 📱 **Phone mode** | Serve over LAN HTTPS and use your iPhone's mic for in-person conversations |
| 🎯 **Focus mode** | Keep the newest text mid-screen ("Center latest") instead of at the bottom edge |
| 🗺️ **Dialect-aware** | Berlinerisch/Hessisch markers ("dit", "ebbes", "gell") are detected and the likely intended forms — including Whisper mis-hearings like "nett" for "net" (nicht) — are hinted to the translator; extend via `dialects.txt` |
| 📚 **Glossary** | Pin names and terms in `glossary.txt` — biases recognition *and* translation |
| ⌨️ **Type to translate** | A text box under the feed — type instead of speaking, mic not required; same context, corrections, and draft+refine pipeline |
| 📖 **Gender lexicons** | Optional: compile dict.cc / FreeDict exports into loanword-gender dictionaries — "caipirinha" gets der/die/das and "problema" gets el/la injected |

## 🚀 Quick start

**You need:** an Apple Silicon Mac (M1+ — MLX runs the speech model on the
GPU), 16 GB+ RAM recommended, ~12 GB free disk, and [Homebrew](https://brew.sh).

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

# 4. Run it
uv run uvicorn server:app --host 127.0.0.1 --port 8710
```

…then open <http://127.0.0.1:8710>. Or skip step 4 and double-click
**`Start AllKlaro.command`** in Finder — it starts Ollama if needed, starts
the server, and opens the page. (If macOS balks, right-click → Open once, or
`chmod +x "Start AllKlaro.command"`.)

⏳ **The first launch is slow**: the Whisper weights (~1.6 GB) and Silero VAD
model (~2 MB) download once. Watch the terminal for `Whisper ready.` — after
that, startup takes seconds and everything runs offline.

Press **Start**, allow the microphone, and talk. Each pause finalizes an
utterance; the green level meter shows whether audio is arriving — if it
never moves, the wrong input device is selected.

## 🎧 Capturing call audio (Zoom / Teams / Meet / videos)

Out of the box AllKlaro hears your **microphone**. To translate the *other*
side of a call (or anything your Mac plays), add the free
[BlackHole](https://existential.audio/blackhole/) loopback driver — a
one-time setup:

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

⚠️ **Gotchas:** if the level meter stays flat while the call is playing, the
audio isn't reaching BlackHole — re-check step 3, the step everyone misses.
Bluetooth headphones must be **members** of the Multi-Output device to hear
the call, and macOS likes to silently switch output to plain headphones when
they connect — re-select "Meeting + BlackHole" after connecting them.

## 🎛️ Controls

Everything lives behind the **⚙︎ gear** (auto-collapsed on phones so the
conversation gets the full screen):

- **Input** — audio device to capture. *You + Them* call mode is the default
  when BlackHole is installed. Device names appear after the first mic
  permission grant.
- **Direction** — auto pairs, forced directions, or multi-target modes. Add
  more languages by extending `LANG_NAMES` in `server.py` plus `<option>`s in
  `static/index.html`.
- **Model** — any installed Ollama chat model; this model's translation is
  the one you keep. Suggested pairings (benchmarked on grammar probes +
  warm latency): everyday calls **Draft `qwen2.5:7b-instruct` + Model
  `gemma3:12b`**; important conversations **Draft `gemma3:12b` + Model
  `qwen2.5:32b-instruct`** (needs ~32 GB+ RAM to keep both warm). Models
  under ~4 GB (3B-class) paraphrase too freely — avoid them even as drafts.
- **Draft model** — optional fast first pass: this model's translation
  appears immediately ("refining…"), then the main model's answer replaces
  it with a green flash. Defaults to the smallest installed model above the
  4 GB trust line; set **Off** for single-pass translation. Edits you make
  always win over the refinement.
- **Pause** — silence that ends an utterance (default 700 ms). Fragments cut
  mid-sentence are auto-merged with the speaker's next utterance.
- **Speak** — auto-read translations aloud (capture mutes itself while
  speaking). Independent of tap-to-hear, which always works.
- **Center latest** — focus mode: newest card and live transcript sit
  mid-screen; history scrolls up.
- **Summarize / Export / Clear** — learner review tools (the summary rides
  along in the export); Clear wipes the conversation after a confirmation.
- **✏️ on any finished translation** — edit it in place; the correction is
  saved for retrieval and immediately fixes the conversation context.
- **Type to translate** — the text bar at the very bottom works without the
  mic; auto modes detect the typed language.
- **Speak** voices: German `de-DE`, English `en-US`, Spanish `es-MX`
  (Latin American).
- **About** — links to the website and repo.
- Tap a card's background for a **full-screen big-text view** to show the
  person you're talking to.
- Settings persist; the WebSocket auto-reconnects; a wake-lock keeps the
  screen on while listening.

## 📱 Phone mode (in-person conversations)

Run **`Start AllKlaro (Phone).command`** — it generates a self-signed
certificate (mic access requires HTTPS off-localhost) and serves on your LAN,
printing the exact URL. On a phone on the same Wi-Fi: open
`https://<mac-ip>:8710`, accept the certificate warning, press Start, allow
the mic, and lay the phone between the speakers. ⚠️ This exposes the app to
everyone on the network — use networks you trust.

## ⚙️ Under the hood

The speed and correctness machinery, for the curious:

- **Speculative transcription** — Whisper starts ~320 ms into a pause;
  discarded if speech resumes. Partials never block real work.
- **Prompt caching** — static system prompt + history as chat turns means
  Ollama re-processes only the new sentence.
- **Continuous-speech splitting** — no pauses (videos, fast talkers)? Split
  at natural micro-pauses after ~8 s instead of stalling.
- **Pair-constrained detection** — "Dutch" detected in a German↔English
  conversation triggers a re-decode pinned to German.
- **Junk defense in depth** — Whisper's temperature-fallback ladder,
  compression-ratio and no-speech signals, a repetition-loop detector, and a
  blocklist of stock hallucinations ("Untertitelung des ZDF…").
- **VAD hysteresis** — starting speech needs Silero probability > 0.5,
  continuing only > 0.35, so quiet word-endings don't chop sentences.
- **Declension guard** — with the gender lexicons built, every final
  translation is checked by *constraint intersection*: each German
  determiner form maps to its complete set of (case, number, gender)
  readings, each noun form to its possible readings from a Wiktionary
  paradigm table (256k forms — plurals, weak nouns, genitive -s), and an
  NP with no consistent reading is wrong in every interpretation: "das
  Termin", "eine Termin", "die Auto", "mit den Lehrer" (dative plural is
  Lehrern), "wegen des Termin" (missing -s), "ein schöne Tag", "la
  problema". Where a preposition pins the case (mit/für/ohne/…) or a
  contraction pins the gender (zum/zur/im/ins), the exact required
  article is named: "mit die Frau" → der Frau. Violations trigger one
  corrective re-ask with the facts stated; the retry is used only if it
  verifies clean. Undecidable contexts stay unchecked (two-way
  prepositions depend on motion semantics, subject/object case needs
  parsing, relative pronouns are recognized and skipped, genitive
  prepositions accept the colloquial dative) — near-zero false positives
  by construction.
- **LanguageTool (optional)** — `uv sync --extra lt` and run with
  `ALLKLARO_LT=1` to add a local LanguageTool (Java) as a second opinion:
  its grammar/agreement findings feed the same corrective re-ask,
  filtered to hard grammar rules so style opinions never trigger
  re-translations.

## 📖 Gender lexicons (optional)

Local models sometimes guess wrong on the grammatical gender of loanwords
("einen Margarita", "la problema"). Feed the compiler any mix of
[dict.cc translation-file exports](https://www.dict.cc/translation_file_request.php)
(.zip) and [FreeDict source tarballs](https://freedict.org/downloads/)
(.src.tar.xz) in one call:

```bash
uv run python build_gender_lexicon.py \
  english-to-german-dictionary.zip \
  freedict-eng-deu-*.src.tar.xz freedict-deu-eng-*.src.tar.xz \
  freedict-eng-spa-*.src.tar.xz freedict-spa-deu-*.src.tar.xz ...
```

Then (optional, German declension checking at full strength) compile the
noun paradigm table — this downloads the Wiktionary-derived
[german-nouns](https://github.com/gambolputty/german-nouns) CSV:

```bash
uv run python build_noun_forms.py
```

This writes per-target lexicons (~27k German, ~1k Spanish noun genders,
~256k German noun forms with case/number readings) to
`~/.cache/allklaro/`. When a sentence being translated into German or
Spanish mentions one of these words, the correct article is injected into
the prompt (caipirinha → der Caipirinha, problem → el problema) —
overriding the model's guess and the built-in rules of thumb. Safety rules:
only unambiguous, same-spelling pairs are kept, so the lexicon can never
push a false-friend word choice ("gift" ≠ das Gift); words any dictionary
lists with two genders (Margarita {m} {f}, la/el radio) are dropped; and
gender-less dictionaries (FreeDict eng-spa) borrow genders from the others
only when every observation agrees. ⚠️ dict.cc data is licensed for
**private use only** — the compiled lexicons stay in `~/.cache` and must
never be committed.

## 🧪 Tests

```bash
uv run pytest                                   # fast unit suite (mocked Whisper/Ollama)
RUN_INTEGRATION=1 uv run pytest -m integration  # real Whisper + Ollama, ~25 s
```

200+ tests cover the VAD state machine, filtering, direction resolution,
merging, echo dedupe, speculation, context, speaker channels, export,
summarize, frontend wiring, the WebSocket protocol including error paths,
typed input, the draft+refine flow, correction retrieval, gender-lexicon
compilation, the constraint-intersection declension guard, and dialect
hints. Integration tests synthesize German and Spanish speech with macOS
`say`, push it through the real pipeline, and verify against the live
models that corrections steer wording, dictionary genders beat the
model's guesses, and dialect mis-hearings keep their meaning.

## 🛟 Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot reach Ollama" banner | `brew services start ollama` (or `ollama serve`) |
| Level meter flat | Wrong Input device, or call audio isn't routed into BlackHole (step 3 above) |
| First translation after a break is slow | The model is reloading; it stays warm for 60 min after each use |
| Utterances cut too early / too late | Drag the **Pause** slider; deeper VAD tuning at the top of `server.py` |
