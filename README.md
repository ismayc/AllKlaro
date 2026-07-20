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
| 📋 **One-tap copy** | Every finished translation carries a copy button — no text selection, a ✓ confirms it landed on the clipboard |
| 📖 **Learner tools** | One-click **Summarize** (summary + vocabulary list) and Markdown **Export** for later review |
| 📱 **Phone mode** | Serve over LAN HTTPS and use your iPhone's mic for in-person conversations |
| 🎯 **Focus mode** | Keep the newest text mid-screen ("Center latest") instead of at the bottom edge |
| 🔎 **Big-text view** | Tap a card's background and the translation fills the screen — made for showing the person across the table |
| 🗺️ **Dialect-aware** | Regional markers are detected and the intended forms hinted to the translator — German (Berlinerisch/Hessisch/Wormser: "dit", "ebbes", "nää", plus Whisper mis-hearings like "nett" for "net") *and* Spanish ("chamba", "ahorita", "plegar", "guay"); extend via `dialects.txt`, one `[lang]` section per language |
| 🎭 **German & Spanish style** | Optional dialect output: Berlinerisch, Hessisch, or Wormser Platt for German; Mexican or Barcelona Spanish — reply to your friends the way they write |
| 🤝 **Address form** | Pin how "you" comes out — du/Sie/ihr in German, tú/usted/ustedes in Spanish — or leave it on Auto and let context decide |
| 📚 **Glossary** | Pin names and terms in `glossary.txt` — biases recognition *and* translation |
| ⌨️ **Type to translate** | A text box under the feed — type instead of speaking, mic not required; same context, corrections, and draft+refine pipeline |
| 📖 **Gender lexicons** | Optional: compile dict.cc / FreeDict exports into loanword-gender dictionaries — "caipirinha" gets der/die/das and "problema" gets el/la injected |
| 🔍 **Long-press dictionary** | Hold any word in the feed for its Wiktionary entry — gender with article, plural, IPA, meanings; inflected forms chain to their base word ("ging" → gehen) |

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

## 🧠 Recommended models

**Recommendations as of 2026-07-19** — the open-model landscape shifts
every few months, so treat this table as a snapshot that will need
adjusting over time, not settled truth; the benchmark command below is
how to re-decide.

Measured with this repo's own probe suite (gender agreement, saved-
correction reuse, context pronouns, dialect and address-form steering —
the things that make AllKlaro AllKlaro), warm, on an Apple Silicon Mac:

| Your RAM | Draft model | Main model |
|---|---|---|
| 16 GB | Off (or `qwen2.5:7b-instruct` if it fits) | `gemma3:12b` |
| 24–32 GB (daily default) | `qwen2.5:7b-instruct` | `gemma3:12b` |
| 32 GB+ (important calls) | `gemma3:12b` | `qwen2.5:32b-instruct` |

Probe scores (2026-07-19): `gemma3:12b` **12/12** — the only model that
follows saved corrections *and* dialect/style instructions;
`qwen2.5:32b-instruct` 10/12 — the strongest raw translation and worth
its tier for high-stakes nuance, but it ignores injected dictionary-
gender notes and writes generic rather than Wormser dialect, so prefer
`gemma3:12b` as main if you lean on those features;
`translategemma:12b` 10/12 — excellent translation fidelity but it is a
strict-translator fine-tune that ignores your corrections and any style
steering; `translategemma:4b` 8/12; `qwen2.5:7b-instruct` 9/12 but ~2.5×
faster, which is exactly the draft job. Earlier rounds rejected
`qwen2.5:14b` (grammar) and all 3B-class general models (paraphrase too
freely even as drafts).

Benchmark any candidate yourself in one command (needs Ollama running
and the model pulled):

```bash
RUN_INTEGRATION=1 ALLKLARO_MODEL=<model> uv run pytest \
  tests/test_integration.py -k "ollama or api_translate" -q
```

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
- **📋 on any finished translation** — copy it to the clipboard with one tap,
  no text selection needed.
- **Type to translate** — the text bar at the very bottom works without the
  mic; auto modes detect the typed language. The 📥 button pastes your
  clipboard and translates it in one tap — made for text copied out of
  WhatsApp; dialect German in the paste gets the same `dialects.txt` hints
  as speech.
- **German style** — Standard, Berlinerisch, Hessisch, or Wormser Platt
  (Rheinhessisch): translations *into* German come out in that dialect
  (the declension guard steps aside, since "dit Haus" is not a mistake
  there).
- **Spanish style** — Standard, Mexicano (celular, computadora, ustedes,
  órale), or Barcelona (móvil, ordenador, vosotros, vale, with the odd
  Catalan loan like plegar).
- **Address (you)** — English hides whether "you" is du, Sie, or ihr (tú,
  usted, or ustedes). Auto lets the conversation context decide; pin it
  when you know who you're talking to. Barcelona style + plural uses
  vosotros instead of the Latin American ustedes. Also accepted by
  `/api/translate` as `"address": "informal" | "formal" | "plural"`.
- **Speak** voices: German `de-DE`, English `en-US`, Spanish `es-MX`
  (Latin American).
- **About** — links to the website and repo.
- **Long-press any word** — its dictionary entry slides up: gender with the
  article, plural, IPA, and meanings. Inflected forms bring their base word
  along ("ging" shows gehen too). Needs a one-time lexicon build:

  ```bash
  uv run python build_wiktionary_lexicon.py        # German + Spanish
  ```

  streams Wiktionary extracts from [kaikki.org](https://kaikki.org)
  (CC BY-SA, ~95 MB per language) into `~/.cache/allklaro/`. Add `en` for
  English lookups.
- Tap a card's background for a **full-screen big-text view** to show the
  person you're talking to.
- Settings persist; the WebSocket auto-reconnects; a wake-lock keeps the
  screen on while listening.

## 📱 Phone mode (in-person conversations)

Run **`Start AllKlaro (iPhone).command`** — it generates a self-signed
certificate (mic access requires HTTPS off-localhost) and serves on your LAN,
printing the exact URL. On an iPhone on the same Wi-Fi: open
`https://<mac-ip>:8710`, accept the certificate warning, press Start, allow
the mic, and lay the phone between the speakers. ⚠️ This exposes the app to
everyone on the network — use networks you trust.

**Add to Home Screen** (Share → Add to Home Screen) installs it like an app
— full screen, with the AllKlaro icon. If the icon shows as a lettered
tile: iOS's icon fetcher ignores the certificate exception you granted in
Safari, so install the certificate once — open `https://<mac-ip>:8710/cert`,
allow the download, install it under **Settings → General → VPN & Device
Management**, then enable it under **Settings → General → About →
Certificate Trust Settings**. Re-add to the home screen afterwards; this
also removes the warning page for good.

### 📲 iOS Shortcut: translate straight from the share sheet

`POST /api/translate` is a stateless endpoint made for Apple Shortcuts, so
a WhatsApp message can be translated without switching apps:

```json
{"text": "Kannste morjen ooch vorbeikommen?",
 "mode": "auto-de-en",          // optional, this is the default
 "de_flavor": "berlin",         // optional: berlin | hessian | worms
 "es_flavor": "mexico",         // optional: mexico | barcelona
 "address": "informal"}         // optional: informal | formal | plural
```

returns `{"source": "de", "target": "en", "translation": "...",
"translations": {"en": "..."}, "display": "🗣️ AllKlaro (DE → EN):\n..."}`
— `display` is the translation with a ready-made caption, for Shortcuts
that want a labeled result.

**Step 0 — prove the server is reachable** (do this before touching
Shortcuts; it isolates the two things that can fail):

1. On the Mac, run **`Start AllKlaro (iPhone).command`**. It prints your
   Mac's actual address, the exact Shortcut URL, and a browser smoke-test
   URL. ⚠️ Every address below written as `<mac-ip>` means *that* printed
   IP — it is different on every network, so never copy an address from
   an example.
2. On the iPhone, open `https://<mac-ip>:8710` in Safari — the app must
   load.
3. Still in Safari, open the printed smoke-test URL
   (`https://<mac-ip>:8710/api/translate?text=Hallo`). You should see
   JSON containing a translation. If steps 2–3 fail, no Shortcut can
   work: check same Wi-Fi, the printed IP, and the certificate (see
   troubleshooting).

**Build the shortcut** (Shortcuts app → **+**):

1. Tap the name at the top → **Rename** → `Translate with AllKlaro`.
2. Search Actions and add **Get Contents of URL**. Tap the blue `URL` placeholder
   and type the full URL: `https://<mac-ip>:8710/api/translate`.
3. Tap the action's **expand arrow (▸ / "Show More")**:
   - **Method** → **POST** (it defaults to GET — a GET here with no
     `text` returns only an error).
   - **Request Body** → **JSON**.
   - **Add new field** → **Text**. Key: `text`. For the value, tap the
     box, then in the bar above the keyboard tap **Shortcut Input** — it
     must appear as a blue pill, not typed words. (No bar? Tap the wand /
     "Select Variable" icon.)
4. In Search Actions, add **Get Dictionary Value**. It should read *Get Value for key in
   Contents of URL*; type `translation` as the key — or `display` for the
   captioned "🗣️ AllKlaro (DE → EN):" version. If "in" doesn't say
   **Contents of URL**, tap it and select that variable.
5. In Search Actions, add **Show Content** (named **Show Result** before iOS 26). It should
   show **Dictionary Value**; wire the variable if it's empty.
6. Tap **ⓘ** (bottom of the editor) → enable **Show in Share Sheet**.
   Closing that sheet reveals a **"Receive … input from"** header at the
   very top of the shortcut. In that header:
   - optionally limit the types to **Text**;
   - tap **"If there's no input: Continue"** and change **Continue** →
     **Get Clipboard**. ⚠️ This is the step that makes Back Tap / Home
     Screen launches translate your *copied* text — without it they send
     an empty request and show a blank result.
7. Test by tapping the shortcut **inside the Shortcuts app** first, with
   some German on the clipboard. Approve the one-time prompts: connect to
   your Mac's address (Allow), Local Network access (Allow), paste from
   WhatsApp (Allow — silence it for good under Settings → Apps →
   Shortcuts → **Paste from Other Apps** → Allow).

**Daily WhatsApp flow** (WhatsApp offers Copy but no Share on message
text): long-press the message → **Copy** (press-and-hold the text itself
first to copy only part of it), then launch the shortcut — the clipboard
fallback kicks in. Fast launchers: **Back Tap** (Settings → Accessibility
→ Touch → Back Tap → Double Tap → the shortcut), the Action Button, a
Home Screen icon, or a Shortcuts widget. In apps whose text selection
does offer Share (Safari, Mail), the share-sheet route works directly.

**Troubleshooting:**

- **Blank result** → the server answered with an error the dictionary
  step drops. Temporarily add a second **Show Content** right after *Get
  Contents of URL* showing **Contents of URL** — the raw JSON names the
  problem (`"No text to translate."` = the clipboard fallback from step 6
  isn't set).
- **"Could not connect" / SSL error** → the self-signed certificate is
  pinned to the Mac's IP at generation time. If your router handed the
  Mac a new IP since then, delete the `certs/` folder, rerun the iPhone
  launcher (it regenerates for the current IP), and reinstall via
  `/cert`. Also confirm phone and Mac share one Wi-Fi network.
- **Back Tap does nothing** → make sure the shortcut works when tapped in
  the app first; Back Tap only launches what already runs. Thick cases
  dampen it — tap firmly with a fingertip. Check Settings →
  Accessibility → Touch → Back Tap still lists the shortcut.
- **Local Network denied** → Settings → Privacy & Security → Local
  Network → **Shortcuts** must be ON.

## 🌍 Anywhere mode (Tailscale)

Phone mode works on your Wi-Fi; **anywhere mode** works from any network
— securely. AllKlaro has no login, so it must never be exposed to the
public internet (no port forwarding!). Instead, [Tailscale](https://tailscale.com)
(free for personal use) puts your Mac and phone on a private encrypted
WireGuard network that follows you around:

1. **Mac:** `brew install --cask tailscale-app`, open the Tailscale app,
   sign in. **iPhone:** install the Tailscale app from the App Store,
   sign in to the same account, allow the VPN profile.
2. Run **`Start AllKlaro (Anywhere).command`** — it fronts the server
   with Tailscale's HTTPS proxy (a real certificate: no `/cert` install,
   no warnings) and prints your Mac's stable private URL, e.g.
   `https://<your-mac>.<tailnet>.ts.net`. If Tailscale asks to enable
   HTTPS certificates for your tailnet, follow the link it prints (a
   one-time admin-console toggle).
3. Point the iOS Shortcut at `https://<that-name>/api/translate`. The
   name never changes — it works at home and away, and survives router
   IP reshuffles that break the LAN phone mode URL.

The MacBook must be on and awake while you're away: Settings → Battery →
prevent automatic sleeping (or `caffeinate -s`), and expect Ollama to
spin the fans when requests arrive. The Tailscale VPN toggle on the
phone must be on — that's the tunnel.

### 🚦 Start the server from your phone

`allklaroctl` is a headless start/stop script made to be run over SSH by
an iOS Shortcut — it brings up Ollama and the server, configures the
Tailscale HTTPS proxy, and waits until everything actually answers before
reporting:

**One-time setup, on the Mac.** System Settings → General → Sharing →
**Remote Login** ON (allow your user only). Then run `tailscale ip -4`
and note the `100.x.y.z` address it prints — you'll need it below.

**One-time setup, on the phone.** ⚠️ The tunnel must be up *before* the
shortcut runs — see *the tunnel can't repair itself*, below. Two ways:

- **Leave Tailscale connected** and just don't toggle it off. Tailscale
  already installs a broad VPN On Demand policy of its own while it's
  enabled, which keeps the VPN alive across reboots, crashes, and app
  updates. Nothing to configure.
- **Or make it explicit with rules:** Tailscale app → tap your profile
  picture (top right) → **VPN On Demand** → enable it, and set **Wi-Fi**
  and **Cellular** to **Always**. (Options are Always / Only On /
  Except On / Never / Do Nothing, per interface.) ⚠️ Custom rules
  *replace* Tailscale's automatic policy, so a wrong rule here can stop
  the VPN connecting at all — if you only want "stay connected," the
  first option is safer.

No `VPN On Demand` item in the menu? Update the app from the App Store.

**Build the shortcut** — one shortcut, named e.g. `StartAllKlaroServer`,
containing these two actions in order:

1. **Run Script Over SSH**
   - **Host:** the Mac's **Tailscale IP** from above (`100.x.y.z`).
     ⚠️ Use the number, *not* the `….ts.net` name — the Shortcuts SSH
     client doesn't resolve MagicDNS, so the name fails to connect here
     even though Safari resolves it fine. Unlike the LAN `<mac-ip>` used
     by phone mode, this `100.x` address is assigned by Tailscale and
     stays the same across networks and router reboots — type it once
     and leave it alone.
   - **Port:** 22 — **User:** your macOS username
   - **Authentication:** password, or an SSH key (Shortcuts can generate
     one; paste its public key into `~/.ssh/authorized_keys` on the Mac)
   - **Script:** `~/repos/AllKlaro/allklaroctl start`
2. **Show Content** — displays the reply ("running — https://…").

Then duplicate the whole shortcut, change `start` to `stop` in its
script, and name it `StopAllKlaroServer` for a remote off-switch.

**The tunnel can't repair itself.** The SSH action reaches the Mac
*through* the tailnet, so Tailscale must already be up on both devices
before a single line of `allklaroctl` runs. The script can't fix a down
tunnel; it isn't running yet. When the phone's VPN toggle is off you get
an opaque SSH connection error that looks like a broken *server* —
which is why the phone-side setup above matters.

If you'd rather connect per-shortcut than leave always-on enabled, add a
**Set VPN** action above Run Script Over SSH and pick the Tailscale entry
from its dropdown. Tailscale installs a VPN profile (listed under
Settings → General → VPN & Device Management), but iOS treats
app-provided VPNs differently from personal ones and Tailscale may not
be offered there — if it isn't, use always-on.

The Mac side needs no such step: Tailscale is set to start on login and
`tailscale serve` persists its config, so an awake, logged-in Mac is
always ready. The `open -a Tailscale` fallback inside `allklaroctl` is
there for local runs from the `.command` launchers, where it *can* help.

Limits: SSH can wake a sleeping Mac on the same network in some setups,
but it cannot power one on — for away-from-home starts the Mac must be
awake (see the sleep settings above).

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
