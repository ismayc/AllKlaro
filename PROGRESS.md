# AllKlaro — pipeline progress

Running list of what has been fixed and what is still open, kept in the order
the evidence justifies rather than the order things were noticed. Every claim
here is measured against the **real 54-minute conversation recording**, never
synthetic `say` audio, which cannot reproduce the load (it never queues).

---

## Done

- [x] **Live partials moved off the Whisper thread** onto Parakeet TDT 0.6B v3
      on its own worker. 884 ms → 71 ms for the same 6 s window; partials
      starved 84 → 0. Finals and speculations deliberately stay on Whisper.
- [x] **Refine pass gated on the translation backlog**, not just
      `whisper_pending` — the queue the next card actually sits in is Ollama's.
- [x] **`transcribe_ms` split into `queue_ms` + `decode_ms`.** Of a ~4 s
      "transcribe", only 2.2 s was decoding. This is what makes the rest of
      this list honest rather than plausible.
- [x] **Speculation reuse on `soft_max` splits.** Every miss was the same
      case: continuous speech cut at a micro-pause, where the speculation
      already held the emitted chunk plus 128 ms of silence. Miss rate
      45% → 6%; **55 → 42 Whisper decodes** for the same 40 utterances.
- [x] **Inverted model pairing no longer restored.** A saved draft larger than
      the main model put the big model on the critical path and left the small
      one cold — the largest single lever found, and a settings value rather
      than code.

---

## The finding that reframes the rest

**The bottleneck has moved off Whisper.** Whisper queue sits at 1–2, `queue_ms`
p50 is 0, decode is ~1.3–2.3 s. Meanwhile `translate_ms` p50 ranges 2.9–7.7 s,
and the refine pass — the other big Ollama consumer — is where the remaining
argument is. Further Whisper-side optimisation has little left to win.

*(The refine timeout is now `REFINE_TIMEOUT_SEC` 20 s, not 10; the "hits its
10 s timeout on 8–42% of utterances" that stood here read a censored metric.
Item 2 has the corrected accounting.)*

---

## Open

### 1. Audio accumulation — largest remaining component, cut by ~3.5 s
- [x] **Confirmed as the dominant term.** Pooled over the stub runs, the
      emitted chunk is **65%** of first-word lag at p50 (5.76 s of 8.71 s).
      The earlier 36–52% was measured against live Ollama, where a 2.9–7.7 s
      translate leg dilutes accumulation's share; both are right about their
      own run.
- [x] **The split points already existed.** `split_at` is updated at *every*
      micro-pause and was then ignored until `SOFT_MAX_SEC` (8 s) elapsed, so
      usable cuts were being found seconds early and discarded.
- [x] **Swept bracketed A-B-A**, on the chunk-count-invariant metric — lag at
      speech-run starts, which a lower cap cannot flatter by manufacturing
      more, shorter utterances:

      | cap | runstart p50 | vs 8.0    | verdict |
      |-----|--------------|-----------|---------|
      | 8.0 | 9092 ms      | —         | baseline |
      | 6.0 | 8631 ms      | −460 ms   | inside its 366 ms control spread |
      | 5.0 | 6827 ms      | −2264 ms  | real |
      | 4.0 | 6417 ms      | −2800 ms  | real, but pays steeply for 400 ms |

- [x] **Default is now 5.0.** Confirmed against live Ollama, where the
      expected cost did not appear: more, shorter chunks did **not** congest
      it. Translate p50 fell 3000 → 1744 ms and its worst case 14.7 s → 4.4 s,
      because a shorter chunk is a shorter prompt. Live first-word lag fell
      **10444 → 6932 ms**, a larger win than the stub predicted.
- [x] **Speculation coverage was *not* the real cost — that reading was
      wrong.** `spec:none` did go 6 → 20 live, but taking that as the cap's
      price does not survive looking at which utterances they are. Every
      `spec:none` is a `soft_max` split; every `pause` split is a hit (19/19
      at cap 8.0, 15/15 at cap 5.0). `specs_shed` is **0 in both arms**, so
      none of it is backlog shedding, and `spec:miss` — the only outcome that
      actually wastes a decode — barely moved, 2 → 3.

      The 20 split into two mechanisms, both pinned by tests in
      `tests/test_vad.py`:

      | mechanism | n (cap 5.0) | dead time | recoverable? |
      |-----------|-------------|-----------|--------------|
      | over-cap: no dip until past 5 s, so the first micro-pause sets `split_at` **and** fires the cut on the same frame | 13 | none | no — and nothing to gain |
      | stale `split_at` from a 6–9 frame dip, cut when `seconds` crosses the cap | 7 | p50 1.45 s | in principle |

      The first is the majority and is **free**: the cut is simultaneous with
      the split decision, so a speculation would submit the same audio at the
      same instant. Only the second leaves usable dead time, and filling it
      means speculating at *every* micro-pause — the redundant decoding the
      backlog gates exist to shed, on a Whisper thread already at 1.11x
      realtime.
- [x] **Sized before building, and it is too small to build.** `queue_ms` p50
      is **0** in every group, so a speculation is worth only the head start,
      not queue avoidance. A complete fix for the recoverable mechanism saves
      at most one decode (p50 **0.68 s**) on **7 of 52** utterances — about
      **92 ms** against a first-word lag p50 of 6.3 s, or **1.5%**. The
      remaining headroom is in translation, not transcription: `translate_ms`
      p50 1744–3243 ms with refine at its 10 s timeout — see item 2.
      Refines also shed more (37 → 49) but `refine_ms` sits at that timeout in
      *both* arms, so that is shedding work already failing.
- [x] **Validated on a second, harder slice — the refine pass was hiding it.**
      At 40:00 with refine on, the bracket gave a 1170/1207 ms gain against a
      1034/810 ms control spread across two runs: not demonstrable. Suppress
      the refine pass (`ALLKLARO_REFINE_MAX_IN_FLIGHT=-1`) and the control
      spread collapses to **82 ms**, whereupon cap 5.0 shows a **2272 ms**
      gain — against demo4's 2264 ms. They agree to 8 ms on different slices.
      The weak slice40 result was a measurement artifact, not a weaker effect.
- [x] **Residual tail cost, roughly halved by removing refine.** Utterances
      unfinished within the 120 s drain, at 40:00: cap 8.0 drops 1 either way;
      cap 5.0 drops 3 with refine on and 2 with it off. **A measurement
      artifact, not a defect**: this is a replay deadline, and in a live call
      those utterances arrive very late rather than vanishing. Recorded so the
      drop counts in a replay summary are not read as lost speech.
- [x] **Translating from partials: sized, and rejected for the same reason
      the speculation fix was.** `tools/partial_headroom.py` measures it
      offline on the real slice — the app's own VAD and partial ASR, no
      Ollama, deterministic — by asking when a partial first becomes good
      enough to translate versus when the chunk is actually emitted.

      | rule | fires on | saving p50 | similarity to final | leading words surviving |
      |---|---|---|---|---|
      | ≥20 chars, no stability required | 84% | **2.88 s** | 0.74 (p10 **0.25**) | p50 40%, **none at all in 35%** |
      | ≥40 chars, 2 stable partials | 10% | 0.99 s | **0.95** | p50 100% |

      The two regimes are the whole answer. Loose enough to fire often and
      the text is not what was said: the worst cases are Parakeet decoding
      *German audio as English* — `"Cool call it kicker males."` for *Ich
      kipp ja immer nicht*, `"Yeah, it's a modest type."` for *Das verdunstet
      aber eigentlich auch eine Menge*. Translating those would put nonsense
      on screen 2.9 s early and then take it back.

      Strict enough to be faithful and it fires on 5 utterances in 51 at
      ~1 s each: **~0.12 s expected, about 1.7% of first-word lag** — the
      same order as the 92 ms that killed the speculation idea above.
- [x] **The blocker is partial *fidelity*, not available time.** Requiring
      just two consecutive partials where one merely *extends* the other
      drops firing from 84% to 14%; requiring three, it never fires at all.
      Parakeet is not extending its guess as audio arrives, it is constantly
      **revising** it. The 2.88 s of headroom is real and stays on the table
      — anything that makes mid-utterance partials trustworthy reopens this,
      and the rig is committed to re-measure it in one run.

### 3. The measurement rig cannot resolve what it is tuning — fixed at p50
- [x] **Translation stage separated from model-server noise.**
      `tools/fake_ollama.py` serves `/api/chat` at a latency you choose,
      defaulting to the measured real p50s (draft 2600 ms, refine 8800 ms).
      `OLLAMA_URL` is now `ALLKLARO_OLLAMA_URL`-overridable so both arms of an
      A/B run identical source, the same rationale as `REFINE_MAX_IN_FLIGHT`.
- [x] **Repeatability measured, not assumed.** Two *identical* arms over the
      real 240 s slice, 42 utterances each: `first_word_lag` p50 differed by
      **130 ms**, against **3600 ms** for two identical live-Ollama arms.
      `translate_ms` p50 delta 0 ms, stdev 17 ms within a run. The rig now
      resolves sub-second effects, so items 1 and 2 are decidable.
- [x] **Residual noise is Whisper, and it lives in the tail.** Every large
      p90 delta between the identical arms is an ASR metric — `decode_ms`
      1022 ms, `transcribe_ms` 828 ms — while `translate_ms` p90 moved 84 ms.
      A fixed-cost ASR stub is the fix, deliberately *not* built: p50
      resolution decides items 1 and 2, and tail precision buys nothing until
      a p90 claim is load-bearing. **This is a decision, not a backlog item**
      — it stays closed until something actually needs a p90.
- [x] **Arms are not bit-identical.** Run 2 produced 41 finals to run 1's 40
      and 32 speculation hits to 31, so a small part of any delta is a
      different utterance set rather than timing. Compare distributions, and
      do not read a sub-100 ms difference as signal. **A standing rule for
      reading this rig, not work to be done.**
- [x] **The live-Ollama spread is worse than the 3600 ms recorded above, and
      it drifts *within* a session.** A-B-A on demo4 (2026-08-07) gave
      identical 10 s control arms of 11798 and 14355 ms first-word lag —
      **2556 ms** — and, less expected, refine counts swung just as hard:
      24% vs 47% kill-rate, 19 vs 10 landed. So "how many refines landed" is
      *not* one of the audio-determined counts that are safe at n=1. An
      in-session pair showed a clean 24%→8% win that the third arm destroyed.
- [ ] **The bind this creates for item 2, and there is no rig answer yet.**
      The stub is what makes lag decidable — but a fixed-cost stub *cannot
      time out*, so it cannot measure anything about the refine timeout, which
      is the one knob item 2 turns. That question is therefore stuck between a
      rig that is precise and blind to it, and one that can see it and cannot
      resolve it. A stub that models a *latency distribution* rather than a
      fixed cost would close this; not built.

*Done second, as planned: it is what makes items 1 and 2 decidable instead of
arguable — but only against the stub. Live-Ollama arms remain unresolvable
below ~2.5 s, which is where item 2's remaining questions live.*

### 4. Output quality, which lag work does not touch
Ground truth now exists: `tools/dump_transcripts.py` over **five** slices of
the recording — 1:00, 10:00, 25:00, 40:00, 50:00 — **259 utterances**, with
auto-detect and forced-`de` side by side. One slice was not enough: the first
version of the loop filter looked perfect on 51 utterances and had a false
positive waiting at 10:00. Rebuild the extra slices the same way as
`demo4.wav` (see the `/test-allklaro-translating` skill), varying the offset.

**The conversation shifts language as it goes**, which no single slice shows:
English is 12% of utterances at 1:00, 6% at 10:00, 19% at 25:00, 31% at 40:00
and **60% at 50:00**. Any behaviour keyed to a fixed source language gets
worse the longer the conversation runs.

- [x] **"German transcribed as English" is largely a misdiagnosis.** Six of
      51 utterances auto-detected as English, and on inspection most are
      *correct* — the conversation is genuinely bilingual. #49 stays
      `"We had..."` even when Whisper is forced to German, and #50/#51 are
      fluent English sentences. Auto-detect was right; the cards were not
      wrong for the reason assumed.
- [x] **The real failure is the mirror image, and it is reachable from the
      UI.** `lang_hint(mode)` pins Whisper's language for any *forced*
      direction (`de-en` → always German) and only returns `None` for
      `auto-*` modes. Pick "German → English" during a bilingual conversation
      and every English utterance is decoded as German. Measured on the same
      audio, that turns *"we had some sort of problem where they came and
      fixed something"* into *"Und so haben wir so ein Problem, wo sie sich
      und die Füße starete, die sich so starete, die Füße starete, die Füße
      starete."*
- [x] **Fixed by falling back, not by changing the default.** The loop filter
      already stopped that text reaching the screen — which quietly made it
      *worse*: the utterance became a silent `discard_empty`, so the English
      was lost rather than mangled. Now, when a forced decode is rejected
      outright and the raw text was substantial (`FORCED_REDO_MIN_CHARS`, 40),
      the audio is decoded once more with the language free, and if the
      detected language is not the pinned one that utterance is translated the
      way it was actually spoken. The mode is untouched — only the utterance
      is redirected — and the card carries the `auto` chip, because the
      direction came from the audio rather than from the user's choice.
      Multi-target modes (`de-en+es`) are excluded: there is no single
      language to fall back to.
      The cost is one extra Whisper decode on the one thread, and only on a
      substantial decode that cleaning threw away entirely — never on the
      silence and short noise that make up most rejected chunks.
- [x] **Decided (Chester, 2026-08-07): the forced modes stay, unchanged, and
      the default stays `auto-de-en`.** The fallback removed the sharp edge
      that made this urgent, and a forced direction still earns its place
      where bilingual drift does not happen — typed input and the iOS
      Shortcut, where you know what language you just pasted. **A decision,
      not a backlog item.**
- [x] **A phrase-level repetition loop passed every filter — now caught by
      `has_phrase_loop`**, which flags any 3-word phrase occurring 3+ times.
      It runs *after* `collapse_repeats`, not before: a "L-L-L-…" tail is 100
      repeated tokens to a word-level check, so testing raw text would throw
      away the real speech in front of a loop the collapser can rescue.
      Validated end to end on the dump — 0 of 51 real utterances dropped, and
      exactly the one garbage utterance removed. Original evidence: That same
      utterance: `is_degenerate` compression 1.33 (needs > 4.0), Whisper's own
      `compression_ratio` 1.57 (drops at > 2.4), `collapse_repeats` unchanged
      (needs 8+ repeats of a unit ≤ 12 chars). It is a ~17-char phrase
      repeated 3–4 times, which is too few repeats and too long a unit for
      all three guards. `cleaned == raw`; nothing caught it. Note this is not
      a `HALLUCINATION_RE` gap as originally written — that is a blocklist of
      stock phrases and was never the mechanism.
- [x] **Loops are their own defect, not a symptom of the forced language** —
      corrected on the wider corpus. Under plain auto-detect, 4 of 259
      utterances are degenerate: "EP" ×32 at 25:00, "PPE" ×60 at 40:00, "I"
      ×50 and "I'd like to" ×5 at 50:00. The first pass looked at one slice,
      found none, and drew the wrong conclusion.
- [x] **Forcing German is damaging, but less uniformly than first written.**
      On the 32 English utterances at 50:00 it produced *zero* loops and only
      2 garbage drops; most came back as fluent, often *correct* German
      ("Yeah, I see them." → "Ja, ah, ich sehe sie."), because a language hint
      makes Whisper translate rather than transcribe. One case went the other
      way entirely — auto-detect emitted "I I I I…" where forced German gave a
      clean "Ja!". So the fix is not simply "always auto": it is worth
      measuring which mode wins per utterance before changing the default.
- [ ] A wrong card is worse than a slow one.

### 6. One mic carries everyone — now marked where the voice changes
Multi-speaker *load* is already covered: the 54-minute recording is a real
group conversation captured through a single mic, which is the only setup
intended, so every lag number here already includes overlap and turn-taking.
That also explains why `soft_max` dominates (22 of 40 splits) — several
people trading turns rarely leave the ~700 ms gap that closes an utterance,
so the audio reads as continuous speech. This is a cause of item 1, not a
separate load risk.

- [x] **`SPEAKERS = {0: "you", 1: "them"}` is a channel tag, not
      diarization** — it records which input stream the audio arrived on, set
      by the client. With one mic for the whole room every card is `"you"`,
      so a group transcript cannot show who said what.
- [x] **Decided (Chester, 2026-08-07): no diarization. A break when the voice
      changes, and nothing more.** Not "who", which would need a third model
      on a GPU that Whisper, Parakeet and Ollama already contend for — just
      where one person stopped and another started. A dashed *new voice* rule
      between cards, drawn from `voiceprint.py`: 24 log-spaced spectral bands
      (timbre, mean-removed so loudness cancels) plus median F0 by
      autocorrelation over voiced frames. numpy on audio already in memory,
      about a millisecond, no model.
- [x] **Compared where the utterances are still in order**, in the frame loop
      rather than in `handle_utterance` — the handlers run concurrently and
      finish out of order, so a comparison made there would sometimes be
      against the wrong neighbour. A merged fragment never shows a break: a
      merge means one person carried on through a micro-pause, which is the
      whole reason it merged.
- [x] **The threshold is not the highest-scoring one, again for the reason
      item 4's detector was not tuned to its best score.** Swept over 6
      synthetic voices x 6 lines (`tools/voice_eval.py`), F1 peaks near 0.15;
      **0.35 ships**. A false mark asserts a speaker change that did not
      happen, while a missed one only omits a divider, so the two errors are
      not worth trading evenly. At 0.35 the synthetic same-speaker p90 is
      0.161 against a different-speaker p10 of 0.295 — the distributions do
      not overlap — and on the real 240 s slice it marks 26% of adjacent
      pairs, a plausible turn rate for three-plus friends.
      ⚠️ **Synthetic voices are an UPPER BOUND.** They have no overlap, no
      varying distance from the microphone, and no two people who sound
      alike. This is the same trap as measuring pipeline load with `say`,
      wearing a different costume. The real slice has no speaker labels, so
      its 26% is descriptive only.
- [x] **One tuning bug found and fixed by the mutation run.** F0 was computed
      over the *loud* frames, the same subset used for the spectrum. That
      coupled two things that should be independent: appending unvoiced noise
      to a vowel shifted the estimate 19 Hz — not because noise frames were
      counted (none pass the voicing test) but because they raised the energy
      percentile and so changed *which voiced frames* survived it. Decoupled,
      the same audio with and without the noise gives the identical answer.
      The naive fix then let weak transitional frames in and pushed
      same-speaker p90 to 0.642, so the voicing threshold was retuned to
      0.45 — where the distributions stop overlapping. At 0.6 precision
      reaches 1.000 but recall falls to 0.69, because too many utterances
      lose their fundamental and fall back to timbre alone.
- [x] **Measured on the real recording, and it does not work. Marks are now
      OFF by default** (`ALLKLARO_VOICE_MARKS=1` to see them anyway).
      The synthetic numbers above were not merely optimistic, they were
      qualitatively wrong.

      **The labels were available after all**, without diarization and
      without anyone listening: the pipeline's own split reasons supply them.
      A `soft_max` cut means the 5 s cap fired while someone was still
      talking, so the *next* utterance continues the **same speaker**; a
      `pause` cut is 700 ms of silence, which is where a turn actually
      changes. Over the full 54 minutes — 706 utterances, 534 continuations,
      171 pauses:

      | threshold | marked across a continuation (same speaker) | marked across a pause |
      |---|---|---|
      | 0.25 | 45.7% | 47.4% |
      | **0.35** (shipped) | **33.7%** | **35.1%** |
      | 0.45 | 24.7% | 29.2% |
      | 0.60 | 11.6% | 16.4% |

      The two columns are the same column. If the signature carried speaker
      identity, continuations would sit far below pauses; they do not. At the
      shipped threshold **one mark in three would land mid-sentence**, and no
      threshold repairs it — by 0.60 the gap is still inside noise and the
      feature marks almost nothing.
- [x] **Two false trails, recorded because both were convincing.** First:
      comparing the two halves of one utterance looked like a label-free
      same-speaker probe, and gave 30% above threshold — until the synthetic
      control ran the same way and gave **33%** for a TTS voice with zero
      speaker variation. Half an utterance is too short and too phonetically
      idiosyncratic to describe a voice, so that probe measures *what was
      said*, not *who said it*. Second: `soft_max` chunks might straddle a
      speaker change and contaminate the set — checked, and pause-only chunks
      were *worse*, not better, so contamination was not the explanation.
- [x] **Why synthetic lied, mechanically.** Timbre survives real audio
      (same-speaker p90 0.251); **F0 does not.** On real speech 6.8% of
      same-speaker pairs differ by within 0.15 of a *full octave* — textbook
      autocorrelation octave errors, which clean synthetic tones never
      produce. At `PITCH_WEIGHT` 0.6 a single octave error contributes 0.6 to
      the distance on its own. Octave-folding the comparison helps the
      symptom and not the disease: folded, same-speaker p90 is 0.361 against
      an adjacent-pair 0.377 — still no separation.
- [ ] **What would actually work, and it is a real dependency.** A pretrained
      speaker encoder (ECAPA/GE2E class) rather than 24 bands and a pitch
      track. torch 2.13 and sklearn are already installed, so the marginal
      cost is a small model, not a runtime — but it is a genuine new
      dependency in an offline app and is Chester's call. The measurement rig
      above is the part worth keeping: it can score any replacement against
      real labels in one run, which is what this section lacked all along.

### 7. A running gist, pinned above the feed
- [x] **Built.** A short summary sits above the conversation and is *folded*
      forward — each refresh sees the previous gist plus only the lines since,
      so cost is flat over a 54-minute call instead of growing with the
      transcript. `fold_gist` / `gist_messages` / `remember_for_gist` in
      `server.py`, rendered by `showGist` in `static/app.js`.
- [x] **Built to lose.** This is a second background job on the Ollama the
      translator is using — the exact shape that makes the refine pass fail —
      so it is gated the same way and more tightly: once a minute
      (`GIST_INTERVAL_SEC`), never while anything is in flight or queued
      (`GIST_MAX_IN_FLIGHT`, `GIST_MAX_QUEUE`, both ≤ the refine thresholds
      and asserted to stay that way), bounded batch, bounded backlog, and a
      reentrancy guard so a slow fold cannot pile up duplicates.
- [x] **A failure costs nothing.** The batch is dropped only once a fold
      succeeds, and is re-identified by uid rather than index so a concurrent
      trim cannot silently lose utterances. A blank or failed answer leaves
      the previous gist on screen.
- [x] **The bullet format had to be spelled out.** Asking gemma3:12b for
      "at most 3 short bullets" produced prose every time; naming the line
      format (`- `) fixed it. Recorded in the prompt itself so it is not
      re-litigated.
- [x] **Tested**: 22 new tests, all 10 mutations of the guarantees caught,
      plus two integration tests against the real model — one that a bilingual
      exchange yields an *English* gist rather than a translation, one that
      earlier context survives a fold.
- [x] **Driven on the real recording, and it was broken in four ways.** A
      240 s slice at 25:00 through the app's own audio path:
      - *It never fired once.* Gated at or below the refine thresholds,
        `in_flight` sat at 3-6 for the whole slice and touched 2 only during
        the drain. Yielding to the backlog forever is not a policy, it is the
        feature not existing — hence `GIST_MAX_STALE_SEC`.
      - *It lost the specifics.* "fly to Hawaii in March" became "their trip"
        in **4 of 8** folds. The prompt now defends names, places, dates and
        numbers explicitly: **8 of 8**.
      - *It grew.* 4 bullets → 8 across two live folds. Told to produce three
        lines *total* rather than three new ones, it holds at 3 over a
        six-fold run (293 → 430 chars).
      - *It leaked the transcript format.* By fold 6 it copied a line through
        verbatim as `- [DE] Ich habe in Opas Bericht gelesen.` Stripped on the
        response, not merely forbidden in the prompt.
- [ ] Still unverified over a *full* call: six folds is not fifty, and slow
      semantic drift (fold 6 already conflated who was displeased about what)
      is the failure mode a 54-minute conversation would expose.
- [x] **A reconnect no longer restarts the gist.** It lives in the WebSocket
      session, so a dropped connection began a second summary underneath text
      describing the first. The client is the only party that survives a
      reconnect, so it hands the gist back in `config` (`gist_text`) and the
      next fold continues. It only ever *seeds an empty* gist — every settings
      change resends config, and none of those may roll a live summary back —
      and it is bounded at `GIST_SEED_MAX_CHARS`, since the value re-enters
      the fold prompt and would otherwise be a client-controlled prompt of any
      length.

### 5. Dialect selection was dormant on the audio path — fixed
- [x] **Confirmed dormant.** Whisper normalises dialect to standard
      orthography. Over **514 word tokens** of the real Berlin recording,
      zero unambiguous dialect markers appeared — only `des`×2 and `mehr`×1,
      both ambiguous — so the hint could not fire once. The archived
      conversation transcript agrees: 0 of 15 utterances.
- [x] **Fixed by using the dialect the user selected** instead of inferring
      one from spelling. The flavor selector only ever steered *output*; it
      now also tells the source side what is being heard. An asserted dialect
      makes the ambiguous entries usable on their own, which is where the
      actual repair lives — `? nett = net (nicht)` was in the lexicon all
      along and unreachable.
- [x] **Verified against the real model**, not just the prompt text:
      "Ich habe das nett verstarne." went from *"I understood that nicely"*
      — the exact inversion — to *"I didn't understand."*
- [x] **Ambiguous entries now name their dialect.** Without this, selecting
      Berlinerisch would have offered the Rhine-Hessian reading of "mehr"
      (= *mer*, wir) to a Berlin speaker: a new error rather than a fix.
- [x] **Dormancy confirmed at 4.4x the original sample.** Across five slices
      of the recording — **2277 German word tokens** — there are **zero**
      unambiguous dialect markers from the 97-key lexicon. Only the ambiguous
      `mehr` ×10 and `des` ×4 appear. The spelling-based hint could never have
      fired, and asserting the selected dialect was the right repair.
- [x] **The gap is open a crack: the first entries keyed on a mis-hearing,
      and they fire.** The transcript route did yield something after all —
      not by looking for dialect spellings, which are absent by construction,
      but by finding an utterance whose *translation* was nonsense. Over the
      full 54 minutes, transcribed with the app's own Whisper (8299 word
      tokens) and cross-checked against an independent decoder of the same
      audio:

          both systems wrote : "Ich kicke ja immer nicht.
                                Meistens *guckt* der Arvid nach."
          actually said      : ick kieke ja immer nich   (kieken = gucken)
          both translated    : kicking a ball

      Two independent decoders making the identical mistake is the audio, not
      one model's quirk — and the speaker gives the meaning away in the next
      breath by using the standard word. So `kicke`, `kickt` and `gekickt`
      are now in `dialects.txt`, keyed on what Whisper *writes*.
      Necessarily **ambiguous and `[berlin]`-tagged**: kicken is an ordinary
      German verb, so the reading is only offered to someone who has said
      they are listening to a Berliner, and hedged even then. Verified
      against gemma3:12b, which is the only test that counts here:

          without the hint : "I never actually do it, though."
          Berlinerisch     : "I never really look, mostly Arvid checks."
- [x] **Phonetic search for more of them does not work, in either
      direction** — `tools/dialect_mishearings.py`, kept as the record.
      Filtering to words absent from the 368k-entry Wiktionary lexicon gives
      five candidates and all five are *English*: the conversation is
      bilingual, so `what`~`wat`, `next`~`net`, `hot`~`hoscht`. Dropping the
      filter is worse — `die`~`dit`, `ist`~`nischt`, `auch`~`aach`, dozens of
      times each. The mis-hearings that matter land on **real German words**,
      which is precisely why nothing downstream catches them, and precisely
      why a dictionary filter cannot find them.
- [ ] **The rest still needs an ear.** One entry from one utterance is not a
      catalogue. The method that worked — read the utterances whose
      translation is implausible, confirm against a second decoder — is
      repeatable but manual, and the recording contains little detectable
      dialect to begin with (zero unambiguous markers in 2277 tokens). Every
      *other* ambiguous entry remains Hessian / Rhine-Hessian.

### 11. Translate less, rather than faster — no latency win, product call open
Came out of a deliberate lateral pass over the product rather than the
pipeline. Every other lever divides the same work up differently; this is the
only one that makes there be less of it, and for someone learning German it
is arguably the better behaviour anyway — translating a sentence you
understood removes the reason to practise.

- [x] **Built and off by default.** `known_words.txt` (absent = feature off)
      holds the listener's vocabulary; an utterance is skipped only if it is
      German, at most `KNOWN_SKIP_MAX_WORDS` (8) long, and *entirely* covered.
      Conservative on purpose — a needless translation costs a little time, a
      wrongly skipped one costs the listener the sentence. The card still
      appears with the heard text, and the ✨ tap fetches the translation if
      the guess was wrong, which is what makes skipping safe to do at all.
- [x] **The headline number was measured on the wrong unit.** 42% of segments
      are ≤3 words and 43% are covered by a 300-word vocabulary — but those
      are *Whisper's* segments. The pipeline's utterances are VAD chunks up to
      5 s with fragments merged, so live the gate fires on **10%**, not 43%.
- [x] **And at 10% it buys no latency.** Deterministic A/B against the stub,
      comparing the 47 utterances translated in both arms:

      | | first-word lag p50 | p90 |
      |---|---|---|
      | baseline | 7730 ms | 11915 ms |
      | skip-known | 8104 ms | 12084 ms |

      Paired per-utterance the median delta is **+15 ms** and 23 of 47
      improved — a coin flip. The skipped work was 10% of utterances and 10%
      of Ollama time, so this is not "the cheap ones were removed"; removing
      a tenth of the load simply does not move the queue.
      ⚠️ The stub draws latency from a distribution that ignores prompt
      length, while real Ollama is faster on short prompts (translate p50 fell
      3000 → 1744 ms when the chunk cap dropped). So the rig is **generous**
      to this idea and it still did not pay.
- [ ] **The product question is untouched by that and is Chester's.** Whether
      a learner wants "Ja." translated is not a latency question, and the
      feature should not be sold as one. Left off pending that call; a real
      deployment would key on a frequency list or his own vocabulary history
      rather than the 300 commonest words of one conversation.

### 12. Cut on the verb, not on a stopwatch — measured, premise wrong, declined
The next lateral idea after 11, and the most plausible-sounding of them:
`SOFT_MAX_SEC` (5 s) cuts an utterance short, German is verb-final, so the cut
lands before the verb and the card cannot be translated — which would also
explain why translating from partials was wrong 35% of the time (item 1). The
intervention would be to close on syntactic completion instead of on elapsed
time. Sized against the real hour first, and it does not survive.

- [x] **The cap does not cut on a stopwatch.** `split_at` holds the *last
      micro-pause* seen, and when the cap fires the VAD emits everything up to
      that pause, not up to the 5 s mark. The app already cuts at the
      speaker's own hesitation. Emitted `soft_max` chunks average 5.5 s and
      run as short as 1.7 s, which is the giveaway.
- [x] **And the utterance is not the card.** The merge from item 8 runs before
      translation, so broken-ness measured on VAD utterances counts fragments
      the pipeline already repairs. Scoring the wrong unit here would have
      overstated the problem by 10 pp.
- [x] **No difference between a cap cut and a real pause.** 54 minutes, 703
      utterances, 522 German. The cap fires on **76% of all splits** — the app
      is almost always cutting on the cap — and its boundaries are still no
      worse than the ones 700 ms of silence produces:

      | | soft_max | pause | p |
      |---|---|---|---|
      | raw utterances broken | 50.9% | 47.9% | 0.59 |
      | merged cards broken | 41.1% | 38.7% | 0.70 |

- [x] **The ceiling is what settles it**, with no significance test involved:
      the boundary this idea wants to wait for is *itself* broken 38.7% of the
      time. Cards are fragments because spontaneous speech is fragmentary, not
      because of the cap. Closing on syntactic completion has ~2.5 pp of
      headroom and costs latency, which is the one thing this pipeline cannot
      spare. **Not built.**
- [x] **The merge from item 8 is confirmed to earn its place** — it takes
      broken cards from 50.4% to 40.7%.
- [x] ⚠️ **The null is a null, not a proof.** The `pause` arm is small — 96
      German utterances, 75 once merged — so the smallest difference this
      design could detect is **15.8 pp** on raw utterances and **18.0 pp** on
      cards. It rules out a large effect and not a small one, and
      `tools/boundary_quality.py` prints that number next to every comparison
      so the point cannot be quietly dropped.
- [x] ⚠️ **No verb-specific rate exists.** Over 522 German utterances the
      judge emitted 332 COMPLETE and 190 MISSING_VERB and *never once*
      MISSING_OTHER, so the three-way collapsed to a binary. Broken-vs-whole
      holds because the validated detector was the union of both broken
      labels, but the narrow claim about the finite verb was not measurable
      this way. Anything that needs it needs a parser, not a prompt.
- [x] **The judge was validated before it was believed.** 51 hand-labelled
      utterances. gemma3:12b is unusable here — precision **0.30**, it calls
      "It's because of the flood." verb-missing — and few-shot moved it to
      0.31. `qwen2.5:14b-instruct` plus a casing rule reproduces the hand
      labels at 86% agreement (precision 0.84, recall 0.93) and matched
      qwen2.5:32b within noise at a quarter of the wall clock.
- [x] **Shipped: merge on casing, not only on punctuation.** German
      capitalises every noun and every sentence start, so a chunk opening
      lowercase is a continuation — free, no model call, precision 0.82
      against hand labels. `looks_finished()` cannot see those, because the
      evidence sits on the *current* chunk while the rule inspects the
      previous one. `continues_previous()` in `server.py`.
      Over the hour: **554 → 500 cards** (German 391 → 346, mean 1.34 → 1.51
      utterances per card). It rejoins exactly the verb-final splits this item
      was aimed at — *"...in dem Bereich des Gartens."* + *"ausgestreut, damit
      das Wasser besser abläuft."*, which separately produced a verb the model
      invented and a participle with no subject.
      ⚠️ Cards get longer: p50 84 chars, p90 234, max 474. Sixteen exceed
      `MERGE_MAX_CHARS` because the cap is checked *before* appending, so a
      299-char card can still grow — pre-existing, more visible now.
      ⚠️ It cannot be scored with the default metric: the casing rule is both
      the intervention and half the detector, so `tools/boundary_quality.py`
      refuses that combination rather than reporting a win it manufactured.

### 13. "What did they just say?" — the ✨ tap widened to a window
Item 12 declined to change where the pipeline cuts, and in doing so measured
the thing that made this worth building: **41% of cards are still fragments**
after the item 8 merge, because that merge only reaches across a 2 s gap and a
German clause routinely spans more. Item 8 already established what a fragment
costs — each half translated with no sight of the other. This does the same
repair on demand, over a stretch the automatic merge cannot reach.

- [x] **Built.** ⏪ in the topbar joins the cards from the last 45 s and asks
      for one translation of the whole passage. `recap_window` in `server.py`,
      `recapWindow` / `requestRecap` in `static/app.js`.
- [x] **The saving is real and it is not about latency.** Verified in Chrome
      against gemma3:12b, on a split of exactly the shape item 12 counted:

      | | |
      |---|---|
      | card 1 alone | "And underneath that, they **put** gravel in that area of the garden…" |
      | card 2 alone | "**spread out** so that the water drains better." |
      | ⏪ together | "And underneath that, they **spread** gravel in the garden area so the water drains better." |

      Card 1 invented a verb because `ausgestreut` had not been said yet;
      card 2 is an orphaned participle with no subject. Neither is fixable by
      a better model — the words were not there. Joining is the only repair.
- [x] **45 s, widened from 15 s** (Chester, 2026-08-09) — 15 s is about
      two cards on real conversation, barely more than the card already on
      screen; the question wants the stretch.
- [x] **Windowed by when cards ARRIVED, not when the words were spoken.** The
      listener asks because of what they just read, and under lag those are
      different instants. Arrival is the one that matches the question.
- [x] **It changes nothing.** Not written to `history`, replaces no card,
      steers no later translation — the passage is already in the context as
      cards, and a second copy would be a translation passed off as speech.
      Pinned by a test, since nothing on screen would reveal the difference.
- [x] **Ungated like the ✨ tap, for the same reason** — somebody is waiting
      on purpose, and the moment the pipeline is behind is exactly the moment
      something got missed. One at a time, though: consecutive taps cover
      almost the same seconds, so a queue of them is one passage racing
      itself. Timeout, `busy`, and `failed` all answer, because a button that
      silently does nothing is indistinguishable from a broken one.
- [x] **The window lives in one place.** `RECAP_WINDOW_MS` in `app.js` only —
      the server never sees a card, just the joined passage, which it bounds
      because a client is not to be trusted about length. A static test ties
      the button's "15 seconds" to the constant, so the tooltip cannot start
      lying about the feature's scope.
- [ ] **Untried on the phone.** Verified at a 400 px viewport in Chrome with
      no overflow, but not yet on real iPhone Safari. The dismiss ✕ is 23×22,
      matching the gist toggle it sits beside and under the 44 px Apple asks
      for — left consistent rather than diverged, but it is a thumb target on
      the surface you actually read.

### 14. A comprehension feedback loop — no input exists, and none is coming
The last of the lateral ideas: learn what the listener actually understands
from what he does, and adapt — feeding, among other things, the known-words
gate of item 11 that is still sitting off pending a product call. It dies
earlier than item 12 did. Item 12 at least had a premise worth measuring;
this one has no input at all.

- [x] **Nothing is recorded.** Word lookups, ✨ taps, ⏪ recaps: every one is
      served and then discarded. `corrections.jsonl` is the only learner
      signal that persists, and `known_words.txt` does not exist, so item 11's
      gate has never been on outside a benchmark.
- [x] **And the interaction it would learn from does not happen.** Eight days
      of real use — `/tmp/allklaro-server.log`, 2026-07-31 to 2026-08-08, the
      `:8710` server, phone over Tailscale:

      | signal | events in 8 days |
      |---|---|
      | word lookups | **4** |
      | hand corrections | **0** |
      | ✨ improve taps | **0** |

      All four lookups are one burst — `anniversary` three times and `Happy`
      submitted as *German*, which cannot have returned anything. So the true
      count of "I did not know that German word" is plausibly zero.
- [x] **Not a broken button.** `/api/correction` validates and appends, is
      covered by `tests/test_corrections.py`, and `corrections.jsonl` has been
      0 bytes since 19 July. The long-press fix the phone needs
      (`-webkit-touch-callout: none`) is in `style.css`. The affordances work;
      they are simply not used, which is what a person listening to another
      person does — attention is on the conversation, not on the app.
- [x] **Even given a signal, the shape of speech is against it.** Over the
      real hour: 5,697 content tokens, **1,376 distinct forms, 61% of them
      appearing exactly once**. A loop pays off when what you learned comes
      back, and within a conversation most words never do. The head is the
      opposite problem — the top 100 forms are 61% of all tokens, and those
      are the words a learner already has. Whatever is left in between is
      where item 11 measured its 10% firing rate and no latency win.
- [x] **Declined.** Not built, and nothing added to record for later: a
      collector with no consumer is how `corrections.jsonl` got to be an empty
      file that three features consult.
- [ ] **What the idea was reaching for, without needing the loop.** The
      learner value here is vocabulary review, and that needs no model of what
      Chester understands — only the transcript, which already exists. 89% of
      the hour's distinct forms have a Wiktionary entry, and a frequency list
      drops the trivial head. That is the roadmap's first item (Anki/vocab
      export), it is unblocked, and it costs one small pure-Python dependency
      (`wordfreq`, not currently installed). Chester's call.

### 15. Measured against a batch service — and item 12 answered the wrong question
Chester compared a live run against a TransyncAI transcript of the **same
recording** and called it dramatically worse. He was right, and the reason is
not the translation model.

- [x] **The gap is chunking.** Same 54 minutes: TransyncAI produced **216
      segments averaging 39 words**; this pipeline produced **554**. Three of
      five hand-aligned passages lost their meaning purely to a cut —
      *"das war ja auch zwischenzeitlich hier mächtig"* without *windig*
      ("pretty intense" instead of "pretty windy"), and *"gibt es da auch
      keine Abdeckung für."* without the pool it refers to.
- [x] **One case is transcription, not translation.** *"Er hat kein Solar da
      drauf"* was heard as *"Die hat keinen Soldaten"* → "She doesn't have any
      soldiers." No translation model can repair a sentence it never heard.
- [x] **AllKlaro wins on dialect**, worth recording because it is the one
      place recent work beat a commercial service: *"und dann hat Uwe
      gekickt"* is rendered by TransyncAI as "Uwe **kicked it off**". The live
      run said "Uwe **looked**", which is correct — Berlinerisch *jekiekt* =
      *geguckt*, confirmed two lines later by *"Meistens **guckt** der Arvid
      nach"*. That is the mis-hearing lexicon earning its place.
- [x] **Item 12 answered the wrong question.** It asked whether the 5 s cap's
      boundaries are worse than a real pause — they are not, and that stands.
      The useful question was whether chunking costs quality *at all*, and
      against a system that does not chunk the answer is plainly yes: 41% of
      cards are fragments against ~0% for turn-level segments. The lever was
      never where to cut; it is how much to rejoin.
- [x] **Still 500 against 216 → closed by item 16.** This box assumed the
      lever was a wider `MERGE_GAP_SEC` or a higher cap, and both buy
      coherence with latency. Measured, that assumption was wrong twice over:
      widening the gap to 6 s barely moves the count (438 → 394 over the
      hour), and merging replaces the card in place, so the price was never
      latency — it is retranslation. The lever that works is the split
      reason; see item 16.
- [x] ⚠️ **A wedged Ollama looks exactly like a quality regression.** The run
      that prompted this was **draft-only**: `gemma3:12b` could not load, so no
      card was ever refined and every ✨/⏪/gist call timed out silently. The
      draft rendered "It's because of the flood." as *"Es ist aufgrund des
      Überschwemmungs."* where the main model gives *"Das liegt am
      Hochwasser."* Cause was `OLLAMA_CONTEXT_LENGTH=131072` — a 128k KV cache
      per model, so draft and main evict each other. Set the context length in
      **Ollama.app's own settings**; the env var alone is overridden by the app.

### 16. The VAD already knows who is still talking — merge on the split reason
The stated goal is to match the batch service's reading experience — ~216
turn-sized segments of ~39 words over the hour — live. Both text rules
together got to 438 cards, and the obvious knobs go no further: over the full
hour, `gap 2→6 s` and `max 300→800` chars move 438 → 394. The text is the
wrong witness — Whisper punctuates a mid-flow cut exactly like a finished
sentence, so `looks_finished` and `continues_previous` both read "…die
Klimaanlage lief." / "Und darunter…" as two complete thoughts.

- [x] **Shipped: `flowed_on()` — merge when the previous chunk was a
      `soft_max` split.** That split is the cap firing on speech that never
      paused: the VAD emits up to the last micro-pause *because the speaker
      was still going*, so the next chunk is the same person continuing. The
      signal was already on `meta` for the stats dump; the rule is one line.
      Over the hour: **438 → 228 cards, 8.1 → 4.2/min, mean 17.2 → 33.0
      words** against the batch service's 216 / 4.0 / ~39. Live on the 240 s
      slice: 30 → **21 cards** (baseline was 35), same 534 source words, all
      translated, 20/21 refined, partials lost 0, lag 5.1–6.8 s.
- [x] **Merging on the clock alone was measured and rejected.** `turn`-style
      merging (gap + size, no evidence) hits the target number exactly — 211
      cards, median 39 words at `gap=1.5s/max=250` — and is the wrong trade:
      sampled cards splice a question onto its answer ("How did you do it?
      No. No, I don't do red eyes anymore."), and German onto English. A
      clock cannot tell a turn change from a breath, and a 39-word card
      containing two speakers is worse than two 17-word cards. Under
      `flowed_on` 7 of 8 sampled multi-utterance cards are single-speaker
      passages. (This section originally claimed the batch service segments
      by turn "because it has diarization" — invented, and wrong: Chester
      confirmed 2026-08-09 it gets the same bare audio stream. Its coherence
      comes from batch processing — whole-recording context with no latency
      constraint — not from speaker knowledge.)
- [x] **The cost is retranslation, not latency.** A merge replaces the card
      and re-translates the grown text (`replaces` in app.js), so first paint
      is unchanged. Merge ops over the hour go 265 → 475, ~79% more translate
      work. Measured live with both models co-resident: no queue growth, no
      refine starvation, lag within the normal 5–9 s band — but that is one
      run on a healthy Ollama, and item 15's warning applies unchanged: a
      wedged Ollama will make this rule *look* like it broke translation.
- [x] **Held up on three unseen segments** (12:00, 28:00, 44:00 of the real
      hour): 21 / 22 / 23 cards per 240 s, mean 22.5–24.5 words, every card
      translated, partials lost 0 throughout, and the long cards read as
      single-speaker passages — including *"Von Gottbergs hießen die, wo man
      da als… als Küchenbammser gearbeitet hat"* as one card, the exact
      stranded pair that motivated the ellipsis rule. One sampled card still
      splices a question onto its answer across a real turn, the known
      diarization residual.
- [x] ⚠️ **Sustained dense speech degrades Ollama globally, and the gates
      absorb it in the right order.** Across three back-to-back segments
      (~12 min of continuous conversation), translate p50 went 1.1 s → 1.5 s
      → 7.5 s and refines landed 43 → 35 → 14 (gated 8 → 19 → 33). The tell
      that this is not the merge's cost: 8-, 22- and 57-char cards took
      9–17 s in the third run — a global slowdown, while card size itself
      costs only ~2.8× (chars tercile p50 1.3 s → 3.5 s). Degradation order
      was refines shed → lag spikes to ~20 s → finals still 100% translated,
      partials lost 0. The standing suspect is still the 128k
      `OLLAMA_CONTEXT_LENGTH` (item 15): fix it in Ollama.app's settings
      before reading any sustained-load number as a pipeline regression.
- [ ] **The last 228 → 216 is a batch-vs-streaming gap, not a diarization
      gap.** (Corrected 2026-08-09 — the first version of this box blamed
      diarization; Transync has none, it processes the same bare audio in
      batch with whole-recording context.) What remains is boundary quality
      under a latency budget: Transync cuts at semantic boundaries it can
      see because it is never mid-stream. The residual splice defect (~1 in
      8 long cards joins a question to its answer) therefore has two
      candidate fixes — smarter text/pause boundary rules (no new
      dependency, closest to what Transync actually does) or a pretrained
      speaker encoder as a same-speaker gate. Chester chose the text rules
      first (2026-08-09) → shipped below; the encoder spike remains
      unmeasured and available (`tools/voice_eval.py --boundaries` is the
      harness).
- [x] **Shipped: a question closes the merge window** (`yields_turn`).
      Mining the hour for the splice signature found it exactly: the
      shipped rule made 12 merges across a `?` boundary without casing
      evidence, all 12 via `flowed_on`, and all 12 are question→answer
      handovers — conversational volleys change speakers faster than the
      ~190 ms micro-pause, so the handover hides inside one continuous
      speech run, which is precisely where `soft_max` is blind. Refusing
      them: 352 → 364 cards over the hour (+3.4%), 12 real two-speaker
      splices gone, precision 12/12 on this recording. A lowercase
      continuation still overrules ("oder?" mid-sentence), and the
      answer-side casing gate means a genuinely continuing question-asker
      who Whisper happens to lowercase still merges.

### 17. The context length is now pinned in two places, and an hour holds
Item 16's sustained-load collapse had a standing suspect: Ollama.app injects
`OLLAMA_CONTEXT_LENGTH=131072` into `ollama serve`, and at 128k the per-model
KV cache is big enough that draft and main evict each other. Both ends are
now fixed.

- [x] **server.py pins `num_ctx: 16384` on every Ollama request** (all five
      call sites, prewarm included — prewarm is what sizes the runner's
      allocation in the first place). A request that says num_ctx gets a
      runner sized to num_ctx *whatever the app's setting is*, so an app
      update or settings reset cannot re-bloat the KV. Sized by the largest
      prompt in the app: the whole-session summary, ~11-12k tokens for an
      hour of this conversation. One shared value on purpose — call sites
      that disagreed would thrash the runner mid-session. Overridable via
      `ALLKLARO_NUM_CTX`. Verified live: gemma3's runner went `-c 131072` →
      `-c 16384` on the first pinned request.
- [x] **Ollama.app's own setting is now 16384 too** (was 131072, changed
      2026-08-09 via its settings store, verified in the relaunched serve
      process env). Defense for clients that do not pin.
- [x] **31 minutes of continuous real conversation under the pin, no
      collapse.** Translate p50 by 10-minute window: 1.10 s → 1.69 s →
      2.20 s, refines landing 82% → 70% → 69%, partials lost 0 throughout,
      192 cards. Under 131072 the same pipeline collapsed to p50 7.5 s with
      tiny cards taking 9-17 s by minute 12, twice in one morning. The slow
      creep is real but a different regime from the collapse; minutes 30-60
      remain unmeasured (below), so "an hour holds" rests on the creep's
      trajectory plus the collapse being gone, not on a full-hour trace.
- [x] ⚠️ **The 62-minute acceptance run was killed at minute 31 by its own
      test harness**: committing a test edit while the run was live —
      `uvicorn --reload` watches `.py` files, the server restarted, and the
      session's WebSocket died. Nothing pipeline-related failed. The rule
      for any future long live run: no `.py` edits, no commits touching
      `.py`, until the run completes. (Non-`.py` files are safe; the
      watcher's default scope is Python sources only.)
- [x] Chester accepted the 31-minute evidence at the time and skipped the
      rerun — then the paragraph work (item 18) forced the question again,
      and item 18's 62-minute replay is now the full-length proof. If a
      future long session degrades anyway, the first check is unchanged:
      `ollama ps` for the CONTEXT column, then per-window translate p50 from
      `/tmp/allklaro-trace.jsonl`.

### 18. The paragraph merge nearly sank the pipeline twice, and now holds an hour
Absorption + the 500-char cap (item 16's follow-through) shipped with a cost
model that was wrong, and it took three runs to get a true one.

- [x] **Storm one: retranslating the whole card on every merge is O(k²).**
      First /takt after absorption: one chain's translates climbed 20 → 39 s
      as the card grew, the concurrent long translates saturated the GPU,
      Whisper starved (queue_ms 5 → 32 s), last-lag 59 s on a slice that
      runs 5-9 s. Fix: stitch — replay the replaced card's finished
      translation as an instant delta, stream only the tail.
- [x] **Storm two: the stitch's fallback fed itself.** Under real pacing a
      chunk arrives every 2-4 s against a 1-2.5 s draft translate, so half
      the links arrived before their base finished, fell back to a full
      retranslate, finished later, and made the next link miss too — within
      two minutes of the 62-min replay nothing stitched (translate p50
      8.7 s at minute 0-2). Fix: a link *waits* for its in-flight base
      (`CHAIN_WAIT_SEC`) — the wait overlaps running work, so every link is
      O(tail) unconditionally. A superseded card also skips history (its
      fragment is not context) and its refine (invisible text), and its
      translation future resolves on every exit path so a chain cannot
      stall.
- [x] **The 62-minute acceptance run passed** (2026-08-09, replay of the
      real hour + 8 min through the full live pipeline, realtime pace):
      797 finals, 497 merges, **300 net cards** (~4.8/min vs the batch
      reference's 4.0), translate p50 **3.6 s flat across all six 10-minute
      windows** (3.0-4.1 s, no trend, recovered after dense stretches), lag
      p50 8.7 s / p90 16 s — against the 2026-08-06 live baseline's 13-18 s
      band. Both prior failure signatures absent over the full hour.
      Replay's "256 utterances never finished" is accounting, not loss:
      superseded cards no longer emit `translation_done`; their text and
      translation live on in their successor.
- [x] **Eyeballed and passed** (2026-08-10, visible /takt on demo4, Chester
      watching): 15 cards vs the 08-06 baseline's 35 on the same audio, all
      translated, partials lost 0. The long merged paragraphs read as single
      flowing translations — no seams, no duplicated fragments. "Ja, ja." /
      "Yeah." sit inline; the one question ("And how warm is it at your
      place?") closed its card and the answer arrived as its own. Per-card
      lag median ~9.7 s (vs the old 13-18 s live band). Two caveats, neither
      a merge-layer regression:
      - *Stitch seam on card #1* (fixed same day): the translation replayed
        its base untranslated (half German, half English). The cold-start /
        chain-wait hypothesis died in diagnosis — a wait timeout falls back
        to a full retranslate and cannot seam. The real cause, reproduced
        byte-for-byte at temperature 0: with the Berlin heard note plus the
        `"gekickt" = geguckt` gloss in the prompt, **qwen2.5:7b answers with
        the corrected German sentence instead of a translation** — it does
        the restoration task, not the translation task. The stitch then
        faithfully replays that untranslated base into every successor, and
        the sheddable refine never overwrites it. Fix: a trailing guard line
        ("the reply is still ONLY the English translation") on **draft-tier
        calls only** (`guard_language`), because the same line — every
        wording and position tried — made gemma3 stop uncrossing the "nett
        verstarne" negation, and gemma never flipped tasks in the first
        place. Guarded qwen now yields "Handover and so on, then Uwe looked
        and said": English, *and* the dialect reading restored. Pinned by a
        unit test and a real-model integration test.
      - One slim "Yeah." → "Ja." card slipped past absorption, most likely
        because a real pause had already finalized its neighbor — 1 slim
        card in 15 against the old shredding.

      The run also pinned down a transcription-layer artifact worth its own
      item: the "Er hat keinen Soldaten Beruf" card is Whisper mishearing
      *"Er hat keinen Solar, er hat keinen Solar"* on an isolated chunk. No
      live caller passes `initial_prompt`; decoding the same slice with the
      preceding sentence as prompt converges on the correct reading at all
      three window positions tried, while isolated decodes are unstable
      ("Solarer Beruf" / "ja" / "also er hat kein solar"). That is item 16's
      batch-context gap in miniature, and threading the previous final's
      tail into `initial_prompt` is the evidence-backed fix candidate —
      guarded against error propagation and cross-language bias, tested,
      and A-B-A bracketed before it ships.

### 19. Auto mode decoded every chunk blind — context now threads into Whisper
The `whisper_prompt()` context lookup was keyed on `lang_hint(mode)`, which
only forced directions have — so the DEFAULT auto-de-en mode fed Whisper no
context at all, ever. Item 16 named batch context as the batch reference's
real edge; this was the same gap inside our own pipeline.

- [x] **Mechanism proven before touching code.** The "Soldaten Beruf" card
      from the 2026-08-10 eyeball run is Whisper mishearing *"Er hat keinen
      Solar, er hat keinen Solar (da drauf)"* on an isolated chunk: decodes
      of that slice are window-position-unstable in isolation ("Solarer
      Beruf" / "ja" / "also er hat kein solar") and converge on the correct
      reading at every position once the preceding sentence is in
      `initial_prompt`.
- [x] **The fix rides the machinery that already existed**: in auto modes
      the prompt now falls back to the previous final's tail (`prev`,
      last 200 chars), whatever language it was in — Whisper detects the
      chunk's language from the audio, not the prompt, so the decode stays
      language-free. Finals, speculation, and both redo paths all go
      through `whisper_prompt()`, so one edit covers them consistently.
- [x] **Verified on the live pipeline** (two demo4 replays): the soldier is
      gone — the transcript now reads "er hat keinen Solar. Er hat keinen
      Solarabruf" (compound still slightly mangled, topic recovered). No
      prompt-induced repetition loops in any final; outcomes and lag inside
      the demo4 noise band (p50 5.8 / 6.5 s across the two runs).
- [x] **A-B-A bracketed over the hour and accepted** (2026-08-28). Three
      realtime replays of the full 54.3-minute recording, arms
      A(off)-B(on)-A(off) via `ALLKLARO_AUTO_PROMPT_CONTEXT`, server
      restarted between arms so each starts equally cold. Both risks came
      back negative and the arms are structurally identical: 710 utterances
      each, 261/262/261 net cards, 429/428/428 merges, chars p50 119/120/118.
      - *Error propagation: not observed, and not growing.* Cards opening
        with 5+ words carried over from the prompt: A1 1, B 1, A2 0; whole-
        card echoes: 0/1/0; repetition loops 0 in all three. Per-10-minute
        windows show no drift, which is the shape propagation would take
        (each decode becomes the next prompt).
      - *Cross-language pull: not observed.* Language switches 131/123/127
        with B inside the baseline spread; German cards 180/179/181, English
        81/83/80.
      - *Lag: inside noise, as expected.* First-word p50 16.3 / 15.2 / 14.9 s:
        the two identical arms differ by 1.4 s, and B sits 0.5 s under
        their mean. Claim no lag benefit for this change; claim no cost.
      - *The win is transcript quality, and it replicates.* English filler
        decoded inside German cards (`no`, `yeah`, `oh`, `okay`, `well`):
        **27 / 8 / 26**. The baseline arms agree to within one token and B
        is a third of both. "no, no" hallucination runs: 5 / 1 / 6. This is
        item #4's German-decoded-as-English failure, and context suppresses
        most of it. The two arms share 85.5% of their words, with 185
        disagreements of 4+ words.
      - *A dialect entry becomes reachable.* Both baseline arms decode the
        Berlinerisch utterance as "Ich kipp ja immer nicht"; B decodes "du,
        ich kicke ja immer nicht". `kicke` is exactly the mis-hearing the
        2026-08-07 `dialects.txt` entry keys on, so the Berlin reading can
        fire with context and is lost without it.
      - ⚠️ *The card that motivated the item is not the evidence.* No arm
        reproduced "Soldaten Beruf": A1 gave "Solarer Beruf", A2 gave
        "Solarabruf" (*Solar* recovered with no context at all), B gave "er
        hat keinen solar er hat keinen Solara Beruf". That decode is
        unstable run to run, so it argues for the mechanism far more weakly
        than the demo4 verification above implies. The replicated filler
        and dialect results are what carry this item.
      - ⚠️ *Rig trap that cost the first attempt.* `tools/replay.py` always
        sends `draft_model`, and `""` means the draft pass is OFF
        (server.py's config handler), so an hour with no `--draft-model`
        runs single-pass on gemma3:12b: translate p50 111 s, first-word lag
        p50 184 s, 227 Ollama read timeouts. That is not the app's
        behavior; `resolvePair()` picks qwen2.5:7b-instruct here. Every arm
        above pins `--model gemma3:12b --draft-model qwen2.5:7b-instruct`.

---

## Closed

Kept in full rather than deleted: each one records a failure mode that
looked like something else at first, and the reasoning is the point.

### 2. Does the refine pass still earn its place? — yes, rarely, and it stays
*Answered. It lands on a minority of utterances and improves a minority of
those, which is worth keeping but not worth waiting for — so the same work is
also available on demand, where there is nothing to wait behind. The long
trail below is here because almost every intermediate answer was wrong for a
reason worth not repeating.*

- [x] **Every refine's actual fate, counted.** `refines_shed` increments in
      *two* places — the gate at `server.py:2653` and the timeout at `:2666` —
      so it cannot distinguish "never tried" from "tried and gave up", and
      `refine_ms` is stamped on every utterance whether the pass ran or not
      (`:2693`). Splitting the live traces by elapsed time separates them:

      | fate | cap 8.0 | cap 5.0 |
      |------|---------|---------|
      | gate skipped it outright (~0 ms) | 13 | 19 |
      | **attempted, timed out at 10 s** | **25 (62%)** | **32 (62%)** |
      | a refine actually landed | 2 (5%) | 1 (2%) |

      The reconstruction agrees with the trace's own `refines_shed` sum to
      within 1–2 events in each arm. The landed count is threshold-sensitive
      at the low end — a 566 ms value sits right at the boundary — so read it
      as "1–6 of 52", not a hard 1; even the generous reading is ~12%.
      **The dominant outcome is a full 10 s of the constrained resource spent
      to produce nothing**, on 62% of utterances in both arms.
- [x] **Measured with the pass switched off entirely, and it earns nothing.**
      Same slice, same cap, refine on vs off: runstart lag 8720 vs 8852 ms —
      132 ms, nothing — while refine *on* delivers two *fewer* completed
      utterances (37/38 against 39/40) and one more drain casualty at cap 5.0.
      It is also the dominant source of run-to-run variance: turning it off
      took the control spread from 810 ms to **82 ms**, which is what finally
      made item 1 measurable on this slice. The rig sees no wording benefit to
      weigh against that, so the burden is now on keeping it, not cutting it.
      Next: decide between dropping it and making it much cheaper, and if it
      stays, judge quality with real models rather than the stub.
- [x] ~~**Against live Ollama it is worse than "a large fraction":
      `refine_ms` p50 is 10003 ms at cap 8.0 and 10001 ms at cap 5.0 — the
      10 s timeout itself, in both arms.**~~ **Superseded — do not quote
      this.** It rests entirely on the 2026-08-06 traces, which do not
      reproduce (see below), and it reads a *censored* metric as if it were
      the real distribution: everything over the timeout is recorded as
      exactly the timeout. The stub point stands — a fixed-cost server cannot
      time out, so it sheds 7 at the same cap and hides the failure mode.
- [x] **Settled: keep the draft, the second pass is what is in question.**
      The two halves of the two-tier design were measured separately and the
      evidence points opposite ways.
      - *Single-pass on the main model is 8.2 s worse, not better.* Bracketed
        A1→B→A2 on slice40 against live Ollama: runstart lag **20052 ms**
        against an 11864 ms two-pass baseline, control spread **1101 ms**.
        Mechanism confirmed — `translate_ms` p50 goes 3243 → 7142 ms (max
        8623 → 18042) with the 12B model in front of the user. B completed
        *more* utterances (51 vs 49) while being far slower, so this is not a
        dropped-work artifact. **The fast draft is what makes a card appear;
        it stays.**
      - *The refine pass buys about one card in twenty to one in a hundred.*
        Delivery is load-dependent: 39% of utterances on slice40 (both A arms
        agreeing exactly) but 6-8% on the earlier demo4 runs. When it does
        land, the main model is materially better on ~13% of en→de utterances
        and *worse* on ~4 of 62 (`South Carolina` → `Südkaroina`, "fly to
        work" → `mit dem Fly`). Net: 0.8%-5% of cards improved, against two
        fewer completed utterances per slice.
- [x] **The wording question is answered, by real models rather than the
      stub.** 254 real utterances translated by draft and main through the
      app's own prompts (`wording_ab.py`): 85% differ, 15% are identical, 31%
      differ substantially. The main model fixes real errors a learner would
      care about — `aufgrund des Überschwemmungs` → `Das liegt an der
      Überschwemmung`, `Vorstellungsscheu` → `Lampenfieber`, `Hetzische` →
      `hessische` — and the draft produced 1 catastrophic output of 254
      (emitted Chinese, leaked a `user` role marker, echoed the prompt) to the
      main model's 0. So "earns nothing" was too strong: it earns a little,
      rarely.
- [x] **The 10 s timeouts are contention, not model swapping.** `server.py:89`
      blamed swapping and `OLLAMA_CONTEXT_LENGTH=131072` made that plausible.
      Falsified directly: with 51.5 GB both models stay co-resident and answer
      in 707 ms (draft) and 1165 ms (main) in steady state. Refine only fails
      when the pipeline is busy — which is exactly when it is also taking
      capacity from the cards ahead of it. The `prewarm_model` call added at
      `server.py:~2591` removed the cold-load failure mode.
- [x] **The "delivery is 39%" figure was computed the wrong way.** It counted
      utterances whose `refine_ms` was under the 10 s timeout — but a refine
      the gate skipped also lands in that bucket, at ~0 ms. On the 2026-08-06
      traces the honest delivery number is **2%**, not 39%. Whenever this is
      re-measured, exclude the ~0 ms rows.
- [x] **...but the 2026-08-06 traces themselves do not reproduce, and the
      pass is in far better shape than they say.** Re-run in-session on the
      *same* slice, cap and timeout (`ALLKLARO_REFINE_TIMEOUT_SEC`, added for
      this):

      | demo4, cap 5.0, 10 s timeout | attempts | landed | timing out | first-word lag p50 |
      |---|---|---|---|---|
      | live trace 2026-08-06 | 33 | 1 | **97%** | 6702 ms |
      | today | 25 | 19 | **24%** | 11798 ms |

      slice10 today agrees with demo4 today (22%, 21 landed), so it is the
      08-06 run that is the outlier, not the slice. Uncensored — timeout
      raised to 60 s — the median attempted refine is **4.6 s** and the ones
      the 10 s ceiling was killing need 14-17 s (max 16.9 s); nothing
      approached 60 s. So "the median refine does not complete" was true of
      that one run and is not true generally.
- [x] **Caveat on today's numbers, and it is not small.** These runs had a
      second full server (his `--reload` instance on 8710) resident with its
      own Whisper and Parakeet copies, on a 48 GB machine. That is the likely
      reason first-word lag reads 11798 ms today against 6702 ms on 08-06.
      The *refine* comparison is still apples-to-apples — both arms today
      carried the same load — but neither lag figure should be quoted against
      a run made without the second server.
- [x] **"Hard-gate it to idle stretches" is not an available option — it is
      already gated that way.** `server.py:2647` attempts a refine only when
      `whisper_pending <= 2` **and** `in_flight <= 2` **and** the utterance is
      not stale. So all 33 attempts at cap 5.0 had *already* passed an
      idleness check, and 32 of them timed out regardless. Idleness at gate
      time does not predict the next 10 seconds, so a stricter gate cannot
      rescue the pass — it can only shed the last 2%. That collapses the
      choice: the two options were never drop-vs-gate, they are drop-vs-keep.
- [x] **Raised the ceiling to 20 s — on mechanism, not on a measured win.**
      The killed refines were slow, not hung: 14-17 s against a 4.6 s median,
      nothing near 60 s. Killing them throws away capacity already spent.
- [x] **The bracket refused to confirm the single pair, and that is the
      finding.** A-B-A on demo4:

      | arm | landed | killed | kill % | first-word lag p50 |
      |-----|--------|--------|--------|--------------------|
      | A1 10 s | 19 | 6 | 24% | 11798 ms |
      | B  20 s | 22 | 2 | 8%  | 11296 ms |
      | A2 10 s | 10 | 9 | 47% | 14355 ms |

      The two *identical* control arms disagree by **23 points of kill-rate
      and 2556 ms of lag**. Against that spread: landed refines +7.5 (spread
      9) and lag -1781 ms (spread 2556) are both **inside noise**. Only
      kill-rate clears, 27 against 23 — and that is near-tautological, since a
      higher ceiling mechanically kills fewer things that exceed it. The
      earlier single pair looked convincing and was not. This is the second
      time this rig has done that (see item 1's slice40 result); the lesson is
      not "pair the arms", it is **bracket them, or claim nothing**.
- [x] **The rig drifts *within* a session.** A2 is much worse than A1 on every
      column despite identical config, so arm order is itself a variable and
      even in-session pairing is not safe. Anything measured here needs A-B-A
      with the spread reported, and effects smaller than ~2.5 s of lag are
      currently unresolvable on this rig at n=1 per arm.
- [x] **Four hypotheses for the timeouts were tested and killed** before the
      censoring turned out to be the whole story, all against real models:
      concurrent refines invisible to the gate (no — `in_flight` is
      decremented after `run_translations` returns, so refines are counted);
      unbounded prompt growth (no — `history` is a `deque(maxlen=6)`, and
      timeout rates are flat across a run, 92%→93%); draft-vs-main contention
      in Ollama (no — 1.0x, measured directly); GPU contention with MLX
      Whisper and Parakeet (no — 0.9x Whisper-only, 1.2x with draft as well,
      and **zero** timeouts in any arm, max 3.3 s).
- [x] **Decided: the pass stays, and the timeouts were the thing to fix.**
      On the 08-06 traces the case for dropping it looked strong — 1 landed
      refine per slice against a 10 s occupancy on 62% of utterances. That
      case rested on a run that does not reproduce. Today the pass lands on
      **42%** of utterances at a 4.6 s median, which is a different feature
      from the one the drop argument described.
- [x] **The refines that land live were captured and read.** `wording_ab.py`
      answered this offline, where every utterance got both answers; this
      captures what the pipeline actually replaced under real pacing
      (`capture_refines.py` records the streamed draft and the
      `translation_revised` text per uid, joined to the trace so only real
      refines count — `enforce_agreement` changes are excluded, and there
      happened to be none). One run: 33 utterances survive merging and
      discards, a refine ran on 11, and **9 changed the text**.
- [x] **Two of the nine fix outright errors, and they are exactly the errors
      a learner would be misled by:**
      - `er hat keinen Solar` — draft *"he doesn't have central heating"*,
        refined *"he doesn't have solar panels"*.
      - `damit es noch in den Skimmer geht` — draft *"into the filter"*,
        refined *"into the skimmer"*; same utterance, `abstellen` went from
        *"adjust that"* to *"turn that off"*.
      A third (`mächtig` → *"pretty intense"*, not the draft's *"really quite
      warm"*) is a fidelity gain. The other five are cosmetic, and one
      (uid 8) is arguably a slight regression. So roughly **3 of 9 are real
      improvements and 2 of those are corrections, not polish** — a better
      hit rate than the offline 13%, on a much smaller sample.
- [x] **The LLM judge failed and should not be trusted here.** Blind pairwise
      judging with `qwen2.5:32b-instruct`, each pair scored twice with the
      candidates swapped, returned **67% inconsistent** — the verdict flipped
      with position on 6 of 9, including the unambiguous solar/central-heating
      pair. Only 3 were decisive (refined 2, draft 1), which is noise at n=9.
      The two-order design is the only reason this was visible; a single-order
      run would have reported a clean 4-2 and been meaningless. Judging
      translation quality here needs a human reading or a much better
      protocol, not a bigger sample of the same judge.
- [x] **Built: the on-demand "improve this card" tap (✨).** The refine pass
      with both of its handicaps removed — no backlog gate, no deadline to
      beat — because a tap is someone waiting on purpose rather than a
      background rewrite of text already on screen. That is exactly the
      condition the offline comparison measured the main model's advantage
      under, and it is also the answer to the draft's 1-in-254 catastrophic
      output: the card it happened on can now be asked again.
      Bounded where it matters and nowhere else: one improve per card, two at
      a time, a generous 90 s ceiling so a wedged Ollama cannot leave the
      button spinning, and a hand correction still outranks the result. The
      guarantee that every tap gets *an answer* — including a failure — is
      tested, and that test is bounded, because when it regresses there is no
      reply at all and an unbounded wait hangs the suite instead of failing.
- [x] **The measurement tools are committed this time.** `wording_ab.py` and
      `capture_refines.py` were scratch scripts and are gone; every number
      above that cites them is recorded but no longer reproducible.
      `tools/capture_refines.py` is now in the repo, and the join it depends
      on is no longer a guess: the server records the refine's *outcome* per
      utterance (`refine` = landed / timeout / gated / error / off,
      `refine_wait_ms`, `refine_changed`, `agreement_changed`), so a
      declension fix can no longer be read as a landed refine and a
      gate-skipped one can no longer be read as a fast success. `refines_shed`
      is split into `refines_gated` and `refines_timeout` for the same reason:
      one number could not distinguish two opposite failures.
- [x] **Read again on the real slice, with the new accounting.** demo4, 20 s
      ceiling, gemma3:12b main + qwen2.5:7b draft:

      | outcome | n |
      |---|---|
      | gated (never attempted) | 20 of 32 |
      | attempted | 12 |
      | landed | 11 |
      | timed out | 1 (8% of attempts) |

      Landed refine p50 **7993 ms**. Of the 11 landed, 11 changed the text —
      but **one of those is only an apostrophe**: gemma writes `it’s` and qwen
      writes `it's`. So **10 substantive**, and `capture_refines.py` now
      reports that split, because the confound scales with how many
      contractions a passage happens to contain.
      ⚠️ **n=1 and unbracketed.** Refine counts swing as hard as lag on this
      rig (24% vs 47% kill-rate between *identical* arms), so read these as
      one run, not as movement from the 42%/4.6 s recorded above.
- [x] **Reading the 10 by hand — and the new finding is a regression, not a
      win.** Three are real improvements: *manchmal* mistranslated as "often"
      → "sometimes"; "quite a bit in between here" → "It was pretty intense
      here at one point" (`mächtig`, the same gain seen before); an invented
      "I'll add that" → "I say" for *Ich sage noch*. Four are cosmetic, one is
      arguably worse ("so not really" → "so not"), and one is the interesting
      failure:

          heard   : "Ich kipp ja immer nicht."
          draft   : "I don't usually do that."
          refined : "I never actually fall over."

      The source is the Berlinerisch mis-hearing from item 5 — *ick kiek ja
      immer nich* ("I never look") arriving as *kipp*. The draft's vaguer
      answer is closer to harmless; the main model confidently built a
      sentence on the wrong word. **A better model makes a dialect mis-hearing
      worse, not better**, because it commits to what it was given. That
      links item 2 to item 5 and is an argument for fixing the transcript
      rather than the translation.

### 10. Dialect words are marked in the heard text — done, lexicon included
- [x] **Built, and deliberately narrow.** Unambiguous forms only, coloured in
      the source line and never in the translation (which is standard German
      by design). Red plus a dotted underline, so it survives colourblindness
      and printing.
- [x] **Measured before building, and the measurement is the design.** Over
      2267 German word tokens of the real recording the lexicon matched **14
      times, every hit ambiguous** ("mehr" ×10, "des" ×4) and none of them
      Berlinerisch. Colouring ambiguous entries would paint ordinary German
      red on a recording with no detectable dialect at all.
- [x] **So it stays dark on the audio path by design**, for the same reason
      item 5 was dormant: Whisper normalises dialect to standard orthography.
      It earns its place on typed input, where the spelling survives — checked
      in the browser: `Ick`, `keene`, `wat`, `dit` marked, `mehr` not, zero
      marks in the translation.
- [x] **The lexicon caught up: the narrowing is live.** All 61 unambiguous
      German entries are tagged, and the Spanish ones with them — 56 name a
      dialect, and 5 deliberately do not. The five are the enclitic
      contractions (`haste`, `biste`, `isset`, `isses`, `kannste`), which are
      colloquial across the whole language rather than regional; untagged is
      the parser's "applies to every dialect" and is the right answer for
      them, not an oversight. Forms genuinely shared between dialects carry
      several tags (`uff` is Berlin *and* Hessian *and* Wormser; `isch`,
      `net`, `mer` are Hessian and Wormser), because under-tagging fails
      silently — it stops marking real dialect the moment someone selects a
      neighbouring one.
      Picking Berlinerisch now marks `ick keene wat dit` and leaves `gell`
      alone; picking Hessisch does the reverse. A tagged entry naming a
      dialect the style selector does not offer is a test failure, since a
      typo like `[berlain]` would otherwise be invisible — the entry would
      simply stop being marked under every real selection.
      The Spanish ambiguous entries were untagged too and had the same defect
      in miniature (glossing Barcelona's `mola` to someone who picked
      Mexicano); tagged in the same pass.

### 8. Sentence fragments were left stranded as their own cards — fixed
- [x] **Cause: an ellipsis was being read as a full stop.** `SENTENCE_END_RE`
      matched `…` and `...`, so the merge rule treated a cut-off utterance as
      a finished sentence and refused to join it to its continuation. But
      Whisper writes an ellipsis precisely when the speaker was *still going*.
- [x] **It was the majority of the merge opportunity, not an edge case.** Over
      the 254 dumped utterances: **58 (23%) end in an ellipsis, and 50 of
      those 58 were `soft_max` splits** — cut at a micro-pause mid-speech.
      Only 28 end in no punctuation at all, which is all the old rule could
      ever catch. Replaying the rule over the real transcripts: **21 merges
      before, 67 after.**
- [x] **Visible on screen**, which is how it was found: "Von Gottbergs hießen
      die, wo man da als…" and "als Küchenbammser gearbeitet hat." arrived as
      two cards, each translated with no sight of the other half. Same slice
      after the fix: 50 cards → 37, short fragments 17 → 11.
- [x] Split out as `looks_finished()`, because "ends with punctuation" and
      "is finished" are different questions and the difference is the bug.

### 9. Live partials could show pure noise — fixed
- [x] **The fast partial path skipped every filter that protects the finals.**
      `maybe_partial` sent Parakeet's output straight to the screen past only
      `HALLUCINATION_RE`; `clean_transcript` ran solely on the *Whisper*
      fallback. Unlike Whisper, Parakeet has no temperature ladder and no
      compression-ratio threshold to fall back on when a decode degenerates.
- [x] **Seen live: 390 consecutive `<unk>`, held on screen for seconds.** And
      no existing filter would have matched even if it had run — with no
      whitespace between the tokens, `collapse_repeats` and `has_phrase_loop`
      both saw one 1950-character "word".
- [x] Fixed with `clean_partial()`: strip the unknown-token literal, then run
      the same repetition checks the finals get. The two are separate
      guarantees — Parakeet also loops on *real* words — and are tested as
      such.
