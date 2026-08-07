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
p50 is 0, decode is ~1.3–2.3 s. Meanwhile `translate_ms` p50 ranges 2.9–7.7 s
and the refine pass hits its 10 s timeout on 8–42% of utterances with 12–32
more shed. Further Whisper-side optimisation has little left to win.

---

## Open

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
- [ ] **The remaining gap cannot be closed from transcripts, which is why it
      is still open.** Cataloguing Berlinerisch mis-hearings needs to know
      what Whisper writes when a Berliner speaks — but Whisper writes standard
      orthography, so the transcript route yields nothing by construction. It
      needs someone listening to the audio next to the text, not another dump.
      Every ambiguous entry remains Hessian / Rhine-Hessian, so the
      Berlinerisch setting still offers only the general comprehension note.

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
- [ ] **Residual noise is Whisper, and it lives in the tail.** Every large
      p90 delta between the identical arms is an ASR metric — `decode_ms`
      1022 ms, `transcribe_ms` 828 ms — while `translate_ms` p90 moved 84 ms.
      A fixed-cost ASR stub is the fix, deliberately *not* built yet: p50
      resolution decides items 1 and 2, and tail precision buys nothing until
      a p90 claim is load-bearing.
- [ ] **Arms are not bit-identical.** Run 2 produced 41 finals to run 1's 40
      and 32 speculation hits to 31, so a small part of any delta is a
      different utterance set rather than timing. Compare distributions, and
      do not read a sub-100 ms difference as signal.

*Done second, as planned: it is what makes items 1 and 2 decidable instead of
arguable.*

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
- [ ] **Speculation coverage is the real cost.** `spec:none` went 6 → 20 live:
      a cut at a 6-frame micro-pause can fire before the 10-frame silence that
      launches a speculation, so `EARLY_SILENCE_FRAMES` / `MICRO_PAUSE_FRAMES`
      want revisiting now that the cap is lower. Refines also shed more
      (37 → 49) but `refine_ms` sits at its 10 s timeout in *both* arms, so
      that is shedding work already failing — see item 2.
- [x] **Validated on a second, harder slice — the refine pass was hiding it.**
      At 40:00 with refine on, the bracket gave a 1170/1207 ms gain against a
      1034/810 ms control spread across two runs: not demonstrable. Suppress
      the refine pass (`ALLKLARO_REFINE_MAX_IN_FLIGHT=-1`) and the control
      spread collapses to **82 ms**, whereupon cap 5.0 shows a **2272 ms**
      gain — against demo4's 2264 ms. They agree to 8 ms on different slices.
      The weak slice40 result was a measurement artifact, not a weaker effect.
- [ ] **Residual tail cost, roughly halved by removing refine.** Utterances
      unfinished within the 120 s drain, at 40:00: cap 8.0 drops 1 either way;
      cap 5.0 drops 3 with refine on and 2 with it off. Note this is a replay
      deadline — in a live call these arrive very late rather than vanish.
- [ ] Translating from partials rather than chunk boundaries is still
      untried, and is the next lever if more is wanted.

### 6. One mic carries everyone, and nothing says who spoke
Multi-speaker *load* is already covered: the 54-minute recording is a real
group conversation captured through a single mic, which is the only setup
intended, so every lag number here already includes overlap and turn-taking.
That also explains why `soft_max` dominates (22 of 40 splits) — several
people trading turns rarely leave the ~700 ms gap that closes an utterance,
so the audio reads as continuous speech. This is a cause of item 1, not a
separate load risk.

- [ ] **`SPEAKERS = {0: "you", 1: "them"}` is a channel tag, not
      diarization** — it records which input stream the audio arrived on, set
      by the client. With one mic for the whole room every card is `"you"`,
      so a group transcript cannot show who said what.
- [ ] Whether that matters is a product call: for *listening along* to a
      conversation it may not, and diarization on a single stream is a real
      cost. Worth deciding before it gets built by accident.

### 2. Does the refine pass still earn its place?
- [ ] It times out or is shed on a large fraction of utterances, spending
      Ollama time rewriting text already on screen — in a pipeline where
      Ollama is now the constraint.
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
- [ ] **Against live Ollama it is worse than "a large fraction":
      `refine_ms` p50 is 10003 ms at cap 8.0 and 10001 ms at cap
      5.0 — the 10 s timeout itself, in both arms.** The median refine does
      not complete. Shed counts were 37 of 40 and 49 of 52. So the pass is
      near-totally ineffective in practice while still costing queue time,
      and the stub hides this completely (it sheds 7 at the same cap, because
      a fixed-cost server cannot time out).
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
      `server.py:~2591` is why delivery is 39% now and was 6-8% before.
- [ ] Remaining call, and it is a product one, not a measurement one: drop the
      refine pass, or hard-gate it to genuinely idle stretches. The rig cannot
      decide whether 1-in-20 better wording is worth two lost utterances.


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
- [ ] **The real failure is the mirror image, and it is reachable from the
      UI.** `lang_hint(mode)` pins Whisper's language for any *forced*
      direction (`de-en` → always German) and only returns `None` for
      `auto-*` modes (`server.py:2000`, used at `server.py:2261`). Pick
      "German → English" during a bilingual conversation and every English
      utterance is decoded as German. Measured on the same audio, that turns
      *"we had some sort of problem where they came and fixed something"*
      into *"Und so haben wir so ein Problem, wo sie sich und die Füße
      starete, die sich so starete, die Füße starete, die Füße starete."*
      Consider defaulting to auto, or falling back when the forced decode
      looks degenerate.
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
