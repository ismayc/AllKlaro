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
- [ ] **Remaining gap: no Berlinerisch mis-hearings are catalogued.** Every
      ambiguous entry is Hessian / Rhine-Hessian, so for the Berlinerisch
      setting the concrete word-level repair still has nothing to offer —
      only the general comprehension note. Closing this needs ground truth:
      what Whisper actually writes when a Berliner speaks, which the real
      recording can supply once some of it is hand-checked.

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

### 1. Audio accumulation — the largest remaining component
- [ ] 5.8 s median chunk = **36–52% of first-word lag**, untouched.
- [ ] Partly irreducible, but 22 of 40 utterances are already `soft_max`
      splits at micro-pauses, so the machinery to cut earlier exists. Test a
      lower `SOFT_MAX_SEC`, or translate from partials instead of waiting for
      a chunk boundary.

### 2. Does the refine pass still earn its place?
- [ ] It times out or is shed on a large fraction of utterances, spending
      Ollama time rewriting text already on screen — in a pipeline where
      Ollama is now the constraint.
- [ ] Make it much cheaper, make it conditional, or drop it and put a single
      good model on the critical path.

### 4. Output quality, which lag work does not touch
- [ ] German audio sometimes transcribed as English words (3 of 35 cards in a
      4-minute run), so the card renders EN→DE.
- [ ] Hallucination loops (a whole card of one repeated syllable) that
      `HALLUCINATION_RE` does not catch.
- [ ] A wrong card is worse than a slow one.
