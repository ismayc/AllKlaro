const feed = document.getElementById("feed");
const hint = document.getElementById("hint");
const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const modeSel = document.getElementById("mode");
const flavorSel = document.getElementById("deFlavor");
const esFlavorSel = document.getElementById("esFlavor");
const addressSel = document.getElementById("address");
const deviceSel = document.getElementById("device");
const modelSel = document.getElementById("model");
const draftSel = document.getElementById("draftModel");
const fontSize = document.getElementById("fontSize");
const speakChk = document.getElementById("speakChk");
const focusChk = document.getElementById("focusChk");
const statsChk = document.getElementById("statsChk");
const pipeline = document.getElementById("pipeline");
const pauseSlider = document.getElementById("pause");
const pauseVal = document.getElementById("pauseVal");
const exportBtn = document.getElementById("exportBtn");
const clearBtn = document.getElementById("clearBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const meterFill = document.getElementById("meterFill");
const partialBar = document.getElementById("partialBar");
const partialText = document.getElementById("partialText");
const bigText = document.getElementById("bigText");
const controls = document.getElementById("controls");
const controlsBtn = document.getElementById("controlsBtn");
const aboutBtn = document.getElementById("aboutBtn");
const aboutBox = document.getElementById("about");
const aboutClose = document.getElementById("aboutClose");
const lookupBox = document.getElementById("lookup");
const lookupWord = document.getElementById("lookupWord");
const lookupBody = document.getElementById("lookupBody");
const lookupSpeak = document.getElementById("lookupSpeak");
const lookupClose = document.getElementById("lookupClose");
const pinBtn = document.getElementById("pinBtn");

let ws = null, audioCtx = null;
let streams = [], workletNodes = [];
let running = false, callMode = false, muted = false;
let wakeLock = null, reconnectDelay = 500;
let lastSummary = "";
const cards = new Map(); // utterance id -> card state
// es-MX: Chester's Spanish-speaking contacts are Latin American.
const TTS_LANG = { de: "de-DE", en: "en-US", es: "es-MX" };
const LANG_NAMES = { de: "German", en: "English", es: "Spanish" };
// Matches server.py's UNSURE_BELOW: under this the detector was close to a
// coin flip, and the chip says so rather than looking decided.
const UNSURE_BELOW = 0.75;
let pinnedSource = ""; // "" = detect the typed language; else force it

// ------------------------------------------------------------------- settings

const CFG_KEY = "translator-settings";
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(CFG_KEY)) || {}; }
  catch { return {}; }
}
function saveSettings() {
  localStorage.setItem(CFG_KEY, JSON.stringify({
    mode: modeSel.value,
    deFlavor: flavorSel.value,
    esFlavor: esFlavorSel.value,
    address: addressSel.value,
    model: modelSel.value,
    draft: draftSel.value,
    deviceLabel: deviceSel.selectedOptions[0]?.textContent || "",
    fontSize: fontSize.value,
    speak: speakChk.checked,
    focus: focusChk.checked,
    stats: statsChk.checked,
    pause: pauseSlider.value,
    pinnedSource,
    controlsHidden: controls.classList.contains("hidden"),
  }));
}
const saved = loadSettings();
if (saved.fontSize) fontSize.value = saved.fontSize;
if (saved.speak) speakChk.checked = true;
if (saved.focus) { focusChk.checked = true; feed.classList.add("focus"); }
if (saved.stats) { statsChk.checked = true; pipeline.classList.remove("hidden"); }
if (saved.mode && [...modeSel.options].some((o) => o.value === saved.mode)) {
  modeSel.value = saved.mode;
}
if (saved.deFlavor &&
    [...flavorSel.options].some((o) => o.value === saved.deFlavor)) {
  flavorSel.value = saved.deFlavor;
}
if (saved.esFlavor &&
    [...esFlavorSel.options].some((o) => o.value === saved.esFlavor)) {
  esFlavorSel.value = saved.esFlavor;
}
if (saved.address &&
    [...addressSel.options].some((o) => o.value === saved.address)) {
  addressSel.value = saved.address;
}
if (saved.pause) pauseSlider.value = saved.pause;
pauseVal.textContent = pauseSlider.value;
pauseSlider.oninput = () => { pauseVal.textContent = pauseSlider.value; };
pauseSlider.onchange = () => { sendConfig(); saveSettings(); };

// Collapsible settings: hidden by default on phone-sized screens so the
// conversation gets the whole viewport; the gear button toggles.
function setControlsHidden(hidden) {
  controls.classList.toggle("hidden", hidden);
  controlsBtn.setAttribute("aria-expanded", String(!hidden));
}
setControlsHidden(saved.controlsHidden ??
                  window.matchMedia("(max-width: 640px)").matches);
controlsBtn.onclick = () => {
  setControlsHidden(!controls.classList.contains("hidden"));
  saveSettings();
};

// The About panel shows which deployed build this page is actually running,
// so a stale phone cache is diagnosable at a glance.
const buildV = new URL(document.querySelector('script[src*="app.js"]').src)
  .searchParams.get("v");
document.getElementById("buildStamp").textContent =
  /^\d+$/.test(buildV) ? new Date(buildV * 1000).toLocaleString() : "unversioned";

aboutBtn.onclick = () => aboutBox.classList.remove("hidden");
aboutClose.onclick = () => aboutBox.classList.add("hidden");
aboutBox.onclick = (e) => { // tap the backdrop (not the card) to dismiss
  if (e.target === aboutBox) aboutBox.classList.add("hidden");
};

function setStatus(cls, text) {
  statusDot.className = "dot" + (cls ? " " + cls : "");
  statusText.textContent = text;
}

fontSize.oninput = () => { feed.style.fontSize = fontSize.value + "px"; saveSettings(); };
feed.style.fontSize = fontSize.value + "px";
speakChk.onchange = saveSettings;
focusChk.onchange = () => {
  feed.classList.toggle("focus", focusChk.checked);
  clearPartial();
  saveSettings();
  feed.scrollTop = feed.scrollHeight;
};
statsChk.onchange = () => {
  pipeline.classList.toggle("hidden", !statsChk.checked);
  sendConfig();   // the server only pushes stats while the overlay is open
  saveSettings();
};

// ------------------------------------------------------- pipeline overlay
//
// "It falls behind when nobody pauses" has several possible causes that feel
// identical from the outside: audio still buffering because no pause has come
// yet, a backlog of utterances, a Whisper thread with a queue on it, or
// partials being starved so the screen sits empty. Each gets its own number.
const pipeSpeech = document.getElementById("pipeSpeech");
const pipeFlight = document.getElementById("pipeFlight");
const pipeQueue = document.getElementById("pipeQueue");
const pipeSkipped = document.getElementById("pipeSkipped");
const pipeChunk = document.getElementById("pipeChunk");
const pipeLag = document.getElementById("pipeLag");

function showStats(s) {
  pipeSpeech.textContent = s.speech_sec.toFixed(1) + "s";
  // Buffering past the soft-max split means continuous speech: the words
  // already spoken cannot reach the screen until a micro-pause shows up.
  pipeSpeech.className = s.speech_sec >= 8 ? "hot" : s.speech_sec >= 4 ? "warm" : "";
  pipeFlight.textContent = s.in_flight;
  pipeFlight.className = s.in_flight > 1 ? "warm" : "";
  pipeQueue.textContent = s.whisper_queue;
  pipeQueue.className = s.whisper_queue > 1 ? "hot" : "";
  pipeSkipped.textContent = s.partials_skipped;
  pipeSkipped.className = s.partials_skipped > 2 ? "warm" : "";
  const last = s.last;
  if (!last) return;
  pipeChunk.textContent = `${last.chunk_sec}s ${last.split || "?"}`;
  pipeChunk.className = last.split === "soft_max" || last.split === "hard_max"
    ? "hot" : "";
  // The felt delay: the age of the chunk's FIRST word when its card landed.
  const felt = (last.first_word_lag_ms / 1000).toFixed(1);
  pipeLag.textContent = `${felt}s`;
  pipeLag.className = last.first_word_lag_ms >= 8000 ? "hot"
    : last.first_word_lag_ms >= 4000 ? "warm" : "";
}

// ----------------------------------------------------------- models & devices

// Models under ~4 GB (≈3B params) translate fast but paraphrase too freely
// to trust even as drafts; prefer the smallest model above this line.
const MIN_DRAFT_BYTES = 4e9;

function modelLabel(name, sizes) {
  const gb = (sizes?.[name] || 0) / 1e9;
  return gb ? `${name} · ${gb.toFixed(1).replace(/\.0$/, "")} GB` : name;
}

// Which (main, draft) pair to actually use, given what's saved and what's
// installed. Pure so it can be tested without a browser.
//
// The draft pass only helps if the draft is faster than the model it drafts
// for; a pairing where it isn't puts the big model on the critical path and
// leaves the small one cold. Saved settings can be inverted like that — hand
// picked, or from before the size-aware default below existed — and it is the
// most expensive mistake available here. Measured over a real 54-minute
// conversation: main qwen2.5:7b (4.7 GB) with a 19 GB draft gave a first-word
// lag p50 of 57 s, against 15 s for main gemma3:12b (8.1 GB) with a 4.7 GB
// draft. So an inverted pair is discarded rather than restored, main included
// — keeping the saved main would just re-pick a draft for the wrong model.
function resolvePair(saved, models, sizes, def) {
  const size = (m) => (sizes?.[m] || 0);
  const savedMain = models.includes(saved.model) ? saved.model : "";
  // "" is a real saved draft: it means the user turned the draft pass off.
  const savedDraft = saved.draft === "" || models.includes(saved.draft)
    ? saved.draft : undefined;
  const inverted = !!savedMain && !!savedDraft
    && size(savedDraft) >= size(savedMain);
  const model = !inverted && savedMain
    ? savedMain
    : (models.includes(def) ? def : models[0] ?? "");
  if (!inverted && savedDraft !== undefined) return { model, draft: savedDraft };
  // Default draft: smallest model that's still trustworthy (see
  // MIN_DRAFT_BYTES) and smaller than the main model; if only tiny models
  // exist, the largest of those is the least-bad choice.
  const smaller = models
    .filter((m) => m !== model && size(m) < size(model))
    .sort((a, b) => size(a) - size(b));
  const draft = smaller.find((m) => size(m) >= MIN_DRAFT_BYTES)
    ?? smaller[smaller.length - 1] ?? "";
  return { model, draft };
}

async function loadModels() {
  try {
    const r = await fetch("/api/models");
    const { models, default: def, sizes, error } = await r.json();
    if (error) showError(error);
    modelSel.innerHTML = "";
    for (const m of models) {
      modelSel.add(new Option(modelLabel(m, sizes), m, m === def, m === def));
    }
    draftSel.innerHTML = "";
    draftSel.add(new Option("Off", ""));
    for (const m of models) draftSel.add(new Option(modelLabel(m, sizes), m));

    const pair = resolvePair(saved, models, sizes, def);
    modelSel.value = pair.model;
    draftSel.value = pair.draft;
    // Deliberately not saved back here: loadDevices() is still in flight, and
    // saveSettings() would write an empty deviceLabel over the saved one. Every
    // load re-resolves, so a repaired pair is what runs either way, and the
    // next settings change persists it.
  } catch {
    showError("Could not list Ollama models — is `ollama serve` running?");
  }
}

// Loopback devices carry the meeting audio; browser echo-cancellation and
// noise suppression would mangle or cancel it, so use them raw.
const isLoopback = (label) => /blackhole|aggregate/i.test(label);

async function loadDevices() {
  const prev = deviceSel.value || null;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices.filter((d) => d.kind === "audioinput");
  deviceSel.innerHTML = "";
  deviceSel.add(new Option("Default microphone", ""));
  let haveBlackhole = false;
  for (const d of inputs) {
    if (d.deviceId === "default" || !d.label) continue;
    if (/blackhole/i.test(d.label)) haveBlackhole = true;
    deviceSel.add(new Option(d.label, d.deviceId));
  }
  if (haveBlackhole) {
    deviceSel.add(new Option("You + Them (mic + BlackHole)", "__call__"));
  }
  const wanted = prev !== null ? null : saved.deviceLabel; // restore only on first load
  const byLabel = wanted &&
    [...deviceSel.options].find((o) => o.textContent === wanted);
  if (byLabel) deviceSel.value = byLabel.value;
  else if (prev && [...deviceSel.options].some((o) => o.value === prev)) {
    deviceSel.value = prev;
  } else if (prev === null && !saved.deviceLabel && haveBlackhole) {
    deviceSel.value = "__call__"; // calls are the primary use case
  }
}
navigator.mediaDevices.addEventListener?.("devicechange", loadDevices);

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", mode: modeSel.value,
                             de_flavor: flavorSel.value,
                             es_flavor: esFlavorSel.value,
                             address: addressSel.value,
                             model: modelSel.value, draft_model: draftSel.value,
                             stats: statsChk.checked,
                             pause_ms: +pauseSlider.value }));
  }
}
// ------------------------------------------------------------ language pin
// Detection is good but not perfect, and a wrong guess on typed text is
// silent. This pins the language of what you type so it never runs.

function autoPair() {
  const parts = modeSel.value.split("-");
  return parts[0] === "auto" && parts.length === 3 ? [parts[1], parts[2]] : null;
}

function renderPin() {
  const pair = autoPair();
  // A forced direction ("German → English") already decides the source.
  pinBtn.classList.toggle("gone", !pair);
  if (!pair) return;
  if (pinnedSource && !pair.includes(pinnedSource)) pinnedSource = "";
  pinBtn.textContent = pinnedSource ? pinnedSource.toUpperCase() : "Auto";
  pinBtn.classList.toggle("pinned", !!pinnedSource);
  pinBtn.title = pinnedSource
    ? `Typing in ${LANG_NAMES[pinnedSource]} — tap to change`
    : "Typed language is detected — tap to pin it instead";
}

pinBtn.onclick = () => {
  const pair = autoPair();
  if (!pair) return;
  const cycle = ["", pair[0], pair[1]];
  pinnedSource = cycle[(cycle.indexOf(pinnedSource) + 1) % cycle.length];
  renderPin();
  saveSettings();
};

if (saved.pinnedSource) pinnedSource = saved.pinnedSource;
renderPin();

modeSel.onchange = () => { renderPin(); sendConfig(); saveSettings(); };
flavorSel.onchange = () => { sendConfig(); saveSettings(); };
esFlavorSel.onchange = () => { sendConfig(); saveSettings(); };
addressSel.onchange = () => { sendConfig(); saveSettings(); };
modelSel.onchange = () => { sendConfig(); saveSettings(); };
draftSel.onchange = () => { sendConfig(); saveSettings(); };
deviceSel.onchange = async () => {
  saveSettings();
  if (running) { stop(); await start(); } // re-open capture on the new device
};

// ------------------------------------------------------------------- feed UI

function showError(msg) {
  const div = document.createElement("div");
  div.className = "error-banner";
  div.textContent = msg;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function langChip(code) {
  return `<span class="lang ${code}">${code.toUpperCase()}</span>`;
}

// The source chip doubles as the fix when the language was guessed wrong:
// in an auto mode it is a button that re-runs the card the other way round.
// Only auto modes get one — a forced direction has nothing to flip.
function sourceChip(msg) {
  const other = msg.targets[0];
  if (!msg.auto || !other || other === msg.source) {
    const span = document.createElement("span");
    span.className = "lang " + msg.source;
    span.textContent = msg.source.toUpperCase();
    return span;
  }
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "lang flip " + msg.source;
  btn.textContent = msg.source.toUpperCase();
  // conf is absent for speech (Whisper reports a language, not a margin).
  const unsure = msg.conf != null && msg.conf < UNSURE_BELOW;
  if (unsure) btn.classList.add("unsure");
  btn.title = (msg.chosen
    ? `Set to ${LANG_NAMES[msg.source]}`
    : `Detected ${LANG_NAMES[msg.source]}${unsure ? ", but it was a close call" : ""}`)
    + ` — tap to redo this as ${LANG_NAMES[other]}`;
  btn.onclick = (e) => { e.stopPropagation(); flipCard(msg.id); };
  return btn;
}

// Ask the server to redo this utterance with the languages swapped. It comes
// back as a fresh card that replaces this one, so the old (wrong) reading
// also drops out of the conversation context.
function flipCard(id) {
  const c = cards.get(id);
  if (!c) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return showError("Not connected — cannot redo that translation.");
  }
  c.card.classList.add("flipping");
  ws.send(JSON.stringify({ type: "retranslate", id,
                           text: c.text, source: c.targets[0] }));
}

function newCard(msg) {
  hint?.remove();
  if (msg.replaces != null) {
    // Server merged a cut-off fragment into this utterance; drop the old card.
    cards.get(msg.replaces)?.card.remove();
    cards.delete(msg.replaces);
  }
  const card = document.createElement("div");
  card.className = "card " + msg.source;
  const orig = document.createElement("div");
  orig.className = "orig";
  const speakerChip = callMode
    ? `<span class="speaker">${msg.speaker === "them" ? "Them" : "You"}</span>` : "";
  orig.innerHTML = speakerChip;
  orig.append(sourceChip(msg), wordSpans(msg.text, msg.source));
  orig.title = "Tap to hear it spoken";
  // Tapping a phrase speaks it aloud in that phrase's own language.
  orig.onclick = (e) => { e.stopPropagation(); speakText(msg.text, msg.source, true); };
  card.append(orig);

  const rows = {};
  for (const t of msg.targets) {
    const row = document.createElement("div");
    row.className = "trans";
    row.innerHTML = langChip(t) + '<span class="cursor">▍</span>';
    row.title = "Tap to hear it spoken";
    row.onclick = (e) => {
      e.stopPropagation();
      speakText(cards.get(msg.id)?.rows[t].text, t, true);
    };
    card.append(row);
    rows[t] = { el: row, text: "" };
  }
  card.onclick = () => showBig(msg.id); // card background still opens big-text
  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
  cards.set(msg.id, {
    id: msg.id, card, rows, source: msg.source, targets: msg.targets,
    text: msg.text, speaker: msg.speaker,
    time: new Date().toLocaleTimeString(),
  });
}

function renderRow(c, target) {
  const row = c.rows[target];
  row.el.innerHTML = langChip(target);
  row.el.append(document.createTextNode(row.text));
  row.el.insertAdjacentHTML("beforeend", '<span class="cursor">▍</span>');
}

// -------------------------------------------------------- editing corrections

async function copyText(text, btn) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      // Plain-http on the LAN has no async clipboard; fall back to execCommand.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    btn.textContent = "✓";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = "📋";
      btn.classList.remove("copied");
    }, 1200);
  } catch {
    showError("Could not copy to the clipboard.");
  }
}

function renderFinalRow(c, target) {
  const row = c.rows[target];
  row.el.innerHTML = langChip(target);
  row.el.append(wordSpans(row.text, target));
  const copyBtn = document.createElement("button");
  copyBtn.className = "copy-btn";
  copyBtn.title = "Copy this translation";
  copyBtn.textContent = "📋";
  copyBtn.onclick = (e) => { e.stopPropagation(); copyText(row.text, copyBtn); };
  row.el.append(copyBtn);
  const btn = document.createElement("button");
  btn.className = "edit-btn";
  btn.title = "Edit this translation";
  btn.textContent = "✏️";
  btn.onclick = (e) => { e.stopPropagation(); startEdit(c, target); };
  row.el.append(btn);
  if (row.edited) {
    row.el.insertAdjacentHTML("beforeend", '<span class="edited">edited</span>');
  }
}

function startEdit(c, target) {
  const row = c.rows[target];
  if (row.el.querySelector(".edit-box")) return; // already editing this row
  const original = row.text;
  row.el.innerHTML = langChip(target);
  const box = document.createElement("textarea");
  box.className = "edit-box";
  box.rows = 2;
  box.value = original;
  box.onclick = (e) => e.stopPropagation(); // don't speak / open big-text
  const finish = (save) => {
    box.onblur = null;
    const value = box.value.trim();
    if (save && value && value !== original) {
      row.text = value;
      row.edited = true;
      sendCorrection(c, target, original, value);
    }
    renderFinalRow(c, target);
  };
  box.onblur = () => finish(true); // tapping elsewhere (or mobile Done) saves
  box.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") finish(false);
  };
  row.el.append(box);
  box.focus();
  box.setSelectionRange(original.length, original.length);
}

async function sendCorrection(c, target, was, corrected) {
  // Fix the live conversation context too, so follow-up translations
  // build on the corrected phrasing.
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "correction", id: c.id, target, corrected }));
  }
  try {
    const r = await fetch("/api/correction", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: c.source, target, text: c.text,
                             model_translation: was, corrected }),
    });
    const { error } = await r.json();
    if (error) showError(error);
  } catch {
    showError("Could not save the correction.");
  }
}

function showBig(id) {
  const c = cards.get(id);
  if (!c) return;
  bigText.innerHTML = "";
  for (const t of c.targets) {
    const p = document.createElement("p");
    p.textContent = c.rows[t].text;
    bigText.appendChild(p);
  }
  bigText.classList.remove("hidden");
}
bigText.onclick = () => bigText.classList.add("hidden");

function speakText(text, lang, interrupt = false) {
  if (!text || !window.speechSynthesis) return;
  if (interrupt) speechSynthesis.cancel(); // a tap always wins over the queue
  const u = new SpeechSynthesisUtterance(text);
  u.lang = TTS_LANG[lang] || "en-US";
  // Mute capture while speaking so the app doesn't translate its own voice.
  u.onstart = () => { muted = true; };
  u.onend = u.onerror = () => { muted = false; };
  speechSynthesis.speak(u);
}

// ------------------------------------------------------------- word lookup

// Long-press a word for its dictionary entry (Wiktionary lexicons built by
// build_wiktionary_lexicon.py). A long-press must not also fire the tap
// actions (speak / big-text), so the click that follows it is swallowed.
const LONG_PRESS_MS = 500;
let suppressClick = false;
document.addEventListener("click", (e) => {
  if (suppressClick) {
    suppressClick = false;
    e.stopPropagation();
    e.preventDefault();
  }
}, true);
// iOS often skips the click after a long-press entirely; the flag must not
// stay armed and swallow the next genuine tap (📋, ✏️, tap-to-speak).
document.addEventListener("pointerdown", () => { suppressClick = false; }, true);

function attachLongPress(el, fn) {
  let timer = null;
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return; // right-click has its own menu
    clearTimeout(timer);
    timer = setTimeout(() => { suppressClick = true; fn(); }, LONG_PRESS_MS);
  });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) {
    el.addEventListener(ev, () => clearTimeout(timer));
  }
}

// Text -> fragment with each word in a long-pressable span.
function wordSpans(text, lang) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const m of text.matchAll(/\p{L}[\p{L}\p{M}'’-]*/gu)) {
    if (m.index > last) frag.append(text.slice(last, m.index));
    const span = document.createElement("span");
    span.className = "w";
    span.textContent = m[0];
    attachLongPress(span, () => openLookup(m[0], lang));
    frag.append(span);
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.append(text.slice(last));
  return frag;
}

const ARTICLES = { de: { m: "der", f: "die", n: "das" },
                   es: { m: "el", f: "la" } };
let lookupCurrent = null;

async function openLookup(word, lang) {
  lookupCurrent = { word, lang };
  lookupWord.textContent = word;
  lookupBody.textContent = "Looking up…";
  lookupBox.classList.remove("hidden");
  let data;
  try {
    const r = await fetch(`/api/lookup?${new URLSearchParams({ word, lang })}`);
    data = await r.json();
  } catch {
    data = { error: "Lookup failed — is the server running?" };
  }
  if (lookupCurrent?.word !== word) return; // superseded by a newer press
  renderLookup(data, lang);
}

function renderLookup(data, lang) {
  lookupBody.innerHTML = "";
  if (data.error || !data.entries?.length) {
    const p = document.createElement("p");
    p.className = "lookup-miss";
    p.textContent = data.error || "No dictionary entry found.";
    lookupBody.append(p);
    return;
  }
  for (const e of data.entries) {
    const div = document.createElement("div");
    div.className = "lookup-entry";
    const head = document.createElement("div");
    head.className = "lookup-headline";
    head.textContent =
      [ARTICLES[lang]?.[e.gender], e.word].filter(Boolean).join(" ");
    div.append(head);
    const meta = [e.pos, e.gender, e.ipa, e.plural && `pl. ${e.plural}`]
      .filter(Boolean).join(" · ");
    if (meta) {
      const m = document.createElement("div");
      m.className = "lookup-meta";
      m.textContent = meta;
      div.append(m);
    }
    const ol = document.createElement("ol");
    for (const s of e.senses) {
      const li = document.createElement("li");
      li.textContent = s;
      ol.append(li);
    }
    div.append(ol);
    lookupBody.append(div);
  }
}

lookupClose.onclick = () => lookupBox.classList.add("hidden");
lookupBox.onclick = (e) => { // tap the backdrop (not the sheet) to dismiss
  if (e.target === lookupBox) lookupBox.classList.add("hidden");
};
lookupSpeak.onclick = () =>
  lookupCurrent && speakText(lookupCurrent.word, lookupCurrent.lang, true);

// Live partial text: bottom bar normally; in focus mode an in-feed card so
// the incoming speech sits near mid-screen with the translations.
let partialCard = null;

function showPartial(text) {
  if (focusChk.checked) {
    partialBar.classList.add("hidden");
    hint?.remove();
    if (!partialCard) {
      partialCard = document.createElement("div");
      partialCard.className = "card partial";
    }
    partialCard.textContent = text;
    feed.appendChild(partialCard); // re-append keeps it below the newest card
    feed.scrollTop = feed.scrollHeight;
  } else {
    partialText.textContent = text;
    partialBar.classList.remove("hidden");
  }
}

function clearPartial() {
  partialBar.classList.add("hidden");
  partialCard?.remove();
}

function handleMessage(msg) {
  if (msg.type === "partial") {
    showPartial(msg.text);
  } else if (msg.type === "final") {
    clearPartial();
    newCard(msg);
  } else if (msg.type === "translation_delta") {
    const c = cards.get(msg.id);
    if (!c || !c.rows[msg.target]) return;
    c.rows[msg.target].text += msg.text;
    renderRow(c, msg.target);
    feed.scrollTop = feed.scrollHeight;
  } else if (msg.type === "translation_done") {
    const c = cards.get(msg.id);
    if (!c) return;
    // Re-render each row without the cursor and with its edit button.
    for (const t of c.targets) renderFinalRow(c, t);
    const secs = ((msg.transcribe_ms + msg.translate_ms) / 1000).toFixed(1);
    const refining = msg.refining ? '<span class="refining">refining…</span> ' : "";
    c.card.insertAdjacentHTML("beforeend",
      `<div class="latency">${refining}${secs}s</div>`);
    if (speakChk.checked) speakText(c.rows[c.targets[0]].text, c.targets[0]);
  } else if (msg.type === "translation_revised") {
    const c = cards.get(msg.id);
    if (!c) return;
    c.card.querySelector(".refining")?.remove();
    for (const [t, text] of Object.entries(msg.texts || {})) {
      const row = c.rows[t];
      if (!row || row.edited) continue; // a user edit outranks the refinement
      row.text = text;
      renderFinalRow(c, t);
      row.el.classList.add("revised"); // brief flash marks the swap
    }
  } else if (msg.type === "stats") {
    showStats(msg);
  } else if (msg.type === "discard") {
    clearPartial();
  } else if (msg.type === "error") {
    showError(msg.message);
  }
}

// ------------------------------------------------------------- export & summary

function conversationItems() {
  return [...cards.values()].filter((c) => c.text).map((c) => ({
    time: c.time,
    speaker: callMode ? (c.speaker === "them" ? "Them" : "You") : "",
    source: c.source,
    target: c.targets[0],
    text: c.text,
    translations: Object.fromEntries(c.targets.map((t) => [t, c.rows[t].text])),
  }));
}

exportBtn.onclick = async () => {
  const items = conversationItems();
  if (!items.length) return showError("Nothing to export yet.");
  const r = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items, summary: lastSummary }),
  });
  const { markdown } = await r.json();
  const blob = new Blob([markdown], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `allklaro-${new Date().toISOString().slice(0, 16).replace(":", "")}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
};

clearBtn.onclick = () => {
  if (!feed.querySelector(".card, .error-banner")) return; // nothing to clear
  if (!confirm("Clear the conversation? This cannot be undone.")) return;
  window.speechSynthesis?.cancel();
  cards.clear();
  lastSummary = ""; // a stale summary must not leak into the next export
  clearPartial();
  feed.replaceChildren(hint);
};

summarizeBtn.onclick = async () => {
  const items = conversationItems();
  if (!items.length) return showError("Nothing to summarize yet.");
  summarizeBtn.disabled = true;
  summarizeBtn.textContent = "Summarizing…";
  try {
    const r = await fetch("/api/summarize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items, model: modelSel.value }),
    });
    const { summary, error } = await r.json();
    if (error) return showError(error);
    lastSummary = summary;
    const div = document.createElement("div");
    div.className = "card summary";
    div.textContent = summary;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
  } catch {
    showError("Summarize failed.");
  } finally {
    summarizeBtn.disabled = false;
    summarizeBtn.textContent = "Summarize";
  }
};

// -------------------------------------------------------------- audio capture

function onPCM(buf, tag) {
  if (ws && ws.readyState === WebSocket.OPEN && !muted) {
    if (callMode) {
      const payload = new Uint8Array(1 + buf.byteLength);
      payload[0] = tag;
      payload.set(new Uint8Array(buf), 1);
      ws.send(payload);
    } else {
      ws.send(buf);
    }
  }
  // Level meter: shows at a glance whether audio is reaching the app.
  const pcm = new Int16Array(buf);
  let peak = 0;
  for (let i = 0; i < pcm.length; i += 8) {
    const v = Math.abs(pcm[i]);
    if (v > peak) peak = v;
  }
  const pct = Math.min(100, (peak / 32768) * 140);
  meterFill.style.width = pct + "%";
  meterFill.classList.toggle("hot", peak > 30000);
}

async function openCapture(deviceId, raw, tag) {
  const audio = {
    echoCancellation: !raw, noiseSuppression: !raw,
    autoGainControl: !raw, channelCount: 1,
  };
  if (deviceId) audio.deviceId = { exact: deviceId };
  const stream = await navigator.mediaDevices.getUserMedia({ audio });
  streams.push(stream);
  const src = audioCtx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(audioCtx, "pcm-capture");
  node.port.onmessage = (e) => onPCM(e.data, tag);
  src.connect(node);
  workletNodes.push(node);
}

// ----------------------------------------------------- websocket & reconnection

function connectWS() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
    ws.onopen = () => { reconnectDelay = 500; sendConfig(); resolve(); };
    ws.onerror = reject;
    ws.onclose = () => { if (running) scheduleReconnect(); };
  });
}

function scheduleReconnect() {
  setStatus("live", "reconnecting…");
  setTimeout(async () => {
    if (!running) return;
    try {
      await connectWS();
      setStatus("live", "listening");
    } catch {
      reconnectDelay = Math.min(5000, reconnectDelay * 2);
      scheduleReconnect();
    }
  }, reconnectDelay);
}

async function acquireWakeLock() {
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
}
document.addEventListener("visibilitychange", () => {
  if (running && document.visibilityState === "visible") acquireWakeLock();
});

// --------------------------------------------------------------- start / stop

async function start() {
  callMode = deviceSel.value === "__call__";
  let captures;
  if (callMode) {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const bh = devices.find((d) => d.kind === "audioinput" && /blackhole/i.test(d.label));
    if (!bh) return showError("BlackHole input not found for call mode.");
    captures = [{ deviceId: "", raw: false, tag: 0 },
                { deviceId: bh.deviceId, raw: true, tag: 1 }];
  } else {
    const label = deviceSel.selectedOptions[0]?.textContent || "";
    captures = [{ deviceId: deviceSel.value, raw: isLoopback(label), tag: 0 }];
  }

  try {
    if (ws) { ws.onclose = null; ws.close(); } // e.g. opened by typed input
    await connectWS();
  } catch {
    showError("Could not reach the server. Is it running?");
    return;
  }

  audioCtx = new AudioContext();
  // iOS Safari creates audio contexts suspended; resume within the tap gesture.
  if (audioCtx.state === "suspended") await audioCtx.resume();
  await audioCtx.audioWorklet.addModule("/static/worklet.js");
  try {
    for (const c of captures) await openCapture(c.deviceId, c.raw, c.tag);
  } catch {
    showError("Could not open that input device. Allow mic access for this page and retry.");
    stopCapture();
    return;
  }
  loadDevices(); // labels become available after first permission grant

  running = true;
  micBtn.classList.add("live");
  micLabel.textContent = "Stop";
  setStatus("live", "listening");
  acquireWakeLock();
}

function stopCapture() {
  workletNodes.forEach((n) => n.disconnect());
  streams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
  audioCtx?.close();
  workletNodes = []; streams = []; audioCtx = null;
}

function stop() {
  running = false;
  stopCapture();
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  micBtn.classList.remove("live");
  micLabel.textContent = "Start";
  clearPartial();
  meterFill.style.width = "0%";
  setStatus("ok", "ready");
}

micBtn.onclick = () => (running ? stop() : start());

// ------------------------------------------------------------- typed input

const typeForm = document.getElementById("typeBar");
const typeInput = document.getElementById("typeInput");

// One-tap paste-and-translate: for text copied out of WhatsApp and the
// like. Falls back to focusing the input if the browser refuses clipboard
// access (then a manual paste is one gesture away).
document.getElementById("pasteBtn").onclick = async () => {
  try {
    const text = (await navigator.clipboard.readText())?.trim();
    if (!text) return typeInput.focus();
    typeInput.value = text;
    typeForm.requestSubmit();
  } catch {
    typeInput.focus();
  }
};

typeForm.onsubmit = async (e) => {
  e.preventDefault();
  const text = typeInput.value.trim();
  if (!text) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    try { await connectWS(); } // typing works without the mic running
    catch { return showError("Could not reach the server. Is it running?"); }
  }
  const payload = { type: "text", text };
  if (pinnedSource && autoPair()?.includes(pinnedSource)) {
    payload.source = pinnedSource;
  }
  ws.send(JSON.stringify(payload));
  typeInput.value = "";
};

Promise.all([loadModels(), loadDevices()]).then(() => setStatus("ok", "ready"));
