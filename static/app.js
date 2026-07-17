const feed = document.getElementById("feed");
const hint = document.getElementById("hint");
const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const modeSel = document.getElementById("mode");
const deviceSel = document.getElementById("device");
const modelSel = document.getElementById("model");
const fontSize = document.getElementById("fontSize");
const speakChk = document.getElementById("speakChk");
const focusChk = document.getElementById("focusChk");
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

let ws = null, audioCtx = null;
let streams = [], workletNodes = [];
let running = false, callMode = false, muted = false;
let wakeLock = null, reconnectDelay = 500;
let lastSummary = "";
const cards = new Map(); // utterance id -> card state
const TTS_LANG = { de: "de-DE", en: "en-US", es: "es-ES" };

// ------------------------------------------------------------------- settings

const CFG_KEY = "translator-settings";
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(CFG_KEY)) || {}; }
  catch { return {}; }
}
function saveSettings() {
  localStorage.setItem(CFG_KEY, JSON.stringify({
    mode: modeSel.value,
    model: modelSel.value,
    deviceLabel: deviceSel.selectedOptions[0]?.textContent || "",
    fontSize: fontSize.value,
    speak: speakChk.checked,
    focus: focusChk.checked,
    pause: pauseSlider.value,
    controlsHidden: controls.classList.contains("hidden"),
  }));
}
const saved = loadSettings();
if (saved.fontSize) fontSize.value = saved.fontSize;
if (saved.speak) speakChk.checked = true;
if (saved.focus) { focusChk.checked = true; feed.classList.add("focus"); }
if (saved.mode && [...modeSel.options].some((o) => o.value === saved.mode)) {
  modeSel.value = saved.mode;
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

// ----------------------------------------------------------- models & devices

async function loadModels() {
  try {
    const r = await fetch("/api/models");
    const { models, default: def, error } = await r.json();
    if (error) showError(error);
    modelSel.innerHTML = "";
    for (const m of models) modelSel.add(new Option(m, m, m === def, m === def));
    if (saved.model && models.includes(saved.model)) modelSel.value = saved.model;
    else if (!models.includes(def) && models.length) modelSel.value = models[0];
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
                             model: modelSel.value, pause_ms: +pauseSlider.value }));
  }
}
modeSel.onchange = () => { sendConfig(); saveSettings(); };
modelSel.onchange = () => { sendConfig(); saveSettings(); };
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
  orig.innerHTML = speakerChip + langChip(msg.source);
  orig.append(document.createTextNode(msg.text));
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

function renderFinalRow(c, target) {
  const row = c.rows[target];
  row.el.innerHTML = langChip(target);
  row.el.append(document.createTextNode(row.text));
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
    c.card.insertAdjacentHTML("beforeend",
      `<div class="latency">${secs}s</div>`);
    if (speakChk.checked) speakText(c.rows[c.targets[0]].text, c.targets[0]);
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

Promise.all([loadModels(), loadDevices()]).then(() => setStatus("ok", "ready"));
