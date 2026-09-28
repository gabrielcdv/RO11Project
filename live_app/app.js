import { createDsp, toTensor, trimSilence } from "./dsp.js";

const MAX_SECONDS = 15;
const EMOJI = {
  anger: "😠", boredom: "🥱", calm: "😌", disgust: "🤢", fear: "😨",
  happiness: "😄", neutral: "😐", sadness: "😢", surprise: "😮",
};

const $ = (id) => document.getElementById(id);
const [{ spec, models }, tables] = await Promise.all(
  ["models/models.json", "models/dsp.json"].map((u) => fetch(u).then((r) => r.json())),
);
const dsp = createDsp(spec, tables);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const sessions = new Map();  // model id -> Promise<InferenceSession>, loaded on first use
let images = null;           // spectrograms of the current recording
let runId = 0;               // ignore results of a prediction superseded by a newer one

function currentModel() {
  const arch = document.querySelector("input[name=arch]:checked").value;
  const data = document.querySelector("input[name=data]:checked").value;
  return models.find((m) => m.id === `${arch}_${data}`);
}

function showModelInfo() {
  const m = currentModel();
  $("model-info").textContent =
    `Trained on ${m.datasets.join(", ")} · ${m.classes.length} emotions · ` +
    `validation accuracy ${Math.round(m.val_accuracy * 100)} % ` +
    `(chance ${Math.round(100 / m.classes.length)} %)`;
}

function session(m) {
  if (!sessions.has(m.id)) {
    sessions.set(m.id, ort.InferenceSession.create(`models/${m.file}`).catch((e) => {
      sessions.delete(m.id);
      throw e;
    }));
  }
  return sessions.get(m.id);
}

// ---------- audio in ----------

let audioCtx = null;

async function decode(blob) {
  const buf = await blob.arrayBuffer();
  audioCtx ??= new AudioContext();
  const decoded = await audioCtx.decodeAudioData(buf);
  // resample to 16 kHz mono (the offline context downmixes the channels)
  const length = Math.ceil(decoded.duration * spec.sr);
  const ctx = new OfflineAudioContext(1, length, spec.sr);
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  return (await ctx.startRendering()).getChannelData(0);
}

async function handleAudio(blob) {
  $("player").src = URL.createObjectURL(blob);
  $("player").hidden = false;
  setStatus("Computing the spectrograms…");
  try {
    const y = trimSilence(await decode(blob), spec.sr);
    const chunks = dsp.splitChunks(y);
    if (!chunks.length) {
      images = null;
      $("result").hidden = true;
      return setStatus("Too short: speak for at least one second.", true);
    }
    images = chunks.map(dsp.melImage);
    drawSpectrograms(images);
    await predict();
  } catch (e) {
    console.error(e);
    setStatus(`Could not read this audio (${e.message}).`, true);
  }
}

let recorder = null;
let timer = null;

async function toggleRecording() {
  if (recorder) return recorder.stop();
  let stream;
  try {
    // raw signal, like the corpus recordings: no browser noise suppression or gain control
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (e) {
    return setStatus(`Microphone unavailable (${e.message}).`, true);
  }
  const parts = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => parts.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(timer);
    recorder = null;
    $("record").textContent = "● Record";
    $("record").classList.remove("recording");
    handleAudio(new Blob(parts, { type: parts[0]?.type }));
  };
  recorder.start();
  const start = performance.now();
  timer = setInterval(() => {
    const s = (performance.now() - start) / 1000;
    $("timer").textContent = `${s.toFixed(1)} s`;
    if (s >= MAX_SECONDS) recorder?.stop();
  }, 100);
  $("record").textContent = "■ Stop";
  $("record").classList.add("recording");
  setStatus("Recording…");
}

// ---------- model ----------

async function predict() {
  if (!images) return;
  const id = ++runId;
  const m = currentModel();
  if (!sessions.has(m.id)) setStatus(`Loading the model (${m.arch === "resnet" ? "45" : "5"} MB)…`);
  try {
    const s = await session(m);
    setStatus("Running the model…");
    const size = m.channels * m.img_size * m.img_size;
    const batch = new Float32Array(images.length * size);
    images.forEach((img, i) => batch.set(toTensor(img, m), i * size));
    const input = new ort.Tensor("float32", batch, [images.length, m.channels, m.img_size, m.img_size]);
    const { logits } = await s.run({ input });
    if (id !== runId) return;
    showResult(m, averageSoftmax(logits.data, images.length, m.classes.length));
  } catch (e) {
    console.error(e);
    if (id === runId) setStatus(`Model error (${e.message}).`, true);
  }
}

// one prediction per 3 s chunk, averaged
function averageSoftmax(logits, n, k) {
  const probs = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    const row = logits.slice(i * k, (i + 1) * k);
    const max = Math.max(...row);
    const exp = row.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    exp.forEach((v, j) => (probs[j] += v / sum / n));
  }
  return probs;
}

// ---------- display ----------

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").style.color = error ? "var(--danger)" : "";
}

function showResult(m, probs) {
  const ranked = m.classes.map((c, i) => [c, probs[i]]).sort((a, b) => b[1] - a[1]);
  const [top, p] = ranked[0];
  $("emoji").textContent = EMOJI[top] ?? "";
  $("top").textContent = top;
  $("conf").textContent = `${Math.round(p * 100)} % confidence`;
  $("bars").replaceChildren(...ranked.map(([c, v]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
    li.querySelector(".name").textContent = c;
    li.querySelector(".fill").style.width = `${v * 100}%`;
    li.querySelector(".pct").textContent = `${Math.round(v * 100)} %`;
    return li;
  }));
  const n = images.length;
  setStatus(`${m.label} · ${n} segment${n > 1 ? "s" : ""} of 3 s${n > 1 ? " averaged" : ""}`);
  $("result").hidden = false;
}

function drawSpectrograms(imgs) {
  $("specs").replaceChildren(...imgs.map(({ width, height, rgb }) => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const data = new ImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      data.data.set(rgb.subarray(3 * i, 3 * i + 3), 4 * i);
      data.data[4 * i + 3] = 255;
    }
    canvas.getContext("2d").putImageData(data, 0, 0);
    return canvas;
  }));
}

// ---------- wiring ----------

document.querySelectorAll("input[name=arch], input[name=data]").forEach((el) =>
  el.addEventListener("change", () => { showModelInfo(); predict(); }));
$("record").addEventListener("click", toggleRecording);
$("upload").addEventListener("change", (e) => e.target.files[0] && handleAudio(e.target.files[0]));
$("record").disabled = !navigator.mediaDevices?.getUserMedia;
showModelInfo();
