// Audio -> model input, reproducing the Python pipeline in the browser:
//   mel_to_rgb() in prepare_*.ipynb   (librosa mel-spectrogram -> z-score -> viridis PNG)
//   eval_tf in the training notebooks (PIL grayscale / bilinear resize -> ToTensor -> Normalize)
// Plain ES module with no browser API, so it also runs under node.

// spec: {sr, duration, min_seg, n_fft, hop, n_mels}; dsp: contents of models/dsp.json
export function createDsp(spec, dsp) {
  const { sr, n_fft: nFft, hop, n_mels: nMels } = spec;
  const chunk = Math.round(spec.duration * sr);
  const minSamples = Math.round(spec.min_seg * sr);
  const nFrames = Math.floor(chunk / hop);     // 300 columns kept, as in mel_to_rgb
  const nBins = nFft / 2 + 1;

  // periodic Hann window (scipy get_window("hann", fftbins=True), used by librosa)
  const window = new Float64Array(nFft);
  for (let n = 0; n < nFft; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / nFft);
  // n_fft = 400 is not a power of two: plain DFT with lookup tables, fast enough for 3 s
  const cos = new Float64Array(nFft);
  const sin = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / nFft);
    sin[i] = Math.sin((2 * Math.PI * i) / nFft);
  }

  // librosa.util.fix_length + melspectrogram(center=True, pad_mode="constant") + power_to_db
  function logMel(segment) {
    const pad = nFft / 2;
    const x = new Float64Array(chunk + 2 * pad);
    x.set(segment.subarray(0, chunk), pad);
    const total = 1 + Math.floor(chunk / hop);  // librosa frame count before [:, :300]
    const db = Array.from({ length: nMels }, () => new Float64Array(total));
    const frame = new Float64Array(nFft);
    const power = new Float64Array(nBins);
    let max = -Infinity;
    for (let t = 0; t < total; t++) {
      for (let n = 0; n < nFft; n++) frame[n] = x[t * hop + n] * window[n];
      for (let k = 0; k < nBins; k++) {
        let re = 0, im = 0;
        for (let n = 0, idx = 0; n < nFft; n++, idx = (idx + k) % nFft) {
          re += frame[n] * cos[idx];
          im -= frame[n] * sin[idx];
        }
        power[k] = re * re + im * im;
      }
      for (let m = 0; m < nMels; m++) {
        const { start, weights } = dsp.mel_filters[m];
        let s = 0;
        for (let j = 0; j < weights.length; j++) s += weights[j] * power[start + j];
        const v = 10 * Math.log10(Math.max(1e-10, s));  // ref = 1.0, amin = 1e-10
        db[m][t] = v;
        if (v > max) max = v;
      }
    }
    for (const row of db) for (let t = 0; t < total; t++) row[t] = Math.max(row[t], max - 80);  // top_db
    return db.map((row) => row.subarray(0, nFrames));
  }

  // The spectrogram image: RGB, width nFrames, height nMels, high frequencies on top.
  function melImage(segment) {
    const db = logMel(segment);
    let sum = 0, sq = 0;
    const count = nMels * nFrames;
    for (const row of db) for (const v of row) sum += v;
    const mean = sum / count;
    for (const row of db) for (const v of row) sq += (v - mean) ** 2;
    const std = Math.sqrt(sq / count);
    const rgb = new Uint8Array(count * 3);
    for (let m = 0; m < nMels; m++) {
      const row = db[nMels - 1 - m];  // norm[::-1]
      for (let t = 0; t < nFrames; t++) {
        const z = (row[t] - mean) / (std + 1e-9);
        const norm = Math.min(Math.max(z, -3), 3) / 6 + 0.5;
        const color = dsp.viridis[Math.min(Math.floor(norm * 256), 255)];  // matplotlib lookup
        rgb.set(color, (m * nFrames + t) * 3);
      }
    }
    return { width: nFrames, height: nMels, rgb };
  }

  // 3 s chunks like prepare_*.ipynb: a trailing chunk shorter than min_seg is dropped
  function splitChunks(y) {
    const out = [];
    for (let start = 0; start < y.length; start += chunk) {
      const seg = y.subarray(start, start + chunk);
      if (seg.length >= minSamples) out.push(seg);
    }
    return out;
  }

  return { melImage, splitChunks };
}

// Drops the silence before and after speech (the corpus clips are cut tightly around the
// sentence, a live recording is not). Frames quieter than peak - topDb are silence; a
// short margin is kept, like the pauses at the edges of the corpus clips (on the EmoDB
// test split, trimming without it costs ~5 points of accuracy, with it ~0).
export function trimSilence(y, sr, topDb = 35, margin = 0.2) {
  const frame = Math.round(0.025 * sr), hop = Math.round(0.01 * sr);
  const rms = [];
  for (let s = 0; s + frame <= y.length; s += hop) {
    let e = 0;
    for (let i = s; i < s + frame; i++) e += y[i] * y[i];
    rms.push(Math.sqrt(e / frame));
  }
  const peak = Math.max(...rms, 1e-10);
  const threshold = peak * 10 ** (-topDb / 20);
  const first = rms.findIndex((r) => r > threshold);
  const last = rms.findLastIndex((r) => r > threshold);
  if (first < 0) return y;
  const pad = Math.round(margin * sr);
  return y.subarray(Math.max(0, first * hop - pad), Math.min(y.length, last * hop + frame + pad));
}

// transforms.Grayscale + Resize((size, size)) + ToTensor + Normalize, as PIL does them.
// Returns a Float32Array laid out (channels, size, size).
export function toTensor(img, { img_size: size, channels, mean, std }) {
  const { width, height, rgb } = img;
  let planes;
  if (channels === 1) {
    const gray = new Uint8Array(width * height);  // PIL convert("L"), fixed-point
    for (let i = 0; i < gray.length; i++) {
      gray[i] = (rgb[3 * i] * 19595 + rgb[3 * i + 1] * 38470 + rgb[3 * i + 2] * 7471 + 0x8000) >> 16;
    }
    planes = [gray];
  } else {
    planes = [0, 1, 2].map((c) => Uint8Array.from({ length: width * height }, (_, i) => rgb[3 * i + c]));
  }
  const out = new Float32Array(channels * size * size);
  planes.forEach((plane, c) => {
    const resized = resizeBilinear(plane, width, height, size, size);
    for (let i = 0; i < resized.length; i++) {
      out[c * size * size + i] = (resized[i] / 255 - mean[c]) / std[c];
    }
  });
  return out;
}

// PIL Image.resize(BILINEAR) for 8-bit images (Resample.c): antialiased when shrinking,
// horizontal pass then vertical pass, 22-bit fixed-point weights, uint8 in between.
const PRECISION_BITS = 32 - 8 - 2;

function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale;  // bilinear filter support = 1
  const coeffs = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = [];
    let total = 0;
    for (let x = xmin; x < xmax; x++) {
      const v = Math.max(0, 1 - Math.abs((x - center + 0.5) / filterScale));
      w.push(v);
      total += v;
    }
    coeffs.push({ xmin, k: w.map((v) => Math.trunc(0.5 + (v / total) * (1 << PRECISION_BITS))) });
  }
  return coeffs;
}

function resample(get, n, coeffs) {
  const out = new Uint8Array(n * coeffs.length);
  for (let line = 0; line < n; line++) {
    coeffs.forEach(({ xmin, k }, o) => {
      let ss = 1 << (PRECISION_BITS - 1);
      for (let j = 0; j < k.length; j++) ss += get(line, xmin + j) * k[j];
      out[line * coeffs.length + o] = Math.min(255, Math.max(0, Math.floor(ss / (1 << PRECISION_BITS))));
    });
  }
  return out;
}

function resizeBilinear(src, w, h, outW, outH) {
  const horiz = resample((y, x) => src[y * w + x], h, coefficients(w, outW));  // (h, outW)
  const vert = resample((x, y) => horiz[y * outW + x], outW, coefficients(h, outH));  // (outW, outH)
  const out = new Uint8Array(outW * outH);
  for (let x = 0; x < outW; x++) for (let y = 0; y < outH; y++) out[y * outW + x] = vert[x * outH + y];
  return out;
}
