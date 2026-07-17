// Captures mic audio, resamples to 16 kHz mono, ships Int16 PCM chunks.
class PCMCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.readPos = 0;
    this.pending = new Float32Array(0);
    this.out = new Int16Array(2048); // 128 ms @ 16 kHz
    this.outLen = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;

    // Append incoming samples to the pending buffer.
    const merged = new Float32Array(this.pending.length + ch.length);
    merged.set(this.pending);
    merged.set(ch, this.pending.length);
    this.pending = merged;

    // Linear-interpolation resample from device rate to 16 kHz.
    while (this.readPos + 1 < this.pending.length) {
      const i = Math.floor(this.readPos);
      const frac = this.readPos - i;
      const s = this.pending[i] * (1 - frac) + this.pending[i + 1] * frac;
      const v = Math.max(-1, Math.min(1, s));
      this.out[this.outLen++] = v < 0 ? v * 32768 : v * 32767;
      this.readPos += this.ratio;
      if (this.outLen === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.outLen = 0;
      }
    }
    const keep = Math.floor(this.readPos);
    this.pending = this.pending.slice(keep);
    this.readPos -= keep;
    return true;
  }
}
registerProcessor("pcm-capture", PCMCapture);
