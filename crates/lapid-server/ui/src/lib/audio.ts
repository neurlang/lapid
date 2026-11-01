// Browser-side audio conversion: anything decodeAudioData understands -> 16 kHz mono 16-bit PCM WAV.

export const TARGET_RATE = 16000;

export async function decodeAudio(blob: Blob): Promise<AudioBuffer> {
  const ctx = new AudioContext();
  try {
    return await ctx.decodeAudioData(await blob.arrayBuffer());
  } finally {
    void ctx.close();
  }
}

/** Downmix + resample with an OfflineAudioContext (the browser's resampler is good and fast). */
export async function toMono(buffer: AudioBuffer, rate = TARGET_RATE): Promise<Float32Array> {
  const length = Math.max(1, Math.ceil(buffer.duration * rate));
  const offline = new OfflineAudioContext(1, length, rate);
  const src = offline.createBufferSource();
  src.buffer = buffer;
  src.connect(offline.destination); // channel count 1 => speaker downmix
  src.start();
  const out = await offline.startRendering();
  return out.getChannelData(0);
}

export function encodeWav(samples: Float32Array, rate = TARGET_RATE): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

export async function toWav16k(blob: Blob): Promise<{ wav: Blob; duration: number }> {
  const decoded = await decodeAudio(blob);
  const mono = await toMono(decoded);
  return { wav: encodeWav(mono), duration: decoded.duration };
}
