// Hold-to-talk recorder -> 16 kHz mono 16-bit WAV (what Muse Voice Transcribe expects).
// onPcm (optional) receives each ~85 ms chunk as 16 kHz PCM16 for realtime streaming.
export class WavRecorder {
  private onPcm?: (pcm: ArrayBuffer) => void
  constructor(onPcm?: (pcm: ArrayBuffer) => void) { this.onPcm = onPcm }

  private ctx: AudioContext | null = null
  private stream: MediaStream | null = null
  private node: ScriptProcessorNode | null = null
  private chunks: Float32Array[] = []

  async start() {
    // iPhone Safari starts an AudioContext "suspended" unless it is created (and resumed) inside the tap itself;
    // after awaiting the microphone permission the tap no longer counts, and no audio would ever arrive
    this.ctx = new AudioContext()
    void this.ctx.resume().catch(() => {})
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
    } catch (e) {
      void this.ctx.close().catch(() => {})
      this.ctx = null
      throw e
    }
    if (this.ctx.state === 'suspended') await this.ctx.resume().catch(() => {})
    const src = this.ctx.createMediaStreamSource(this.stream)
    this.node = this.ctx.createScriptProcessor(4096, 1, 1)
    this.chunks = []
    const rate = this.ctx.sampleRate
    this.node.onaudioprocess = (e) => {
      const c = new Float32Array(e.inputBuffer.getChannelData(0))
      this.chunks.push(c)
      if (this.onPcm) this.onPcm(toPcm16(downsample(c, rate, 16000)))
    }
    src.connect(this.node)
    this.node.connect(this.ctx.destination)
  }

  async stop(): Promise<Blob> {
    const rate = this.ctx?.sampleRate ?? 48000
    this.node?.disconnect()
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.ctx?.close()
    const n = this.chunks.reduce((s, c) => s + c.length, 0)
    const all = new Float32Array(n)
    let o = 0
    for (const c of this.chunks) { all.set(c, o); o += c.length }
    return encodeWav(downsample(all, rate, 16000), 16000)
  }
}

function downsample(x: Float32Array, from: number, to: number) {
  if (from === to) return x
  const ratio = from / to
  const out = new Float32Array(Math.floor(x.length / ratio))
  for (let i = 0; i < out.length; i++) {
    const s = Math.floor(i * ratio), e = Math.min(x.length, Math.floor((i + 1) * ratio))
    let acc = 0
    for (let j = s; j < e; j++) acc += x[j]
    out[i] = acc / Math.max(1, e - s)
  }
  return out
}

function toPcm16(x: Float32Array) {
  const out = new Int16Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = Math.max(-1, Math.min(1, x[i])) * 0x7fff
  return out.buffer
}

function encodeWav(x: Float32Array, rate: number) {
  const buf = new ArrayBuffer(44 + x.length * 2)
  const v = new DataView(buf)
  const str = (o: number, s: string) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)))
  str(0, 'RIFF'); v.setUint32(4, 36 + x.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true)
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true)
  str(36, 'data'); v.setUint32(40, x.length * 2, true)
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 0x7fff, true)
  return new Blob([buf], { type: 'audio/wav' })
}
