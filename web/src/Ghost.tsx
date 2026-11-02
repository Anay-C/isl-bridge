// Ghost signer (Talk mode, 👻): the hearing person's reply is signed by a solid, bright ghost figure drawn
// from the reference recordings (same body-space trajectories as the Learn page's ghost, /api/coach/<sign>),
// instead of playing the video clip. Drawn as a signer FACING the viewer (like the clip, not mirrored), at
// full strength on a dimmed camera so it is easy to read. Frames are interpolated for smooth motion.
import { useEffect, useRef } from 'react'
import { HAND_EDGES } from './vision'

export type GhostRef = { fps: number; frames: { arm: number[][]; lh: number[][] | null; rh: number[][] | null }[] }

const ARM_EDGES = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5]]  // shoulders, elbows, wrists
export const GHOST_SPEED = 0.85  // a little slower than the recording: easier to read
const TAIL_MS = 350  // hold the last pose briefly between signs

const cache = new Map<string, Promise<GhostRef | null>>()
export function loadGhost(sign: string): Promise<GhostRef | null> {
  let p = cache.get(sign)
  if (!p) {
    p = fetch(`/api/coach/${encodeURIComponent(sign)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((g) => (g && g.frames?.length ? fillGaps(g) : null))
      .catch(() => null)
    cache.set(sign, p)
  }
  return p
}
// hand tracking drops out for a few frames in the recordings: hold the last seen hand (<= 0.25 s) so it
// does not flicker in and out
function fillGaps(g: GhostRef): GhostRef {
  const max = Math.round(g.fps * 0.25)
  for (const side of ['lh', 'rh'] as const) {
    let last: number[][] | null = null, gap = 0
    for (const f of g.frames) {
      if (f[side]) { last = f[side]; gap = 0 }
      else if (last && ++gap <= max) f[side] = last
    }
  }
  return g
}
export const ghostMs = (g: GhostRef) => (g.frames.length / g.fps / GHOST_SPEED) * 1000 + TAIL_MS

const lerp = (a: number[][] | null, b: number[][] | null, t: number) =>
  !a ? null : !b ? a : a.map((q, i) => [q[0] + (b[i][0] - q[0]) * t, q[1] + (b[i][1] - q[1]) * t])

export default function Ghost({ g, label, caption }: { g: GhostRef | null; label: string; caption: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !g) return
    const t0 = performance.now()
    let raf = 0
    const frame = () => {
      const dpr = window.devicePixelRatio || 1
      const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr)
      if (c.width !== W || c.height !== H) { c.width = W; c.height = H }
      const ctx = c.getContext('2d')!
      ctx.clearRect(0, 0, W, H)
      const pos = Math.min(g.frames.length - 1, ((performance.now() - t0) / 1000) * g.fps * GHOST_SPEED)
      const i = Math.floor(pos), t = pos - i
      const a = g.frames[i], b = g.frames[Math.min(i + 1, g.frames.length - 1)]
      const arm = lerp(a.arm, b.arm, t)!
      const lh = lerp(a.lh, b.lh, t), rh = lerp(a.rh, b.rh, t)
      // the body tracker lags fast hands: end each forearm at its hand's wrist so hands never float loose
      if (lh) arm[4] = lh[0]
      if (rh) arm[5] = rh[0]
      // body space: shoulders at x = +-0.5, y = 0. Measured over all 227 references: hands stay within
      // x -0.9..0.8 and y -0.9..2.3 (99%); nose at y -0.58 (median), head top ~-1.0. Fit x -1.2..1.2, y -1.1..2.45.
      const sc = Math.min(W / 2.4, H / 3.6)
      const cx = W / 2, cy = H / 2 - 0.67 * sc
      const pt = (q: number[]) => [cx + q[0] * sc, cy + q[1] * sc]
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      // head + torso, so the hands read against a body
      ctx.fillStyle = 'rgba(120, 220, 255, 0.18)'
      ctx.strokeStyle = 'rgba(150, 230, 255, 0.9)'
      ctx.lineWidth = Math.max(2, sc * 0.03)
      ctx.beginPath()
      ctx.ellipse(cx, cy - 0.62 * sc, 0.28 * sc, 0.36 * sc, 0, 0, Math.PI * 2)
      ctx.fill(); ctx.stroke()
      const [sl, sr] = [pt(arm[0]), pt(arm[1])]
      ctx.beginPath()
      ctx.moveTo(sl[0], sl[1]); ctx.lineTo(sr[0], sr[1])
      ctx.lineTo(cx + (sr[0] - cx) * 0.8, cy + 2.0 * sc); ctx.lineTo(cx + (sl[0] - cx) * 0.8, cy + 2.0 * sc)
      ctx.closePath()
      ctx.fill(); ctx.stroke()
      ctx.shadowColor = 'rgba(120, 220, 255, 1)'
      ctx.shadowBlur = 16 * dpr
      const lines = (pts: number[][], edges: number[][], color: string, lw: number) => {
        ctx.strokeStyle = color
        ctx.lineWidth = lw
        ctx.beginPath()
        for (const [p, q] of edges) {
          const [u, v] = [pt(pts[p]), pt(pts[q])]
          ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1])
        }
        ctx.stroke()
      }
      lines(arm, ARM_EDGES, 'rgba(140, 225, 255, 1)', Math.max(6, sc * 0.11))
      for (const h of [lh, rh]) {
        if (!h) continue
        lines(h, HAND_EDGES, 'rgba(235, 250, 255, 1)', Math.max(3, sc * 0.045))
        ctx.fillStyle = 'rgba(255, 255, 255, 1)'
        for (const q of h) { const [x, y] = pt(q); ctx.beginPath(); ctx.arc(x, y, Math.max(2, sc * 0.022), 0, Math.PI * 2); ctx.fill() }
      }
      ctx.shadowBlur = 0
      raf = requestAnimationFrame(frame)
    }
    frame()
    return () => cancelAnimationFrame(raf)
  }, [g])
  return (
    <div className="ghost-stage">
      <canvas ref={ref} />
      <div className="clip-label">{label}</div>
      <div className="caption">{caption}</div>
    </div>
  )
}
