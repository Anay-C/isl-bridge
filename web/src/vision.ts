// MediaPipe Tasks in the browser. MUST match ml/extract.py: same .task models (hand + pose lite),
// unmirrored frames, VIDEO mode, left/right hand assigned by distance to pose wrists 15/16.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'

export type Pt = { x: number; y: number; z: number; visibility?: number }
// t: capture time (ms, performance.now) so the server measures the camera's real frame rate even when Wi-Fi
// delivers frames in bursts
export type Frame = { w: number; h: number; t: number; pose: number[] | null; lh: number[] | null; rh: number[] | null }
export type Raw = { pose: Pt[] | null; lh: Pt[] | null; rh: Pt[] | null }

let hand: HandLandmarker | null = null
let pose: PoseLandmarker | null = null
let lastTs = 0

export async function loadVision() {
  if (hand && pose) return  // already loaded (e.g. switching camera <-> screen)
  const fileset = await FilesetResolver.forVisionTasks('/wasm')
  const make = async (delegate: 'GPU' | 'CPU') => {
    hand = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate },
      runningMode: 'VIDEO',
      numHands: 2,
      // defaults (0.5) drop the second hand when hands touch or overlap (HELP: thumb on the other palm)
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    })
    pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: '/models/pose_landmarker_lite.task', delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
    })
  }
  try {
    await make('GPU')
  } catch {
    await make('CPU')
  }
}

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y)

function assignHands(hands: Pt[][], p: Pt[] | null): [Pt[] | null, Pt[] | null] {
  if (!hands.length) return [null, null]
  if (!p) {
    let l: Pt[] | null = null, r: Pt[] | null = null
    for (const h of hands) { if (h[0].x < 0.5) r = h; else l = h } // unmirrored: signer's right is image left
    return [l, r]
  }
  const lw = p[15], rw = p[16]
  if (hands.length === 1) {
    const h = hands[0]
    return dist(h[0], lw) < dist(h[0], rw) ? [h, null] : [null, h]
  }
  const [a, b] = hands
  return dist(a[0], lw) + dist(b[0], rw) <= dist(b[0], lw) + dist(a[0], rw) ? [a, b] : [b, a]
}

// 4 decimals = 0.06 px at 640 px: no effect on features, about half the bytes per frame over Wi-Fi
const r4 = (v: number) => Math.round(v * 1e4) / 1e4
const flat = (pts: Pt[] | null, withVis = false) =>
  pts ? pts.flatMap((q) => (withVis ? [r4(q.x), r4(q.y), r4(q.z), r4(q.visibility ?? 0)] : [r4(q.x), r4(q.y), r4(q.z)])) : null

export function detect(video: HTMLVideoElement): { frame: Frame; raw: Raw } | null {
  if (!hand || !pose || video.readyState < 2) return null
  let ts = performance.now()
  if (ts <= lastTs) ts = lastTs + 1 // strictly increasing timestamps
  lastTs = ts
  const pr = pose.detectForVideo(video, ts)
  const hr = hand.detectForVideo(video, ts)
  const p = (pr.landmarks[0] as Pt[] | undefined) ?? null
  const [lh, rh] = assignHands(hr.landmarks as Pt[][], p)
  return {
    frame: { w: video.videoWidth, h: video.videoHeight, t: Math.round(ts), pose: flat(p, true), lh: flat(lh), rh: flat(rh) },
    raw: { pose: p, lh, rh },
  }
}

export const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]]
const POSE_EDGES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24]]

// Display-only smoothing (the server always gets the raw landmarks, exactly like training). Adaptive EMA in the
// spirit of the One Euro filter: still hands are smoothed hard (no jitter), fast hands follow with little lag.
const prevDraw: Record<'pose' | 'lh' | 'rh', Pt[] | null> = { pose: null, lh: null, rh: null }
function smooth(key: 'pose' | 'lh' | 'rh', pts: Pt[] | null): Pt[] | null {
  const prev = prevDraw[key]
  if (!pts || !prev || prev.length !== pts.length) { prevDraw[key] = pts; return pts }
  let speed = 0
  for (let i = 0; i < pts.length; i++) speed = Math.max(speed, Math.hypot(pts[i].x - prev[i].x, pts[i].y - prev[i].y))
  const a = Math.min(1, 0.25 + speed * 25)  // 0.25 when still; no smoothing above ~3% of the frame per frame
  const out = pts.map((q, i) => ({ ...q, x: prev[i].x + a * (q.x - prev[i].x), y: prev[i].y + a * (q.y - prev[i].y) }))
  prevDraw[key] = out
  return out
}

export function draw(ctx: CanvasRenderingContext2D, rawIn: Raw, active: boolean) {
  const raw = { pose: smooth('pose', rawIn.pose), lh: smooth('lh', rawIn.lh), rh: smooth('rh', rawIn.rh) }
  const { width: W, height: H } = ctx.canvas
  ctx.clearRect(0, 0, W, H)
  const line = (pts: Pt[], edges: number[][], color: string, lw: number) => {
    ctx.strokeStyle = color
    ctx.lineWidth = lw
    ctx.lineCap = 'round'
    for (const [i, j] of edges) {
      ctx.beginPath()
      ctx.moveTo(pts[i].x * W, pts[i].y * H)
      ctx.lineTo(pts[j].x * W, pts[j].y * H)
      ctx.stroke()
    }
  }
  if (raw.pose) line(raw.pose, POSE_EDGES, 'rgba(255,255,255,0.55)', 3)
  const c = active ? '#ffb020' : '#3ddc97'
  if (raw.lh) line(raw.lh, HAND_EDGES, c, 3)
  if (raw.rh) line(raw.rh, HAND_EDGES, c, 3)
}
