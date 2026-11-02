// Learn mode (?learn): Ghost Coach. The reference signer's arms + hands are drawn as a "ghost" onto the
// learner's own body (scaled to their shoulders) in sync with the reference clip; the learner copies it,
// the server scores the attempt (recogniser + trajectory comparison) and returns stars + tips.
import { useEffect, useRef, useState, type MutableRefObject } from 'react'
import { HAND_EDGES, type Raw } from './vision'

type HowTo = { steps: string[]; tip: string; confidence: 'high' | 'medium' | 'low' }
type Ghost = { fps: number; frames: { arm: number[][]; lh: number[][] | null; rh: number[][] | null }[]; howto?: HowTo | null }
export type PracticeResult = {
  label: string; stars: number; score: number; tips: string[]; bad_hands: string[]; mirrored?: boolean; top: string[]
}
export type Overlay = (ctx: CanvasRenderingContext2D, raw: Raw) => void

type Props = {
  clips: string[]
  result: PracticeResult | null
  send: (m: object) => void
  overlay: MutableRefObject<Overlay | null>
  pretty: (l: string) => string
  capture: () => string | null  // current camera frame as a JPEG data URL
}
type LensResult = { object: string; hindi: string; sign: string | null; clip: string | null; learnable: boolean }

const LESSONS: Record<string, string[]> = {
  'Hospital basics': ['Hello', 'Thank you', 'Pain', 'Doctor', 'Medicine', 'Help', 'Water', 'Fever', 'Yes', 'No',
    'Please', 'Sorry', 'Hospital', 'Headache', 'Emergency'],
  'Feelings': ['happy', 'sad', 'sick', 'Tired', 'good', 'bad', 'hot', 'cold'],
  'People': ['I', 'you', 'Mother', 'Father', 'Doctor', 'Nurse', 'Friend', 'Family', 'Baby'],
  'Time': ['Today', 'Tomorrow', 'Yesterday', 'Morning', 'Night', 'Week', 'Time'],
  'Everyday': ['Good afternoon', 'Good evening', 'Good night', 'Man', 'Woman', 'Boy', 'Girl', 'Teacher', 'Student',
    'School', 'Bank', 'Market', 'Bus', 'Train', 'Car', 'Book', 'Door', 'Key', 'Clock', 'Television'],
  'Days & colours': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'Red', 'Blue', 'Green', 'Yellow', 'Black', 'White'],
}
const ARM_EDGES = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5]] // shoulders, elbows, wrists
const STARS_KEY = 'isl-learn-stars'

function loadStars(): Record<string, number> {
  try { return JSON.parse(localStorage.getItem(STARS_KEY) ?? '{}') } catch { return {} }
}

export default function Learn({ clips, result, send, overlay, pretty, capture }: Props) {
  const lessons = Object.fromEntries(Object.entries(LESSONS).map(([k, v]) => [k, v.filter((w) => clips.includes(w))]))
  lessons['All words'] = clips
  const [lesson, setLesson] = useState('Hospital basics')
  const [word, setWord] = useState(lessons['Hospital basics'][0] ?? clips[0])
  const [ghost, setGhost] = useState<Ghost | null>(null)
  const [slow, setSlow] = useState(false)
  const [showGhost, setShowGhost] = useState(true)
  const [trying, setTrying] = useState(false)
  const [stars, setStars] = useState(loadStars)
  const ref = useRef<HTMLVideoElement>(null)
  const badRef = useRef<{ hands: string[]; mirrored: boolean }>({ hands: [], mirrored: false })
  const [lens, setLens] = useState<LensResult | 'looking' | 'error' | null>(null)
  const runLens = async () => {
    const image = capture()
    if (!image) return
    setLens('looking')
    try {
      const r = await fetch('/api/lens', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image }) })
      if (!r.ok) throw new Error()
      setLens(await r.json())
    } catch { setLens('error') }
  }

  useEffect(() => {
    setGhost(null)
    fetch(`/api/coach/${encodeURIComponent(word)}`).then((r) => (r.ok ? r.json() : null)).then(setGhost)
  }, [word])
  useEffect(() => { if (ref.current) ref.current.playbackRate = slow ? 0.5 : 1 }, [slow, word])

  // practice result arrives from the server
  useEffect(() => {
    if (!result || result.label !== word) return
    setTrying(false)
    badRef.current = { hands: result.bad_hands, mirrored: !!result.mirrored }
    setStars((s) => {
      const next = { ...s, [word]: Math.max(s[word] ?? 0, result.stars) }
      try { localStorage.setItem(STARS_KEY, JSON.stringify(next)) } catch { /* private mode */ }
      return next
    })
  }, [result])

  // ghost drawer, called by the camera loop after the learner's own skeleton
  useEffect(() => {
    overlay.current = (ctx, raw) => {
      const v = ref.current
      if (!ghost || !showGhost || !v || !ghost.frames.length) return
      const f = ghost.frames[Math.min(ghost.frames.length - 1, Math.floor(v.currentTime * ghost.fps))]
      const { width: W, height: H } = ctx.canvas
      let cx = W / 2, cy = H * 0.45, sc = W * 0.22
      const p = raw.pose
      if (p && (p[11].visibility ?? 1) > 0.3 && (p[12].visibility ?? 1) > 0.3) {
        cx = ((p[11].x + p[12].x) / 2) * W
        cy = ((p[11].y + p[12].y) / 2) * H
        sc = Math.hypot((p[11].x - p[12].x) * W, (p[11].y - p[12].y) * H)
      }
      const pt = (q: number[]) => [cx + q[0] * sc, cy + q[1] * sc]
      const lines = (pts: number[][], edges: number[][], color: string, lw: number) => {
        ctx.strokeStyle = color
        ctx.lineWidth = lw
        ctx.beginPath()
        for (const [i, j] of edges) {
          const [a, b] = [pt(pts[i]), pt(pts[j])]
          ctx.moveTo(a[0], a[1])
          ctx.lineTo(b[0], b[1])
        }
        ctx.stroke()
      }
      ctx.save()
      ctx.lineCap = 'round'
      ctx.shadowColor = 'rgba(120, 220, 255, 0.9)'
      ctx.shadowBlur = 18
      lines(f.arm, ARM_EDGES, 'rgba(120, 220, 255, 0.55)', Math.max(4, sc * 0.07))
      const { hands, mirrored } = badRef.current
      const flag = (side: 'left' | 'right') => hands.includes(mirrored ? (side === 'left' ? 'right' : 'left') : side)
      for (const [side, h] of [['left', f.lh], ['right', f.rh]] as const) {
        if (!h) continue
        const bad = flag(side)
        ctx.shadowColor = bad ? 'rgba(255, 170, 60, 0.95)' : 'rgba(120, 220, 255, 0.9)'
        lines(h, HAND_EDGES, bad ? 'rgba(255, 170, 60, 0.9)' : 'rgba(160, 235, 255, 0.85)', Math.max(2, sc * 0.03))
      }
      ctx.restore()
    }
    return () => { overlay.current = null }
  }, [ghost, showGhost, overlay])

  const pick = (w: string) => {
    setWord(w)
    setTrying(false)
    badRef.current = { hands: [], mirrored: false }
    send({ type: 'practice', label: null })
  }
  const tryIt = () => {
    setTrying(true)
    badRef.current = { hands: [], mirrored: false }
    send({ type: 'practice', label: word })
  }
  const list = lessons[lesson] ?? []
  const next = () => pick(list[(list.indexOf(word) + 1) % list.length] ?? word)
  const done = list.filter((w) => (stars[w] ?? 0) >= 3).length
  const shown = result && result.label === word && !trying ? result : null

  return (
    <section className="learn">
      <div className="learn-top">
        <select value={lesson} onChange={(e) => { setLesson(e.target.value); pick(lessons[e.target.value][0]) }}>
          {Object.keys(lessons).map((k) => <option key={k}>{k}</option>)}
        </select>
        <span className="learn-progress">{done}/{list.length} mastered</span>
      </div>

      <div className="learn-ref">
        <video ref={ref} key={word} src={`/clips/${encodeURIComponent(word)}.mp4`} autoPlay loop muted playsInline />
        <div className="learn-word">{pretty(word)}</div>
      </div>

      {ghost?.howto && (
        <div className="howto">
          <ol>{ghost.howto.steps.map((s) => <li key={s}>{s}</li>)}</ol>
          {ghost.howto.tip && <p className="tip">💡 {ghost.howto.tip}</p>}
          <div className="howto-foot">
            <small>Steps written by Muse Spark from the video{ghost.howto.confidence === 'low' ? ' (low confidence, trust the video)' : ''}</small>
            <button className="ghost" aria-label="Read the steps aloud" onClick={() => {
              const u = new SpeechSynthesisUtterance(ghost.howto!.steps.join('. '))
              u.lang = 'en-IN'
              speechSynthesis.cancel()
              speechSynthesis.speak(u)
            }}>🔊</button>
          </div>
        </div>
      )}

      <div className="learn-controls">
        <button className={slow ? 'on' : ''} onClick={() => setSlow(!slow)}>🐢 Slow</button>
        <button className={showGhost ? 'on' : ''} onClick={() => setShowGhost(!showGhost)}>👻 Ghost</button>
        <button onClick={runLens} title="Hold an object up to the camera">🔍 Lens</button>
        {trying
          ? <button className="primary" onClick={() => { setTrying(false); send({ type: 'practice', label: null }) }}>Your turn: sign it now… (cancel)</button>
          : <button className="primary" onClick={tryIt}>✋ Try it</button>}
      </div>

      {lens && (
        <div className="lens">
          {lens === 'looking' && <p>Looking at what you're holding…</p>}
          {lens === 'error' && <p>Couldn't recognise that. Hold the object closer, in good light.</p>}
          {typeof lens === 'object' && (<>
            <div className="lens-head"><b>{lens.object || 'Nothing clear'}</b>{lens.hindi && <span> · {lens.hindi}</span>}
              <button className="ghost" aria-label="Close" onClick={() => setLens(null)}>✕</button></div>
            {lens.clip
              ? <video src={lens.clip} autoPlay loop muted playsInline />
              : lens.object && <p>No single-sign video for “{lens.object}” in the ISLRTC dictionary yet.</p>}
            {lens.sign && <p>Sign: <b>{pretty(lens.sign)}</b></p>}
            {lens.learnable && lens.sign && (
              <button className="primary" onClick={() => { setLesson('All words'); pick(lens.sign!); setLens(null) }}>Learn this sign with the ghost →</button>
            )}
          </>)}
        </div>
      )}

      {shown && (
        <div className={`learn-result s${shown.stars}`}>
          <div className="stars" aria-label={`${shown.stars} of 3 stars`}>{'★'.repeat(shown.stars)}{'☆'.repeat(3 - shown.stars)}</div>
          <div className="score-bar"><div style={{ width: `${shown.score}%` }} /></div>
          <ul>{shown.tips.map((t) => <li key={t}>{t}</li>)}</ul>
          {shown.stars < 3 && shown.top[0] && shown.top[0] !== word && <small>It looked more like “{pretty(shown.top[0])}”.</small>}
          {shown.mirrored && <small>Signing left-handed? That's fine, it was compared mirrored.</small>}
          <div className="row">
            <button onClick={tryIt}>Again</button>
            <button className="primary" onClick={next}>Next word →</button>
          </div>
        </div>
      )}

      <div className="chips">
        {list.map((w) => (
          <button key={w} className={`${w === word ? 'cur' : ''} ${(stars[w] ?? 0) >= 3 ? 'done' : ''}`} onClick={() => pick(w)}>
            {pretty(w)} {stars[w] ? '★'.repeat(stars[w]) : ''}
          </button>
        ))}
      </div>
    </section>
  )
}
