// Guided teaching (✋ Teach, ⚡ Quick setup): the words to teach come one at a time, big on the camera view.
// For each word the ghost first shows the sign (a mirror image of the reference signer over a dimmed camera);
// the camera's keypoints are NOT sent meanwhile, so copying along is not counted. Then the person signs it
// TEACH_REPS times, lowering the hands after each; every rep becomes a personal sample (server: "enroll").
// No ghost recording (a brand-new sign like a name or a medicine): straight to "sign it your way".
import Ghost, { ghostMs, type GhostRef, type HowTo } from './Ghost'
import type { Pt, Raw } from './vision'

// watch: the ghost plays (frames muted) · ready: waiting for the hands to go down (muted) · sign: reps count ·
// saving: last rep in, waiting for the self-test · done: learned (short ✓, then the next word) ·
// check: self-test weak, teach again or keep · finished: summary (frames flow again: signing closes it)
export type GuidePhase = 'watch' | 'ready' | 'sign' | 'saving' | 'done' | 'check' | 'finished'
export const TEACHING: GuidePhase[] = ['watch', 'ready', 'sign', 'saving']  // the server is counting reps
export type TeachReport = { label: string; kept: number; dropped: number; self_test: number; confused_with: string[] }
export type Guide = {
  words: string[]; i: number; phase: GuidePhase
  label: string | null  // the server's name for the current word (set by its first enroll_progress)
  n: number  // reps counted for the current word
  results: Record<string, 'ok' | 'weak'>  // words taught in this session
  report?: TeachReport  // the current word's self-test
  watched: number  // "watch again" count: restarts the demo
}
export const DEMO_PLAYS = 2  // the ghost shows the sign twice, then it is the person's turn

// Same rule as server/segmenter.py: a wrist is raised when it is less than 1.2 shoulder widths below the shoulder
// line, or less than 2.2 and not hanging below its elbow (pose 13 / 14). Only detected hands count.
export function handsUp(raw: Raw, W: number, H: number) {
  const p = raw.pose
  if (!p) return false
  const sw = Math.hypot((p[11].x - p[12].x) * W, (p[11].y - p[12].y) * H) + 1e-6
  const shY = ((p[11].y + p[12].y) / 2) * H
  const up = (h: Pt[] | null, elbow: number) => {
    if (!h) return false
    const y = h[0].y * H, drop = (y - shY) / sw
    return drop < 1.2 || (drop < 2.2 && y - p[elbow].y * H < 0.5 * sw)
  }
  return up(raw.lh, 13) || up(raw.rh, 14)
}

type StageProps = {
  guide: Guide; demo: GhostRef | null | undefined; mirror: boolean; signing: boolean; reps: number; hint: string
  blocked: boolean; pretty: (l: string) => string; onReady: () => void; onShowAgain: () => void
}
// over the camera: the word, what to do now, reps done; the ghost demo; a small looping ghost while signing
export function TeachStage({ guide, demo, mirror, signing, reps, hint, blocked, pretty, onReady, onShowAgain }: StageProps) {
  const word = pretty(guide.label ?? guide.words[guide.i])
  const { phase, n } = guide
  const sub = phase === 'watch' ? (demo === undefined ? 'Loading the sign…' : '👻 Watch the ghost, then copy it like a mirror')
    : phase === 'ready' ? '✋ Your turn: lower your hands to start'
      : phase === 'sign' ? (signing ? '● Signing… lower your hands when done'
        : n === 0 ? (demo ? 'Your turn: sign it, then lower your hands' : 'Sign it your way, then lower your hands')
          : `✓ ${n} done · sign it again (${reps - n} more)`)
        : phase === 'saving' ? 'Saving…'
          : phase === 'done' ? '✓ Learned'
            : phase === 'check' ? 'Your reps looked different from each other'
              : `Learned ${Object.keys(guide.results).length} ${Object.keys(guide.results).length === 1 ? 'sign' : 'signs'}`
  const counting = phase === 'ready' || phase === 'sign' || phase === 'saving'
  return (<>
    {phase === 'watch' && demo && (
      <Ghost key={`${guide.i}-${guide.watched}`} g={demo} loop mirror={mirror} className="ghost-stage teach-demo" />
    )}
    <div className="teach-top">
      <div className="teach-head">
        <span className="teach-word">{phase === 'finished' ? '🎉 All done' : word}</span>
        {guide.words.length > 1 && phase !== 'finished' && <span className="teach-count">{guide.i + 1} / {guide.words.length}</span>}
      </div>
      <div className={`teach-sub ${phase === 'sign' && signing ? 'live' : ''}`} role="status">{sub}</div>
      {counting && (
        <div className="rep-dots" aria-label={`${n} of ${reps} done`}>
          {Array.from({ length: reps }, (_, k) => <i key={k} className={k < n ? 'on' : ''} />)}
        </div>
      )}
      {hint && phase !== 'watch' && <div className="teach-hint">{hint}</div>}
    </div>
    {phase === 'watch' && demo && (
      <div className="teach-go">
        <div className="teach-timer"><div key={`${guide.i}-${guide.watched}`} style={{ animationDuration: `${DEMO_PLAYS * ghostMs(demo)}ms` }} /></div>
        <button className="primary" onClick={onReady}>I'm ready ✋</button>
      </div>
    )}
    {(phase === 'ready' || phase === 'sign') && demo && !blocked && (
      <button className="ghost-mini-btn" onClick={onShowAgain} aria-label="Watch the ghost again">
        <Ghost g={demo} loop mirror={mirror} className="ghost-mini" />
        <small>👻 Watch again</small>
      </button>
    )}
    {phase === 'done' && <div className="teach-flash"><b>✓</b><span>Learned “{word}”</span></div>}
    {phase === 'check' && <div className="teach-flash warn"><b>!</b><span>“{word}” varied a lot</span></div>}
  </>)
}

type CardProps = {
  guide: Guide; howto: HowTo | null | undefined; hasDemo: boolean; reps: number; pretty: (l: string) => string
  onShowAgain: () => void; onSkip: () => void; onStop: () => void; onAgain: () => void; onNext: () => void
  onClose: () => void; onJump: (i: number) => void
}
// the side panel while teaching: how to sign the word, progress over the words, skip / stop
export function TeachCard({ guide, howto, hasDemo, reps, pretty, onShowAgain, onSkip, onStop, onAgain, onNext, onClose, onJump }: CardProps) {
  const many = guide.words.length > 1
  const word = pretty(guide.label ?? guide.words[guide.i])
  const done = Object.keys(guide.results).length
  const weak = Object.values(guide.results).filter((r) => r === 'weak').length
  const chips = many && (
    <div className="chips teach-chips">
      {guide.words.map((w, k) => (
        <button key={w} className={`${guide.results[w] === 'ok' ? 'done' : guide.results[w] === 'weak' ? 'weak' : ''} ${k === guide.i && guide.phase !== 'finished' ? 'cur' : ''}`}
          onClick={() => onJump(k)} title="Teach this word">
          {guide.results[w] === 'ok' ? '✓ ' : guide.results[w] === 'weak' ? '⚠ ' : ''}{pretty(w)}
        </button>
      ))}
    </div>
  )
  if (guide.phase === 'finished') return (
    <div className="teach-card">
      <b className="teach-title">🎉 ISL Bridge learned your signing</b>
      <p>{done} {done === 1 ? 'sign' : 'signs'} taught{weak ? ` (${weak} marked ⚠ could be clearer: tap one to teach it again)` : ''}. They are saved
        in your private profile, and the app now reads <i>your</i> way of signing them.</p>
      <p>Try it now: sign <b>I · PAIN · HEAD</b>, then lower your hands.</p>
      <div className="teach-buttons"><button className="primary" onClick={onClose}>💬 Start the conversation</button></div>
      {chips}
    </div>
  )
  const r = guide.report
  return (
    <div className="teach-card">
      <div className="teach-card-head">
        <b>✋ Teaching ISL Bridge your signing</b>
        {many && <small>word {guide.i + 1} of {guide.words.length}</small>}
      </div>
      {many && <div className="teach-progress" aria-hidden="true"><div style={{ width: `${(done / guide.words.length) * 100}%` }} /></div>}
      {guide.phase === 'check' && r ? (<>
        <p>⚠ Self-test: <b>{r.self_test} of {r.kept}</b> of your “{word}” reps were recognised
          {r.confused_with.length > 0 && <>; it sometimes looks like <b>{r.confused_with.map(pretty).join(', ')}</b></>}.
          Teach it again and sign it the same way each time, or keep it.</p>
        <div className="teach-buttons">
          <button className="primary" onClick={onAgain}>↻ Teach it again</button>
          <button onClick={onNext}>{guide.i + 1 < guide.words.length ? 'Keep it, next word →' : 'Keep it'}</button>
        </div>
      </>) : (<>
        <p>{guide.phase === 'watch'
          ? <>The ghost shows <b>{word}</b>. Watch, then copy it like a mirror.</>
          : <>Sign <b>{word}</b> {reps} times{hasDemo ? ' like the ghost' : ', your way'}. Lower your hands (or hold still) after each one.</>}</p>
        {howto?.steps?.length ? (
          <div className="howto">
            <ol>{howto.steps.map((s) => <li key={s}>{s}</li>)}</ol>
            {howto.tip && <p className="tip">💡 {howto.tip}</p>}
            <small>Steps written by Muse Spark from the sign video{howto.confidence === 'low' ? ' (low confidence: trust the ghost)' : ''}</small>
          </div>
        ) : null}
      </>)}
      <div className="teach-buttons">
        {hasDemo && (guide.phase === 'ready' || guide.phase === 'sign') && <button onClick={onShowAgain}>👻 Watch again</button>}
        {many && guide.phase !== 'check' && <button onClick={onSkip}>Skip word →</button>}
        <button onClick={onStop}>{many ? 'Stop (continue later)' : 'Cancel'}</button>
      </div>
      {chips}
    </div>
  )
}
