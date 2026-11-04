import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import { detect, draw, loadVision, type Raw } from './vision'
import { WavRecorder } from './wav'
import Record from './Record'
import Learn, { type Overlay, type PracticeResult } from './Learn'
import Cutout from './Cutout'
import Ghost, { ghostMs, loadGhost, type GhostRef } from './Ghost'
import { DEMO_PLAYS, TEACHING, TeachCard, TeachStage, handsUp, type Guide } from './Teach'

type Cand = { label: string; p: number }
// cands: the top-3 per sign that Muse Spark chose from; chosen: what it picked (shows context corrections);
// sid: the server's id of a signed sentence ("✏️ Wrong?" reports refer to it)
type Turn = { who: 'deaf' | 'hearing'; text: string; sub?: string; ms?: number; playlist?: Clip[]; cands?: Cand[][]; chosen?: string[]; typed?: boolean; sid?: number; reported?: boolean
  // hearing turns: shown as soon as the words are known (captions first), the ISL signs follow
  waiting?: boolean; failed?: boolean }
type Fix = { turn: number; sid?: number; signs: string[]; meant: string[]; note: string } | null

// Every device gets its own private profile (taught signs). Anay's own setup (localhost / home Wi-Fi) keeps
// "anay"; a tester opening the public link gets a random id. ?user=NAME picks a profile explicitly (remembered).
function initialUser() {
  const clean = (s: string) => s.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40)
  const q = clean(new URLSearchParams(location.search).get('user') ?? '')
  if (q) { localStorageSet('isl-user', q); return q }
  const saved = localStorageGet('isl-user')
  if (saved) return saved
  const home = /^(localhost|127\.|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(location.hostname)
  const id = home ? 'anay' : `t-${Math.random().toString(36).slice(2, 8)}`
  localStorageSet('isl-user', id)
  return id
}
function newProfile() {
  localStorageSet('isl-user', `t-${Math.random().toString(36).slice(2, 8)}`)
  localStorageSet('isl-setup-later', null)
  location.href = location.pathname  // drop ?user= so the new id sticks
}
// Opened through the fixed tester address (deploy/forward, ?via=): remember it. If this link dies (the laptop's
// tunnel was replaced), the app offers to reopen through it, which leads to the new link with the same profile.
function initialHome() {
  const v = new URLSearchParams(location.search).get('via') ?? ''
  if (/^https:\/\/[a-z0-9-]+\.static\.hf\.space$/.test(v)) { localStorageSet('isl-home', v); return v }
  return localStorageGet('isl-home')
}
// camera failures in plain words (the raw DOMException names mean nothing to a tester)
function cameraHelp(e: Error) {
  if (!navigator.mediaDevices) return 'The camera needs a secure (https) link. Open the https:// address.'
  if (e.name === 'NotAllowedError' || e.name === 'SecurityError')
    return 'Camera blocked. Allow the camera for this site (tap the 🔒 or ⓘ next to the address, then Permissions), then tap Retry.'
  if (e.name === 'NotFoundError' || e.name === 'OverconstrainedError') return 'No camera found on this device.'
  if (e.name === 'NotReadableError' || e.name === 'AbortError') return 'The camera is busy in another app or tab. Close it, then tap Retry.'
  return `Camera error: ${e.message}`
}
type Clip = { sign?: string; clip?: string; cutout?: string; cutout_box?: number[]; fingerspell?: string; dictionary?: boolean; picture?: string }
type Enroll = { label: string; n: number; reps: number } | null
const TEACH_REPS = 5
// when teaching ends, a leftover "Learned …" status goes back to the normal prompt (errors stay)
const afterTeaching = (st: string) => (st.startsWith('Learned') ? 'Ready — sign to the camera' : st)
// Quick setup: the words a hospital demo needs most. Teaching each 5x makes the app learn YOUR way of signing
// (guided one word at a time: the ghost shows it, then the person signs it, see Teach.tsx).
const QUICK_WORDS = ['Hello', 'Thank you', 'I', 'you', 'Pain', 'Head', 'Stomach', 'Fever', 'Doctor', 'Medicine',
  'Help', 'Water', 'Yes', 'No', 'Please', 'Today', 'Yesterday', 'Where', 'What', 'Emergency']
type Summary = {
  chief_complaint: string; symptoms: { symptom: string; since: string; severity: string }[]
  medicines_mentioned: string[]; allergies: string[]; to_confirm: string[]
  summary_english: string; summary_hindi: string; _error?: string
}
// HELP alone is everyday clinic talk ("can you help me?"), so it only counts together with PAIN / HURT
const SOS_SIGNS = new Set(['Emergency', 'Ambulance', 'Police'])
const SOS_PAIRS = [['Help', 'Pain'], ['Help', 'Hurt']]
const SOS_TEXT = {
  en: 'I am Deaf and I need help. Please call 112.',
  hi: 'मैं बधिर हूँ, मुझे मदद चाहिए। कृपया 112 पर कॉल करें।',
}
const SCENARIO_LIST: [string, string][] = [['hospital', '🏥 Hospital'], ['pharmacy', '💊 Pharmacy'], ['general', '💬 General']]

const pretty = (l: string) => l.replace(' large', '').replace(' little', '').replace('Store or Shop', 'Shop')

// Text-to-speech. Phones block speech on a page that has never been tapped (Chrome Android fails with
// "not-allowed"; iPhone needs the first speak() inside a tap), and a signer may never touch the screen.
// So the first tap anywhere unlocks speech, and a "tap to turn on the voice" bar shows until then.
const hasTTS = typeof window !== 'undefined' && 'speechSynthesis' in window
let voiceUnlocked = false
function unlockVoice() {
  if (voiceUnlocked || !hasTTS) return
  voiceUnlocked = true
  const u = new SpeechSynthesisUtterance(' ')
  u.volume = 0
  speechSynthesis.speak(u)
}
function pageWasTapped() {
  const ua = (navigator as Navigator & { userActivation?: { hasBeenActive: boolean } }).userActivation
  return voiceUnlocked || (ua ? ua.hasBeenActive : false)
}
function pickVoice(lang: 'en' | 'hi') {
  const vs = speechSynthesis.getVoices()
  const norm = (l: string) => l.replace('_', '-').toLowerCase()
  const want = lang === 'hi' ? 'hi' : 'en'
  return vs.find((v) => norm(v.lang) === `${want}-in` && v.localService) ?? vs.find((v) => norm(v.lang) === `${want}-in`)
    ?? vs.find((v) => norm(v.lang).startsWith(want) && v.default) ?? vs.find((v) => norm(v.lang).startsWith(want))
}
function speak(text: string, lang: 'en' | 'hi', onFail?: (error: string) => void) {
  if (!hasTTS) { onFail?.('this browser has no text-to-speech'); return }
  const u = new SpeechSynthesisUtterance(text)
  u.lang = lang === 'hi' ? 'hi-IN' : 'en-IN'
  const v = pickVoice(lang)
  if (v) u.voice = v
  u.onerror = (e) => { if (e.error !== 'interrupted' && e.error !== 'canceled') onFail?.(e.error) }
  // Chrome bug: speak() right after cancel() is sometimes dropped silently, so only cancel when needed
  const busy = speechSynthesis.speaking || speechSynthesis.pending
  if (busy) speechSynthesis.cancel()
  speechSynthesis.resume()  // Chrome Android can get stuck "paused" after the tab was in the background
  if (busy) setTimeout(() => speechSynthesis.speak(u), 120)
  else speechSynthesis.speak(u)
}

export default function App() {
  const video = useRef<HTMLVideoElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const ws = useRef<WebSocket | null>(null)
  const rec = useRef<WavRecorder | null>(null)
  const speechSock = useRef<WebSocket | null>(null)
  const [live, setLive] = useState('')
  const [source, setSource] = useState<'camera' | 'screen'>('camera')
  // camera picker: every camera on the device; the choice is remembered. Back cameras are not mirrored.
  const [cams, setCams] = useState<MediaDeviceInfo[]>([])
  const [camId, setCamId] = useState(() => localStorageGet('isl-cam') ?? '')
  const [activeCam, setActiveCam] = useState('')
  const [mirror, setMirror] = useState(true)
  const recent = useRef<string[]>([])  // last confident signs, for SOS pairs
  const framing = useRef<Framing>({ n: 0, far: 0, near: 0, oneHand: 0, hint: '', bad: 0, block: false })
  const [frameHint, setFrameHint] = useState('')
  const [frameBlock, setFrameBlock] = useState(false)  // camera too close: recognition paused, outline shown
  const canShare = typeof navigator.mediaDevices?.getDisplayMedia === 'function'
  const signingRef = useRef(false)
  const [status, setStatus] = useState('Starting camera…')
  const [signing, setSigning] = useState(false)
  const [pending, setPending] = useState<Cand[][]>([])
  const [interim, setInterim] = useState<string[]>([])  // live guesses while signing
  const [composing, setComposing] = useState(false)  // hands down 2 s, Muse Spark is writing the sentence
  // voice: null = fine; otherwise speech is blocked until a tap (the text is what still needs saying)
  const [voiceBlocked, setVoiceBlocked] = useState<{ text: string } | null>(() => (pageWasTapped() ? null : { text: '' }))
  useEffect(() => {
    const onTap = () => { unlockVoice(); setVoiceBlocked((b) => (b && !b.text ? null : b)) }
    window.addEventListener('pointerdown', onTap, true)
    return () => window.removeEventListener('pointerdown', onTap, true)
  }, [])
  const sayAloud = (text: string, lang: 'en' | 'hi') => speak(text, lang, (error) => {
    send({ type: 'client_log', msg: `speech failed: ${error} | ${navigator.userAgent}` })
    setVoiceBlocked({ text })
  })
  // guided teaching (Teach.tsx). The ref is the source of truth for the WebSocket and camera-loop callbacks.
  const [guide, setGuide] = useState<Guide | null>(null)
  const guideRef = useRef<Guide | null>(null)
  const putGuide = (g: Guide | null) => { guideRef.current = g; setGuide(g) }
  const downFrames = useRef(0)  // 'ready': camera frames in a row with the hands down
  const readyAt = useRef(0)  // when 'ready' began: hands that never drop (seated, forearms on a table) start after 2 s
  const armRef = useRef<() => void>(() => {})
  const [turns, setTurns] = useState<Turn[]>([])
  const [lang, setLang] = useState<'en' | 'hi'>('en')
  const [scenario, setScenario] = useState('hospital')
  const [user, setUser] = useState(initialUser)
  // first visit: welcome + how to test + "help improve" choice (keypoints only, never video). Until it is
  // answered nothing is logged.
  const [welcome, setWelcome] = useState(() => localStorageGet('isl-welcome') !== '1')
  const [logOk, setLogOk] = useState(() => localStorageGet('isl-log') !== '0')
  const logRef = useRef(logOk && !welcome)
  const [server, setServer] = useState<{ metaAi: boolean; version: string; public: boolean }>({ metaAi: true, version: '', public: false })
  const [camError, setCamError] = useState('')
  const [camTry, setCamTry] = useState(0)
  const [home] = useState(initialHome)
  const [lost, setLost] = useState(false)  // no server connection for 20 s
  const [fix, setFix] = useState<Fix>(null)
  const [noteOpen, setNoteOpen] = useState(false)
  const [note, setNote] = useState('')
  const [toast, setToast] = useState('')
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(''), 4500); return () => clearTimeout(t) } }, [toast])
  const [vocab, setVocab] = useState<string[]>([])
  const [enroll, setEnroll] = useState<Enroll>(null)
  const [teachOpen, setTeachOpen] = useState(false)
  const [teachLabel, setTeachLabel] = useState('')
  const [personal, setPersonal] = useState<Record<string, number>>({})
  const [sos, setSos] = useState<'offer' | 'alert' | null>(null)
  const [summary, setSummary] = useState<Summary | 'loading' | null>(null)
  const [typeOpen, setTypeOpen] = useState(false)
  const [typed, setTyped] = useState('')
  const [recording, setRecording] = useState(false)
  const [playlist, setPlaylist] = useState<Clip[]>([])
  const [clipIdx, setClipIdx] = useState(0)
  const [caption, setCaption] = useState('')
  // 👻 Ghost signer: replies are signed by a bright ghost figure instead of the video clips
  const [ghostOn, setGhostOn] = useState(() => localStorageGet('isl-ghost') === '1')
  const [ghost, setGhost] = useState<{ idx: number; sign: string; g: GhostRef | null } | null>(null)  // g null = no recording
  const recordMode = new URLSearchParams(location.search).has('record')
  const learnMode = new URLSearchParams(location.search).has('learn')
  const overlay = useRef<Overlay | null>(null)
  const [practice, setPractice] = useState<PracticeResult | null>(null)
  const [clips, setClips] = useState<string[]>([])
  useEffect(() => { fetch('/api/vocab').then((r) => r.json()).then((d) => setClips(d.clips)).catch(() => {}) }, [])
  const [setupLater, setSetupLater] = useState(() => localStorageGet('isl-setup-later') === '1')
  const [menuOpen, setMenuOpen] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)
  // the conversation scrolls inside its own panel (the camera stays put); keep the newest turn in view
  const chatRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const c = chatRef.current
    if (!c || !(turns.length || live)) return
    // a turn grows after it is added (sentence, sign chips, pictures), so scroll again once it has settled
    // scroll to the TOP of the newest bubble (clamped at the end), so on a small phone screen the sentence itself
    // shows first rather than the bottom of a tall bubble
    const go = () => {
      const bs = c.querySelectorAll('.bubble'), b = bs[bs.length - 1] as HTMLElement | undefined
      const top = b ? b.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop - 4 : c.scrollHeight
      c.scrollTo({ top, behavior: 'smooth' })
    }
    go()
    const t1 = setTimeout(go, 250), t2 = setTimeout(go, 1000)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [turns, live])
  const langRef = useRef(lang)
  langRef.current = lang

  const send = (m: object) => {
    const sock = ws.current
    if (sock?.readyState !== 1) return false
    sock.send(JSON.stringify(m))
    return true
  }
  const scenarioRef = useRef(scenario)
  scenarioRef.current = scenario
  // scenario change: tell the open connection (reconnecting would drop the conversation history)
  useEffect(() => { send({ type: 'scenario', scenario }) }, [scenario])

  // ---- WebSocket (reconnects by itself, e.g. if the server restarts during a demo) -------------
  useEffect(() => {
    let closed = false
    let retry: ReturnType<typeof setTimeout> | undefined
    let lostTimer: ReturnType<typeof setTimeout> | undefined
    const connect = () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const sock = new WebSocket(`${proto}://${location.host}/ws/sign`)
    ws.current = sock
    sock.onopen = () => {
      clearTimeout(lostTimer); lostTimer = undefined; setLost(false)
      sock.send(JSON.stringify({ type: 'hello', user, scenario: scenarioRef.current, mode: learnMode ? 'learn' : 'talk', log: logRef.current }))
      setPending([])
      setEnroll(null)
      // reconnected while teaching: the new session starts the current word again (reps so far were not saved)
      const g = guideRef.current
      if (g && TEACHING.includes(g.phase)) {
        sock.send(JSON.stringify({ type: 'enroll', label: g.words[g.i], reps: TEACH_REPS, restart: true }))
        downFrames.current = 0
        readyAt.current = performance.now()
        putGuide({ ...g, label: null, n: 0, phase: g.phase === 'watch' ? 'watch' : 'ready' })
      }
      setStatus((st) => (st === 'Disconnected from server' || st.startsWith('Reconnecting') ? 'Ready — sign to the camera' : st))
    }
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data)
      if (m.type === 'ready') {
        setVocab(m.vocab); setPersonal(m.personal)
        setServer({ metaAi: m.meta_ai !== false, version: m.version ?? '', public: !!m.public })
        // the public server gave this device another profile id (the requested one is reserved): keep it
        if (m.user && m.user !== user) { localStorageSet('isl-user', m.user); setUser(m.user) }
        if (!m.model && !recordMode) setStatus('Model not trained yet')
      }
      else if (m.type === 'deleted') {
        setToast(`Deleted everything saved for this profile (${m.files} files). Starting fresh…`)
        setTimeout(newProfile, 1800)
      }
      else if (m.type === 'feedback_ok') setToast(m.learned?.length ? `Thanks! Saved, and it learned your “${m.learned.map(pretty).join('”, “')}”.` : 'Thanks! Your report is saved.')
      else if (m.type === 'summary') setSummary(m)
      else if (m.type === 'practice_result') setPractice(m)
      else if (m.type === 'personal') { setPersonal(m.personal); setVocab(m.vocab) }
      else if (m.type === 'enroll_done') {
        setPersonal((p) => ({ ...p, [m.label]: m.kept }))
        // guided teaching: learned well -> a short ✓, then the next word; weak self-test -> teach again or keep
        const g = guideRef.current
        if (g && TEACHING.includes(g.phase) && m.label === g.label) {
          const ok = m.self_test >= m.kept - 1
          putGuide({ ...g, phase: ok ? 'done' : 'check', report: m, results: { ...g.results, [g.words[g.i]]: ok ? 'ok' : 'weak' } })
        }
      }
      else if (m.type === 'interim') { if (signingRef.current) setInterim(m.signs) }
      else if (m.type === 'composing') setComposing(true)
      else if (m.type === 'state') {
        setSigning(m.signing); signingRef.current = m.signing; if (!m.signing) setInterim([]); else setComposing(false)
        if (m.signing && guideRef.current?.phase === 'finished') { putGuide(null); setStatus(afterTeaching) }  // they started talking: show the conversation
      }
      else if (m.type === 'sign' && learnMode) { /* Learn page only reacts to practice results */ }
      else if (m.type === 'sign') {
        setInterim([])
        setPending((p) => [...p, m.top])
        const sure = m.top[0].p > 0.5 ? m.top[0].label : null
        recent.current = [...recent.current, ...(sure ? [sure] : [])].slice(-4)
        if (sure && (SOS_SIGNS.has(sure) || SOS_PAIRS.some(([a, b]) => recent.current.includes(a) && recent.current.includes(b))))
          setSos((s) => s ?? 'offer')
      }
      else if (m.type === 'sentence') {
        setComposing(false)
        setPending((p) => p.slice(m.n ?? p.length))  // signs made while Spark was writing stay pending
        const text = langRef.current === 'hi' && m.hindi ? m.hindi : m.english
        setTurns((t) => [...t, { who: 'deaf', text, sub: m.chosen_signs.map(pretty).join(' · '), ms: m._latency_ms, cands: m.cands, chosen: m.chosen_signs, sid: m.sid }])
        sayAloud(text, langRef.current)
      } else if (m.type === 'enroll_progress') {
        setEnroll(m.n >= m.reps ? null : m)
        if (m.n >= m.reps) { setVocab((v) => (v.includes(m.label) ? v : [...v, m.label])); setStatus(`Learned “${m.label}” ✓`) }
        const g = guideRef.current
        if (g && TEACHING.includes(g.phase)) {
          // n = 0 starts the word (and names it, e.g. "me" -> "I"); a late rep of a word left behind is ignored
          if (m.n === 0) putGuide({ ...g, label: m.label, n: 0 })
          else if (m.label === g.label && m.n > g.n) {
            try { navigator.vibrate?.(m.n >= m.reps ? [40, 60, 40] : 40) } catch { /* no vibration */ }
            putGuide({ ...g, n: m.n, phase: m.n >= m.reps ? 'saving' : g.phase })
          }
        }
      } else if (m.type === 'error') setStatus(m.message)
    }
    sock.onclose = () => {
      if (closed) return
      setStatus('Reconnecting to server…')
      setComposing(false)
      setSigning(false); signingRef.current = false
      if (!lostTimer) lostTimer = setTimeout(() => setLost(true), 20000)
      retry = setTimeout(connect, 2000)
    }
    }
    connect()
    return () => { closed = true; clearTimeout(retry); clearTimeout(lostTimer); ws.current?.close() }
  }, [user])

  // ---- Camera + MediaPipe loop ---------------------------------------------------------------
  // source 'screen' = a shared tab/window (e.g. an ISL video playing on YouTube) through the same live pipeline
  useEffect(() => {
    let stop = false
    let stream: MediaStream | null = null
    ;(async () => {
      stream = source === 'screen'
        ? await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 30 } }, audio: false })
        : await navigator.mediaDevices.getUserMedia({
          video: { ...(camId ? { deviceId: { exact: camId } } : { facingMode: 'user' }), width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        }).catch((e) => {
          if (!camId) throw e
          setCamId(''); localStorageSet('isl-cam', null)  // saved camera is gone (unplugged, other phone)
          throw new Error('saved camera not found, switching to the default one')
        })
      if (stop) { stream.getTracks().forEach((t) => t.stop()); return }
      setCamError('')
      const track = stream.getVideoTracks()[0]
      if (source === 'camera') {
        const st = track.getSettings()
        setActiveCam(st.deviceId ?? '')
        setMirror(st.facingMode !== 'environment')  // selfie view for front / laptop cameras only
        // labels are only filled in after camera permission, so list the cameras now
        navigator.mediaDevices.enumerateDevices().then((d) => setCams(d.filter((x) => x.kind === 'videoinput'))).catch(() => {})
      }
      track.onended = () => setSource('camera')  // user pressed "Stop sharing"
      const v = video.current!
      v.srcObject = stream
      await v.play()
      setStatus('Loading hand tracking…')
      await loadVision()
      setStatus(source === 'screen' ? 'Watching the shared screen…' : 'Ready — sign to the camera')
      let lastTime = -1
      const loop = () => {
        if (stop) return
        // only new camera frames: phones redraw at 60-120 Hz but the camera gives ~30 fps; re-running the
        // models on the same frame wastes the phone's compute (worse tracking) and skews the frame rate
        const fresh = v.currentTime !== lastTime
        lastTime = v.currentTime
        const r = fresh ? detect(v) : null
        if (r) framing.current = checkFraming(r.raw, v, framing.current, setFrameHint, setFrameBlock, source === 'camera')
        if (r) {
          // teaching: the person's turn starts once the hands are down, so a sign copied along with the ghost
          // never becomes a half rep
          const g = guideRef.current
          if (g?.phase === 'ready') {
            downFrames.current = handsUp(r.raw, v.videoWidth, v.videoHeight) ? 0 : downFrames.current + 1
            if (downFrames.current >= 3 || performance.now() - readyAt.current > 2000) armRef.current()
          }
          // too close to read the signs, or teaching but not the person's turn (the ghost is playing): send
          // "nobody there" so no guesses are made from fragments of a sign and copying along is not counted
          const mute = framing.current.block || (g !== null && g.phase !== 'sign' && g.phase !== 'finished')
          send(mute ? { type: 'frame', w: r.frame.w, h: r.frame.h, t: r.frame.t, pose: null, lh: null, rh: null } : { type: 'frame', ...r.frame })
          const c = canvas.current!
          if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight }
          const ctx = c.getContext('2d')!
          draw(ctx, r.raw, signingRef.current)
          overlay.current?.(ctx, r.raw)
        }
        requestAnimationFrame(loop)
      }
      loop()
    })().catch((e) => {
      if (source === 'screen') { setStatus(`Screen share error: ${e.message}`); setSource('camera'); return }
      const help = cameraHelp(e)
      setStatus(help)
      setCamError(help)
    })
    return () => { stop = true; stream?.getTracks().forEach((t) => t.stop()) }
  }, [source, camId, camTry])

  // ---- Hearing person: hold to talk -> clips --------------------------------------------------
  // Realtime: mic PCM streams to /ws/speech while held (live captions); falls back to uploading the WAV.
  const startTalk = async () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const sock = new WebSocket(`${proto}://${location.host}/ws/speech`)
    sock.binaryType = 'arraybuffer'
    const queue: ArrayBuffer[] = []
    sock.onopen = () => { sock.send(JSON.stringify({ type: 'start', scenario })); queue.splice(0).forEach((b) => sock.send(b)) }
    let sentEnd = false, done = false
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data)
      if (m.type === 'partial') setLive(m.text)
      else if (m.type === 'transcript') { setLive(''); showWords(m.text) }
      else if (m.type === 'result') { done = true; setLive(''); showReply(m); sock.close() }
      else if (m.type === 'error') {
        if (speechSock.current === sock) speechSock.current = null  // still talking: upload the WAV on release
        if (sentEnd) {
          done = true; setLive('')
          settleWaiting((t) => ({ ...t, waiting: false, failed: true }))
          setStatus(m.message === 'no speech heard' ? 'No speech heard — hold the mic and speak' : `Speech error: ${m.message}`)
        }
        sock.close()
      }
    }
    sock.onerror = () => { if (speechSock.current === sock) speechSock.current = null }
    sock.onclose = () => {
      if (sentEnd && !done) { setLive(''); settleWaiting((t) => ({ ...t, waiting: false, failed: true })); setStatus('Speech translation failed — try again or type ⌨️') }
    }
    ;(sock as WebSocket & { markEnd?: () => void }).markEnd = () => { sentEnd = true }
    speechSock.current = sock
    const r = new WavRecorder((pcm) => {
      if (sock.readyState === WebSocket.OPEN) sock.send(pcm)
      else if (sock.readyState === WebSocket.CONNECTING) queue.push(pcm)
    })
    rec.current = r
    talkStart.current = r.start().catch((e) => {
      rec.current = null; sock.close(); speechSock.current = null
      setStatus(`Microphone error: ${(e as Error).message}`)
      throw e
    })
    setRecording(true)
  }
  const talkStart = useRef<Promise<void> | null>(null)
  const stopTalk = async () => {
    if (!rec.current) return
    setRecording(false)
    try { await talkStart.current } catch { return }  // released before the mic finished starting
    if (!rec.current) return
    const wav = await rec.current.stop()
    rec.current = null
    const sock = speechSock.current
    speechSock.current = null
    if (sock && sock.readyState === WebSocket.OPEN) {
      setStatus('Translating to ISL…')
      ;(sock as WebSocket & { markEnd?: () => void }).markEnd?.()
      sock.send(JSON.stringify({ type: 'end' }))
      return
    }
    setLive('')
    if (wav.size <= 44 + 16000) { setStatus('Hold the mic button while you speak'); return }  // < 0.5 s of audio
    const fd = new FormData()  // realtime unavailable: upload the recording instead
    fd.append('audio', wav, 'speech.wav')
    fd.append('scenario', scenario)
    await toSign(fetch('/api/speech', { method: 'POST', body: fd }))
  }

  // hearing person's reply (spoken or typed) -> captions + ISL clips
  // captions first: the hearing person's words appear at once; showReply fills in the ISL signs (Spark, 4-10 s)
  const showWords = (text: string, typed = false) => setTurns((t) => [...t, { who: 'hearing', text, typed, waiting: true }])
  const settleWaiting = (f: (t: Turn) => Turn) => setTurns((ts) => {
    const i = ts.map((t) => t.who === 'hearing' && t.waiting).lastIndexOf(true)
    return i < 0 ? ts : ts.map((t, j) => (j === i ? f(t) : t))
  })
  const showReply = (j: { caption: string; playlist: Clip[]; latency_ms: { total: number } }, typed = false) => {
    const turn: Turn = { who: 'hearing', text: j.caption, sub: j.playlist.map((c: Clip) => c.sign ? pretty(c.sign) : `“${c.fingerspell}”`).join(' · '), ms: j.latency_ms.total, playlist: j.playlist, typed }
    setTurns((ts) => {
      const i = ts.map((t) => t.who === 'hearing' && t.waiting).lastIndexOf(true)
      return i < 0 ? [...ts, turn] : ts.map((t, k) => (k === i ? turn : t))
    })
    send({ type: 'reply', text: j.caption })
    setCaption(j.caption)
    setPlaylist(j.playlist)
    setClipIdx(0)
    setStatus('Ready — sign to the camera')
  }
  const toSign = async (req: Promise<Response>, typed = false) => {
    setStatus('Translating to ISL…')
    try {
      const r = await req
      if (!r.ok) {
        const t = await r.text()
        let msg = t
        try { msg = JSON.parse(t).error ?? t } catch { /* plain text */ }
        throw new Error(msg)
      }
      showReply(await r.json(), typed)
    } catch (e) {
      settleWaiting((t) => ({ ...t, waiting: false, failed: true }))
      setStatus(`Translation error: ${(e as Error).message.slice(0, 120)}`)
    }
  }
  const replyText = (text: string) =>
    fetch('/api/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, scenario }) })
  const retryReply = (i: number) => {  // a reply that could not be put into ISL: tap its bubble to try again
    const t = turns[i]
    setTurns((ts) => ts.map((x, j) => (j === i ? { ...x, waiting: true, failed: false } : x)))
    toSign(replyText(t.text), t.typed)
  }
  const sendTyped = () => {
    const text = typed.trim()
    if (!text) return
    setTyped('')
    setTypeOpen(false)
    showWords(text, true)
    toSign(replyText(text), true)
  }

  const captureFrame = () => {  // unmirrored camera frame for ISL Lens
    const v = video.current
    if (!v || !v.videoWidth) return null
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    return c.toDataURL('image/jpeg', 0.8)
  }

  const nextClip = useCallback(() => setClipIdx((i) => i + 1), [])
  const current = playlist[clipIdx]
  useEffect(() => {
    if (current && !current.clip) { const t = setTimeout(nextClip, 1400); return () => clearTimeout(t) }
  }, [current, nextClip])
  // ghost signer: preload every sign of the reply, then play each one; signs without a reference recording
  // (e.g. ISLRTC dictionary words) fall back to their video clip
  // warm the HTTP cache with every clip / picture of the reply, so each sign starts without a blank gap
  useEffect(() => {
    for (const c of playlist) for (const u of [c.cutout ?? c.clip, c.picture]) if (u) fetch(u).catch(() => {})
  }, [playlist])
  const ghostable = (c?: Clip) => !!(ghostOn && c?.sign && !c.dictionary && clips.includes(c.sign))
  useEffect(() => { if (ghostOn) playlist.forEach((c) => ghostable(c) && loadGhost(c.sign!)) }, [playlist, ghostOn])
  useEffect(() => {
    if (!ghostable(current)) return
    let cancel = false, t: ReturnType<typeof setTimeout> | undefined
    loadGhost(current!.sign!).then((g) => {
      if (cancel) return
      setGhost({ idx: clipIdx, sign: current!.sign!, g })
      if (g) t = setTimeout(nextClip, ghostMs(g))
    })
    return () => { cancel = true; clearTimeout(t) }
  }, [current, clipIdx, ghostOn])
  // decided from the current sign directly (not after an effect), so the video never flashes between signs
  const loaded = ghost && ghost.idx === clipIdx && ghost.sign === current?.sign ? ghost : undefined
  const ghostNow = ghostable(current) && loaded?.g !== null ? { g: loaded?.g ?? null } : null

  // ---- Guided teaching: one word at a time; the ghost shows it, then the person signs it TEACH_REPS times ----
  // The server counts reps from the moment a word opens, but the camera frames stay muted until it is the
  // person's turn (phase 'sign'). Skipping or stopping half-way: the server puts the sign's old samples back.
  const openWord = (g: Guide, i: number) => {
    downFrames.current = 0
    send({ type: 'enroll', label: g.words[i], reps: TEACH_REPS, restart: true })  // not connected: sent on reconnect
    putGuide({ ...g, i, phase: 'watch', label: null, n: 0, report: undefined, watched: 0 })
  }
  const startGuide = (words: string[]) => {
    const list = [...new Set(words.map((w) => w.trim()).filter(Boolean))]
    if (!list.length) return
    if (guideRef.current && TEACHING.includes(guideRef.current.phase)) send({ type: 'cancel_enroll' })
    setPending([])  // the server drops unspoken signs when teaching starts
    setTeachOpen(false)
    setTeachLabel('')
    stopPlayback()
    openWord({ words: list, i: 0, phase: 'watch', label: null, n: 0, results: {}, watched: 0 }, 0)
  }
  const leaveWord = () => {  // stopped half-way: the server restores what the profile had for the sign
    const g = guideRef.current
    if (g && TEACHING.includes(g.phase)) send({ type: 'cancel_enroll' })
  }
  const closeGuide = () => { putGuide(null); setEnroll(null); setStatus(afterTeaching) }
  const stopGuide = () => { leaveWord(); closeGuide() }
  const nextWord = () => {
    const g = guideRef.current
    if (!g) return
    leaveWord()
    if (g.i + 1 < g.words.length) { openWord(g, g.i + 1); return }
    if (g.words.length > 1) { putGuide({ ...g, phase: 'finished' }); return }
    closeGuide()
    const w = pretty(g.label ?? g.words[0])
    const r = g.results[g.words[0]]
    if (r === 'ok') setToast(`Learned “${w}” ✓ It now reads your way of signing it.`)
    else if (r === 'weak') setToast(`Saved “${w}”. If it still misreads it, teach it again from ✋ Teach.`)
  }
  const jumpTo = (i: number) => { const g = guideRef.current; if (g) { leaveWord(); openWord(g, i) } }
  const teachAgain = () => { const g = guideRef.current; if (g) openWord(g, g.i) }
  const showAgain = () => {
    const g = guideRef.current
    if (g && (g.phase === 'ready' || g.phase === 'sign')) putGuide({ ...g, phase: 'watch', watched: g.watched + 1 })
  }
  const toReady = () => {
    const g = guideRef.current
    if (g?.phase === 'watch') { downFrames.current = 0; readyAt.current = performance.now(); putGuide({ ...g, phase: 'ready' }) }
  }
  armRef.current = () => {  // hands are down: the person's turn, reps count from now
    const g = guideRef.current
    if (g?.phase === 'ready') putGuide({ ...g, phase: 'sign' })
  }
  // the ghost of the current word (undefined while loading, null = no recording: sign it your way)
  const guideWord = guide ? guide.words[guide.i] : ''
  const [demo, setDemo] = useState<{ word: string; g: GhostRef | null } | null>(null)
  useEffect(() => {
    if (!guideWord) return
    let live = true
    loadGhost(guideWord).then((g) => { if (live) setDemo({ word: guideWord, g }) })
    const g = guideRef.current
    const after = g?.words[g.i + 1]
    if (after) loadGhost(after)  // ready before it is needed
    return () => { live = false }
  }, [guideWord])
  const demoNow = demo && demo.word === guideWord ? demo.g : undefined
  // the ghost plays DEMO_PLAYS times, then it is the person's turn
  useEffect(() => {
    if (guide?.phase !== 'watch' || demoNow === undefined) return
    const t = setTimeout(toReady, demoNow ? DEMO_PLAYS * ghostMs(demoNow) : 0)
    return () => clearTimeout(t)
  }, [guide?.phase, guide?.i, guide?.watched, demoNow])
  // learned: a short ✓, then the next word (saving: the self-test normally comes back within milliseconds)
  useEffect(() => {
    if (guide?.phase !== 'done' && guide?.phase !== 'saving') return
    const t = setTimeout(nextWord, guide.phase === 'done' ? 1400 : 6000)
    return () => clearTimeout(t)
  }, [guide?.phase, guide?.i])

  const quickWords = QUICK_WORDS.filter((w) => vocab.includes(w))
  const quickLeft = quickWords.filter((w) => (personal[w] ?? 0) < 3)  // not taught yet (a correction adds 1)
  const quickLabel = quickLeft.length && quickLeft.length < quickWords.length
    ? `⚡ Continue quick setup (${quickLeft.length} words left)` : `⚡ Quick setup: the ${quickWords.length} key hospital words`
  const quickSetup = () => { if (quickWords.length) startGuide(quickLeft.length ? quickLeft : quickWords) }

  const shareLocation = () => {
    const open = (extra: string) => window.open(`https://wa.me/?text=${encodeURIComponent(SOS_TEXT.en + ' ' + SOS_TEXT.hi + extra)}`, '_blank')
    if (!navigator.geolocation) return open('')
    navigator.geolocation.getCurrentPosition(
      (p) => open(` My location: https://maps.google.com/?q=${p.coords.latitude},${p.coords.longitude}`),
      () => open(''), { timeout: 5000 })
  }

  const toggleGhost = () => { const v = !ghostOn; setGhostOn(v); localStorageSet('isl-ghost', v ? '1' : '0') }
  // "help improve": the server keeps this person's sign keypoints only while it is on
  const setConsent = (v: boolean) => {
    setLogOk(v); logRef.current = v; localStorageSet('isl-log', v ? '1' : '0')
    send({ type: 'consent', log: v })
  }
  const acceptWelcome = () => {
    unlockVoice()  // this tap also lets the phone speak later sentences
    setVoiceBlocked(null)
    setWelcome(false); localStorageSet('isl-welcome', '1')
    setConsent(logOk)
  }
  // "✏️ Wrong?" on a signed sentence: what the person really signed, sign by sign, plus an optional note
  const openFix = (i: number) => {
    const t = turns[i]
    const signs = t.chosen?.length === t.cands?.length ? t.chosen! : (t.cands ?? []).map((c) => c[0]?.label ?? '')
    setFix({ turn: i, sid: t.sid, signs, meant: signs.map(pretty), note: '' })
  }
  const sendFix = () => {
    if (!fix) return
    const byPretty = new Map(vocab.map((v) => [pretty(v).toLowerCase(), v]))
    const signs = fix.meant.map((m, i) => ({ i, meant: byPretty.get(m.trim().toLowerCase()) ?? m.trim() }))
      .filter((x, i) => x.meant && x.meant !== fix.signs[i])
    if (!signs.length && !fix.note.trim()) { setFix(null); return }
    if (!send({ type: 'feedback', sid: fix.sid, signs, note: fix.note.trim(), device: navigator.userAgent })) {
      setToast('Not connected to the server - try again in a moment'); return
    }
    setTurns((ts) => ts.map((t, i) => (i === fix.turn ? { ...t, reported: true } : t)))
    setFix(null)
  }
  const deleteMyData = () => {
    if (!confirm('Delete everything ISL Bridge saved for this profile: taught signs, sign keypoints and reports? This cannot be undone.')) return
    if (!send({ type: 'delete_me' })) setToast('Not connected to the server - try again in a moment')
  }
  const sendNote = () => {
    if (note.trim() && !send({ type: 'feedback', note: note.trim(), device: navigator.userAgent })) {
      setToast('Not connected to the server - try again in a moment'); return
    }
    setNote(''); setNoteOpen(false)
  }
  const talkMode = !learnMode && !recordMode
  const stopPlayback = () => { setPlaylist([]); setClipIdx(0) }
  const credit = (
    <p className="credit">Sign videos: ISL Dictionary, Indian Sign Language Research and Training Centre (ISLRTC), DEPwD, Ministry of Social Justice &amp; Empowerment · INCLUDE dataset (AI4Bharat), signers of St. Louis School for the Deaf, Adyar · CISLR (IIT Kanpur) · ISL-CSLTR (Elakkiya &amp; Natarajan) · Powered by Meta Muse Spark, Voice Transcribe &amp; SAM 3.1 · <button className="linkish" onClick={() => setAboutOpen(true)}>About &amp; privacy</button></p>
  )

  return (
    <div className={`app ${talkMode ? 'mode-talk' : 'mode-panel'}`}>
      <header className="top">
        <a className="brand" href="?" aria-label="ISL Bridge"><span className="logo" aria-hidden="true">🤟</span>ISL<span>Bridge</span></a>
        <div className="top-actions">
          <select className="wide-only" aria-label="Scenario" value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {SCENARIO_LIST.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <button className="pill" aria-label="Language of the spoken sentences" title="Spoken language" onClick={() => setLang(lang === 'en' ? 'hi' : 'en')}>{lang === 'en' ? 'EN' : 'हिं'}</button>
          {canShare && <button className={`pill wide-only ${source === 'screen' ? 'on' : ''}`} title="Test on a video: share a tab playing ISL (e.g. YouTube)"
            onClick={() => setSource(source === 'screen' ? 'camera' : 'screen')}>{source === 'screen' ? '📷 Camera' : '📺 Video'}</button>}
          {!learnMode && <button className={`pill wide-only ${ghostOn ? 'on' : ''}`} title="Ghost signer: replies are signed by a ghost figure instead of videos" onClick={toggleGhost}>👻</button>}
          <a className="pill wide-only" href={learnMode ? '?' : '?learn'}>{learnMode ? '💬 Talk' : '🎓 Learn'}</a>
          <button className="pill wide-only" title="Send feedback to the ISL Bridge team" onClick={() => setNoteOpen(true)}>💬 Feedback</button>
          <button className="pill wide-only" aria-label="About and privacy" onClick={() => setAboutOpen(true)}>ⓘ</button>
          <button className="pill narrow-only" aria-label="Menu" onClick={() => setMenuOpen(true)}>☰</button>
          <button className="pill sos-pill" onClick={() => setSos('offer')}>SOS</button>
        </div>
      </header>

      <section className={`stage ${signing ? 'live' : ''} ${source === 'camera' && cams.length > 1 ? 'has-cams' : ''}`}>
        <video ref={video} playsInline muted className={source === 'camera' ? (mirror ? 'mirror' : '') : 'contain'} />
        <canvas ref={canvas} className={`${source === 'camera' ? (mirror ? 'mirror' : '') : 'contain'} overlay`} />
        {source === 'camera' && cams.length > 1 && !current && !guide && (
          <select className="cam-select" aria-label="Choose camera" value={activeCam}
            onChange={(e) => { setCamId(e.target.value); localStorageSet('isl-cam', e.target.value) }}>
            {cams.map((c, i) => <option key={c.deviceId || i} value={c.deviceId}>📷 {camName(c, i)}</option>)}
          </select>
        )}
        {frameHint && !current && !frameBlock && !guide && <div className="frame-hint" role="status">{frameHint}</div>}
        {frameBlock && !current && (
          <div className="frame-gate" role="alert">
            <svg className="silhouette" viewBox="0 0 120 160" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
              <ellipse cx="60" cy="38" rx="13" ry="16" />
              <path d="M32 74 Q60 60 88 74 L96 112 M32 74 L24 112 M40 160 L42 96 M80 160 L78 96" />
              <path d="M24 112 Q22 92 34 84 M96 112 Q98 92 86 84" />
            </svg>
            <div className="gate-text">
              <b>📏 Move the phone back</b>
              <span>Your head, shoulders and both hands must fit in the picture, like the outline. Signs are not read until then.</span>
            </div>
          </div>
        )}
        {camError && !current && source === 'camera' && !(lost && home) && (
          <button className="cam-retry" onClick={() => { setCamError(''); setStatus('Starting camera…'); setCamTry((n) => n + 1) }}>↻ Retry camera</button>
        )}
        {/* the server is gone (its link may have been replaced): retrying here can't work, the fixed address can */}
        {lost && home && !current && (
          <button className="cam-retry" onClick={() => { location.href = home }}>Connection lost · ↻ Reopen ISL Bridge</button>
        )}
        {!current && !guide && <div className="status">{enroll ? `Teaching “${enroll.label}” — sign it (${enroll.n}/${enroll.reps})` : signing ? (learnMode ? '● Signing…' : '● Signing… lower your hands when you finish') : composing && !learnMode ? '✍️ Writing the sentence…' : status}</div>}
        {!current && !signing && !enroll && !guide && talkMode && source === 'camera' && <div className="privacy" title="Hand and body keypoints are computed on this device; only those numbers are sent">🔒 Video stays on this device</div>}
        {enroll && !guide && <div className="enroll-bar"><div style={{ width: `${(enroll.n / enroll.reps) * 100}%` }} /></div>}
        {enroll && !guide && clips.includes(enroll.label) && (
          <video className="teach-ref" key={enroll.label} src={`/clips/${encodeURIComponent(enroll.label)}.mp4`} autoPlay loop muted playsInline />
        )}
        {enroll && !guide && !recordMode && (
          <div className="teach-actions">
            <button className="ghost" onClick={() => { send({ type: 'cancel_enroll' }); setEnroll(null) }}>Stop</button>
          </div>
        )}
        {guide && !current && (
          <TeachStage guide={guide} demo={demoNow} mirror={mirror} signing={signing} reps={TEACH_REPS} hint={frameBlock ? '' : frameHint}
            blocked={frameBlock} pretty={pretty} onReady={toReady} onShowAgain={showAgain} />
        )}
        {interim.length > 0 && !enroll && !learnMode && !current && <div className="interim">{interim.map(pretty).join(' · ')}…</div>}
        {current && ghostNow && <Ghost key={clipIdx} g={ghostNow.g} label={pretty(current.sign ?? '')} caption={caption} />}
        {current && !ghostNow && (
          <div className={`ar ${current.cutout ? 'cut' : 'card'}`}>
            {current.cutout
              ? <Cutout key={clipIdx} src={current.cutout} box={current.cutout_box} onEnded={nextClip} />
              : current.clip
                ? <video key={clipIdx} className="ar-clip" src={current.clip} autoPlay muted playsInline onEnded={nextClip} />
                : <div key={clipIdx} className="ar-spell">{current.fingerspell}</div>}
            <div className="ar-label">{current.sign ? pretty(current.sign) : 'Fingerspelling'}{current.dictionary && <small>ISLRTC dictionary</small>}</div>
            {current.picture && <PictureCard key={current.picture} src={current.picture} />}
            <div className="ar-caption">{caption}</div>
          </div>
        )}
        {current && (
          <div className="ar-top">
            <div className="ar-dots" aria-label={`sign ${clipIdx + 1} of ${playlist.length}`}>
              {playlist.map((_, i) => <i key={i} className={i < clipIdx ? 'done' : i === clipIdx ? 'now' : ''} />)}
            </div>
            <button className="ar-stop" onClick={stopPlayback} aria-label="Stop signing">✕</button>
          </div>
        )}
      </section>

      <div className="side">
        {voiceBlocked && talkMode && (
          <button className="voice-bar" onClick={() => {
            unlockVoice()
            const text = voiceBlocked.text || (lang === 'hi' ? 'आवाज़ चालू है' : 'Voice is on')
            setVoiceBlocked(null)
            sayAloud(text, lang)
          }}>🔊 {voiceBlocked.text ? 'Tap to hear the sentence (the phone blocked the voice)' : 'Tap once to turn on the voice'}</button>
        )}
        {recordMode && <Record user={user} enroll={enroll} send={send} />}
        {learnMode && clips.length > 0 && <Learn clips={clips} result={practice} send={send} overlay={overlay} pretty={pretty} capture={captureFrame} />}

        {talkMode && guide && (
          <section className="chat" aria-label="Teaching">
            <TeachCard guide={guide} howto={demoNow?.howto} hasDemo={!!demoNow} reps={TEACH_REPS} pretty={pretty}
              onShowAgain={showAgain} onSkip={nextWord} onStop={stopGuide} onAgain={teachAgain} onNext={nextWord}
              onClose={closeGuide} onJump={jumpTo} />
            {credit}
          </section>
        )}

        {talkMode && !guide && pending.length > 0 && (
          <section className="pending" aria-label="Signs read so far">
            {pending.map((c, i) => (
              <div key={i} className="cand">
                <b>{pretty(c[0].label)}</b>
                <small>{c.slice(1).map((x) => (
                  <button key={x.label} className="alt" title="Tap if this is what you meant (the app learns it, privately)"
                    onClick={() => { send({ type: 'correct', index: i, label: x.label }); setPending((p) => p.map((q, j) => (j === i ? [{ label: x.label, p: 1 }] : q))) }}
                  >{pretty(x.label)}</button>
                ))}</small>
              </div>
            ))}
            <button className="ghost" aria-label="Remove the last sign" onClick={() => { send({ type: 'undo' }); setPending((p) => p.slice(0, -1)) }}>⌫</button>
            <button className="ghost" onClick={() => send({ type: 'speak_now' })}>Speak ▶</button>
          </section>
        )}

        {talkMode && !guide && (
          <section className="chat" ref={chatRef} aria-live="polite">
            {!setupLater && Object.keys(personal).length < 5 && quickWords.length > 0 && turns.length < 2 && !enroll && (
              <div className="setup-card">
                <b>✋ Teach ISL Bridge your signing first</b>
                <span>Everyone signs a little differently, and a new phone only knows other people's way. The {quickWords.length} key
                  hospital words come one at a time: a ghost shows each sign, then you sign it 5 times (about 8 minutes).
                  After that it reads <i>your</i> signs. Stays private in your profile.</span>
                <div className="row">
                  <button className="primary" onClick={quickSetup}>{quickLeft.length && quickLeft.length < quickWords.length ? `⚡ Continue (${quickLeft.length} words left)` : '⚡ Start quick setup'}</button>
                  <button className="ghost" onClick={() => { setSetupLater(true); localStorageSet('isl-setup-later', '1') }}>Later</button>
                </div>
              </div>
            )}
            {turns.length === 0 && !live && (
              <div className="onboard">
                <div><b>1</b><span>🤟 <strong>Sign</strong> to the camera, then lower your hands. The phone speaks your sentence.</span></div>
                <div><b>2</b><span>🎤 The hearing person <strong>holds the mic</strong> and talks. Their words come back as ISL signs.</span></div>
                <div><b>3</b><span>✋ <strong>Teach</strong> the app your own way of signing in 5 repetitions.</span></div>
              </div>
            )}
            {turns.map((t, i) => (
              <div key={i} className={`bubble ${t.who}`} title={t.who === 'hearing' ? 'Tap to sign it again' : 'Tap to speak it again'}
                onClick={() => { if (t.who === 'deaf') sayAloud(t.text, lang); else if (t.failed) retryReply(i); else if (t.playlist?.length) { setCaption(t.text); setPlaylist(t.playlist); setClipIdx(0) } }}>
                <div className="who">{t.who === 'deaf' ? '🤟 Signed' : t.typed ? '⌨️ Typed' : '🎤 Said'}</div>
                <div className="text">{t.text}</div>
                {t.who === 'deaf' && t.cands?.length
                  ? <SignTrail cands={t.cands} chosen={t.chosen ?? []} />
                  : t.who === 'hearing' && t.playlist?.length
                    ? <div className="trail">{t.playlist.map((c, j) => <span key={j} className={`tchip ${c.dictionary ? 'dict' : ''} ${c.sign ? '' : 'spell'}`}>{c.sign ? pretty(c.sign) : c.fingerspell}</span>)}</div>
                    : t.waiting ? <small className="waiting">⏳ Putting it into ISL signs…</small>
                      : t.failed ? <small className="failed">⚠ Couldn't show this in ISL signs. Tap to try again.</small>
                        : t.sub && <small>{t.sub}</small>}
                {t.ms ? <small className="ms">{t.who === 'deaf' ? 'Muse Spark' : 'to ISL'} · {(t.ms / 1000).toFixed(1)} s{t.who === 'hearing' && t.playlist?.length ? ' · tap to replay ▶' : ''}</small> : null}
                {t.who === 'deaf' && t.sid !== undefined && (t.reported
                  ? <small className="reported">✓ Reported, thanks</small>
                  : <button className="fix-btn" onClick={(e) => { e.stopPropagation(); openFix(i) }}>✏️ Wrong? Tell us what you signed</button>)}
              </div>
            ))}
            {live && <div className="bubble hearing live"><div className="who">🎤 Listening…</div><div className="text">{live}</div></div>}
            {credit}
          </section>
        )}
        {!talkMode && credit}

        {talkMode && !guide && (
          <footer className="controls">
            <button className="ctl" onClick={() => setTeachOpen(true)}><span aria-hidden="true">✋</span><small>Teach</small></button>
            {server.metaAi ? (
              <button
                className={`mic ${recording ? 'rec' : ''}`}
                onPointerDown={startTalk} onPointerUp={stopTalk} onPointerLeave={() => recording && stopTalk()}
                onContextMenu={(e) => e.preventDefault()}
              >{recording ? 'Listening… release to send' : '🎤 Hold to speak'}</button>
            ) : (
              <button className="mic" onClick={() => setTypeOpen(true)} title="The demo's Meta AI credit is used up, so voice is off">⌨️ Type the reply (voice is off)</button>
            )}
            <button className="ctl" aria-label="Type a reply" onClick={() => setTypeOpen(true)}><span aria-hidden="true">⌨️</span><small>Type</small></button>
            <button className="ctl" disabled={turns.length < 2} onClick={() => { setSummary('loading'); send({ type: 'summary' }) }}><span aria-hidden="true">📋</span><small>Summary</small></button>
          </footer>
        )}
      </div>

      {welcome && talkMode && (
        <div className="sheet welcome-sheet">
          <div className="sheet-body welcome">
            <h3>🤟 Welcome to ISL Bridge</h3>
            <p>A conversation app for a Deaf person who signs <b>Indian Sign Language</b> and a hearing person, on one phone. This is a test version: your feedback makes it better.</p>
            <ol className="steps">
              <li><b>Stand back.</b> Prop the phone up so your head, shoulders and both hands are in the picture, in good light.</li>
              <li><b>Sign, then lower your hands.</b> One word or a short sentence, e.g. <i>HELLO</i>, <i>DOCTOR</i>, <i>PAIN · HEAD</i>. The phone speaks it.</li>
              <li><b>Wrong?</b> Tap <b>✏️ Wrong?</b> under the sentence and fix the words. It learns your signing, and we see what failed.</li>
              <li><b>Reply:</b> the hearing person holds <b>🎤</b> and talks (English or Hindi), or types ⌨️. The words come back as ISL signs.</li>
              <li><b>✋ Teach</b> makes the app learn your own signing: a ghost shows each word, then you sign it 5 times. Don't know ISL? <b>🎓 Learn</b> has every sign to practise.</li>
            </ol>
            <label className="consent">
              <input type="checkbox" checked={logOk} onChange={(e) => setLogOk(e.target.checked)} />
              <span><b>Help improve ISL Bridge:</b> save the hand and body keypoints of my signs (numbers only, never video) and my reports. You can switch this off any time (ⓘ / ☰).</span>
            </label>
            <p className="fine">🔒 Your camera video never leaves this phone. Speech you record is sent to Meta AI to be transcribed. Translations can be wrong, so don't rely on it for anything urgent.</p>
            <button className="primary" onClick={acceptWelcome}>Start</button>
          </div>
        </div>
      )}
      {fix && (
        <div className="sheet" onClick={() => setFix(null)}>
          <div className="sheet-body" onClick={(e) => e.stopPropagation()}>
            <h3>✏️ What did you sign?</h3>
            <p>Change each word that was read wrong and leave the right ones. The app learns your corrections (privately), and the team sees what went wrong.</p>
            <div className="fix-signs">
              {fix.signs.map((sg, i) => (
                <label key={i}><small>Sign {i + 1}, read as <b>{pretty(sg)}</b></small>
                  <input list="isl-vocab" value={fix.meant[i]} onFocus={(e) => e.target.select()} aria-label={`What sign ${i + 1} really was`}
                    onChange={(e) => setFix({ ...fix, meant: fix.meant.map((x, j) => (j === i ? e.target.value : x)) })} />
                </label>
              ))}
            </div>
            <datalist id="isl-vocab">{vocab.map((v) => <option key={v} value={pretty(v)} />)}</datalist>
            <textarea placeholder="Anything else? e.g. “I signed 3 signs, it read 2”, “it was slow”" value={fix.note}
              onChange={(e) => setFix({ ...fix, note: e.target.value })} />
            <button onClick={sendFix}>Send</button>
            <button className="secondary" onClick={() => setFix(null)}>Cancel</button>
          </div>
        </div>
      )}
      {noteOpen && (
        <div className="sheet" onClick={() => setNoteOpen(false)}>
          <div className="sheet-body" onClick={(e) => e.stopPropagation()}>
            <h3>💬 Feedback</h3>
            <p>What was confusing, slow or wrong? What would make it useful? It goes to the ISL Bridge team, with the last few sentences of this conversation.</p>
            <textarea autoFocus value={note} onChange={(e) => setNote(e.target.value)} placeholder="Your feedback" />
            <button onClick={sendNote} disabled={!note.trim()}>Send</button>
          </div>
        </div>
      )}
      {toast && <div className="toast" role="status">{toast}</div>}
      {typeOpen && (
        <div className="sheet" onClick={() => setTypeOpen(false)}>
          <div className="sheet-body" onClick={(e) => e.stopPropagation()}>
            <h3>Type your reply</h3>
            <p>English or Hindi. It is shown to the Deaf person in ISL signs (app signs + the ISLRTC dictionary).</p>
            <input autoFocus placeholder="e.g. You need an X-ray before the operation" value={typed}
              onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && sendTyped()} />
            <button onClick={sendTyped}>Show in ISL</button>
          </div>
        </div>
      )}
      {menuOpen && (
        <div className="sheet" onClick={() => setMenuOpen(false)}>
          <div className="sheet-body menu" onClick={(e) => e.stopPropagation()}>
            <h3>Settings</h3>
            <h4>Where are you?</h4>
            <div className="seg">
              {SCENARIO_LIST.map(([v, l]) => <button key={v} className={scenario === v ? 'on' : ''} onClick={() => setScenario(v)}>{l}</button>)}
            </div>
            {!learnMode && (
              <button className="menu-row" onClick={toggleGhost}>
                <span>👻 Ghost signer<small>Replies are signed by a bright skeleton figure instead of videos</small></span>
                <span className={`toggle ${ghostOn ? 'on' : ''}`} aria-hidden="true" />
              </button>
            )}
            {canShare && (
              <button className="menu-row" onClick={() => { setSource(source === 'screen' ? 'camera' : 'screen'); setMenuOpen(false) }}>
                <span>{source === 'screen' ? '📷 Back to the camera' : '📺 Test on a video'}<small>Share a browser tab that plays ISL, e.g. YouTube</small></span>
              </button>
            )}
            <a className="menu-row" href={learnMode ? '?' : '?learn'}>
              <span>{learnMode ? '💬 Back to the conversation' : '🎓 Learn ISL'}<small>{learnMode ? '' : 'Ghost Coach: copy the signer, get stars and tips'}</small></span>
            </a>
            <button className="menu-row" onClick={() => { setMenuOpen(false); setWelcome(true) }}>
              <span>📝 How to test<small>What to try, and how to report mistakes</small></span>
            </button>
            <button className="menu-row" onClick={() => { setMenuOpen(false); setNoteOpen(true) }}>
              <span>💬 Send feedback<small>Anything that was confusing, slow or wrong</small></span>
            </button>
            <button className="menu-row" onClick={() => setConsent(!logOk)}>
              <span>🔬 Help improve ISL Bridge<small>Save the hand and body keypoints of my signs (numbers, never video)</small></span>
              <span className={`toggle ${logOk ? 'on' : ''}`} aria-hidden="true" />
            </button>
            <button className="menu-row" onClick={() => { setMenuOpen(false); setAboutOpen(true) }}>
              <span>ⓘ About, privacy &amp; credits</span>
            </button>
            <p className="profile-row">👤 Profile <code>{user}</code> · <button className="linkish" onClick={() => { if (confirm('Start with a new, empty profile on this device? Signs you taught stay with the old one.')) newProfile() }}>start fresh</button> · <button className="linkish" onClick={deleteMyData}>delete my data</button></p>
          </div>
        </div>
      )}
      {aboutOpen && (
        <div className="sheet" onClick={() => setAboutOpen(false)}>
          <div className="sheet-body about" onClick={(e) => e.stopPropagation()}>
            <h3>🤟 ISL Bridge</h3>
            <p>A two-way conversation between a Deaf person who signs Indian Sign Language and a hearing person, on one phone.</p>
            <ul>
              <li>🔒 <b>Your signing video never leaves this device.</b> Hand and body keypoints are computed on the phone; only those numbers go to the server.</li>
              <li>✋ Signs you teach are stored only in your private profile on the server and are never used to train the shared model.</li>
              <li>🔬 While <b>Help improve</b> is on, the keypoints of your signs and your “Wrong?” reports are saved, so the team can find mistakes and improve the recogniser. Speech you record is sent to Meta AI to be transcribed.</li>
              <li>⚠️ Machine translation can be wrong. For anything important, confirm with the Deaf person; the visit summary lists what to double-check.</li>
            </ul>
            <h4>Powered by Meta AI</h4>
            <ul>
              <li><b>Muse Spark</b>: turns the recogniser's top-3 guesses + the conversation into a natural English / Hindi sentence; rewrites speech into ISL sign order (with a dictionary search tool); visit summary; ISL Lens (vision); how-to steps for each sign.</li>
              <li><b>Muse Voice Transcribe</b>: the hearing person's speech, live (English, Hindi, mixed).</li>
              <li><b>SAM 3.1</b>: cuts the human signer out of each sign video so they appear in your camera view.</li>
              <li><b>PyTorch</b>: the sign recogniser (keypoint Transformer + personal prototypes).</li>
            </ul>
            <button className="menu-row" onClick={() => setConsent(!logOk)}>
              <span>🔬 Help improve ISL Bridge<small>Save the hand and body keypoints of my signs and my reports (numbers, never video), so the team can fix mistakes</small></span>
              <span className={`toggle ${logOk ? 'on' : ''}`} aria-hidden="true" />
            </button>
            <p className="profile-row">👤 Profile <code>{user}</code> · <button className="linkish" onClick={() => { if (confirm('Start with a new, empty profile on this device? Signs you taught stay with the old one.')) newProfile() }}>start fresh</button> · <button className="linkish" onClick={deleteMyData}>delete my data</button>
              {server.version && <> · build <code>{server.version}</code></>}</p>
            {credit}
            <div className="row">
              <button onClick={() => { setAboutOpen(false); setWelcome(true) }} className="secondary">📝 How to test</button>
              <button onClick={() => setAboutOpen(false)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {teachOpen && (
        <div className="sheet" onClick={() => setTeachOpen(false)}>
          <div className="sheet-body" onClick={(e) => e.stopPropagation()}>
            <h3>Teach the app your sign</h3>
            <p>A ghost shows the sign, then you sign it {TEACH_REPS} times, lowering your hands (or holding still) after each one.
              Works for your own way of signing a word, or brand-new signs like your name or your medicine (no ghost for those: sign it your way).</p>
            {quickWords.length > 0 && <button onClick={quickSetup}>{quickLabel}</button>}
            <p className="private">🔒 Saved only to your profile. It is never used to train the shared model.</p>
            {Object.keys(personal).length > 0 && (
              <div className="my-signs">
                <h4>My signs</h4>
                {Object.entries(personal).map(([l, n]) => (
                  <span key={l} className="my-sign">{pretty(l)} <small>{n}×</small>
                    <button className="ghost" title="Re-teach" onClick={() => startGuide([l])}>↻</button>
                    <button className="ghost" title="Delete" onClick={() => send({ type: 'forget', label: l })}>✕</button>
                  </span>
                ))}
              </div>
            )}
            <input placeholder="New sign, e.g. Insulin, Priya" value={teachLabel} onChange={(e) => setTeachLabel(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && startGuide([teachLabel])} />
            <button onClick={() => startGuide([teachLabel])}>Start</button>
            <div className="chips">{vocab.map((v) => <button key={v} onClick={() => startGuide([v])}>{pretty(v)}</button>)}</div>
          </div>
        </div>
      )}
      {summary && (
        <div className="sheet" onClick={() => setSummary(null)}>
          <div className="sheet-body summary" onClick={(e) => e.stopPropagation()}>
            <h3>Visit summary for the doctor</h3>
            {summary === 'loading' ? <p>Writing summary…</p> : (<>
              {summary.chief_complaint && <p><b>Main problem:</b> {summary.chief_complaint}</p>}
              {summary.symptoms.length > 0 && (
                <table><thead><tr><th>Symptom</th><th>Since</th><th>Severity</th></tr></thead>
                  <tbody>{summary.symptoms.map((x, i) => <tr key={i}><td>{x.symptom}</td><td>{x.since || '—'}</td><td>{x.severity || '—'}</td></tr>)}</tbody></table>
              )}
              {summary.medicines_mentioned.length > 0 && <p><b>Medicines:</b> {summary.medicines_mentioned.join(', ')}</p>}
              {summary.allergies.length > 0 && <p><b>Allergies:</b> {summary.allergies.join(', ')}</p>}
              <p className="pre">{lang === 'hi' && summary.summary_hindi ? summary.summary_hindi : summary.summary_english}</p>
              {summary.to_confirm.length > 0 && <p className="warn">Please confirm with the patient: {summary.to_confirm.join('; ')}</p>}
              <p className="fine">Auto-generated from a sign-language translation. Check with the patient before acting on it.</p>
              <button onClick={() => window.print()}>🖨 Print / save PDF</button>
            </>)}
          </div>
        </div>
      )}
      {sos === 'offer' && (
        <div className="sos-offer">
          <b>Need emergency help?</b>
          <button className="danger" onClick={() => { setSos('alert'); sayAloud(SOS_TEXT[lang], lang) }}>Show alert</button>
          <button className="danger" onClick={shareLocation}>Send location (WhatsApp)</button>
          <button className="ghost" aria-label="Close" onClick={() => setSos(null)}>✕</button>
        </div>
      )}
      {sos === 'alert' && (
        <div className="sos-alert" onClick={() => setSos(null)}>
          <div>{SOS_TEXT.en}</div>
          <div>{SOS_TEXT.hi}</div>
          <button onClick={(e) => { e.stopPropagation(); shareLocation() }}>📍 Send my location</button>
          <small>Tap anywhere to close</small>
        </div>
      )}
    </div>
  )
}

// Muse Image picture card for a word the Deaf person may not know as a sign (X-ray, inhaler...). New concepts are
// drawn on the server in the background (~8 s), so retry quietly until the picture exists.
function PictureCard({ src }: { src: string }) {
  const [n, setN] = useState(0)
  const [ok, setOk] = useState(false)
  if (n > 14) return null
  return <img key={n} className={`ar-pic ${ok ? 'on' : ''}`} src={n ? `${src}?r=${n}` : src} alt=""
    onLoad={() => setOk(true)} onError={() => { setTimeout(() => setN((x) => x + 1), 1500) }} />
}

// Which sign Muse Spark picked for each recognised sign; "✨" marks a sign it changed from the recogniser's
// top-1 to one of its alternatives because the conversation made that one more plausible.
function SignTrail({ cands, chosen }: { cands: Cand[][]; chosen: string[] }) {
  const aligned = chosen.length === cands.length
  if (!aligned) return <div className="trail">{chosen.map((c, i) => <span key={i} className="tchip">{pretty(c)}</span>)}</div>
  return (
    <div className="trail">
      {cands.map((c, i) => {
        const top = c[0]?.label ?? ''
        const fixed = chosen[i] !== top && c.some((x) => x.label === chosen[i])
        return (
          <span key={i} className={`tchip ${fixed ? 'fixed' : ''}`} title={c.map((x) => `${pretty(x.label)} ${Math.round(x.p * 100)}%`).join(' · ')}>
            {fixed && <s>{pretty(top)}</s>}{pretty(chosen[i])}{fixed && <i aria-label="changed by context"> ✨</i>}
          </span>
        )
      })}
    </div>
  )
}

// Framing guide. The model was trained on signers whose shoulders fill ~14-20% of the frame width; live tests
// showed phones held ~3x closer (shoulders ~50% of the frame), which loses the second hand and distorts the
// geometry by perspective. Hints are smoothed over ~1-2 s so they do not flicker.
// Gate (1 Nov): Anay's failed attempts had the shoulders at 70% of the frame width and on its bottom edge (only the
// face + one hand visible): the recogniser then guesses from fragments (cos 0.36-0.45 = noise). Past that point
// recognition is paused and an outline shows where to stand (hysteresis so it does not flicker).
type Framing = { n: number; far: number; near: number; oneHand: number; hint: string; bad: number; block: boolean }
function checkFraming(raw: Raw, v: HTMLVideoElement, f: Framing, setHint: (h: string) => void,
  setBlock: (b: boolean) => void, gate: boolean): Framing {
  const p = raw.pose
  let near = 0, far = 0, one = 0, bad = 0
  if (!p) far = 1
  else {
    const sw = Math.hypot((p[11].x - p[12].x) * v.videoWidth, (p[11].y - p[12].y) * v.videoHeight) / v.videoWidth
    const shoulderY = (p[11].y + p[12].y) / 2  // 0 = top, 1 = bottom of the frame
    near = sw > 0.34 ? 1 : 0
    far = sw < 0.1 ? 1 : 0
    bad = sw > 0.5 || shoulderY > 0.86 ? 1 : 0
    one = (raw.lh ? 1 : 0) + (raw.rh ? 1 : 0) === 1 ? 1 : 0
  }
  // running mean over the first 20 frames (so a phone held too close is caught within ~0.2 s, before any sign is
  // guessed), then ~1 s exponential smoothing at 30 fps
  const n = f.n + 1
  const sm = (prev: number, x: number) => (n <= 20 ? prev + (x - prev) / n : 0.95 * prev + 0.05 * x)
  const g: Framing = { n, near: sm(f.near, near), far: sm(f.far, far), oneHand: sm(f.oneHand, one), hint: f.hint,
    bad: sm(f.bad, bad), block: f.block }
  const block = gate && g.n >= 5 && (f.block ? g.bad > 0.3 : g.bad > 0.65)
  if (block !== f.block) { setBlock(block); g.block = block }
  const hint = g.n < 30 ? '' : g.near > 0.6 ? '📏 Move the phone back: head, shoulders and both hands should fit with space around them'
    : g.far > 0.6 ? '📏 Come a little closer to the camera' : ''
  if (hint !== f.hint) { setHint(hint); g.hint = hint }
  return g
}

function localStorageGet(k: string) {
  try { return localStorage.getItem(k) } catch { return null }
}
function localStorageSet(k: string, v: string | null) {
  try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v) } catch { /* private mode */ }
}
// short names for the picker. Android: "camera2 1, facing front" -> "Front"; iPhone: "Back Ultra Wide Camera" ->
// "Back Ultra Wide"; laptops: "HP True Vision FHD Camera (04f2:b6ab)" -> "HP True Vision FHD Camera"
function camName(c: MediaDeviceInfo, i: number) {
  const android = c.label.match(/^camera\d* (\d+), facing (front|back)/i)
  if (android) {
    const side = android[2].toLowerCase() === 'front' ? 'Front' : 'Back'
    return android[1] === '0' || android[1] === '1' ? side : `${side} ${android[1]}`
  }
  const name = c.label.replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)/i, '').replace(/^(Front|Back)(.*) Camera$/i, '$1$2').trim()
  return (name || `Camera ${i + 1}`).slice(0, 28)
}
