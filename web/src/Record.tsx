// Record mode (?record): collect your own takes of each sign, with the INCLUDE reference video to copy.
// "Missing" signs (Pain, Help…) become training data; INCLUDE signs become the new-signer test set
// for the before/after personalisation metric. Saved server-side under data/recordings/<user>/<label>/.
import { useEffect, useState } from 'react'

type Props = {
  user: string
  enroll: { label: string; n: number; reps: number } | null
  send: (m: object) => void
}

export default function Record({ user, enroll, send }: Props) {
  const [lists, setLists] = useState<{ include_signs: string[]; cislr_signs: string[] } | null>(null)
  const [which, setWhich] = useState<'cislr_signs' | 'include_signs'>('cislr_signs')
  const [idx, setIdx] = useState(0)
  const [counts, setCounts] = useState<Record<string, number>>({})
  const [refI, setRefI] = useState(0)

  const refresh = () => fetch(`/api/recordings/${user}`).then((r) => r.json()).then(setCounts)
  useEffect(() => { fetch('/api/vocab').then((r) => r.json()).then(setLists); refresh() }, [])
  useEffect(() => { if (!enroll) refresh() }, [enroll])

  if (!lists) return null
  const words = lists[which]
  const word = words[idx % words.length]
  const reps = which === 'cislr_signs' ? 20 : 5
  const target = which === 'cislr_signs' ? 20 : 5
  const recording = enroll?.label === word

  return (
    <section className="record">
      <div className="record-tabs">
        <button className={which === 'cislr_signs' ? 'on' : ''} onClick={() => { setWhich('cislr_signs'); setIdx(0) }}>Extra signs (optional)</button>
        <button className={which === 'include_signs' ? 'on' : ''} onClick={() => { setWhich('include_signs'); setIdx(0) }}>INCLUDE signs (test, ×5)</button>
      </div>
      <div className="record-word">
        <button className="ghost" onClick={() => setIdx((i) => (i - 1 + words.length) % words.length)}>◀</button>
        <h2>{word} <small>{counts[word] ?? 0}/{target}</small></h2>
        <button className="ghost" onClick={() => setIdx((i) => (i + 1) % words.length)}>▶</button>
      </div>
      {which === 'include_signs' && (
        <video className="ref" key={word + refI} src={`/api/ref/${encodeURIComponent(word)}?i=${refI}`} autoPlay loop muted playsInline
          onClick={() => setRefI((i) => i + 1)} />
      )}
      {which === 'cislr_signs' && <p className="hint">No INCLUDE video — learn it from the ISLRTC dictionary, then sign it {reps}× with small natural variation.</p>}
      {recording
        ? <button className="danger" onClick={() => send({ type: 'cancel_enroll' })}>Recording {enroll!.n}/{enroll!.reps} — stop</button>
        : <button className="primary" onClick={() => send({ type: 'enroll', label: word, reps, save: true, personalise: false })}>● Record {reps} takes</button>}
      <div className="chips">
        {words.map((w, i) => (
          <button key={w} className={(counts[w] ?? 0) >= target ? 'done' : ''} onClick={() => setIdx(i)}>{w} {counts[w] ?? 0}</button>
        ))}
      </div>
    </section>
  )
}
