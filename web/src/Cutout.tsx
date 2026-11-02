// AR interpreter: plays a SAM 3.1 "stacked" clip (colour on top, mask below, plain H.264 so every phone can
// decode it) and recombines the halves into a transparent signer on a canvas, drawn over the live camera.
import { useEffect, useRef } from 'react'

// box: the signer's bounding box (0..1 of the colour half, ml/cutout_boxes.py): only that region is drawn, so the
// signer fills the overlay instead of standing small in the middle of the original video frame.
export default function Cutout({ src, onEnded, box }: { src: string; onEnded: () => void; box?: number[] }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const ended = useRef(onEnded)
  ended.current = onEnded

  useEffect(() => {
    const v = document.createElement('video')
    Object.assign(v, { src, muted: true, playsInline: true, autoplay: true })
    v.setAttribute('playsinline', '')
    // iOS only decodes videos that are in the document; keep it in but invisible
    Object.assign(v.style, { position: 'fixed', width: '1px', height: '1px', opacity: '0', pointerEvents: 'none' })
    document.body.appendChild(v)
    const off = document.createElement('canvas')
    const octx = off.getContext('2d', { willReadFrequently: true })!
    let raf = 0
    const draw = () => {
      const c = canvas.current
      if (c && v.readyState >= 2 && v.videoWidth) {
        const w = v.videoWidth, h = v.videoHeight / 2
        const [bx0, by0, bx1, by1] = box && box.length === 4 ? box : [0, 0, 1, 1]
        const x = Math.round(bx0 * w), y = Math.round(by0 * h)
        const cw = Math.max(1, Math.round((bx1 - bx0) * w)), ch = Math.max(1, Math.round((by1 - by0) * h))
        if (off.width !== w) { off.width = w; off.height = h * 2 }
        if (c.width !== cw || c.height !== ch) { c.width = cw; c.height = ch }
        octx.drawImage(v, 0, 0)
        const top = octx.getImageData(x, y, cw, ch)
        const mask = octx.getImageData(x, h + y, cw, ch).data
        for (let i = 3; i < top.data.length; i += 4) top.data[i] = mask[i - 3] // mask luminance -> alpha
        c.getContext('2d')!.putImageData(top, 0, 0)
      }
      raf = requestAnimationFrame(draw)
    }
    v.onended = () => ended.current()
    v.play().catch(() => {})
    draw()
    return () => { cancelAnimationFrame(raf); v.pause(); v.removeAttribute('src'); v.load(); v.remove() }
  }, [src, box?.join(',')])

  return <canvas ref={canvas} className="cutout" />
}
