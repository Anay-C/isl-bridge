# 🤟 ISL Bridge

**A two-way conversation between a Deaf person who signs Indian Sign Language and a hearing person, on one phone.**
The Deaf person signs to the camera and the phone speaks a natural sentence in English or Hindi. The hearing person
holds the mic and talks, and their words come back as ISL: a real human signer cut out by SAM 3.1 stands in the
camera view, with captions and picture cards.

Built for the Meta Global AI Developer Hackathon 2026 (Indie track) by Anay Chauhan (VIT Pune).

> Pitch it honestly: ISL Bridge recognises **202 everyday and hospital signs**, reads whole signed sentences, and
> learns **your** way of signing a word from a few repetitions. It is not a general ISL translator (yet).

**Live demo: https://anaychauhan-isl-bridge.static.hf.space** (any phone or laptop with a camera). It is served
from the developer's laptop, so it can occasionally be offline; [Run it](#run-it) starts it on your own machine.

---

## Why

India has **more than 6 million ISL users** (Ethnologue, via [Careers360, 2022](https://news.careers360.com/indian-sign-language-md-university-mdu-rohtak-rci-diploma-courses-interpreter-jobs-aioat/amp)),
and Census 2011 counted **5.07 million people with a hearing disability**
([MOSPI, *Disabled Persons in India* 2016](https://mospi.gov.in/sites/default/files/publication_reports/Disabled_persons_in_India_2016.pdf)).
There are only a few hundred certified ISL interpreters. In a hospital, a Deaf patient usually
has to fall back on written notes, but many Deaf adults were schooled orally and read little Hindi or English. ISL
also varies by region and school, so a single fixed model will never understand everyone.

ISL Bridge is designed around those facts:
- **Two-way**: the doctor's reply comes back in ISL, not as text only.
- **Personal**: the app learns *your* variant of a sign from 5 repetitions, privately.
- **Context-aware**: it uses the conversation to choose between similar-looking signs.
- **Visual**: pictures for words without a familiar sign (X-ray, inhaler...) instead of English fingerspelling.

## What it does

| | Feature | How |
|---|---|---|
| 🤟→🔊 | **Sign → speech**: sign a sentence, lower your hands, and the phone speaks it (EN / हिं) | MediaPipe keypoints on the phone → PyTorch recogniser (top-3 per sign) → **Muse Spark** picks the sensible reading using the conversation + scenario |
| 🎤→🤟 | **Speech → sign**: the hearing person holds the mic | **Muse Voice Transcribe** (realtime, live captions, EN/HI code-switching) → **Muse Spark** rewrites into ISL order (time-topic-comment, question word last) and searches the ISLRTC dictionary with **tool calling** → playlist of sign clips |
| 🧍 | **AR interpreter**: the signer stands inside your camera view | **SAM 3.1** video segmentation cut the human signer out of every sign clip |
| 🖼️ | **Picture cards** for X-ray, injection, syrup, inhaler... | **Muse Image** draws a simple pictogram once per concept; Spark decides which words need one |
| ✋ | **Teach a sign** (5 reps), guided: the word appears big on screen, a ghost shows the sign, then you sign it. ⚡ Quick setup walks through 20 hospital words one at a time. Also brand-new signs (your name, your medicine) | Personal prototypes on the model's embedding sphere; private to your profile. Keypoints are not sent while the ghost plays, so copying along never counts |
| 👆 | **Tap to correct**: tap the right alternative and the app learns it | Adds a private sample |
| 📋 | **Visit summary** for the doctor in English + Devanagari Hindi, with "please confirm" items | Muse Spark (structured output) |
| 🆘 | **SOS**: EMERGENCY / AMBULANCE / POLICE, or HELP + PAIN → full-screen alert + WhatsApp location | Never sends anything by itself |
| 🎓 | **Learn ISL (Ghost Coach)**: a ghost skeleton of the signer is drawn on *your* body; stars + tips | Trajectory comparison + recogniser; how-to steps written by Muse Spark from each video |
| 🔍 | **ISL Lens**: show an object to the camera and get its sign | Muse Spark vision |
| 👻 | **Ghost signer**: replies signed by a skeleton figure instead of video | Body-space trajectories |

**Privacy:** the signing video never leaves the phone. Hand and body keypoints are computed on the device and only
those numbers are sent. Taught signs are stored per user and never used to train the shared model.

## Meta AI in ISL Bridge

| Meta model | Used for |
|---|---|
| **Muse Spark** (`muse-spark-1.3`, Responses API) | 1) sentence from top-3 sign candidates + conversation + scenario (structured output, speculative calls while you sign); 2) speech → ISL gloss with a `search_dictionary` **tool** over 2,311 ISLRTC dictionary signs; 3) picture decisions; 4) bilingual visit summary; 5) ISL Lens (image understanding); 6) how-to steps for 227 signs written from the videos |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Realtime streaming ASR over WebSocket while the mic is held, keyword biasing with the app's sign words, English/Hindi |
| **SAM 3.1** (`sam-3.1`, video) | Segmenting the signer in every sign clip ("person"), turned into a portable stacked colour+mask H.264 that the phone recombines into a transparent signer |
| **Muse Image** (`muse-image-1.0`) | Picture cards for words the Deaf person may not know as a sign |
| **PyTorch** | The sign recogniser: keypoint Transformer + cosine classifier (CosFace margin) trained on 5 ISL sources; personal prototypes live on the same unit sphere |

## Architecture

```
 Phone browser (React + TypeScript, PWA-style)                      FastAPI server (Python)
 ┌──────────────────────────────────────────────┐   keypoints only   ┌──────────────────────────────────────────┐
 │ camera → MediaPipe Tasks (hand + pose lite)  │ ──── WebSocket ──▶ │ segmenter (hands down = end of sentence)  │
 │ framing guide, live skeleton, AR overlay     │                    │ sign spotting: DP over windows → signs    │
 │ Web Speech API speaks the sentence           │ ◀── sentence ───── │ recogniser (PyTorch) top-3 + personal     │
 │                                              │                    │ Muse Spark: top-3 + history → sentence    │
 │ hold-to-talk mic (16 kHz PCM)                │ ──── WebSocket ──▶ │ Muse Voice Transcribe (realtime)          │
 │ interpreter: SAM cut-out / clip / picture    │ ◀── playlist ───── │ Muse Spark → ISL gloss (+ dictionary tool)│
 └──────────────────────────────────────────────┘                    │ Muse Image pictures, SAM 3.1 cut-outs     │
                                                                     └──────────────────────────────────────────┘
```

- **Features** (`ml/features.py`, the single spec shared by training and the live server): shoulder-centred,
  shoulder-width-scaled pose + both hands in body space and hand-local space, presence flags, 48 frames.
- **Model** (`ml/model.py`): 4-layer Transformer encoder → 256-d unit embedding → cosine classifier, CosFace margin.
- **Training data**: INCLUDE (AI4Bharat), CISLR (IIT Kanpur), ISLRTC dictionary, ISL-CSLTR (7 signers) and
  sign clips mined from CSLTR sentences: several communities, because variants matter more than model size.
- **Continuous signing** (`ml/spotting.py`): people do not drop their hands between signs, so the whole utterance
  is decoded with dynamic programming over multi-length windows (score = seconds × (cos − 0.55)), which beat
  greedy peak-picking by +23.5 points on unseen signers. A short stretch (or one that matches a single sign well
  as a whole) is preferred as one sign, so a fragment of a sign cannot win (BOOK used to be read as WHAT).
- **Segmentation** (`server/segmenter.py`): a stretch starts when a hand is raised relative to the body and keeps
  the arm rising and falling around it (the training clips include them; adding them back was worth +9 points),
  and a hand lost by the hand tracker mid-sign is followed through the pose wrist.
- **Personalisation** (`ml/recognizer.py`): score = blend of class weight and the user's nearest samples; a taught
  sign counts at full strength only when its samples are trusted (the model agrees, or another rep corroborates).
- **Latency**: live guesses every 0.5 s while signing; Muse Spark starts speculatively, so the sentence is
  usually spoken 0.5–1.5 s after the hands go down.

## Results (measured, held-out data)

| What | Result |
|---|---|
| Single signs, **signers never seen in training**, through the exact live pipeline (CSLTR signers 0 & 3; mean of 3 training seeds) | **69.6 % top-1, 80.4 % top-3** (n = 34 per seed) |
| Words found in continuous signed sentences, same unseen signers (mean of 3 seeds) | 58.8 % of vocabulary words (n = 76), precision 58 % |
| **Every sign tested**: 624 held-out INCLUDE + CISLR clips (178 signs), streamed through the live server path | **79.2 % top-1, 85.3 % top-3**; 109 of 178 signs at ≥ 90 %; the app reacts to 99.4 % of signs |
| Robustness: all 202 signs streamed over the real WebSocket by 8 clients at once, malformed messages mixed in | 0 dropped sessions, 0 server exceptions, 0 lost sentences |
| A signer from another region (ISLVT, Maharashtra, continuous sentences, never trained on) | 23.9 % of vocabulary words: regional variants are why personalisation is built in |
| **Personalisation**, 26 unseen signers, a camera setup the generic model cannot read (hands only) | app signs **3.8 % → 75.0 %** top-1 after ONE example per sign; brand-new signs 0 → 58.4 %; 26/26 signers improved |
| **Context** (937 signed turns, real recogniser noise, judged blind) | meaning preserved **64.0 % → 74.6 %** with Muse Spark + conversation (+10.6 pts, 95 % CI +7.5…+13.7); Spark *without* context: 50.4 % |
| Ghost Coach stars on held-out clips | correct sign → 3★ in 95 %; wrong sign → 1★ in 98 % |
| Latency | sign recognised 0.2–0.7 s after hands down; sentence spoken 0.5–3 s after; speech → ISL ~4 s after release |
| Cost | ~$0.001 per signed sentence; ~$11 of Meta API spend for the whole project including all SAM cut-outs |

Caveats we state up front: the unseen-signer sets are small; the context eval uses synthetic dialogues judged by
Muse Spark; ISL varies by region, so accuracy on a new person depends on how close their variant is to the
training communities. That is exactly why personalisation is built in.

## Try it / share it with testers

**Tester address: https://anaychauhan-isl-bridge.static.hf.space** (works on any phone, anywhere, while the
laptop is sharing). Double-click `deploy\share.bat` (or `.venv-ml/Scripts/python deploy/share.py`): it starts a
public-mode server and a Cloudflare quick tunnel, and keeps that fixed Hugging Face address pointing at the tunnel's
current link, also through Wi-Fi drops. Testers get a welcome screen with a consent choice, their own private profile,
**✏️ Wrong?** under every sentence, feedback and *delete my data*. `ml/tester_report.py` summarises what they found.
Full guide: [TESTING.md](TESTING.md). A hosted version (Hugging Face Space) is packaged in `deploy/build_space.py`.

## Run it

```bash
# Python 3.12 venv with torch, mediapipe, fastapi, uvicorn, openai, websockets (see ml/ and server/)
.venv-ml/Scripts/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
cd web && npm install && npx vite --host        # https://localhost:5173 (phone: https://<laptop-ip>:5173)
```
`.env` needs `MODEL_API_KEY` (Meta Model API). Optional: `ISL_DEVICE=cuda` (GPU inference),
`ISL_MODEL=a.pt,b.pt,c.pt` (ensemble). Pages: `/` conversation, `/?learn` Ghost Coach, `/?record` data collection.

End-to-end regression: `ml/test_features.py <url>` (25 checks over the real WebSocket + HTTP, incl. consent,
reports, limits). Evaluations: `ml/test_signs.py` (every sign, held-out clips, live server path, per-sign report),
`ml/test_ws.py` (every sign streamed over the real WebSocket, fuzzed), `ml/eval_live.py` (unseen signers, live
pipeline), `ml/eval_islvt.py` (signer from another region), `ml/eval_words.py`, `ml/eval_personal_emergency.py`,
`ml/eval_context.py`, `server/smoke_meta.py` (every Meta API call once).

## Data & credits

- **INCLUDE** dataset, AI4Bharat (CC BY 4.0), signers of St. Louis School for the Deaf, Adyar, Chennai
- **ISL Dictionary**, Indian Sign Language Research and Training Centre (ISLRTC), DEPwD, Ministry of Social Justice &
  Empowerment (research / ISL-technology use with acknowledgement, non-commercial)
- **CISLR**, IIT Kanpur (AFL-3.0) · **ISL-CSLTR**, Elakkiya & Natarajan (CC BY 4.0) · Mendeley ISL emergency words
  (CC BY 4.0, evaluation only)
- Meta **Muse Spark, Muse Voice Transcribe, SAM 3.1, Muse Image**, **PyTorch**; Google **MediaPipe** Tasks

## Limitations & next steps

- 202 signs; no facial grammar yet (ISL marks questions with the eyebrows), no fingerspelling recogniser.
- Accuracy for a new person depends on their regional variant: more Deaf signers in training is the main lever
  (the FDMSE-ISL dataset, 20 Deaf signers × 2,002 words, is the next step).
- Must be reviewed with Deaf users before any clinical use; it supports communication, it does not replace a
  qualified interpreter.
