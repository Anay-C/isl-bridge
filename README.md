# 🤟 ISL Bridge

ISL Bridge lets a Deaf person who signs Indian Sign Language and a hearing person have a conversation on one phone,
both ways.

- The Deaf person signs to the camera, and the phone says the sentence out loud in English or Hindi.
- The hearing person holds the mic and talks, and the reply comes back in ISL: a real human signer (cut out of the
  sign video with SAM 3.1) shows up inside the camera view, with captions and picture cards.

I'm Anay Chauhan, a student at VIT Pune, and I built this for the Meta Global AI Developer Hackathon 2026 (Indie
track).

To be clear about what it can do: it recognises **202 everyday and hospital signs**, it can read a whole signed
sentence, and it learns *your* way of signing a word after a few repetitions. It is not a full ISL translator.

**Live demo: https://anaychauhan-isl-bridge.static.hf.space** (works on any phone or laptop with a camera). It runs
off my laptop, so it's sometimes offline. If it is, see [Run it](#run-it) to start it yourself.

---

## Why I built it

India has **more than 6 million ISL users** (Ethnologue, via [Careers360, 2022](https://news.careers360.com/indian-sign-language-md-university-mdu-rohtak-rci-diploma-courses-interpreter-jobs-aioat/amp)),
and Census 2011 counted **5.07 million people with a hearing disability**
([MOSPI, *Disabled Persons in India* 2016](https://mospi.gov.in/sites/default/files/publication_reports/Disabled_persons_in_India_2016.pdf)).
There are only a few hundred certified ISL interpreters. So at a hospital, a Deaf patient usually ends up writing
notes, but a lot of Deaf adults went to oral schools and can't read much Hindi or English. ISL also changes from
region to region and school to school, so one fixed model won't understand everyone.

That's why the app works the way it does:
- **Two-way**: the doctor's reply comes back in ISL, not just as text.
- **Personal**: you can teach it your version of a sign with 5 repetitions, and it stays private to you.
- **Uses context**: it looks at the conversation so far to pick between signs that look alike.
- **Pictures**: words without a common sign (X-ray, inhaler...) get a picture instead of English fingerspelling.

## What it does

| | Feature | How it works |
|---|---|---|
| 🤟→🔊 | **Sign → speech**: sign a sentence, drop your hands, and the phone says it (EN / हिं) | MediaPipe keypoints on the phone → my PyTorch recogniser (top-3 guesses per sign) → **Muse Spark** picks the reading that makes sense given the conversation and scenario |
| 🎤→🤟 | **Speech → sign**: the hearing person holds the mic | **Muse Voice Transcribe** (realtime, live captions, handles EN/HI mixing) → **Muse Spark** reorders it into ISL grammar (time-topic-comment, question word at the end) and looks up the ISLRTC dictionary through **tool calling** → a playlist of sign clips |
| 🧍 | **AR interpreter**: the signer stands inside your camera view | I used **SAM 3.1** video segmentation to cut the signer out of every sign clip |
| 🖼️ | **Picture cards** for X-ray, injection, syrup, inhaler... | **Muse Image** draws a simple pictogram once per word; Spark decides which words need one |
| ✋ | **Teach a sign** (5 reps): the word shows up big, a ghost shows the sign, then you sign it. ⚡ Quick setup goes through 20 hospital words one by one. You can also add new signs (your name, your medicine) | Personal prototypes on the model's embedding sphere, private to your profile. Nothing is sent while the ghost is playing, so copying along doesn't count as a rep |
| 👆 | **Tap to correct**: tap the right word and the app learns it | Adds a private sample |
| 📋 | **Visit summary** for the doctor in English + Hindi (Devanagari), with "please confirm" items | Muse Spark (structured output) |
| 🆘 | **SOS**: EMERGENCY / AMBULANCE / POLICE, or HELP + PAIN → full-screen alert + WhatsApp location | It never sends anything on its own |
| 🎓 | **Learn ISL (Ghost Coach)**: a ghost skeleton of the signer is drawn over *your* body, with stars and tips | Compares your movement with the signer's + the recogniser; the how-to steps were written by Muse Spark from each video |
| 🔍 | **ISL Lens**: show an object to the camera and get its sign | Muse Spark vision |
| 👻 | **Ghost signer**: replies signed by a skeleton figure instead of video | Body-space trajectories |

**Privacy:** the video never leaves the phone. The hand and body keypoints are worked out on the phone and only
those numbers get sent. Signs you teach are saved under your own profile and are never used to train the shared
model.

## How I used Meta AI

| Meta model | What I used it for |
|---|---|
| **Muse Spark** (`muse-spark-1.3`, Responses API) | 1) turning the top-3 sign guesses + conversation + scenario into a sentence (structured output, started early while you're still signing); 2) speech → ISL gloss, with a `search_dictionary` **tool** over 2,311 ISLRTC dictionary signs; 3) deciding which words get pictures; 4) the bilingual visit summary; 5) ISL Lens (image understanding); 6) how-to steps for 227 signs, written from the videos |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Realtime speech-to-text over WebSocket while the mic is held, biased towards the app's sign words, English/Hindi |
| **SAM 3.1** (`sam-3.1`, video) | Cutting out the signer ("person") in every sign clip. I pack colour + mask into one H.264 video and the phone puts them back together into a transparent signer |
| **Muse Image** (`muse-image-1.0`) | Picture cards for words the Deaf person may not know a sign for |
| **PyTorch** | The sign recogniser: keypoint Transformer + cosine classifier (CosFace margin), trained on 5 ISL sources. The personal prototypes live on the same unit sphere |

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

- **Features** (`ml/features.py`): one feature spec that both training and the live server use. Pose + both
  hands, centred on the shoulders and scaled by shoulder width, in body space and hand-local space, with presence
  flags, 48 frames.
- **Model** (`ml/model.py`): 4-layer Transformer encoder → 256-d unit embedding → cosine classifier with a CosFace
  margin.
- **Training data**: INCLUDE (AI4Bharat), CISLR (IIT Kanpur), the ISLRTC dictionary, ISL-CSLTR (7 signers) and sign
  clips I cut out of CSLTR sentences. I went for many different communities because sign variants mattered more
  than model size.
- **Continuous signing** (`ml/spotting.py`): people don't drop their hands between signs, so I decode the whole
  utterance with dynamic programming over windows of different lengths (score = seconds × (cos − 0.55)). This beat
  greedy peak-picking by 23.5 points on unseen signers. Short stretches, or ones that match a single sign well as a
  whole, are kept as one sign so a piece of a sign can't win (BOOK used to come out as WHAT).
- **Segmentation** (`server/segmenter.py`): a stretch starts when a hand goes up relative to the body, and it keeps
  the arm going up and coming down around the sign, because the training clips include that (adding it back was
  worth 9 points). If the hand tracker loses a hand mid-sign, it follows the pose wrist instead.
- **Personalisation** (`ml/recognizer.py`): the score is a blend of the class weight and your nearest samples. A
  taught sign only counts fully when its samples look trustworthy (the model agrees, or another rep backs it up).
- **Latency**: it makes live guesses every 0.5 s while you sign, and Muse Spark starts early, so the sentence is
  usually spoken 0.5–1.5 s after your hands go down.

## Results (measured on held-out data)

| What I tested | Result |
|---|---|
| Single signs from **signers the model never saw**, through the same pipeline as the live app (CSLTR signers 0 & 3; average of 3 training seeds) | **69.6 % top-1, 80.4 % top-3** (n = 34 per seed) |
| Vocabulary words found in continuous signed sentences, same unseen signers (average of 3 seeds) | 58.8 % of words (n = 76), precision 58 % |
| **Every sign**: 624 held-out INCLUDE + CISLR clips (178 signs) streamed through the live server | **79.2 % top-1, 85.3 % top-3**; 109 of 178 signs at ≥ 90 %; the app responds to 99.4 % of signs |
| Load test: all 202 signs over the real WebSocket, 8 clients at once, with broken messages mixed in | 0 dropped sessions, 0 server errors, 0 lost sentences |
| A signer from another region (ISLVT, Maharashtra, continuous sentences, not in training) | 23.9 % of words. Regional variants are the reason I built personalisation |
| **Personalisation**: 26 unseen signers, filmed in a setup the base model can't read (hands only) | **3.8 % → 75.0 %** top-1 after ONE example per sign; brand-new signs 0 → 58.4 %; all 26 signers improved |
| **Context** (937 signed turns with real recogniser mistakes, judged blind) | meaning kept **64.0 % → 74.6 %** with Muse Spark + the conversation (+10.6 points, 95 % CI +7.5…+13.7). Spark *without* the conversation: 50.4 % |
| Ghost Coach stars on held-out clips | correct sign → 3★ 95 % of the time; wrong sign → 1★ 98 % of the time |
| Latency | sign recognised 0.2–0.7 s after hands down; sentence spoken 0.5–3 s after; speech → ISL about 4 s after letting go of the mic |
| Cost | about $0.001 per signed sentence; about $11 of Meta API credit for the whole project, all SAM cut-outs included |

Things to keep in mind: the unseen-signer test sets are small, the context test uses made-up dialogues judged by
Muse Spark, and since ISL differs by region, how well it works for a new person depends on how close their signing
is to the people in the training data. That's the whole reason personalisation is there.

## Try it / share it with testers

**Tester link: https://anaychauhan-isl-bridge.static.hf.space** (any phone, anywhere, while my laptop is sharing).
To share it I double-click `deploy\share.bat` (or run `.venv-ml/Scripts/python deploy/share.py`). It starts a
public-mode server and a Cloudflare quick tunnel, and keeps the fixed Hugging Face address pointing at the tunnel's
current link, even when the Wi-Fi drops. Testers get a welcome screen where they choose whether to share data, their
own private profile, a **✏️ Wrong?** button under every sentence, a feedback form and *delete my data*.
`ml/tester_report.py` sums up what they ran into. More in [TESTING.md](TESTING.md). There's also a hosted version
(Hugging Face Space) packaged in `deploy/build_space.py`.

## Run it

```bash
# Python 3.12 venv with torch, mediapipe, fastapi, uvicorn, openai, websockets (see ml/ and server/)
.venv-ml/Scripts/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
cd web && npm install && npx vite --host        # https://localhost:5173 (phone: https://<laptop-ip>:5173)
```
You need a `.env` with `MODEL_API_KEY` (Meta Model API). Optional: `ISL_DEVICE=cuda` (GPU inference),
`ISL_MODEL=a.pt,b.pt,c.pt` (ensemble). Pages: `/` conversation, `/?learn` Ghost Coach, `/?record` data collection.

Tests: `ml/test_features.py <url>` runs 25 end-to-end checks over the real WebSocket + HTTP (consent, reports,
limits and so on). Evaluations: `ml/test_signs.py` (every sign, held-out clips, live server path, per-sign report),
`ml/test_ws.py` (every sign over the real WebSocket, with fuzzing), `ml/eval_live.py` (unseen signers, live
pipeline), `ml/eval_islvt.py` (signer from another region), `ml/eval_words.py`, `ml/eval_personal_emergency.py`,
`ml/eval_context.py`, `server/smoke_meta.py` (calls every Meta API once).

## Data & credits

- **INCLUDE** dataset, AI4Bharat (CC BY 4.0), signers of St. Louis School for the Deaf, Adyar, Chennai
- **ISL Dictionary**, Indian Sign Language Research and Training Centre (ISLRTC), DEPwD, Ministry of Social Justice &
  Empowerment (research / ISL-technology use with acknowledgement, non-commercial)
- **CISLR**, IIT Kanpur (AFL-3.0) · **ISL-CSLTR**, Elakkiya & Natarajan (CC BY 4.0) · Mendeley ISL emergency words
  (CC BY 4.0, evaluation only)
- Meta **Muse Spark, Muse Voice Transcribe, SAM 3.1, Muse Image**, **PyTorch**; Google **MediaPipe** Tasks

## Limitations & what's next

- 202 signs. No facial grammar yet (ISL marks questions with the eyebrows) and no fingerspelling recogniser.
- How well it works for a new person depends on their regional variant. More Deaf signers in the training data is
  the biggest fix; the FDMSE-ISL dataset (20 Deaf signers × 2,002 words) is next.
- It has to be checked with Deaf users before anyone uses it in a clinic. It's meant to help communication, not to
  replace a qualified interpreter.
