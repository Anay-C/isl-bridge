# ISL Conversation App — Meta Global AI Developer Hackathon (Indie track)

> **Current state, how to run, results and known traps: read `HANDOFF.md` first.**

## Goal
Win the Indie track ($150k first prize). Build a two-way Indian Sign Language (ISL) conversation app:
- **Sign → speech:** deaf user signs to the phone camera; app speaks natural Hindi/English.
- **Speech → sign:** hearing person speaks; app shows captions + real ISL signer clips.

**Hard deadline:** project submission Thu 5 Nov 2026, 7:30 AM IST (Nov 4, 9:00 PM ET). Target: submit Tue 3 Nov night.
**Judging:** Impact 30%, Use of Meta AI tooling 25%, Demo (working + video) 25%, Technical depth 20%.
Owner: Anay (student, VIT Pune). Machine: Windows, RTX 5050 8 GB laptop GPU, webcam. Python + PyCharm.

## The three differentiators (never cut these)
1. **Personalisation in ~5 repetitions:** encoder produces an embedding per sign; user signs a word 3–5 times → stored
   personal prototype; prediction blends classifier logits with cosine similarity to prototypes. Users can also add
   brand-new signs (name sign, medicine). Measure accuracy on a new signer **before vs after** personalisation.
2. **Two-way:** speech → Muse Voice Transcribe → Muse Spark rewrites into ISL gloss order → play INCLUDE signer clips
   (background removed with SAM 3.1 video) with captions.
3. **Context-aware sentence building:** recogniser sends top-3 candidates per sign + conversation history + scenario
   to Muse Spark, which picks the sensible sequence and outputs fluent Hindi/English.

Stretch (only if Days 1–3 succeed): eyebrow-raise question detection (face landmarks), SOS via WhatsApp on HELP/PAIN/POLICE, fingerspelling.
Cut order if behind: stretch goals → SAM cutouts (play plain clips). Never cut personalisation or two-way.

## Architecture
```
Phone browser (TypeScript, React+Vite, PWA)
  camera → MediaPipe Tasks (Hand Landmarker + Pose Landmarker, VIDEO mode) → keypoints only
        → WebSocket → FastAPI server (Python)
              recogniser (PyTorch) → top-3 signs → Muse Spark → sentence → back to phone
  phone speaks sentence (Web Speech API speechSynthesis; Meta has no TTS)
  hearing person's mic → server → Muse Voice Transcribe → Muse Spark (ISL gloss order) → clip playlist → phone
```
Folders: `ml/` (download, extract, train, personalise, export), `server/` (FastAPI), `web/` (React), `data/` (gitignored).

## Critical consistency rules (accuracy collapses if broken)
- **Same keypoint models in training and live app:** MediaPipe Tasks `hand_landmarker.task` + `pose_landmarker_lite.task`
  (or full — pick one and use it everywhere). Do NOT train on legacy MediaPipe Holistic if the browser uses Tasks.
  Model URLs:
  - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  - https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
- **Feed unmirrored frames** to MediaPipe in both pipelines (mirror only the on-screen preview with CSS).
- **Assign left/right hand by distance to pose wrists** (pose landmarks 15 = left wrist, 16 = right wrist), not by
  MediaPipe's handedness label (that label assumes mirrored selfie input).
- VIDEO mode needs strictly increasing timestamps per landmarker instance; keep a running offset across videos.
- Same normalisation in Python and TS: centre on shoulder midpoint, scale by shoulder width; resample each sign to a
  fixed length (e.g. 48 frames); store missing hands as zeros + presence flag.
- Windows: multiprocessing uses spawn → create landmarkers inside the worker initializer, guard with `if __name__ == "__main__":`.

## Data: INCLUDE (AI4Bharat)
- Zenodo record 4010759 (https://zenodo.org/records/4010759), API: https://zenodo.org/api/records/4010759 (lists files, sizes, md5).
- 263 word signs, ~4,292 videos, 15 category zips (multi-part, e.g. `Adjectives_1of8.zip`), 56.8 GB total. CC BY 4.0 —
  credit AI4Bharat and the St. Louis School for the Deaf (Adyar, Chennai) signers in app + submission.
- Categories: Adjectives, Animals, Clothes, Colours, Days_and_Time, Electronics, Greetings, Home, Jobs,
  Means_of_Transportation, People, Places, Pronouns, Seasons, Society. Download only what the demo needs first
  (suggested: Greetings, Pronouns, People, Places, Jobs, Days_and_Time).
- Label = video's parent folder name with leading "N. " stripped. Official train/test split is a separate Google Drive
  link (see Zenodo/GitHub README https://github.com/AI4Bharat/INCLUDE); if unavailable, stratified split by video.
- Reference accuracy (OpenHands, arXiv 2110.05877, MediaPipe Holistic keypoints): LSTM 83.0%, Transformer 90.4%,
  ST-GCN 91.2%, SL-GCN 93.5% on INCLUDE; 8–53 ms per clip on CPU. Expect lower on a new signer → personalisation.

## Demo scenario & vocabulary
Pick ONE scenario (pharmacy or hospital visit), 30–50 signs. Prefer INCLUDE words; record missing ones (pain, medicine…)
~20 samples each. Learn signs from ISLRTC dictionary videos; follow INCLUDE's variants. Get a Deaf ISL user to review and
ideally sign in the demo video. Pitch honestly: "recognises N everyday signs at X% accuracy", not "translates ISL".

## Meta Model API (base https://api.meta.ai/v1, Bearer MODEL_API_KEY, OpenAI-SDK compatible)
- **Muse Spark:** `muse-spark-1.3` (Standard mode; do NOT use Contributor mode for user data). Responses API `POST /v1/responses`,
  tool calling, structured output, image/video understanding. ~$1.25 / M input tokens, $4.25 / M output.
- **Muse Voice Transcribe:** `muse-voice-transcribe-1.0`; realtime `wss://api.meta.ai/v1/asr/realtime`, file `POST /v1/asr/transcribe`.
  Code-switching, speaker diarization, keyword biasing, 25 evaluated languages — **verify Hindi/Marathi support on the models page first.** $0.18/hour.
- **SAM 3.1:** `sam-3.1` via `POST /v1/responses`. Content parts: `input_text` (one short noun phrase, e.g. "person") +
  `input_image` (`image_url`, URL or base64 data URL) or `input_video` (`video_url` or `file_id` from `POST /v1/files`, purpose `user_data`).
  `metadata`: `include_confidence: "true"`, `mask_encoding: "one_bit"`, `propagation_direction`. Video: MP4/MOV H.264/H.265,
  ≤15,000 frames, tracks ≤16 objects/frame, billed per frame (~$0.20 / 1,000 frames, min 50 frames). Output is special-token
  text (`<0f>0<|box;...|><|mask;...|>`); parse with `pip install meta-sam-parser` (≥0.0.6 for confidence).
- Docs: https://dev.meta.ai/docs/overview · SAM: https://dev.meta.ai/docs/sam/segmenting · Pricing: https://dev.meta.ai/docs/pricing-rate-limits
- $150 hackathon credit; set a spend limit on the Billing page; credits show under Billing → Free credits.

## 5-day plan
| Day | Work |
|---|---|
| Fri 30 Oct | Download chosen INCLUDE categories, extract keypoints (multiprocess), list all signs, choose vocab. **Go/no-go tonight:** quick model on 20 signs + webcam test on Anay; ≥~80% (after adding his own samples) → build. |
| Sat 31 Oct | Train encoder+classifier (small Transformer/GRU on normalised keypoints, augment: scale/rotate/shift, time-stretch, frame drop); prototypes + personalisation; before/after metric. Export model. |
| Sun 1 Nov | Live web app: MediaPipe in browser, sign segmentation (hands-to-rest / motion energy), top-3 → Muse Spark → spoken sentence. |
| Mon 2 Nov | Speech→sign: Voice Transcribe, gloss reordering, SAM cutouts of clips, clip player + captions. Deaf reviewer. |
| Tue 3 Nov | Metrics (accuracy, before/after personalisation, end-to-end latency), polish, 2-min demo video, submission text. **Submit.** |

## Working agreements
- Ask before deleting files or installing large things; keep `data/` out of git; never commit API keys (use `.env`).
- Prefer small, runnable steps; print clear progress; make long jobs resumable (skip already-processed videos).
