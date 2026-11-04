# ISL Bridge: submission

Each section below matches one box in the submission form. Copy the text under the heading as it is (plain text).

## Project name
ISL Bridge

## Tagline
Two-way Indian Sign Language conversations on one phone: sign to it and it speaks; talk to it and a real signer
appears in your camera view. It learns your way of signing from 5 repetitions.

## Project Description
ISL Bridge is a two-way Indian Sign Language conversation on one phone. Sign to it and it speaks; speak to it and a real signer answers inside your camera view. It learns your way of signing in 5 repetitions.

The problem
India has over 6 million ISL users and only a few hundred certified interpreters. At a hospital, a Deaf patient usually falls back on written notes, but many Deaf adults were schooled orally and read little Hindi or English. ISL also varies by region, so a single fixed model fails many real signers.

What it does
- Sign → speech: the patient signs naturally (no pauses between signs) and lowers their hands. The phone speaks a fluent English or Hindi sentence, usually 0.5–1.5 s later.
- Speech → sign: the doctor holds the mic and talks. Live captions appear, and the reply plays as real human signers cut out of sign videos, standing inside the patient's camera view, with picture cards for words like "X-ray" or "after food".
- Learns your signing: a guided flow (a ghost shows each sign, you copy it five times) adapts the app to your regional variant, or adds new signs such as your name or your medicine.
- For the visit: a bilingual visit summary for the doctor, SOS on the EMERGENCY sign, and Ghost Coach so families can learn ISL.

Results (held-out data)
202 everyday and hospital signs. 79.2% top-1 on 624 held-out clips; 69.6% on signers never seen in training. One personal example per sign took unseen signers from 3.8% to 75.0%. Conversation context raised meaning preserved from 64.0% to 74.6%.

Private by design: the video never leaves the phone; only body keypoints are sent. About $0.001 per sentence. It runs in the browser, with nothing to install.

Honest scope: it recognises 202 signs, not all of ISL, and it supports interpreters rather than replacing them.

## Public GitHub Repository
https://github.com/Anay-C/isl-bridge

## 3-Minute Demo Video
Your YouTube link (upload kit at the end of this file).

## Describe your use of Meta products to build the submitted project
Muse Spark (muse-spark-1.3, Responses API) does six jobs:
1. Sentence building: the recogniser's top-3 guesses per sign + the conversation so far + the scenario become one fluent sentence (structured output). Calls start speculatively while the person is still signing, so the sentence is usually spoken 0.5–1.5 s after the hands go down.
2. Speech → ISL: rewrites the doctor's words into ISL word order (time-topic-comment, question word last) and calls our search_dictionary tool over 2,311 ISLRTC dictionary signs for words outside the app's vocabulary.
3. Picture decisions: which words need a picture card instead of a sign.
4. Visit summary: English + Devanagari Hindi, with "please confirm" items (structured output).
5. ISL Lens: image understanding. Show an object, get its sign.
6. Teaching content: step-by-step how-to text for 227 signs, written from the sign videos (video understanding), shown in Ghost Coach and the guided teaching.
We also used it to build our evaluation: 397 synthetic dialogues, and a blind judge for 937 signed turns.

Muse Voice Transcribe (realtime, over WebSocket): live captions while the doctor holds the mic, English/Hindi code-switching, keyword-biased with the app's sign vocabulary. Moving from file transcription to realtime cut the speech → sign delay from 10.3 s to about 4 s.

SAM 3.1 (video): segmented the signer in every sign clip ("person"). We pack colour + mask into one portable H.264 video that the phone recombines into a transparent signer standing inside the live camera view: an AR interpreter made of real signers.

Muse Image: pictograms for hospital concepts without a familiar sign (X-ray, inhaler, after food), drawn once and cached.

PyTorch: the sign recogniser (keypoint Transformer + cosine classifier) and the personal prototypes behind 5-repetition personalisation. React powers the web app.

The whole project used about $11 of Meta API credit, including every SAM cut-out and picture.

## Why are Meta's models important to your solution?
Recognising signs is only half the problem. Many ISL signs look alike, sentences follow ISL grammar, and the recogniser is noisy on a new signer. Meta's models turn that into a real conversation:

- Context is what makes it work. With Muse Spark reading the recogniser's top-3 guesses together with the conversation, meaning preserved rose from 64.0% to 74.6% across 937 signed turns (+10.6 points, 95% CI +7.5 to +13.7). The same model without the conversation scored 50.4%: the gain comes from the model plus context.
- Real ISL back to the patient. Muse Spark knows ISL word order and calls our dictionary tool for unknown words, and SAM 3.1 let us show real human signers inside the camera view instead of an avatar or AI-generated video, which can invent signs that are not real ISL.
- How doctors in India actually talk. Muse Voice Transcribe captions Hindi, English and mixed speech live.
- No sign for a word? A picture. Muse Image draws pictograms, so the patient isn't left with English words they may not read.
- One API for text, speech, images and video let one student build the whole two-way system during the hackathon, at about $0.001 per sentence: cheap enough for a public hospital.

## Feedback on partner technologies (optional)
Leave it empty, or: We didn't use Fireworks, Modal or Baseten; everything ran on the Meta Model API directly, with PyTorch training on a laptop GPU.

## YouTube upload
- **Title:** ISL Bridge: two-way Indian Sign Language conversation on one phone | Meta AI Hackathon
- **Visibility:** Public (Unlisted also works for judges). Audience: not made for kids. Turn on captions.
- **Description** (fix the chapter times to your final edit; each chapter must be at least 10 s):

```
Sign to it and it speaks; speak to it and a real signer answers inside your camera view. ISL Bridge is a two-way Indian Sign Language conversation app for hospitals, built for the Meta Global AI Developer Hackathon 2026 (Indie track). In this demo I sign the patient's part; the doctor speaks off camera.

Code: https://github.com/Anay-C/isl-bridge

0:00 Intro
0:34 Sign → speech
0:55 Speech → sign
1:27 It learns your signing
1:50 More features
2:04 How it works
2:22 Results

Built with Meta Muse Spark, Muse Voice Transcribe, SAM 3.1, Muse Image and PyTorch, plus Google MediaPipe.
Sign data: INCLUDE (AI4Bharat) and the signers of St. Louis School for the Deaf, Adyar; ISLRTC dictionary; CISLR (IIT Kanpur); ISL-CSLTR.
```

---

## More detail (for judges reading the repo)

### Results (held-out, measured; small samples flagged)
- Every sign tested: 624 held-out clips of 178 signs streamed through the live server path: **79.2% top-1 / 85.3%
  top-3**, 109 signs at ≥ 90%, and the app reacts to 99.4% of signs (it was 69.7% before we found that the live
  segmenter cut off the start and end of each sign; fixed, no extra latency).
- Signers never seen in training, through the exact live pipeline (mean of 3 training seeds): **69.6% top-1 / 80.4%
  top-3** on single signs (n = 34); 58.8% of vocabulary words found inside continuous signed sentences. A signer
  from another region (Maharashtra) gets only 23.9%: ISL varies, which is why personalisation is built in.
- Robustness: all 202 signs streamed over the real WebSocket by 8 clients at once with malformed messages mixed in:
  0 dropped sessions, 0 server exceptions, 0 lost sentences.
- Personalisation on 26 signers never seen, filmed in a setup the generic model cannot read: app signs **3.8% →
  75.0%** top-1 after ONE example per sign; brand-new signs 0 → 58.4%; every one of the 26 signers improved.
- Context: across 937 signed turns with real recogniser noise, meaning preserved **64.0% → 74.6%** with Muse Spark +
  conversation context (+10.6 points, 95% CI +7.5…+13.7). Spark *without* context made it worse (50.4%): the
  context is what helps.
- Ghost Coach: a correct sign earns 3★ in 95% of held-out clips; a wrong sign gets 1★ in 98%.
- Whole project Meta API spend ≈ $11, including every SAM cut-out and picture; ~$0.001 per signed sentence.

### Technical depth
- One feature spec shared by training and live inference; same MediaPipe Tasks models in Python and the browser;
  hands assigned by distance to the pose wrists (the handedness label assumes a mirrored selfie).
- Continuous signing: a dynamic-programming decoder over multi-length windows explains the whole utterance as a
  sequence of signs (window score = seconds × (cos − 0.55)); +23.5 points over greedy peak picking on unseen signers.
- Hands-down detection relative to the body (wrist hanging below the elbow), so it works for both close-up and
  full-body framing; each stretch keeps the arm rising and falling around the sign, as in the training clips
  (+9 points), and a hand the tracker loses mid-sign is followed through the pose wrist (silent signs 3.0% → 0.6%).
- Training on five ISL sources (INCLUDE, CISLR, ISLRTC dictionary, ISL-CSLTR, clips mined from CSLTR sentences),
  because variants matter more than model size: adding 5 CSLTR signers took unseen CSLTR signers from 21% to 88%
  on clean clips.
- Guided teaching: keypoints are not sent while the ghost demonstrates, so copying along never counts as a
  repetition; a trust rule keeps a mis-cut repetition from hijacking common signs.

### Challenges
Our first live test was "garbage" although the test set said ~85%: the training data came from one school in
Chennai, people do not drop their hands between signs, the phone was held 3× closer than the training videos, and
the second hand was lost whenever the hands overlapped. Every fix was measured on signers the model had never seen
and through the exact live pipeline, not on the dataset's own test split (that gap was 40 points at one point).

### What's next
Facial grammar (eyebrows mark questions in ISL), a fingerspelling recogniser, many more Deaf signers in training
(FDMSE-ISL: 20 Deaf signers × 2,002 words), and testing with Deaf users at a real clinic.

### Credits
INCLUDE (AI4Bharat) and the signers of St. Louis School for the Deaf, Adyar · ISL Dictionary of the Indian Sign
Language Research and Training Centre (ISLRTC), DEPwD, Ministry of Social Justice & Empowerment · CISLR (IIT
Kanpur) · ISL-CSLTR (Elakkiya & Natarajan) · Mendeley ISL emergency words (evaluation only) · Meta Muse Spark,
Voice Transcribe, SAM 3.1, Muse Image, PyTorch · Google MediaPipe.
