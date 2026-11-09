# ISL Bridge: submission

What I put in each box of the hackathon submission form, plus more detail for anyone reading the repo.

## Project name
ISL Bridge

## Tagline
Two-way Indian Sign Language conversations on one phone: sign to it and it speaks; talk to it and a real signer
appears in your camera view. It learns your way of signing from 5 repetitions.

## Project Description
ISL Bridge is a two-way Indian Sign Language conversation on one phone. Sign to it and it speaks; talk to it and a real signer answers inside your camera view. It learns your way of signing in 5 repetitions.

The problem
India has over 6 million ISL users and only a few hundred certified interpreters. At a hospital, a Deaf patient usually ends up writing notes, but many Deaf adults went to oral schools and can't read much Hindi or English. ISL also changes from region to region, so one fixed model fails a lot of real signers.

What it does
- Sign → speech: the patient signs normally (no pauses between signs) and drops their hands. The phone says a proper English or Hindi sentence, usually 0.5–1.5 s later.
- Speech → sign: the doctor holds the mic and talks. Captions show up live, and the reply plays as real human signers cut out of sign videos, standing inside the patient's camera view, with picture cards for words like "X-ray" or "after food".
- Learns your signing: a guided flow (a ghost shows each sign, you copy it five times) adapts the app to your regional variant, or adds new signs like your name or your medicine.
- For the visit: a summary for the doctor in two languages, SOS on the EMERGENCY sign, and Ghost Coach so family members can learn ISL.

Results (held-out data)
202 everyday and hospital signs. 79.2% top-1 on 624 held-out clips; 69.6% on signers the model never saw. One personal example per sign took unseen signers from 3.8% to 75.0%. Using the conversation as context raised meaning kept from 64.0% to 74.6%.

Private: the video never leaves the phone, only body keypoints are sent. About $0.001 per sentence. It runs in the browser, nothing to install.

What it isn't: it recognises 202 signs, not all of ISL, and it's meant to support interpreters, not replace them.

## Public GitHub Repository
https://github.com/Anay-C/isl-bridge

## 3-Minute Demo Video
YouTube link (upload details at the end of this file).

## Describe your use of Meta products to build the submitted project
Muse Spark (muse-spark-1.3, Responses API) does six things:
1. Building sentences: the recogniser's top-3 guesses per sign + the conversation so far + the scenario go in, and one natural sentence comes out (structured output). I start the call while the person is still signing, so the sentence is usually spoken 0.5–1.5 s after the hands go down.
2. Speech → ISL: it rewrites the doctor's words into ISL word order (time-topic-comment, question word last) and calls my search_dictionary tool over 2,311 ISLRTC dictionary signs for words the app doesn't have.
3. Pictures: deciding which words need a picture card instead of a sign.
4. Visit summary: English + Hindi (Devanagari), with "please confirm" items (structured output).
5. ISL Lens: image understanding. Show it an object, get the sign.
6. Teaching content: step-by-step how-to text for 227 signs, written from the sign videos (video understanding), shown in Ghost Coach and in the guided teaching.
I also used it for my evaluation: 397 synthetic dialogues, and as a blind judge for 937 signed turns.

Muse Voice Transcribe (realtime, over WebSocket): live captions while the doctor holds the mic, English/Hindi mixing, biased towards the app's sign words. Switching from file transcription to realtime brought the speech → sign delay down from 10.3 s to about 4 s.

SAM 3.1 (video): cut out the signer in every sign clip ("person"). I pack colour + mask into one H.264 video that the phone puts back together into a transparent signer standing inside the live camera view, so the interpreter is made of real signers.

Muse Image: pictograms for hospital things that don't have a well-known sign (X-ray, inhaler, after food), drawn once and cached.

PyTorch: the sign recogniser (keypoint Transformer + cosine classifier) and the personal prototypes behind 5-repetition personalisation. The web app is React.

The whole project used about $11 of Meta API credit, including every SAM cut-out and picture.

## Why are Meta's models important to your solution?
Recognising signs is only half of it. A lot of ISL signs look alike, sentences follow ISL grammar, and the recogniser makes mistakes on a new signer. Meta's models are what turn that into an actual conversation:

- Context is what makes it work. When Muse Spark reads the recogniser's top-3 guesses together with the conversation, meaning kept went from 64.0% to 74.6% over 937 signed turns (+10.6 points, 95% CI +7.5 to +13.7). The same model without the conversation got 50.4%, so the gain comes from the model plus the context.
- Real ISL back to the patient. Muse Spark knows ISL word order and calls my dictionary tool for unknown words, and SAM 3.1 let me show real human signers in the camera view instead of an avatar or AI-generated video, which can make up signs that aren't real ISL.
- How doctors in India actually talk. Muse Voice Transcribe captions Hindi, English and mixed speech live.
- No sign for a word? A picture. Muse Image draws pictograms, so the patient isn't left with English words they might not be able to read.
- Having one API for text, speech, images and video meant I could build the whole two-way system by myself during the hackathon, at about $0.001 per sentence, which is cheap enough for a public hospital.

## Feedback on partner technologies (optional)
I didn't use Fireworks, Modal or Baseten; everything ran on the Meta Model API directly, with PyTorch training on my laptop GPU.

## YouTube upload
- **Title:** ISL Bridge: two-way Indian Sign Language conversation on one phone | Meta AI Hackathon
- **Visibility:** Public. Audience: not made for kids. Captions on.
- **Description:**

```
Sign to it and it speaks; speak to it and a real signer answers inside your camera view. ISL Bridge is a two-way Indian Sign Language conversation app for hospitals, built for the Meta Global AI Developer Hackathon 2026 (Indie track). In this demo I sign the patient's part.

Code: https://github.com/Anay-C/isl-bridge


Built with Meta Muse Spark, Muse Voice Transcribe, SAM 3.1, Muse Image and PyTorch, plus Google MediaPipe.
Sign data: INCLUDE (AI4Bharat) and the signers of St. Louis School for the Deaf, Adyar; ISLRTC dictionary; CISLR (IIT Kanpur); ISL-CSLTR.
```

---

## More detail

### Results (held-out, measured; small samples noted)
- Every sign: 624 held-out clips of 178 signs streamed through the live server: **79.2% top-1 / 85.3% top-3**, 109
  signs at ≥ 90%, and the app responds to 99.4% of signs. (It was 69.7% until I found that the live segmenter was
  cutting off the start and end of each sign. Fixing it added no latency.)
- Signers the model never saw, through the same pipeline as the live app (average of 3 training seeds): **69.6%
  top-1 / 80.4% top-3** on single signs (n = 34); 58.8% of vocabulary words found inside continuous signed sentences.
  A signer from another region (Maharashtra) only gets 23.9%. ISL varies, which is why personalisation is built in.
- Load test: all 202 signs over the real WebSocket from 8 clients at once, with broken messages mixed in: 0 dropped
  sessions, 0 server errors, 0 lost sentences.
- Personalisation on 26 unseen signers, filmed in a setup the base model can't read: **3.8% → 75.0%** top-1 after
  ONE example per sign; brand-new signs 0 → 58.4%; all 26 signers improved.
- Context: over 937 signed turns with real recogniser mistakes, meaning kept went **64.0% → 74.6%** with Muse Spark +
  the conversation (+10.6 points, 95% CI +7.5…+13.7). Spark *without* the conversation made it worse (50.4%), so it's
  the context that helps.
- Ghost Coach: a correct sign gets 3★ on 95% of held-out clips; a wrong sign gets 1★ on 98%.
- Total Meta API spend ≈ $11, including every SAM cut-out and picture; ~$0.001 per signed sentence.

### Technical details
- One feature spec for both training and live use, and the same MediaPipe Tasks models in Python and in the
  browser. Hands are matched to left/right by distance to the pose wrists, because MediaPipe's handedness label
  assumes a mirrored selfie.
- Continuous signing: a dynamic-programming decoder over windows of different lengths explains the whole utterance
  as a sequence of signs (window score = seconds × (cos − 0.55)); 23.5 points better than greedy peak picking on
  unseen signers.
- Hands-down is detected relative to the body (wrist below the elbow), so it works whether you're close to the
  camera or fully in frame. Each stretch keeps the arm going up and coming down around the sign, like in the training
  clips (+9 points), and a hand the tracker loses mid-sign is followed through the pose wrist (missed signs 3.0% →
  0.6%).
- Trained on five ISL sources (INCLUDE, CISLR, ISLRTC dictionary, ISL-CSLTR, clips cut from CSLTR sentences),
  because sign variants matter more than model size: adding 5 CSLTR signers took unseen CSLTR signers from 21% to
  88% on clean clips.
- Guided teaching: nothing is sent while the ghost is demonstrating, so copying along never counts as a rep, and a
  trust rule stops one badly cut rep from taking over a common sign.

### Challenges
My first live test was garbage even though the test set said ~85%. The training data came from one school in
Chennai, people don't drop their hands between signs, I was holding the phone 3× closer than in the training videos,
and the second hand got lost whenever the hands overlapped. After that I measured every fix on signers the model had
never seen, through the same pipeline as the live app, not on the dataset's own test split (at one point the gap
between the two was 40 points).

### What's next
Facial grammar (eyebrows mark questions in ISL), a fingerspelling recogniser, a lot more Deaf signers in training
(FDMSE-ISL: 20 Deaf signers × 2,002 words), and testing with Deaf users at a real clinic.

### Credits
INCLUDE (AI4Bharat) and the signers of St. Louis School for the Deaf, Adyar · ISL Dictionary of the Indian Sign
Language Research and Training Centre (ISLRTC), DEPwD, Ministry of Social Justice & Empowerment · CISLR (IIT
Kanpur) · ISL-CSLTR (Elakkiya & Natarajan) · Mendeley ISL emergency words (evaluation only) · Meta Muse Spark,
Voice Transcribe, SAM 3.1, Muse Image, PyTorch · Google MediaPipe.
