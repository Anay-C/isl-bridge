# ISL Bridge: submission text (draft, paste into the form)

## Name
ISL Bridge

## Tagline (one line)
Two-way Indian Sign Language conversations on one phone: sign to it and it speaks; talk to it and a real signer
appears in your camera view. It learns *your* way of signing from 5 repetitions.

## The problem
India has over 6 million Indian Sign Language users and only a few hundred certified interpreters. At a hospital
counter, a Deaf patient usually writes notes, but many Deaf adults were schooled orally and read little Hindi or
English. And ISL is not one language-in-a-box: signs vary by region and school, so a single fixed model fails on
many real people (we measured this: our first model scored 21% on signers from another community).

## What we built
A phone web app plus a server.
- **Sign → speech.** The Deaf person signs naturally (no pauses between signs needed) and lowers their hands. The
  app finds the signs inside the stretch of signing, keeps the top-3 guesses per sign, and Muse Spark picks the
  sensible reading using the scenario (hospital, pharmacy) and the conversation so far, then the phone speaks a
  fluent English or Hindi sentence. Typically spoken 0.5–1.5 s after the hands go down.
- **Speech → sign.** The doctor holds the mic. Muse Voice Transcribe streams live captions (English, Hindi,
  mixed). Muse Spark rewrites the sentence into ISL order and, for words outside the app's vocabulary, searches the
  ISLRTC sign dictionary through a tool call. The reply plays as real signers: SAM 3.1 cut the human signer out of
  every clip, so they stand inside the patient's camera view, with captions. Words that have no familiar sign
  (X-ray, inhaler, after food) get a picture card drawn by Muse Image.
- **Personalisation.** "Teach a sign" takes 5 repetitions: your own variant of a word, or a brand-new sign such as
  your name or your medicine. It is guided: each word appears big on screen, a ghost signer shows it, then you sign
  it (Quick setup walks through 20 hospital words this way, about 8 minutes). Tap-to-correct also teaches. Stored privately per user, never used to train the
  shared model. A trust rule keeps a mis-cut repetition from hijacking common signs.
- **For the visit**: a bilingual visit summary for the doctor (with "please confirm" items), SOS (alert + WhatsApp
  location), Ghost Coach to learn signs (a ghost skeleton on your own body, stars and tips), ISL Lens (show an
  object, get its sign).
- **Privacy**: the signing video never leaves the phone; only hand and body keypoints are sent.

## How we used Meta AI
- **Muse Spark** (6 roles): context-aware sentence building from noisy top-3 candidates (structured output,
  speculative calls while the person is still signing); speech → ISL gloss with a `search_dictionary` tool over 2,311
  ISLRTC dictionary signs; deciding which words need a picture; a bilingual visit summary; ISL Lens (image
  understanding); step-by-step how-to text for 227 signs, written from the sign videos.
- **Muse Voice Transcribe**: realtime streaming transcription over WebSocket while the mic is held, keyword-biased
  with the app's sign vocabulary; English/Hindi.
- **SAM 3.1** (video): segmented the signer in every sign clip ("person"), packed as a portable colour+mask video
  that the phone turns back into a transparent signer standing in the live camera view.
- **Muse Image**: picture cards for hospital concepts without a familiar sign, drawn once and cached.
- **PyTorch**: the sign recogniser (keypoint Transformer + cosine classifier), plus personal prototypes that live
  on the same embedding sphere as the class weights.

## Results (held-out, measured; small samples flagged)
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

## Technical depth (highlights)
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

## Challenges
Our first live test was "garbage" although the test set said ~85%: the training data came from one school in
Chennai, people do not drop their hands between signs, the phone was held 3× closer than the training videos, and
the second hand was lost whenever the hands overlapped. Every fix was measured on signers the model had never seen
and through the exact live pipeline, not on the dataset's own test split (that gap was 40 points at one point).

## What's next
Facial grammar (eyebrows mark questions in ISL), a fingerspelling recogniser, many more Deaf signers in training
(FDMSE-ISL: 20 Deaf signers × 2,002 words), and testing with Deaf users at a real clinic.

## Credits
INCLUDE (AI4Bharat) and the signers of St. Louis School for the Deaf, Adyar · ISL Dictionary of the Indian Sign
Language Research and Training Centre (ISLRTC), DEPwD, Ministry of Social Justice & Empowerment · CISLR (IIT
Kanpur) · ISL-CSLTR (Elakkiya & Natarajan) · Mendeley ISL emergency words (evaluation only) · Meta Muse Spark,
Voice Transcribe, SAM 3.1, Muse Image, PyTorch · Google MediaPipe.
