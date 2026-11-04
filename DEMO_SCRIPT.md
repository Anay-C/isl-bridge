# Demo video (under 3 min): solo script

You record it alone on the laptop: you play both the Deaf patient and the doctor (on-screen labels say so). Record in
pieces, stitch them, and add the voice-over last. Goal: judges see a *working* two-way conversation, the Meta models
doing real work, and honest numbers.

Length: the limit is probably 3 min (other teams' notes; the official page doesn't say), so check the submission form.
At a calm pace (~140 words a minute) this runs ~2:45, or ~2:30 without the (optional) part. Plan about 2 hours in all.

## Setup (~25 min, once)
1. **Laptop on a table, you ~1 m back**: head, shoulders and both hands in the webcam view with space around them;
   plain background; light from the front. Keep a mouse within reach: 🎤 is press-and-hold.
2. **Chrome → http://127.0.0.1:8010** (the same app the testers use, served by this laptop: no internet hop for the
   camera). Make the window wide: camera left, conversation right. Click once (turns the voice on). Scenario
   **🏥 Hospital**, language EN.
3. **Screen recorder**: Snipping Tool (Win + Shift + R → Record; turn on system sound AND microphone) or Game Bar
   (Win + Alt + R; Win + Alt + M for the mic). Record 10 s and play it back: the app's voice and your voice must both
   be in it.
4. **⚡ Quick setup with the recorder running** (✋ Teach → ⚡ Quick setup: 20 hospital words × 5 reps, ~8-10 min; each
   word comes up big, the ghost shows it twice, then you sign it 5 times, lowering your hands after each). Part 5 uses
   ~10 s of this recording. If ✋ Teach → My signs already lists an old "Help", delete it first (it had stray reps).
5. **Teach Insulin with the recorder running**: ✋ Teach → type **Insulin** → Start (no ghost: it's a new sign),
   5 reps.
6. **Cache the reply**: ⌨️ Type → "Do you have a fever? Take this medicine after food." so its sign clips and picture
   are ready. Then **reload the page**: the conversation starts empty (your taught signs stay).
7. **Make the two stills**: the diagram (part 7) and the results card (part 8). The text for both is at the end.

## Shoot (in pieces)
**A. Main take** (parts 1, 3, 4, the end of 5, and 6). Recorder on, then in one conversation:
1. Sign **PAIN · HEAD · YESTERDAY**, lower your hands → the app speaks.
2. As the doctor: click and hold **🎤 Hold to speak**, say *"Do you have a fever? Take this medicine after food."*,
   release → live captions, then the cut-out signer and the picture play in your camera view.
3. As the patient: sign **I · INSULIN** → the app speaks (e.g. "I take insulin").
4. Tap a different guess under one of your signs. Switch to **हिं** and sign **THANK YOU**. Click **📋 Summary**.
   Sign **EMERGENCY** → SOS.

A mistake doesn't mean starting over: pause, repeat that step, and cut the bad bit later.

**B. You to camera** (part 2): phone on a stand at eye level, quiet room, 3-4 takes; keep the best.

**C. The teaching clips** (part 5): already recorded in setup steps 4 and 5.

**D. Voice-over, last**: watch your edit and read each part below. Windows Sound Recorder or your phone's voice
recorder, close to your mouth, in a quiet room.

## Edit (Clipchamp, built into Windows)
- Order: 1 (from A) → 2 (B) → 3 (A again, slower, zoomed on the sign chips) → 4 (A) → 5 (C at 4×, then A's
  I · INSULIN) → 6 (A, fast cuts) → 7 diagram → 8 results card → credits.
- On-screen labels: **"Me as the patient"** on part 1, **"Me as the doctor"** on part 4, **"4×"** on sped-up clips.
- Subtitles for the whole video (Clipchamp's auto-subtitles; fix "Muse Spark", "SAM 3.1", "ISL").
- Keep the live sound where the app speaks and where you speak as the doctor; lower it under the voice-over elsewhere.

## The speech

### 1 · Cold open (0:00–0:08)
**On screen:** you sign PAIN · HEAD · YESTERDAY and lower your hands; the app says e.g. *"I've had a headache since
yesterday."* Label: "Me as the patient: signing in Indian Sign Language".
**Say:** nothing. Let it land.

### 2 · You, to camera (0:08–0:31)
> Hi, I'm Anay, a student at VIT Pune. India has over six million sign language users, and only a few hundred
> certified interpreters. So at a hospital, a Deaf patient usually writes notes, but many Deaf adults read little
> English or Hindi. I built ISL Bridge: sign to it and it speaks; speak to it and it signs back.

### 3 · Sign → speech (0:31–0:52)
**On screen:** the cold open again, slower: live guesses appear while you sign, then the chat bubble's sign chips.
> The app tracks my hands and body; only those points leave my device, never the video. My model finds each sign,
> with no pauses needed, and keeps its top three guesses. Muse Spark weighs them against the conversation so far,
> and the app speaks a natural sentence.

### 4 · Speech → sign (0:52–1:22)
**On screen:** label "Me as the doctor". You hold 🎤 and say the line (live sound); captions appear, then the cut-out
signer plays inside your camera view, with the picture card for "after food".
**You, live:** *"Do you have a fever? Take this medicine after food."*
> Muse Voice Transcribe captions the doctor live. Muse Spark rewrites the sentence into ISL word order. These are
> real signers, not AI avatars: SAM 3.1 cut them out of their videos and into my camera view. "After food" has no
> familiar sign, so Muse Image drew a picture.

### 5 · It learns your signing (1:22–1:45)
**On screen:** from the Quick setup recording: a word comes up big, the ghost shows it, you sign it 5 times (4×),
✓ Learned. Then teaching Insulin (4×). Then, from the main take, you sign I · INSULIN and the app speaks.
> ISL varies by region, so the app learns your signing: a word comes up, a ghost shows the sign, and you copy it
> five times. You can even add new signs. Insulin wasn't in its vocabulary; I taught it in five tries.

### 6 · More, quickly (1:45–1:59) (optional)
**On screen:** fast cuts: tapping a different guess; the Hindi sentence; 📋 visit summary; EMERGENCY → SOS.
> Tap another guess to correct a sign, and it learns. It speaks Hindi too, writes a visit summary for the doctor,
> and raises an SOS on the EMERGENCY sign.

### 7 · Under the hood (1:59–2:17)
**On screen:** the diagram.
> Under the hood, MediaPipe keypoints feed my PyTorch Transformer. Each sign becomes an embedding, which is how it
> learns a new sign from a few examples. Muse Spark does six jobs here, including searching the ISL dictionary with
> a tool call.

### 8 · Results and close (2:17–2:45)
**On screen:** the results card, then credits (3 s, no speech).
> It knows 202 signs: 79% right on held-out clips, 70% for signers it has never seen. For new signers in a hard
> camera setup, one personal example per sign lifted accuracy from 4% to 75%. ISL Bridge isn't a human interpreter,
> but it can be there when no interpreter is.

## Text for the stills

**Diagram**
- Sign → speech: camera → MediaPipe hand + body keypoints (on the device; the video never leaves it) → PyTorch
  Transformer: top-3 per sign, plus your personal signs → Muse Spark picks the reading that fits the conversation →
  the app speaks (EN / हिं).
- Speech → sign: mic → Muse Voice Transcribe (live captions) → Muse Spark: ISL word order + ISLRTC dictionary tool
  call → real signer clips cut out by SAM 3.1, picture cards by Muse Image.

**Results card** (same numbers as SUBMISSION.md; all measured on held-out data)
- 202 everyday and hospital signs
- 79.2% top-1 / 85.3% top-3: 624 held-out clips of 178 signs, through the live app
- 69.6% top-1 / 80.4% top-3: signers never seen in training (n = 34, mean of 3 training runs)
- 3.8% → 75.0% after ONE personal example per sign: 26 unseen signers, hands-only framing the base model can't read
- Meaning kept 64% → 75% with Muse Spark + conversation context (937 signed turns)
- ≈ $11 Meta API spend for the whole project

**Credits:** Built with Muse Spark · Muse Voice Transcribe · SAM 3.1 · Muse Image · PyTorch · MediaPipe.
Sign data: INCLUDE (AI4Bharat) and the signers of St. Louis School for the Deaf, Adyar · ISLRTC dictionary ·
CISLR (IIT Kanpur) · ISL-CSLTR.

## Tips
- Sign at a natural pace; lower the hands fully at the end of a sentence (0.5 s pause → spoken).
- If the app's sentence differs a little from the script, keep the take; redo it only if the meaning is wrong.
- If the app mis-hears a sign, tap the right alternative: that is a feature, show it.
- Don't use AI-generated sign videos: they invent signs that are not real ISL.
- Honest wording: "recognises 202 everyday and hospital signs", never "translates ISL".
- Keep the sharing window ("ISL Bridge - sharing with testers") open while you record: closing it stops the app.
