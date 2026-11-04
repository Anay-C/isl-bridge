# Demo video (under 3 min): speech and shot list

Goal: judges see a *working* two-way conversation, the Meta models doing real work, and honest numbers.
The app is on screen nearly the whole time; the explaining is a voice-over, plus ~25 s of you to camera and one
diagram. The limit is probably 3 min (other teams' notes; the official page doesn't say): check the submission form.
At a calm pace (~140 words a minute) this runs ~2:45, or ~2:30 without the (optional) part. Read it aloud with a timer
before recording.

Record the laptop screen (wide layout: camera left, conversation right) or the phone screen + a second camera
showing the people. Keep captions on.

## Before recording (15 min, once)
1. Server + web running; open the app on the demo device; tap once (turns the voice on).
2. ☰ / header → scenario **🏥 Hospital**, language EN (switch to हिं for one sentence).
3. **Re-teach your signs**: ✋ Teach → ⚡ Quick setup (20 hospital words × 5 reps, ~8-10 min; each word comes up
   on screen, the ghost shows it twice, then you sign it 5 times, lowering your hands after each). Delete the old "Help"
   first (✋ Teach → My signs → ✕) — it contained two stray reps. **Screen-record this run**: part 5 uses ~10 s of it.
4. Rehearse part 5 once: ✋ Teach → type **Insulin** → Start (no ghost: it's a new sign), 5 reps, then sign
   I · INSULIN. Teaching it again on camera replaces the rehearsal.
5. Type the doctor's line once (⌨️) so its sign clips and picture card are cached (no waiting on camera).
6. Framing: upper body + both hands in view with space around them; plain background; good light from the front.
7. A friend plays the doctor (one line, part 4).
8. Make the two stills: the diagram (part 7) and the results card (part 8). Text for both is at the end.
9. Do a full dry run. If a sign fails twice, teach it (5 reps) and move on.

## The video

### 1 · Cold open (0:00–0:08)
**On screen:** at the hospital desk you sign **PAIN · HEAD · YESTERDAY** and lower your hands. The phone says
e.g. *"I've had a headache since yesterday."* Caption: "Signed in Indian Sign Language → spoken by the phone".
**Say:** nothing. Let it land.

### 2 · You, to camera (0:08–0:34)
> Hi, I'm Anay, a student at VIT Pune. India has over six million sign language users, and only a few hundred
> certified interpreters. So at a hospital, a Deaf patient usually writes notes, but many Deaf adults read little
> English or Hindi. I built ISL Bridge: sign to the phone and it speaks; speak to it and it signs back.

### 3 · Sign → speech (0:34–0:53)
**On screen:** the cold open again, slower: live guesses appear while you sign, then the chat bubble with its sign
chips (tap one to show its alternatives).
> The phone tracks my hands and body; only those points leave it, never the video. My model finds each sign, with
> no pauses needed, and keeps its top three guesses. Muse Spark weighs them against the conversation so far, and the
> phone speaks a natural sentence.

### 4 · Speech → sign (0:53–1:22)
**On screen:** the doctor holds 🎤 and speaks; live captions. Then the cut-out signer plays inside your camera view,
with the picture card for "after food".
**Doctor:** *"Do you have a fever? Take this medicine after food."*
> Muse Voice Transcribe captions the doctor live. Muse Spark rewrites the sentence into ISL word order. These are
> real signers, not AI avatars: SAM 3.1 cut them out of their videos and into my camera view. "After food" has no
> familiar sign, so Muse Image drew a picture.

### 5 · It learns your signing (1:22–1:45)
**On screen:** from your Quick setup recording: a word comes up big, the ghost shows it, you sign it 5 times (speed it
up, label "4×"), ✓ Learned. Then teaching **Insulin** (4×). Then, live, you sign **I · INSULIN** and the phone says
e.g. *"I take insulin."*
> ISL varies by region, so the app learns your signing: a word comes up, a ghost shows the sign, and you copy it
> five times. You can even add new signs. Insulin wasn't in its vocabulary; I taught it in five tries.

### 6 · More, quickly (1:45–1:59) (optional)
**On screen:** fast cuts: tap a different guess under a sign; switch to हिं (a sentence spoken in Hindi);
📋 visit summary; sign EMERGENCY → SOS.
> Tap another guess to correct a sign, and it learns. It speaks Hindi too, writes a visit summary for the doctor,
> and raises an SOS on the EMERGENCY sign.

### 7 · Under the hood (1:59–2:17)
**On screen:** the diagram.
> Under the hood, MediaPipe keypoints feed my PyTorch Transformer. Each sign becomes an embedding, which is how it
> learns a new sign from a few examples. Muse Spark does six jobs here, including searching the ISL dictionary with
> a tool call.

### 8 · Results and close (2:17–2:46)
**On screen:** the results card, then credits (3 s, no speech).
> It knows 202 signs: 79% right on held-out clips, 70% for signers it has never seen. For new signers in a hard
> camera setup, one personal example per sign lifted accuracy from 4% to 75%. ISL Bridge isn't a human interpreter,
> but it can be there when no interpreter is.

## Text for the stills

**Diagram**
- Sign → speech: camera → MediaPipe hand + body keypoints (on the phone; the video never leaves it) → PyTorch
  Transformer: top-3 per sign, plus your personal signs → Muse Spark picks the reading that fits the conversation →
  the phone speaks (EN / हिं).
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
- Record the voice-over while watching your edited recording: cleaner audio, and it's easy to fit the time. Keep the
  phone's voice and the doctor's line as live sound.
- Burn in captions: judges may watch muted, and the app is for Deaf people.
- Sign at a natural pace; lower the hands fully at the end of a sentence (0.5 s pause → spoken).
- If the phone's sentence differs a little from the script, keep the take; re-record only if the meaning is wrong.
- If the phone mis-hears a sign live, tap the right alternative — that is a feature, show it.
- Label anything sped up ("4×").
- Don't use AI-generated sign videos: they invent signs that are not real ISL.
- Honest wording: "recognises 202 everyday and hospital signs", never "translates ISL".
