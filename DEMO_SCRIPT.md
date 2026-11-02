# 2-minute demo video: shot list

Goal: judges see a *working* two-way conversation, the Meta models doing real work, and honest numbers.
Record the laptop screen (wide layout: camera left, conversation right) or the phone screen + a second camera
showing the people. Keep captions on. Total 2:00.

## Before recording (15 min, once)
1. Server + web running; open the app on the demo device; tap once (turns the voice on).
2. ☰ / header → scenario **🏥 Hospital**, language EN (switch to हिं for one sentence).
3. **Re-teach your signs**: ✋ Teach → ⚡ Quick setup (20 hospital words × 5 reps, ~10 min). Delete the old "Help"
   first (✋ Teach → My signs → ✕) — it contained two stray reps.
4. Type each planned doctor reply once (⌨️) so its dictionary clips and pictures are cached (no waiting on camera).
5. Framing: upper body + both hands in view with space around them; plain background; good light from the front.
6. Do a full dry run. If a sign fails twice, teach it (5 reps) and move on.

## Shots
| Time | Shot | Say / caption |
|---|---|---|
| 0:00–0:12 | Hospital reception. A Deaf patient (Anay) faces the doctor; neither knows the other's language. | "6 million people in India use Indian Sign Language. There are only a few hundred certified interpreters." |
| 0:12–0:35 | Patient signs **HELLO**, lowers hands → phone speaks "Hello". Then signs **PAIN HEAD YESTERDAY** → live guesses appear while signing → phone speaks "I have had a headache since yesterday." Show the chat bubble's sign chips (✨ if Spark corrected one). | "The phone recognises the signs on-device keypoints, then Muse Spark uses the conversation to choose between similar signs and speaks a natural sentence." |
| 0:35–0:55 | Doctor holds 🎤: "Do you have fever? Take this medicine after food." Live captions while talking. Then the **SAM 3.1 cut-out signer** appears inside the patient's camera view signing YOU FEVER… with the **picture card** for after-food. | "Muse Voice Transcribe hears the doctor; Muse Spark rewrites it into ISL grammar; SAM 3.1 cut a real signer out of every clip so the interpreter stands right in your camera view." |
| 0:55–1:15 | **Personalisation**: patient teaches their name sign or their medicine (e.g. "Insulin") — 5 quick reps, self-test ✓ — then uses it in a sentence and it's recognised. | "ISL varies by region. Teach the app your way of signing in 5 repetitions. It stays private on your profile." |
| 1:15–1:25 | Tap an alternative under a sign to correct it (it learns). Switch to हिं: the sentence is spoken in Hindi. | |
| 1:25–1:40 | 📋 **Visit summary** (EN + Hindi, "please confirm" items) → print. Quick flash of **SOS** (EMERGENCY sign → alert). | |
| 1:40–1:52 | 🎓 Learn: the ghost skeleton on your body, ★★★. (Optional: ISL Lens on a medicine strip.) | "And for families who want to learn ISL." |
| 1:52–2:00 | Results card: 79% top-1 over 178 signs tested on held-out clips, 70% (80% top-3) on signers never seen; 3.8% → 75% after one personal example; meaning preserved 64% → 75% with context. Credits. | "Built with Muse Spark, Voice Transcribe, SAM 3.1, Muse Image and PyTorch." |

## Tips
- Sign at a natural pace; lower the hands fully at the end of a sentence (0.5 s pause → spoken).
- If the phone mis-hears a sign live, tap the right alternative — that is a feature, show it.
- Don't use AI-generated sign videos: they invent signs that are not real ISL.
- Honest wording: "recognises 202 everyday and hospital signs", never "translates ISL".
