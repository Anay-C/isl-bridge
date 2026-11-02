---
title: ISL Bridge
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Two-way Indian Sign Language conversations on one phone
---

# 🤟 ISL Bridge (test version)

**Open the app full screen: https://{SPACE_HOST}** (on a phone or a laptop with a camera, in Chrome, Safari or Edge).

A conversation app for a Deaf person who signs **Indian Sign Language (ISL)** and a hearing person, on one phone:

- **Sign → speech:** sign to the camera, then lower your hands; the phone speaks a natural English or Hindi sentence.
- **Speech → sign:** the hearing person holds 🎤 and talks (or types); the words come back as ISL sign videos with captions.
- **Learns your signing:** ✋ Teach a word 5 times, or tap ✏️ *Wrong?* under a sentence and fix it.

## How to test (5 minutes)
1. Prop the phone up so your head, shoulders and both hands are in the picture, in good light.
2. Sign one word or a short sentence (e.g. HELLO, DOCTOR, PAIN · HEAD), then lower your hands.
3. If it was wrong, tap **✏️ Wrong?** and type what you signed. That is the most useful feedback.
4. Hold **🎤** and say something like "Take this medicine after food".
5. Don't know ISL? **🎓 Learn** (☰ menu) shows each sign to copy, with a ghost coach.

It recognises about 200 everyday and hospital signs. Accuracy for a new signer is roughly 70-80% per sign, and
much higher after teaching the app a few of your own signs. It is a research prototype: don't rely on it for
anything urgent.

## Privacy
- Your camera video never leaves your device: hand and body keypoints are computed in the browser, and only
  those numbers are sent to the server.
- While **Help improve ISL Bridge** is on (you choose on the first screen and can switch it off at any time), the
  keypoints of your signs and your reports are saved, so the team can find mistakes and improve the recogniser.
- Speech you record is sent to Meta AI (Muse Voice Transcribe) to be transcribed.

## Built with
Meta **Muse Spark** (sentences from the recogniser's guesses + context, ISL word order, visit summary, ISL Lens),
**Muse Voice Transcribe** (live captions), **SAM 3.1** (signer cut-outs), Muse Image (picture cards); a PyTorch
keypoint Transformer with personal prototypes; MediaPipe Tasks in the browser.

## Credits
Sign videos and training data: ISL Dictionary of the Indian Sign Language Research and Training Centre (ISLRTC),
DEPwD, Ministry of Social Justice & Empowerment (used for ISL technology with acknowledgement, no profit) ·
INCLUDE dataset (AI4Bharat, CC BY 4.0), signers of St. Louis School for the Deaf, Adyar · CISLR (IIT Kanpur,
AFL-3.0) · ISL-CSLTR (Elakkiya & Natarajan, CC BY 4.0).

Build `{VERSION}`.
