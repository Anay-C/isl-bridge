# Testing ISL Bridge with other people

How I share ISL Bridge with testers, what they see, and how I read what they report.

## 1. Sharing it

Double-click **`deploy\share.bat`** (or run `.venv-ml/Scripts/python deploy/share.py`). After about a minute it
prints:

```
  ISL Bridge is online for testers:
      https://anaychauhan-isl-bridge.static.hf.space
```

**Testers always get the same address: https://anaychauhan-isl-bridge.static.hf.space.** It works on any phone or
laptop with a camera (Chrome, Safari, Edge), from anywhere, with no certificate warning, install or account.

- **How it works.** The app runs **on my laptop** and is reached through a Cloudflare quick tunnel
  (`data/tools/cloudflared.exe`, signed by Cloudflare) at a random `https://<words>.trycloudflare.com` link, which
  changes now and then. The fixed address is a free static page on my Hugging Face account (Space
  `AnayChauhan/isl-bridge`, source in `deploy/forward/`). It checks the app is answering and then forwards to it, and
  `share.py` updates it within seconds whenever the link changes. It also passes along each tester's profile id, so
  they keep their taught signs when the link changes. Current link: `data/share_url.txt`. Every link with its time:
  `data/share_links.log`. What the watchdog did: `data/share.log`.
- The laptop has to stay on, plugged in and online. Closing the lid puts it to sleep (offline until it wakes). After
  a reboot, double-click `share.bat` again; the fixed address catches up by itself.
- **When the Wi-Fi drops** (it does, often for 6-14 minutes): the fixed address shows "ISL Bridge is offline for a
  moment" and keeps retrying. A tester who is already in the app sees **"Connection lost · Reopen ISL Bridge"** after
  about 20 s, and tapping it takes them back into the app through the fixed address, same profile. Drops of up to
  ~12 minutes keep the same link (cloudflared reconnects ~15 s after the internet is back). After a longer drop
  Cloudflare deletes the tunnel ("Tunnel not found"), and `share.py` makes a new link straight away and puts it behind
  the fixed address. When I killed the tunnel on purpose, the tester was back in the app on the new link in ~90 s with
  the same profile. If the server crashes it gets restarted within seconds (same link).
- **Stop sharing:** close the window, or double-click **`deploy\stop_share.bat`** (this also works when the window is
  hidden). The fixed address then shows "offline for a moment".
- **Changed the code but want to keep the link:** `.venv-ml/Scripts/python deploy/share.py --reload` rebuilds the web
  app, restarts only the server (~15 s) and re-publishes the fixed-address page. Testers' pages reconnect on their
  own with their profile; only the conversation context from that moment is lost.
- It runs its own public-mode server on `127.0.0.1:8010`, separate from my dev setup (`:8000` + Vite `:5173`).
- Public mode means limits per device (40 replies/min, 8 Lens photos/min, 40 Spark calls/min per signer), debug
  routes switched off, and my own `anay` profile can't be opened through the link.
- **Spend cap:** `META_SPEND_CAP_USD=50` in `.env` covers everything the laptop serves. Once it's reached, the app
  keeps working without Meta AI: sentences come out word by word, replies are signed word by word from the app's
  clips and the ISLRTC dictionary, and replies are typed instead of spoken. I also set a spend limit on the Meta
  billing page to be safe.
- Running the app without my laptop would need a paid Hugging Face PRO account (Docker Spaces aren't free any more).
  It's all ready for that: `deploy/build_space.py` (Dockerfile, Space card, private-dataset sync of tester data, see
  `server/sync.py`) deploys to `AnayChauhan/isl-bridge-app` in one command. Then the laptop cap goes down to $20
  (the Space gets $30) and the fixed address points at the app Space.

The message I send:
> Please try ISL Bridge, an Indian Sign Language conversation app: https://anaychauhan-isl-bridge.static.hf.space.
> Allow the camera and read the 1-minute intro. Then sign to it, and tap "Wrong?" whenever it gets a sign wrong.
> Takes 5 minutes.

## 2. What a tester sees

1. **Welcome** (first visit only): what the app is, 5 things to try, and a **"Help improve ISL Bridge"** checkbox
   (on by default): "save the hand and body keypoints of my signs (numbers only, never video) and my reports".
   Nothing is logged until they tap **Start** (that tap also unlocks the phone's voice).
2. **Their own private profile** (`t-xxxxxx`, kept on that device). Nobody else's taught signs affect them.
3. **Sign → sentence:** sign, drop the hands, and the phone says the sentence. Under each sentence there's **✏️ Wrong?
   Tell us what you signed**: they type the right words (autocomplete from the 202 signs) and can add a note. The app
   learns the correction for them, and I get a labelled clip.
4. **Reply:** hold 🎤 and speak (live captions; the words show 0.6 s after letting go and the signs follow in 4-10 s),
   or type with ⌨️. If a reply couldn't be signed it says so, and tapping it tries again.
5. **☰ menu / ⓘ:** How to test, 💬 Send feedback, Help improve (on/off), profile id, *start fresh*, *delete my
   data* (removes their profile, keypoints and reports from my laptop right away).
6. **✋ Teach / ⚡ Quick setup (guided):** the words come one at a time, big on the camera view. A ghost shows each
   sign twice ("copy it like a mirror"; nothing counts yet), then the tester signs it 5 times, dropping the hands
   after each one (dots fill up, a small ghost keeps looping in the corner, how-to steps underneath). Once it's
   learned: ✓ and on to the next word. If the reps were too different: "teach it again" or "keep it". They can Skip,
   Stop (and continue later; Quick setup then only offers the words left) or tap any word to jump to it.
7. 🎓 **Learn** (Ghost Coach) for testers who don't know ISL yet.

Good things to ask testers to try: single words (HELLO, DOCTOR, WATER, PAIN), a short sentence (I · PAIN · HEAD),
✋ teaching one of their own signs and then using it, a spoken reply, and the visit summary (📋).

## 3. Reading what they found

```
.venv-ml/Scripts/python ml/tester_report.py            # testers only; --all includes anay; --since 2026-11-04
```
It prints the testers, signs logged, taught signs, the **most common mistakes (read as → actually signed)**, notes,
and how many labelled clips were saved. Raw data:

| What | Where |
|---|---|
| "Wrong?" reports + notes (one JSON line each, with the sentence, top-3 per sign, the fix and recent conversation) | `data/feedback/<profile>.jsonl` |
| Keypoints of each corrected sign (labelled live data to train / evaluate on) | `data/feedback/<profile>/*.npz` (`meant`, `span`, `top`) |
| Every sign from testers with "help improve" on | `data/live_log/<profile>/*.npz` |
| Taught signs | `data/users/<profile>.json` |
| Server log (errors, `[privacy]` deletions) | `data/share_server.log` |

## 4. Automated checks (before sending out a new link)

```
.venv-ml/Scripts/python ml/test_features.py https://<link>.trycloudflare.com data   # 25 checks over the internet
.venv-ml/Scripts/python ml/test_features.py http://127.0.0.1:8000                   # LAN server (20 checks)
.venv-ml/Scripts/python ml/test_ws.py --url ws://127.0.0.1:8010/ws/sign --clients 6 --fuzz --speed 2 --limit 60
.venv-ml/Scripts/python deploy/test_share.py                                        # share watchdog, 16 cases, 1 s
```
On my Wi-Fi, Git Bash `curl` fails for HTTPS (exit 35/28) while browsers, Python and PowerShell work, so I check the
link with Python or PowerShell.

`test_features.py` covers: health, hello/consent, sign → sentence, nothing logged with consent off, "Wrong?" with
and without keypoints, notes, malformed reports, undo, tap-to-correct, forget, summary, switching scenario, teaching
a new sign, cancel, Learn-page practice, delete my data, typed reply → ISL, per-device limits (including a forged
X-Forwarded-For, which only passes through the tunnel), debug route off, and the protected `anay` profile. It uses
real Muse Spark (a few cents per run).

Latest results: 25/25 through the public link; 20/20 on the LAN server; 21/21 on the Space bundle with Meta AI
switched off; 6 clients + fuzzing: 0 dropped connections, 0 server errors, 48/60 first signs right. Browser tests
(headless Chrome) of the full tester journey, the link-change journey (tunnel killed → "Reopen" → new link, same
profile) and the guided teaching flow all passed. A real tunnel survived a 5-minute outage and kept the same link.
The slowest part is the first load over the tunnel (up to ~30 s to "Ready" on a fresh browser).

## 5. Known limits to tell testers

- About 200 signs (hospital + everyday). A sign that isn't on the list comes out as the closest one it knows.
- A new signer gets roughly 70-80% of signs right; regional variants and fast signing do worse. ✋ Teach (5 reps)
  or a few "Wrong?" corrections fix their own signs.
- Stand back so the head, shoulders and both hands are in view; the app pauses and shows an outline if you're too
  close.
- Once a sentence has been spoken (0.5 s after the hands drop), use "Wrong?" instead of ⌫ undo.
- Speech → signs takes 4-10 s (Muse Spark puts it into ISL word order); the captions show straight away.
- Not tested on a real iPhone yet. The iPhone microphone start was fixed without a device to test on (the
  AudioContext is created inside the tap).
- The app runs on my laptop, so when my Wi-Fi drops the app is offline (the fixed address says so and waits).
  After the link changes, testers see the welcome screen again and make the "Help improve" choice again, because to
  the browser each new link is a new site. Their profile and taught signs carry over.
