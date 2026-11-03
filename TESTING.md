# Testing ISL Bridge with other people

How to put ISL Bridge in front of testers, what they see, and how to read what they report.
(Last checked end to end on 3 Nov 2026: every check below passed through the public link.)

## 1. Share it (Anay)

Double-click **`deploy\share.bat`** (or run `.venv-ml/Scripts/python deploy/share.py`). After ~1 minute it prints:

```
  ISL Bridge is online for testers:
      https://anaychauhan-isl-bridge.static.hf.space
```

**Send testers that fixed address: https://anaychauhan-isl-bridge.static.hf.space.** It never changes. It works on
any phone or laptop with a camera (Chrome, Safari, Edge), anywhere: no certificate warning, no install, no account.

- **How it works.** The app runs **on this laptop**, reached through a Cloudflare quick tunnel
  (`data/tools/cloudflared.exe`, signed by Cloudflare) at a random `https://<words>.trycloudflare.com` link. That
  link changes now and then (below). The fixed address is a free static page on your Hugging Face account (Space
  `AnayChauhan/isl-bridge`, source in `deploy/forward/`). It checks the app answers, then forwards there, and
  `share.py` updates it within seconds whenever the link changes. It also keeps each tester's profile id, so a
  tester keeps their taught signs across link changes. Current link: `data/share_url.txt`. Every link, with times:
  `data/share_links.log`. What the watchdog did: `data/share.log`.
- Keep the laptop on, plugged in and online. Closing the lid puts it to sleep (offline until it wakes). After a
  reboot, double-click `share.bat` again; the fixed address follows by itself.
- **When the Wi-Fi drops** (4 times on 3 Nov, 6-14 minutes each): the fixed address shows "ISL Bridge is offline
  for a moment" and keeps retrying by itself. A tester already in the app gets **"Connection lost · Reopen ISL
  Bridge"** after ~20 s. Tapping it goes through the fixed address back into the app, with the same profile.
  Drops up to ~12 minutes keep the same link (cloudflared reconnects ~15 s after the internet is back). After a
  longer drop, Cloudflare deletes the tunnel ("Tunnel not found"), and `share.py` makes a new link at once and
  puts it behind the fixed address. Tested: tunnel killed -> tester back in the app on the new link in ~90 s,
  same profile. A crashed server is restarted within seconds (same link).
- **Stop sharing:** close the window, or double-click **`deploy\stop_share.bat`** (also works when the window is
  hidden, e.g. when it was started for you by Claude Code). The fixed address then shows "offline for a moment".
- **Changed the code? Keep the link:** `.venv-ml/Scripts/python deploy/share.py --reload` rebuilds the web app,
  restarts only the server (~15 s) and re-publishes the fixed-address page. Testers' pages reconnect by themselves
  with their profile; only the conversation context of that moment restarts.
- It runs a separate public-mode server on `127.0.0.1:8010`. Your own setup (`:8000` + Vite `:5173`) is untouched.
- Public mode means per-device limits (40 replies/min, 8 Lens photos/min, 40 Spark calls/min per signer),
  debug routes off, and your `anay` profile can't be opened from the link.
- **Spend cap:** `META_SPEND_CAP_USD=50` in `.env` covers everything this laptop serves ($10.9 spent on 3 Nov).
  Past it the app keeps working without Meta AI: sentences word by word, replies signed word by word from the
  app's clips and the ISLRTC dictionary, typed replies instead of voice. To be fully safe, also set a spend limit
  on the Meta billing page.
- Running the app with the laptop off would need a paid Hugging Face PRO account (Docker Spaces stopped being
  free). Everything for it is ready: `deploy/build_space.py` (Dockerfile, Space card, private-dataset sync of tester
  data, see `server/sync.py`), deploying to `AnayChauhan/isl-bridge-app`. One command once PRO is active. Then lower
  the laptop cap to $20 (the Space gets $30), and point the fixed address at the app Space.

Message to send:
> Please try ISL Bridge, an Indian Sign Language conversation app: https://anaychauhan-isl-bridge.static.hf.space.
> Allow the camera and read the 1-minute intro. Then sign to it, and tap "Wrong?" whenever it gets a sign wrong.
> Takes 5 minutes.

## 2. What a tester sees

1. **Welcome** (first visit only): what the app is, 5 steps to try, and a **"Help improve ISL Bridge"** checkbox
   (on by default): "save the hand and body keypoints of my signs (numbers only, never video) and my reports".
   Nothing is logged until they tap **Start** (that tap also unlocks the phone's voice).
2. **Their own private profile** (`t-xxxxxx`, kept on that device). Nobody else's taught signs affect them.
3. **Sign → sentence:** sign, lower the hands, and the phone speaks the sentence. Each sentence has **✏️ Wrong? Tell us what you
   signed**: they retype the wrong words (autocomplete from the 202 signs) and can add a note. The app learns the
   correction for them, and you get a labelled clip.
4. **Reply:** hold 🎤 and speak (live captions; the words show up 0.6 s after releasing, and the signs follow in
   4-10 s), or type ⌨️. A reply that could not be signed says so; tapping it retries.
5. **☰ menu / ⓘ:** How to test, 💬 Send feedback, Help improve (on/off), profile id, *start fresh*, *delete my
   data* (removes their profile, keypoints and reports from the laptop at once).
6. 🎓 **Learn** (ghost coach) for testers who don't know ISL yet.

Good things to ask testers to try: single words (HELLO, DOCTOR, WATER, PAIN), a short sentence (I · PAIN ·
HEAD), ✋ Teach one of their own signs and use it, a spoken reply, the visit summary (📋).

## 3. Read what they found

```
.venv-ml/Scripts/python ml/tester_report.py            # testers only; --all includes anay; --since 2026-11-04
```
It prints testers, signs logged, taught signs, the **most common mistakes (read as → really signed)**, notes, and
how many labelled clips were saved. Raw data:

| What | Where |
|---|---|
| "Wrong?" reports + notes (one JSON line each, with the sentence, top-3 per sign, the fix, recent conversation) | `data/feedback/<profile>.jsonl` |
| Keypoints of each corrected sign (labelled live data: train / evaluate on it) | `data/feedback/<profile>/*.npz` (`meant`, `span`, `top`) |
| Every sign of testers with "help improve" on | `data/live_log/<profile>/*.npz` |
| Taught signs | `data/users/<profile>.json` |
| Server log (errors, `[privacy]` deletions) | `data/share_server.log` |

## 4. Automated end-to-end checks (run before sending a new link)

```
.venv-ml/Scripts/python ml/test_features.py https://<link>.trycloudflare.com data   # 24 checks over the internet
.venv-ml/Scripts/python ml/test_features.py http://127.0.0.1:8000                   # your LAN server (20 checks)
.venv-ml/Scripts/python ml/test_ws.py --url ws://127.0.0.1:8010/ws/sign --clients 6 --fuzz --speed 2 --limit 60
.venv-ml/Scripts/python deploy/test_share.py                                        # share watchdog, 16 cases, 1 s
```
Check the link from this laptop with Python or PowerShell, not Git Bash `curl`: on this Wi-Fi, curl fails for HTTPS
(exit 35/28) while browsers, Python and PowerShell work.
`test_features.py` covers: health, hello/consent, sign → sentence, consent off = nothing logged, "Wrong?" with and
without keypoints, notes, malformed reports, undo, tap-to-correct, forget, summary, scenario switch, teaching a new
sign, cancel, Learn-page practice, delete my data, typed reply → ISL, per-device limits (incl. a forged
X-Forwarded-For), debug route off, and the protected `anay` profile. It uses real Muse Spark (a few cents).
Results on 3 Nov: 24/24 through the public link (twice in a row), 20/20 on the LAN server, 21/21 on the Space bundle
with Meta AI switched off; 6 clients + fuzzing: 0 dropped connections, 0 server errors, 48/60 first signs right.
Headless-Chrome journeys (scratchpad): welcome → sign → Wrong? → menu → typed reply 12/12 via the link and via
Vite; hold-to-speak → caption at 0.6 s → signs, plus the Learn page, 6/6. Later on 3 Nov, after the fixed address:
24/24 through the new link; link-change journey 9/9 (fixed address → app with a profile id → tunnel killed → "Reopen"
after 26 s → new link, same profile; a fresh visit meanwhile shows "offline for a moment" and then opens by itself);
watchdog 16/16; a real tunnel survived a 5-minute outage with the same link.

## 5. Known limits to tell testers

- About 200 signs (hospital + everyday). Signs outside the list come out as the closest one it knows.
- A new signer gets roughly 70-80% of signs right; regional variants and fast signing do worse. ✋ Teach (5 reps)
  or a few "Wrong?" corrections fix their own signs.
- Stand back so head, shoulders and both hands are visible; the app pauses and shows an outline if too close.
- After a sentence is spoken (0.5 s after the hands drop) use "Wrong?", not ⌫ undo.
- Speech → signs takes 4-10 s (Muse Spark puts it into ISL word order); captions show at once.
- Not tested on a real iPhone yet (only Chrome/Android-style headless tests). The iPhone microphone start was
  fixed blind (AudioContext created inside the tap).
- The app runs on Anay's laptop: when its Wi-Fi drops, the app is offline (the fixed address says so and waits).
  After a link change, testers see the welcome screen once more and make the "Help improve" choice again: to the
  browser, each new link is a new site. Their profile and taught signs carry over.
