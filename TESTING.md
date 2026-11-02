# Testing ISL Bridge with other people

How to put ISL Bridge in front of testers, what they see, and how to read what they report.
(Last checked end to end on 3 Nov 2026: every check below passed through the public link.)

## 1. Share it (Anay)

Double-click **`deploy\share.bat`** (or run `.venv-ml/Scripts/python deploy/share.py`). After ~20 s it prints:

```
  ISL Bridge is online for testers:
      https://<four-random-words>.trycloudflare.com
```

Send that link. It works on any phone or laptop with a camera (Chrome, Safari, Edge), anywhere. No
certificate warning, no app install, no account.

- The link is served **from this laptop** through a Cloudflare quick tunnel (`data/tools/cloudflared.exe`, signed by
  Cloudflare). Keep the window open and the laptop on, plugged in and online. Closing the lid or a reboot takes the
  link down; run the script again and you get a **new** link (also written to `data/share_url.txt`).
- A watchdog restarts the server or the tunnel by itself if one dies (a new tunnel = a new link, printed again).
- **Stop sharing:** close the window, or double-click **`deploy\stop_share.bat`** (also works when the window is
  hidden).
- It runs a separate public-mode server on `127.0.0.1:8010`. Your own setup (`:8000` + Vite `:5173`) is untouched.
- Public mode means per-device limits (40 replies/min, 8 Lens photos/min, 40 Spark calls/min per signer),
  debug routes off, and your `anay` profile can't be opened from the link.
- **Spend cap:** `META_SPEND_CAP_USD=50` in `.env` covers everything this laptop serves ($10.9 spent on 3 Nov).
  Past it the app keeps working without Meta AI: sentences word by word, replies signed word by word from the
  app's clips and the ISLRTC dictionary, typed replies instead of voice. To be fully safe, also set a spend limit
  on the Meta billing page.
- A permanent link with the laptop off would need a paid Hugging Face PRO account (Docker Spaces stopped being
  free). Everything for it is ready: `deploy/build_space.py` (Dockerfile, Space card, private-dataset sync of tester
  data, see `server/sync.py`). One command once PRO is active. Then lower the laptop cap to $20 (the Space gets $30).

Message to send:
> Please try ISL Bridge, an Indian Sign Language conversation app: <link>. Allow the camera and read the
> 1-minute intro. Then sign to it, and tap "Wrong?" whenever it gets a sign wrong. Takes 5 minutes.

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
```
`test_features.py` covers: health, hello/consent, sign → sentence, consent off = nothing logged, "Wrong?" with and
without keypoints, notes, malformed reports, undo, tap-to-correct, forget, summary, scenario switch, teaching a new
sign, cancel, Learn-page practice, delete my data, typed reply → ISL, per-device limits (incl. a forged
X-Forwarded-For), debug route off, and the protected `anay` profile. It uses real Muse Spark (a few cents).
Results on 3 Nov: 24/24 through the public link (twice in a row), 20/20 on the LAN server, 21/21 on the Space bundle
with Meta AI switched off; 6 clients + fuzzing: 0 dropped connections, 0 server errors, 48/60 first signs right.
Headless-Chrome journeys (scratchpad): welcome → sign → Wrong? → menu → typed reply 12/12 via the link and via
Vite; hold-to-speak → caption at 0.6 s → signs, plus the Learn page, 6/6.

## 5. Known limits to tell testers

- About 200 signs (hospital + everyday). Signs outside the list come out as the closest one it knows.
- A new signer gets roughly 70-80% of signs right; regional variants and fast signing do worse. ✋ Teach (5 reps)
  or a few "Wrong?" corrections fix their own signs.
- Stand back so head, shoulders and both hands are visible; the app pauses and shows an outline if too close.
- After a sentence is spoken (0.5 s after the hands drop) use "Wrong?", not ⌫ undo.
- Speech → signs takes 4-10 s (Muse Spark puts it into ISL word order); captions show at once.
- Not tested on a real iPhone yet (only Chrome/Android-style headless tests). The iPhone microphone start was
  fixed blind (AudioContext created inside the tap).
