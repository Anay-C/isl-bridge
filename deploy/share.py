"""Share ISL Bridge with testers from this laptop: one public https link (Cloudflare quick tunnel).

    .venv-ml/Scripts/python deploy/share.py              # build the web app, start server + tunnel, print the link
    .venv-ml/Scripts/python deploy/share.py --no-build   # reuse web/dist
    .venv-ml/Scripts/python deploy/share.py --reload     # new code, SAME link: rebuild, restart only the server
    .venv-ml/Scripts/python deploy/share.py --stop       # take the link offline (= deploy/stop_share.bat)

- A public-mode server (ISL_PUBLIC=1: per-device rate limits, debug routes off, the $50 spend cap from .env)
  on 127.0.0.1:8010, serving the built web app. Only the tunnel reaches it. Your own setup (port 8000 + Vite on
  5173) is untouched; both write to the same data/ folder (profiles, reports, logs, spend ledger).
- cloudflared (data/tools/cloudflared.exe, signed by Cloudflare, Inc.) quick tunnel: no account, a random
  https://<words>.trycloudflare.com address that lives as long as this script runs. It is printed and written to
  data/share_url.txt (every link it has given, with times: data/share_links.log).
- Watchdog: restarts the server within seconds if it dies (same link). If the laptop's internet drops, it keeps
  the tunnel and waits: cloudflared reconnects by itself when the internet is back. It only starts a new tunnel
  (= a NEW link, printed again) when cloudflared has died, or when the internet works but the link has still
  not answered for 2 minutes.
- Keep the laptop on, plugged in and online. Its power plan never sleeps on its own (checked 3 Nov 2026), but
  closing the lid usually does (Windows default), which takes the link offline until it wakes up.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PY = sys.executable
CF = DATA / "tools" / "cloudflared.exe"
URL_FILE = DATA / "share_url.txt"
LINKS_LOG = DATA / "share_links.log"
# the link cloudflared prints - not api.trycloudflare.com, which shows up in its error messages when offline
URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")
CHECK_S = 20            # seconds between checks that the link answers
NEW_TUNNEL_AFTER = 6    # failed checks while the internet works (~2 min) before giving up on a link


def log(msg):
    print(f"[share {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url, timeout=8):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except Exception:
        return None


def health(base, timeout=8):
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def online(timeout=5):
    """Can this laptop reach Cloudflare at all? False while its Wi-Fi / internet is down."""
    try:
        socket.create_connection(("api.trycloudflare.com", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def version():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    return f"{sha.strip() or 'nogit'}-{time.strftime('%m%d-%H%M')}"


def build_web():
    log("building the web app ...")
    subprocess.run("npm run build", cwd=ROOT / "web", shell=True, check=True, stdout=subprocess.DEVNULL)


class Server:
    def __init__(self, port, device):
        self.port, self.device, self.p = port, device, None

    def start(self):
        env = {**os.environ, "ISL_PUBLIC": "1", "ISL_VERSION": version(), "ISL_DEVICE": self.device,
               "PYTHONUNBUFFERED": "1"}
        out = open(DATA / "share_server.log", "a", encoding="utf-8")
        self.p = subprocess.Popen([PY, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port",
                                   str(self.port), "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"],
                                  cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        for _ in range(120):
            if get(f"http://127.0.0.1:{self.port}/api/health") == 200:
                log(f"server up on 127.0.0.1:{self.port} (public mode, log: data/share_server.log)")
                return
            if self.p.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError("the server did not start - see data/share_server.log")

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.terminate()


class Tunnel:
    def __init__(self, port):
        self.port, self.p, self.url = port, None, None

    def start(self):
        """Start cloudflared; return its link (once it answers, or after ~1.5 min). RuntimeError if no link."""
        self.url = None
        out = open(DATA / "share_tunnel.log", "a", encoding="utf-8")
        self.p = subprocess.Popen([str(CF), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace")

        def read():  # keep draining the pipe (a full pipe would block cloudflared)
            for line in self.p.stdout:
                out.write(line)
                out.flush()
                m = URL_RE.search(line)
                if m and not self.url:
                    self.url = m.group(0)
        threading.Thread(target=read, daemon=True).start()
        for _ in range(120):
            if self.url or self.p.poll() is not None:
                break
            time.sleep(0.5)
        if not self.url:
            self.stop()
            raise RuntimeError("cloudflared gave no link (see data/share_tunnel.log)")
        for _ in range(30):  # the new name takes a few seconds to start answering
            if get(f"{self.url}/api/health", timeout=5) == 200:
                break
            time.sleep(2)
        return self.url

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.terminate()


def announce(url):
    URL_FILE.write_text(url + "\n", encoding="utf-8")
    with open(LINKS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {url}\n")
    bar = "=" * 72
    print(f"\n{bar}\n  ISL Bridge is online for testers:\n\n      {url}\n\n"
          f"  Open it on a phone (Chrome / Safari) or a laptop with a camera. Message to send:\n"
          f"  \"Please try ISL Bridge, an Indian Sign Language conversation app: {url}\n"
          f"   Allow the camera, read the 1-minute intro, then sign and tap 'Wrong?' when it gets a sign wrong.\"\n"
          f"  (also saved in data/share_url.txt; reports: data/feedback/, summary: ml/tester_report.py)\n"
          f"  Stop sharing: close this window, or double-click deploy\\stop_share.bat\n{bar}\n", flush=True)


def open_tunnel(tun):
    """Start a tunnel and announce its link. Never gives up: waits out an offline laptop, then retries at once."""
    wait = 15
    while True:
        try:
            return announce(tun.start())
        except RuntimeError as e:
            if not online():
                log("no link yet: this laptop has no internet - waiting for it to come back")
                while not online():
                    time.sleep(10)
                continue
            log(f"no link yet ({e}) - trying again in {wait} s")
            time.sleep(wait)
            wait = min(wait * 2, 300)


def watch(srv, tun):
    """Keep the server and the link up (see the module docstring for when the link changes)."""
    fails, offline, last = 0, False, time.time()
    while True:
        time.sleep(2)
        if not srv.alive():
            log("server stopped - restarting it (the link stays the same)")
            try:
                srv.start()
            except RuntimeError as e:
                log(f"{e} - trying again")
                time.sleep(10)
            continue
        if time.time() - last < CHECK_S:
            continue
        last = time.time()
        if not tun.alive():
            log("cloudflared stopped - starting a new tunnel (the link changes)")
            open_tunnel(tun)
            fails, offline = 0, False
        elif get(f"{tun.url}/api/health", timeout=15) == 200:
            if fails or offline:
                log(f"the link works again: {tun.url}")
            fails, offline = 0, False
        elif not online():
            if not offline:
                log("this laptop lost its internet connection - keeping the tunnel; the same link should "
                    "come back by itself when the internet does")
            fails, offline = 0, True
        else:
            fails += 1
            if fails >= NEW_TUNNEL_AFTER:
                log("the internet works but the link has not answered for 2 minutes - starting a new tunnel "
                    "(the link changes)")
                tun.stop()
                time.sleep(2)
                open_tunnel(tun)
                fails, offline = 0, False


def reload(port, build):
    """New code, same link: rebuild the web app, stop only the server; the running watchdog restarts it."""
    url = URL_FILE.read_text(encoding="utf-8").strip() if URL_FILE.exists() else ""
    local = f"http://127.0.0.1:{port}"
    if not url or not health(local):
        sys.exit("nothing is sharing - start it with deploy\\share.bat")
    if build:
        build_web()
    ps = (f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
          "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    log("old server stopped; waiting for the watchdog to start the new code ...")
    time.sleep(3)
    for _ in range(60):
        h = health(local)
        if h:
            seen = health(url, timeout=15)
            log(f"reloaded: version {h.get('version')}; the link is unchanged: {url}"
                + ("" if seen else " (it did not answer from here just now - check again in a minute)"))
            return
        time.sleep(2)
    sys.exit("the server did not come back within 2 minutes - see data/share_server.log")


# share.py + its server + tunnel (taskkill /T = the whole tree), then anything still on the port
STOP_PS = r"""
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'deploy[\\/]share\.(py|bat)' -and $_.CommandLine -notmatch '--stop' } | ForEach-Object { taskkill /F /T /PID $_.ProcessId | Out-Null; $_.ProcessId }
Get-NetTCPConnection -LocalPort PORT -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue; $_.OwningProcess }
Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | Where-Object { $_.CommandLine -match '127\.0\.0\.1:PORT' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }
"""


def stop(port):
    """Stop a running share from anywhere (its window may be hidden). Windows only."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", STOP_PS.replace("PORT", str(port))],
                         capture_output=True, text=True).stdout.split()
    URL_FILE.unlink(missing_ok=True)
    log(f"stopped {len(out)} process(es): the public link is offline" if out else "nothing was sharing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default=os.environ.get("ISL_DEVICE", "cpu"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--reload", action="store_true",
                    help="restart only the server of a running share with the current code (the link stays)")
    ap.add_argument("--stop", action="store_true", help="stop a running share (server + tunnel) and exit")
    a = ap.parse_args()
    if a.stop:
        return stop(a.port)
    if a.reload:
        return reload(a.port, build=not a.no_build)
    if not CF.exists():
        sys.exit(f"missing {CF}: download cloudflared-windows-amd64.exe from "
                 "https://github.com/cloudflare/cloudflared/releases and save it there")
    if not a.no_build:
        build_web()
    srv, tun = Server(a.port, a.device), Tunnel(a.port)
    srv.start()
    open_tunnel(tun)
    try:
        watch(srv, tun)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        tun.stop()
        srv.stop()
        URL_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
