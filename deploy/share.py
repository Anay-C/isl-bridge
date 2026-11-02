"""Share ISL Bridge with testers from this laptop: one public https link (Cloudflare quick tunnel).

    .venv-ml/Scripts/python deploy/share.py              # build the web app, start server + tunnel, print the link
    .venv-ml/Scripts/python deploy/share.py --no-build   # reuse web/dist

- A public-mode server (ISL_PUBLIC=1: per-device rate limits, debug routes off, the $50 spend cap from .env)
  on 127.0.0.1:8010, serving the built web app. Only the tunnel reaches it. Your own setup (port 8000 + Vite on
  5173) is untouched; both write to the same data/ folder (profiles, reports, logs, spend ledger).
- cloudflared (data/tools/cloudflared.exe, signed by Cloudflare, Inc.) quick tunnel: no account, a random
  https://<words>.trycloudflare.com address that lives as long as this script runs. It is printed and written to
  data/share_url.txt.
- Watchdog: restarts the server or the tunnel if one dies or the link stops answering (a new tunnel = a new
  link; it is printed again). Ctrl+C stops both; from anywhere: deploy/stop_share.bat (= share.py --stop).
- Keep the laptop on, plugged in and online. Its power plan never sleeps on its own (checked 3 Nov 2026), but
  closing the lid usually does (Windows default), which takes the link down until this script is restarted.
"""
import argparse
import os
import re
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
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def log(msg):
    print(f"[share {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url, timeout=8):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except Exception:
        return None


def version():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    return f"{sha.strip() or 'nogit'}-{time.strftime('%m%d-%H%M')}"


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
            raise RuntimeError("no tunnel address - see data/share_tunnel.log")
        for _ in range(60):  # the new name takes a few seconds to resolve
            if get(f"{self.url}/api/health") == 200:
                return self.url
            time.sleep(2)
        return self.url

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.terminate()


def announce(url):
    URL_FILE.write_text(url + "\n", encoding="utf-8")
    bar = "=" * 72
    print(f"\n{bar}\n  ISL Bridge is online for testers:\n\n      {url}\n\n"
          f"  Open it on a phone (Chrome / Safari) or a laptop with a camera. Message to send:\n"
          f"  \"Please try ISL Bridge, an Indian Sign Language conversation app: {url}\n"
          f"   Allow the camera, read the 1-minute intro, then sign and tap 'Wrong?' when it gets a sign wrong.\"\n"
          f"  (also saved in data/share_url.txt; reports: data/feedback/, summary: ml/tester_report.py)\n"
          f"  Stop sharing: close this window, or double-click deploy\\stop_share.bat\n{bar}\n", flush=True)


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
    ap.add_argument("--stop", action="store_true", help="stop a running share (server + tunnel) and exit")
    a = ap.parse_args()
    if a.stop:
        return stop(a.port)
    if not CF.exists():
        sys.exit(f"missing {CF}: download cloudflared-windows-amd64.exe from "
                 "https://github.com/cloudflare/cloudflared/releases and save it there")
    if not a.no_build:
        log("building the web app ...")
        subprocess.run("npm run build", cwd=ROOT / "web", shell=True, check=True, stdout=subprocess.DEVNULL)
    srv, tun = Server(a.port, a.device), Tunnel(a.port)
    srv.start()
    announce(tun.start())
    fails = 0
    try:
        while True:
            time.sleep(20)
            if not srv.alive():
                log("server stopped - restarting it")
                srv.start()
            if not tun.alive():
                log("tunnel stopped - starting a new one (the link changes)")
                announce(tun.start())
                fails = 0
                continue
            fails = 0 if get(f"{tun.url}/api/health", timeout=15) == 200 else fails + 1
            if fails >= 3:  # alive but unreachable for a minute: new tunnel
                log("the link stopped answering - starting a new tunnel (the link changes)")
                tun.stop()
                time.sleep(2)
                announce(tun.start())
                fails = 0
    except KeyboardInterrupt:
        log("stopping")
    finally:
        tun.stop()
        srv.stop()
        URL_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
