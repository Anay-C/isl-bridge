"""Simulated-clock tests of deploy/share.py's watchdog: when does the link change, when does it stay?

    .venv-ml/Scripts/python deploy/test_share.py      # no network, no processes; ~1 s
"""
import importlib.util
import sys
import time as real_time
from pathlib import Path

spec = importlib.util.spec_from_file_location("share", Path(__file__).resolve().parent / "share.py")
share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share)


class Stop(Exception):
    pass


class Clock:
    def __init__(self, end):
        self.t, self.end = 0.0, end

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.t > self.end:
            raise Stop

    def strftime(self, fmt, *a):
        return real_time.strftime(fmt, *a)


class FakeServer:
    def __init__(self, world):
        self.world, self.starts = world, 0

    def alive(self):
        return not self.world.server_dead(self.world.clock.t) or self.starts > 0

    def start(self):
        self.starts += 1
        self.world.logs.append((self.world.clock.t, "server start"))


class FakeTunnel:
    def __init__(self, world):
        self.world, self.n, self.url, self.dead = world, 0, "https://first-link.trycloudflare.com", False

    def start(self):
        if not self.world.online(self.world.clock.t):
            raise RuntimeError("cloudflared gave no link")
        self.n += 1
        self.dead = False
        self.url = f"https://link-{self.n}.trycloudflare.com"
        return self.url

    def running(self):
        return not self.dead and not self.world.cf_dies(self.world.clock.t, self.n)

    def alive(self):
        return self.running() and not self.world.deleted(self.world.clock.t, self.n)

    def stop(self):
        self.dead = True


class World:
    def __init__(self, end, online=lambda t: True, link_up=lambda t, url: True, cf_dies=lambda t, n: False,
                 server_dead=lambda t: False, deleted=lambda t, n: False):
        self.clock, self.logs, self.announced = Clock(end), [], []
        self.online, self.link_up, self.cf_dies, self.server_dead = online, link_up, cf_dies, server_dead
        self.deleted = deleted

    def run(self):
        share.time = self.clock
        share.online = lambda timeout=5: self.online(self.clock.t)
        share.get = lambda url, timeout=8: 200 if self.online(self.clock.t) and self.link_up(self.clock.t, url) \
            else None
        share.announce = lambda url: self.announced.append((self.clock.t, url))
        share.log = lambda m: self.logs.append((self.clock.t, m))
        self.srv, self.tun = FakeServer(self), FakeTunnel(self)
        try:
            share.watch(self.srv, self.tun)
        except Stop:
            pass
        return self


results = []


def check(name, cond, detail=""):
    cond = bool(cond)
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


# A. 8-minute Wi-Fi outage (like 3 Nov 09:20); cloudflared reconnects 40 s after the internet returns
w = World(1200, online=lambda t: not 100 <= t < 580,
          link_up=lambda t, url: url.startswith("https://first-link.") and not 100 <= t < 620).run()
msgs = [m for _, m in w.logs]
check("A outage: the link is kept (no new tunnel)", not w.announced, w.announced)
check("A outage: says it lost the internet, once", sum("lost its internet" in m for m in msgs) == 1, msgs)
check("A outage: says the link works again", any("works again" in m for m in msgs), msgs)

# B. internet fine, but the link stops answering for good at t=100 -> one new tunnel after ~2 min
w = World(900, link_up=lambda t, url: not url.startswith("https://first-link.") or t < 100).run()
check("B dead link: exactly one new tunnel", len(w.announced) == 1, w.announced)
check("B dead link: replaced 2-2.5 min after it died", w.announced and 200 <= w.announced[0][0] <= 250,
      w.announced)

# C. cloudflared crashes at t=100 -> new tunnel at the next check
w = World(600, cf_dies=lambda t, n: n == 0 and t >= 100).run()
check("C cloudflared died: new tunnel within 25 s", len(w.announced) == 1 and w.announced[0][0] <= 125,
      w.announced)

# D. cloudflared dies during a 5-minute outage -> waits, then a new link soon after the internet returns
w = World(1200, online=lambda t: not 100 <= t < 400, cf_dies=lambda t, n: n == 0 and t >= 100).run()
check("D died while offline: one new link", len(w.announced) == 1, w.announced)
check("D died while offline: link within 30 s of the internet returning",
      w.announced and 400 <= w.announced[0][0] <= 430, w.announced)
check("D died while offline: said it is waiting for the internet",
      any("waiting for it to come back" in m for _, m in w.logs), w.logs)

# E. server dies at t=50 -> restarted within ~2 s, same link
w = World(300, server_dead=lambda t: t >= 50).run()
starts = [t for t, m in w.logs if m == "server start"]
check("E server died: restarted within 3 s, link unchanged", starts and starts[0] <= 53 and not w.announced,
      (starts, w.announced))

# F. flaky link (every other check fails) never triggers a new tunnel
w = World(1200, link_up=lambda t, url: int(t // 20) % 2 == 0).run()
check("F flaky link: no new tunnel", not w.announced, w.announced)

# H. 14-minute outage; Cloudflare deleted the tunnel ("Tunnel not found" right after the internet returns)
w = World(1500, online=lambda t: not 100 <= t < 940,
          link_up=lambda t, url: url.startswith("https://first-link.") and t < 100 or not url.startswith("https://first-link."),
          deleted=lambda t, n: n == 0 and t >= 945).run()
msgs = [m for _, m in w.logs]
check("H tunnel deleted: one new link", len(w.announced) == 1, w.announced)
check("H tunnel deleted: new link within 25 s of 'Tunnel not found'", w.announced and w.announced[0][0] <= 970,
      w.announced)
check("H tunnel deleted: says Cloudflare deleted it", any("deleted the tunnel" in m for m in msgs), msgs)

# G. the link regex ignores cloudflared's error message when offline
err = 'failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": dial tcp: no such host'
box = "2026-11-03T03:59:31Z INF |  https://sitting-bye-celebs-legend.trycloudflare.com                    |"
check("G regex: ignores api.trycloudflare.com", share.URL_RE.search(err) is None)
check("G regex: finds the real link",
      (m := share.URL_RE.search(box)) and m.group(0) == "https://sitting-bye-celebs-legend.trycloudflare.com")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
