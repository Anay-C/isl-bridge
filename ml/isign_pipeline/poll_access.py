"""Exit 0 (printing 'ACCESS GRANTED') once the HF token can read the gated iSign CSV. Never prints the token."""
import time, urllib.request, urllib.error
tok = next(l.split("=", 1)[1].strip().strip('"').strip("'") for l in open(r"C:\Meta Hackathon\.env", encoding="utf-8") if l.strip().startswith("HF_TOKEN="))
url = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/iSign_v1.1.csv"
for i in range(2880):  # up to ~24 h
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-99", "Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status in (200, 206):
                print("ACCESS GRANTED", flush=True)
                raise SystemExit(0)
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403):
            print("unexpected HTTP", e.code, flush=True)
    except Exception as e:  # network blip
        pass
    time.sleep(30)
print("TIMEOUT: no access after 6 h", flush=True)
