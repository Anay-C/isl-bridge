"""One-shot check of every Meta Model API call the app makes. Run right after adding MODEL_API_KEY to .env:

    .venv-ml/Scripts/python server/smoke_meta.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
import muse  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # Hindi output on the Windows console

WAV = ROOT / "data" / "test_speech.wav"  # "Hello. Do you have a fever? Please take this medicine tomorrow morning."
VOCAB = {"Hello", "you", "Fever", "Medicine", "Tomorrow", "Morning", "Pain", "Doctor", "What", "Where", "When"}


def show(name, out):
    err = out.pop("_error", None) if isinstance(out, dict) else None
    print(f"\n== {name}: {'FAIL ' + err if err else 'ok'}")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:900])
    return err is None


async def main():
    if not muse.KEY:
        sys.exit("MODEL_API_KEY is empty in .env")
    ok = True
    cands = [[{"label": "Head", "p": 0.5}, {"label": "Hello", "p": 0.3}, {"label": "Hot", "p": 0.2}],
             [{"label": "Pain", "p": 0.6}, {"label": "Bad", "p": 0.3}, {"label": "Sick", "p": 0.1}],
             [{"label": "Yesterday", "p": 0.7}, {"label": "Tomorrow", "p": 0.2}, {"label": "Today", "p": 0.1}]]
    ok &= show("Muse Spark: sentence from top-3 candidates",
               await muse.build_sentence(cands, ["Hearing person: What is the problem?"], "hospital"))
    try:
        ok &= show("Muse Spark: ISL gloss", await muse.to_gloss("Do you have fever since yesterday?", VOCAB, "hospital"))
    except Exception as e:
        ok = show("Muse Spark: ISL gloss", {"_error": repr(e)}) and False
    ok &= show("Muse Spark: visit summary", await muse.summarise(
        ["Hearing person: What is the problem?", "Deaf person: I have a headache since yesterday.",
         "Hearing person: Any fever?", "Deaf person: Yes, a little fever."], "hospital"))
    try:
        asr, ms = await muse.transcribe_wav(WAV.read_bytes(), keywords=sorted(VOCAB))
        ok &= show(f"Voice Transcribe ({ms} ms)", asr)
    except Exception as e:
        body = getattr(getattr(e, "response", None), "text", "")
        ok = show("Voice Transcribe", {"_error": f"{e!r} {body[:400]}"}) and False
    print("\nALL OK" if ok else "\nSOME CALLS FAILED (see above)")


asyncio.run(main())
