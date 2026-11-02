"""Picture cards (Muse Image): a simple pictogram next to a word the Deaf person may not know as a sign.

Many medical words have no ISL sign, or only a dictionary sign most Deaf people have never seen (X-ray, ultrasound,
inhaler...). Fingerspelling them in English does not help someone with little English, so the speech -> sign
playlist also shows a picture of the thing. Muse Spark decides which gloss items are picturable (muse.to_gloss,
"picture" = a short visual description); this module draws them with muse-image-1.0 ($0.01 each, ~8 s) and
caches them in data/pictures/<slug>.png, so each concept is generated once. Common hospital concepts are
pre-generated (python server/pictures.py) so a live reply never waits for a picture; a new concept is drawn in
the background and the phone shows it as soon as it exists.
"""
import asyncio
import base64
import re
from pathlib import Path

import muse

CACHE = Path(__file__).resolve().parent.parent / "data" / "pictures"
STYLE = ("Simple friendly flat pictogram of {what}. Thick dark outlines, a few soft flat colours, plain white "
         "background, centred, one object, no text, no letters, no numbers, no people signing.")
_tasks: dict[str, asyncio.Task] = {}
_sem = asyncio.Semaphore(4)


def slug(word):
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")[:60] or "x"


def url_for(word):
    return f"/pictures/{slug(word)}.png"


def exists(word):
    return (CACHE / f"{slug(word)}.png").exists()


async def _generate(word, what):
    dst = CACHE / f"{slug(word)}.png"
    async with _sem:
        if dst.exists():
            return dst
        r = await muse._client.with_options(timeout=60, max_retries=1).images.generate(
            model="muse-image-1.0", prompt=STYLE.format(what=what or word), n=1)
        muse._log_spend("image", 0.01, images=1)
        d = r.data[0]
        if d.b64_json:
            img = base64.b64decode(d.b64_json)
        else:
            import httpx
            async with httpx.AsyncClient(timeout=60) as c:
                img = (await c.get(d.url)).content
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(img)
        tmp.replace(dst)
        return dst


def common_key(word):
    """The COMMON entry for a word ("X-Ray", "x ray" -> "X-ray"), else None."""
    want = slug(word).replace("-", "")
    return next((k for k in COMMON if slug(k).replace("-", "") == want), None)


def ensure(word, what=""):
    """Start drawing `word` in the background if it is not cached yet. Returns its URL either way."""
    s = slug(word)
    if not exists(word) and muse._client is not None and (s not in _tasks or _tasks[s].done()):
        async def run():
            try:
                await _generate(word, what)
            except Exception as e:  # a missing picture must never break a reply
                print(f"[pictures] {word}: {e!r}"[:300], flush=True)
        _tasks[s] = asyncio.create_task(run())
    return url_for(word)


# Hospital / pharmacy concepts that are hard to sign or fingerspell: drawn once, up front.
COMMON = {
    "X-ray": "a chest X-ray film", "injection": "a medical syringe with a needle", "blood test": "a blood test tube with red blood and a small needle",
    "ultrasound": "an ultrasound scanner probe and screen", "MRI": "an MRI scanner machine", "CT scan": "a CT scanner ring machine",
    "ECG": "an ECG heart-rate line on a monitor", "blood pressure": "a blood pressure cuff with a gauge", "thermometer": "a digital thermometer",
    "stethoscope": "a stethoscope", "tablet": "a strip of tablets", "capsule": "two medicine capsules", "syrup": "a bottle of cough syrup with a measuring spoon",
    "ointment": "a tube of ointment cream", "inhaler": "an asthma inhaler", "drip": "an IV drip bag on a stand", "insulin": "an insulin pen",
    "bandage": "a rolled bandage", "plaster": "a plaster cast on an arm", "stitches": "stitches on a small wound", "wheelchair": "a wheelchair",
    "stretcher": "a hospital stretcher", "operation": "a surgical tray with a scalpel, scissors and forceps", "operation theatre": "an operation theatre with an operating table",
    "ward": "a hospital ward with beds", "ICU": "an intensive care bed with monitors", "emergency": "a hospital emergency entrance with a red cross",
    "reception": "a hospital reception desk", "pharmacy": "a pharmacy counter with medicine shelves", "prescription": "a doctor's prescription paper with a pen",
    "report": "a medical report on a clipboard", "appointment": "a calendar with a marked date", "token": "a queue token number slip",
    "fasting": "an empty plate with a crossed fork and spoon", "after food": "a plate of rice and curry, then an arrow, then one white medicine pill (eat first, then the pill)",
    "before food": "one white medicine pill, then an arrow, then a plate of rice and curry (the pill first, then eat)", "morning": "a rising sun", "afternoon": "a bright midday sun", "night": "a crescent moon and stars",
    "urine test": "a urine sample cup", "dentist": "a tooth and a dental mirror", "eye drops": "an eye drop bottle over an eye", "glasses": "a pair of spectacles",
    "hearing aid": "a behind-the-ear hearing aid", "vaccine": "a vaccine vial and syringe", "mask": "a medical face mask", "gloves": "medical gloves",
    "allergy": "a sneezing nose with pollen", "diabetes": "a glucose meter with a drop of blood", "pregnancy": "a pregnant woman's silhouette",
    "physiotherapy": "a knee with an exercise band", "X-ray room": "an X-ray machine in a room",
}


async def _pregenerate():
    todo = [(w, d) for w, d in COMMON.items() if not exists(w)]
    print(f"{len(COMMON) - len(todo)} cached, drawing {len(todo)} (~${0.01 * len(todo):.2f})", flush=True)
    for i, f in enumerate(asyncio.as_completed([_generate(w, d) for w, d in todo]), 1):
        try:
            p = await f
            print(f"[{i}/{len(todo)}] {p.name}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAILED {e!r}"[:200], flush=True)


if __name__ == "__main__":
    asyncio.run(_pregenerate())
