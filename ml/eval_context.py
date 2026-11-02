"""Does context-aware sentence building (differentiator #3) actually fix recogniser mistakes? Measured.

    python ml/eval_context.py preds          # real top-3 outputs of the live model on HELD-OUT clips, per sign
    python ml/eval_context.py gen --n 250    # Muse Spark writes hospital/pharmacy dialogues (Deaf turns as ISL gloss)
    python ml/eval_context.py eval           # decode every Deaf turn 3 ways + Spark judges the meaning
    python ml/eval_context.py report

Each Deaf turn is "signed" by sampling, for every gloss sign, one real top-3 output the model produced on a
held-out clip of that sign, so recogniser noise is the model's actual confusion pattern. Conditions:
  A  top-1 only      the recogniser's best guess per sign, words joined (no language model)
  B  Spark, no ctx   build_sentence(candidates) with no history, generic scenario
  C  Spark + ctx     build_sentence(candidates, conversation so far, scenario)   <- what the app does
Metrics: sign accuracy (order-aware LCS of chosen signs vs the true gloss, / gloss length) and meaning
preserved (Muse Spark as judge: produced sentence vs the intended sentence: same / partial / wrong).
All stages are resumable (jsonl files in data/context_eval/). Spend goes to data/meta_spend.json.
"""
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "context_eval"
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "server"))
os.environ.setdefault("META_SPEND_TAG", "eval_context")
BUDGET_USD = 15.0  # hard stop for this experiment

SETTINGS = [
    "hospital reception: registering, describing why they came", "doctor consultation about symptoms",
    "pharmacy: buying and asking about medicine", "emergency room, urgent problem",
    "follow-up visit: feeling better or worse since last time", "nurse giving medicine / injection on the ward",
    "booking the next appointment and paying the bill", "parent bringing a sick child to the doctor",
    "elderly patient with family member at the clinic", "asking where the toilet / bathroom / room is",
]


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def append_jsonl(p, rows):
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- 1. real recogniser outputs on held-out clips ------------------------------------------------
def stage_preds():
    import torch
    from features import npz_frames
    from recognizer import Recognizer
    ck = torch.load(ROOT / "data" / "model.pt", weights_only=False)
    rec = Recognizer(ROOT / "data" / "model.pt")
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(v["include_signs"]) | set(v["cislr_signs"])  # same as the server
    roots = {"include": ROOT / "data" / "keypoints", "cislr": ROOT / "data" / "cislr" / "keypoints"}
    keys = ck["test_include"] + ck["test_cislr"]
    embs = rec.embed_frames([npz_frames(roots[k.split("/")[0]] / k.split("/", 1)[1]) for k in keys])
    preds = {}
    for k, e in zip(keys, embs):
        top = rec.predict(e, k=3, personalise=False)
        preds.setdefault(k.split("/")[1], []).append([{"label": t["label"], "p": round(t["p"], 3)} for t in top])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds.json").write_text(json.dumps(preds, indent=0))
    n = sum(map(len, preds.values()))
    top1 = sum(p[0]["label"] == l for l, ps in preds.items() for p in ps) / n
    top3 = sum(any(c["label"] == l for c in p) for l, ps in preds.items() for p in ps) / n
    print(f"{n} held-out clips, {len(preds)} signs | top-1 {top1:.1%} top-3 {top3:.1%}")


# ---- 2. dialogues from Muse Spark ---------------------------------------------------------------
GEN_INSTRUCTIONS = """You write realistic short conversations at Indian hospitals, clinics and pharmacies between a
Deaf patient who signs Indian Sign Language (ISL) and a hearing person (receptionist, doctor, nurse, pharmacist).
For every Deaf turn give (a) "gloss": the ISL signs in ISL order (time first, topic before comment, question
words last, no articles or copula), using ONLY the allowed sign names, 1-5 signs, and (b) "english": exactly what
the Deaf person means, as one natural first-person sentence that follows from those signs and the conversation.
Hearing turns are plain English. Each conversation has 4-8 turns and at least 2 Deaf turns. Vary people, problems
and wording; keep it plausible for a real Indian clinic."""


def gen_schema(signs):
    return {
        "type": "object",
        "properties": {"conversations": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "turns": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["deaf", "hearing"]},
                        "english": {"type": "string"},
                        "gloss": {"type": "array", "items": {"type": "string", "enum": signs}},
                    },
                    "required": ["speaker", "english", "gloss"], "additionalProperties": False}},
            },
            "required": ["setting", "turns"], "additionalProperties": False}}},
        "required": ["conversations"], "additionalProperties": False,
    }


async def stage_gen(n, per_call=6, conc=4):
    import muse
    preds = json.loads((OUT / "preds.json").read_text())
    signs = sorted(preds)
    path = OUT / "dialogues.jsonl"
    have = len(load_jsonl(path))
    sem = asyncio.Semaphore(conc)
    schema = gen_schema(signs)

    async def one(i):
        async with sem:
            if muse.spent() > BUDGET_USD:
                return
            setting = SETTINGS[i % len(SETTINGS)]
            payload = {"setting": setting, "count": per_call, "allowed_signs": signs, "variation_seed": i}
            try:
                out = await muse._spark_json(GEN_INSTRUCTIONS, payload, "dialogues", schema, max_tokens=4000)
            except Exception as e:
                print("gen failed:", e)
                return
            rows = []
            for c in out["conversations"]:
                turns = [t for t in c["turns"] if t["speaker"] == "hearing" or t["gloss"]]
                if sum(t["speaker"] == "deaf" for t in turns) >= 1:
                    rows.append({"setting": setting, "turns": turns})
            append_jsonl(path, rows)
            print(f"call {i}: +{len(rows)} dialogues (spent ${muse.spent():.2f})", flush=True)

    calls = max(0, -(-(n - have) // per_call))
    await asyncio.gather(*(one(have + i) for i in range(calls)))
    print(f"{len(load_jsonl(path))} dialogues in {path}")


# ---- 3. decode Deaf turns 3 ways + judge ---------------------------------------------------------
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


JUDGE_INSTRUCTIONS = """You grade a sign-language translation app. For each item, compare the app's output with what
the Deaf person actually meant. "same" = the meaning a hearing person would understand is the same (wording may
differ); "partial" = main point right but a detail wrong or missing; "wrong" = the meaning is different or unclear.
Be strict about who, what, when, negation and questions."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"grades": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"},
                                         "grade": {"type": "string", "enum": ["same", "partial", "wrong"]}},
        "required": ["id", "grade"], "additionalProperties": False}}},
    "required": ["grades"], "additionalProperties": False,
}


async def stage_eval(limit, conc=4, seed=0):
    import muse
    preds = json.loads((OUT / "preds.json").read_text())
    dialogues = load_jsonl(OUT / "dialogues.jsonl")[:limit]
    path = OUT / "results.jsonl"
    done = {r["id"] for r in load_jsonl(path)}
    sem = asyncio.Semaphore(conc)
    norm = lambda w: w.strip().lower()

    async def dialogue(di, d):
        rng = random.Random(f"{seed}/{di}")
        history, rows = [], []
        for ti, t in enumerate(d["turns"]):
            if t["speaker"] == "hearing":
                history.append(f"Hearing person: {t['english']}")
                continue
            gloss = [g for g in t["gloss"] if g in preds]
            rid = f"{di}/{ti}"
            if not gloss:
                continue
            cands = [rng.choice(preds[g]) for g in gloss]
            top1 = [c[0]["label"] for c in cands]
            row = {"id": rid, "setting": d["setting"], "gloss": gloss, "intended": t["english"],
                   "in_top3": sum(any(c["label"] == g for c in cs) for g, cs in zip(gloss, cands)) / len(gloss),
                   "A": {"signs": top1, "text": " ".join(top1)}}
            if rid not in done:
                async with sem:
                    if muse.spent() > BUDGET_USD:
                        return rows
                    b = await muse.build_sentence(cands, [], "An everyday conversation.")
                    c = await muse.build_sentence(cands, history, f"A Deaf patient in India: {d['setting']}.")
                row["B"] = {"signs": b["chosen_signs"], "text": b["english"], "err": b.get("_error")}
                row["C"] = {"signs": c["chosen_signs"], "text": c["english"], "err": c.get("_error")}
                for k in "ABC":
                    row[k]["sign_acc"] = lcs([norm(x) for x in row[k]["signs"]], [norm(x) for x in gloss]) / len(gloss)
                rows.append(row)
                history.append(f"Deaf person: {c['english']}")  # the app's own output, as in real use
            else:
                history.append(f"Deaf person: {t['english']}")
        return rows

    tasks = [dialogue(i, d) for i, d in enumerate(dialogues)]
    for k, fut in enumerate(asyncio.as_completed(tasks), 1):
        rows = await fut
        append_jsonl(path, rows)
        if k % 10 == 0:
            print(f"{k}/{len(tasks)} dialogues (spent ${muse.spent():.2f})", flush=True)

    # judge every condition's output vs the intended meaning, 25 items per call
    rows = load_jsonl(path)
    jpath = OUT / "judged.jsonl"
    judged = {(j["id"], j["cond"]) for j in load_jsonl(jpath)}
    todo = [(r, c) for r in rows for c in "ABC" if (r["id"], c) not in judged and c in r]
    items = [{"id": f"{r['id']}|{c}", "meant": r["intended"], "app_output": r[c]["text"]} for r, c in todo]
    random.Random(seed).shuffle(items)  # judge never sees which condition an item came from

    async def judge(batch):
        async with sem:
            if muse.spent() > BUDGET_USD:
                return
            try:
                out = await muse._spark_json(JUDGE_INSTRUCTIONS, {"items": batch}, "grades", JUDGE_SCHEMA, max_tokens=2000)
            except Exception as e:
                print("judge failed:", e)
                return
            append_jsonl(jpath, [{"id": g["id"].split("|")[0], "cond": g["id"].split("|")[1], "grade": g["grade"]}
                                 for g in out["grades"] if "|" in g["id"]])

    await asyncio.gather(*(judge(items[i:i + 25]) for i in range(0, len(items), 25)))
    stage_report()


def stage_report():
    rows = {r["id"]: r for r in load_jsonl(OUT / "results.jsonl")}
    grades = {}
    for j in load_jsonl(OUT / "judged.jsonl"):
        grades[(j["id"], j["cond"])] = j["grade"]
    n = len(rows)
    rep = {"deaf_turns": n, "signs": sum(len(r["gloss"]) for r in rows.values()),
           "truth_in_top3": round(sum(r["in_top3"] for r in rows.values()) / n, 3)}
    names = {"A": "top-1 only", "B": "Muse Spark, no context", "C": "Muse Spark + context"}
    print(f"\n{n} Deaf turns, {rep['signs']} signs; true sign in recogniser top-3: {rep['truth_in_top3']:.1%}")
    for k, name in names.items():
        acc = sum(r[k]["sign_acc"] for r in rows.values()) / n
        g = [grades[(i, k)] for i in rows if (i, k) in grades]
        same = g.count("same") / len(g) if g else 0
        ok = (g.count("same") + g.count("partial")) / len(g) if g else 0
        rep[k] = {"name": name, "sign_accuracy": round(acc, 3), "meaning_same": round(same, 3),
                  "meaning_same_or_partial": round(ok, 3), "judged": len(g)}
        print(f"  {k} {name:24s} sign acc {acc:6.1%} | meaning same {same:6.1%} | same+partial {ok:6.1%} (n={len(g)})")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print("->", OUT / "report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preds", "gen", "eval", "report"])
    ap.add_argument("--n", type=int, default=250, help="gen: target number of dialogues")
    ap.add_argument("--limit", type=int, default=10_000, help="eval: max dialogues")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.stage == "preds":
        stage_preds()
    elif args.stage == "gen":
        asyncio.run(stage_gen(args.n))
    elif args.stage == "eval":
        asyncio.run(stage_eval(args.limit))
    else:
        stage_report()


if __name__ == "__main__":
    main()
