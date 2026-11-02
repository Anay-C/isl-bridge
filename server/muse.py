"""Meta Model API calls: Muse Spark (sentence building, ISL gloss) and Muse Voice Transcribe."""
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BASE = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
ASR = "muse-voice-transcribe-1.0"
KEY = os.getenv("MODEL_API_KEY", "")

# live conversation: fail fast and fall back rather than hang (SDK default is 600 s with 2 retries)
_client = AsyncOpenAI(api_key=KEY, base_url=BASE, timeout=40.0, max_retries=1) if KEY else None

# Spend ledger: every call adds its tokens / audio seconds here (data/meta_spend.json), so the hackathon
# credit can be tracked per purpose. Prices: https://dev.meta.ai/docs/pricing-rate-limits
PRICE_IN, PRICE_OUT, PRICE_ASR_HOUR = 1.25e-6, 4.25e-6, 0.18
LEDGER = Path(__file__).resolve().parent.parent / "data" / "meta_spend.json"
TAG = os.getenv("META_SPEND_TAG", "app")  # purpose label for the ledger (e.g. "eval_context")


def _log_spend(kind, usd, **counts):
    try:
        led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        e = led.setdefault(f"{TAG}/{kind}", {"calls": 0, "usd": 0.0})
        e["calls"] += 1
        e["usd"] = round(e["usd"] + usd, 6)
        for k, v in counts.items():
            e[k] = e.get(k, 0) + v
        led["_total_usd"] = round(sum(v["usd"] for k, v in led.items() if not k.startswith("_")), 4)
        tmp = LEDGER.with_suffix(".tmp")  # atomic: a full disk must not wipe the ledger
        tmp.write_text(json.dumps(led, indent=1))
        tmp.replace(LEDGER)
    except Exception:
        pass  # never break a conversation over bookkeeping


def spent():
    try:
        return json.loads(LEDGER.read_text()).get("_total_usd", 0.0)
    except Exception:
        return 0.0

SENTENCE_INSTRUCTIONS = """You are the voice of a Deaf person who signs Indian Sign Language (ISL) to a phone.
A sign recogniser gives you, for each sign in order, its top candidate words with confidences.
ISL grammar differs from English/Hindi (topic-comment, time first, question word last, no articles/copula).
Using the scenario and the conversation so far, choose the most sensible candidate for each sign
(prefer high confidence, but override when context makes another candidate clearly more plausible),
then say what the signer means as ONE short, natural, first-person sentence, as they would say it out loud.
Never add facts that are not signed. If signs are nonsense together, keep it literal and short.
Return both English and Hindi (Devanagari)."""

SENTENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_signs": {"type": "array", "items": {"type": "string"}},
        "english": {"type": "string"},
        "hindi": {"type": "string"},
        "is_question": {"type": "boolean"},
    },
    "required": ["chosen_signs", "english", "hindi", "is_question"],
    "additionalProperties": False,
}

GLOSS_INSTRUCTIONS = """Convert what a hearing person said (English, Hindi or mixed) into Indian Sign Language gloss
for a Deaf reader. Use ISL order: time words first, then topic, then comment; question words (WHAT, WHERE,
WHEN, WHO, HOW) at the end; drop articles, copulas and tense endings.
Each gloss item is ONE of: "sign" = a word from the app vocabulary (preferred, fastest); or "dictionary" = the id
of a single-sign entry of the ISLRTC ISL Dictionary, chosen from "dictionary_candidates" or found with the
search_dictionary tool (search in English, also for Hindi words); or "fingerspell" = names, numbers and words
with no sign. Leave the other two fields "". Only pick a dictionary entry whose meaning matches what was said
(e.g. not "Operation (Maths)" for surgery). Keep it short (<= 8 items).
"picture": a short visual description (e.g. "a chest X-ray film", "a strip of tablets") when the item is a
concrete thing a Deaf patient may not know as a sign and that can be drawn: medical tests and procedures,
equipment, medicine forms (tablet, syrup, injection, inhaler), hospital places, and medicine timing (morning,
night, before/after food). Otherwise "" (people, actions, feelings, question words, names, numbers).
Also give a clean English caption of what was said."""

GLOSS_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "gloss": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"sign": {"type": "string"}, "dictionary": {"type": "string"},
                               "fingerspell": {"type": "string"}, "picture": {"type": "string"}},
                "required": ["sign", "dictionary", "fingerspell", "picture"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["caption", "gloss"],
    "additionalProperties": False,
}


SUMMARY_INSTRUCTIONS = """You write a short visit note for a doctor or nurse from a conversation between a Deaf patient
(signing ISL through a translation app) and a hearing person at a clinic in India.
Use ONLY what the conversation says. Never guess a diagnosis or add symptoms, durations or medicines that
were not said; use "" or [] when something was not mentioned. Machine translation of signs can be wrong,
so list anything ambiguous under "to_confirm". Keep every field brief.
summary_hindi must be Hindi written in Devanagari script (not romanised)."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "chief_complaint": {"type": "string"},
        "symptoms": {"type": "array", "items": {
            "type": "object",
            "properties": {"symptom": {"type": "string"}, "since": {"type": "string"}, "severity": {"type": "string"}},
            "required": ["symptom", "since", "severity"], "additionalProperties": False}},
        "medicines_mentioned": {"type": "array", "items": {"type": "string"}},
        "allergies": {"type": "array", "items": {"type": "string"}},
        "to_confirm": {"type": "array", "items": {"type": "string"}},
        "summary_english": {"type": "string"},
        "summary_hindi": {"type": "string"},
    },
    "required": ["chief_complaint", "symptoms", "medicines_mentioned", "allergies", "to_confirm",
                 "summary_english", "summary_hindi"],
    "additionalProperties": False,
}


async def _spark_json(instructions, payload, name, schema, max_tokens=1000, attempts=2, tools=None, max_rounds=2):
    if _client is None:
        raise RuntimeError("MODEL_API_KEY not set in .env")
    t0 = time.perf_counter()
    tool_specs = [t["spec"] for t in tools] if tools else None
    rounds = 0
    impl = {t["spec"]["name"]: t["fn"] for t in tools} if tools else {}
    for i in range(attempts):  # empty / cut-off reply (reasoning used the token budget): retry with 2x budget
        inp = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for rnd in range(max_rounds + 1):  # tool calls: run them, feed results back, ask again
            kw = {"tools": tool_specs} if tool_specs and rnd < max_rounds else {}
            r = await _client.responses.create(
                model=SPARK,
                instructions=instructions,
                input=inp,
                reasoning={"effort": "minimal"},
                max_output_tokens=max_tokens * (2 ** i),
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                **kw,
            )
            u = r.usage
            if u is not None:
                _log_spend("spark", u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT,
                           input_tokens=u.input_tokens, output_tokens=u.output_tokens)
            calls = [o for o in r.output if o.type == "function_call"]
            if not calls:
                break
            rounds = rnd + 1
            inp += [o.model_dump(exclude_none=True) for o in r.output]
            for c in calls:
                try:
                    res = impl[c.name](**json.loads(c.arguments))
                except Exception as e:
                    res = {"error": str(e)}
                inp.append({"type": "function_call_output", "call_id": c.call_id,
                            "output": json.dumps(res, ensure_ascii=False)})
        try:
            out = json.loads(r.output_text)
            break
        except json.JSONDecodeError:
            if i == attempts - 1:
                raise RuntimeError(f"Muse Spark returned no JSON (status {r.status})")
    out["_latency_ms"] = round((time.perf_counter() - t0) * 1000)
    if tools:
        out["_tool_rounds"] = rounds
    return out


async def build_sentence(candidates, history, scenario):
    """candidates: [[{"label","p"},...] per sign]."""
    payload = {
        "scenario": scenario,
        "conversation_so_far": history[-6:],
        "signs": [[{"word": c["label"], "confidence": round(c["p"], 2)} for c in cands] for cands in candidates],
    }
    try:
        return await _spark_json(SENTENCE_INSTRUCTIONS, payload, "signed_sentence", SENTENCE_SCHEMA)
    except Exception as e:  # never leave the user without a sentence
        words = [c[0]["label"] for c in candidates if c]
        return {"chosen_signs": words, "english": " ".join(words), "hindi": "", "is_question": False,
                "_error": str(e)}


LENS_INSTRUCTIONS = """A learner of Indian Sign Language points a phone camera at something. Name the main object
they are showing (ignore the person holding it), as the simplest everyday English word a sign dictionary would
use (e.g. "medicine" for a pill strip, "phone" for a mobile, "water" for a water bottle), plus up to 3
alternative words from general to specific, and the Hindi word. If one of the app's signs names the object
(same meaning, e.g. "Cell phone" for a smartphone), set app_sign to it, else "". If no clear object is visible,
set object to ""."""


def lens_schema(app_signs):
    return {
        "type": "object",
        "properties": {"object": {"type": "string"}, "alternatives": {"type": "array", "items": {"type": "string"}},
                       "hindi": {"type": "string"}, "app_sign": {"type": "string", "enum": [""] + sorted(app_signs)}},
        "required": ["object", "alternatives", "hindi", "app_sign"], "additionalProperties": False,
    }


async def identify_object(image_data_url, app_signs=()):
    """Camera frame (data URL) -> {"object", "alternatives", "hindi", "app_sign"} via Muse Spark vision."""
    if _client is None:
        raise RuntimeError("MODEL_API_KEY not set in .env")
    t0 = time.perf_counter()
    r = await _client.responses.create(
        model=SPARK, instructions=LENS_INSTRUCTIONS, reasoning={"effort": "minimal"}, max_output_tokens=600,
        input=[{"role": "user", "content": [{"type": "input_text", "text": "App signs: " + ", ".join(sorted(app_signs))},
                                            {"type": "input_image", "image_url": image_data_url}]}],
        text={"format": {"type": "json_schema", "name": "lens", "schema": lens_schema(app_signs), "strict": True}})
    u = r.usage
    if u is not None:
        _log_spend("spark", u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT,
                   input_tokens=u.input_tokens, output_tokens=u.output_tokens)
    out = json.loads(r.output_text)
    out["_latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


DICT_TOOL = {"type": "function", "name": "search_dictionary", "strict": True,
             "description": "Search the ISLRTC Indian Sign Language dictionary (2,311 single-sign videos). "
                            "Query in English (one word or a short phrase). Returns entries with ids.",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                            "required": ["query"], "additionalProperties": False}}


async def to_gloss(text, vocab, scenario, dict_candidates=None, search=None):
    """search: fn(query) -> [{"entry","id"}]; enables Spark's search_dictionary tool."""
    payload = {"scenario": scenario, "said": text, "vocabulary": sorted(vocab),
               "dictionary_candidates": dict_candidates or {}}
    tools = [{"spec": DICT_TOOL, "fn": lambda query: {"results": search(query)}}] if search else None
    out = await _spark_json(GLOSS_INSTRUCTIONS, payload, "isl_gloss", GLOSS_SCHEMA, tools=tools)
    vocab = set(vocab)
    for g in out["gloss"]:  # never trust ids blindly
        if g.get("sign") and g["sign"] not in vocab:
            g["fingerspell"], g["sign"] = g.get("fingerspell") or g["sign"], ""
    out["gloss"] = [g for g in out["gloss"] if g.get("sign") or g.get("dictionary") or g.get("fingerspell")]
    return out


async def summarise(history, scenario):
    """Visit note for the doctor from the whole conversation ("Deaf person: ..." / "Hearing person: ...")."""
    payload = {"scenario": scenario, "conversation": history}
    try:
        return await _spark_json(SUMMARY_INSTRUCTIONS, payload, "visit_note", SUMMARY_SCHEMA, max_tokens=1500)
    except Exception as e:  # without Spark, still hand over the raw transcript
        return {"chief_complaint": "", "symptoms": [], "medicines_mentioned": [], "allergies": [], "to_confirm": [],
                "summary_english": "\n".join(history), "summary_hindi": "", "_error": str(e)}


class RealtimeASR:
    """Streaming Voice Transcribe: feed 16 kHz 16-bit mono PCM while the person talks; finish() returns the text.

    Protocol (dev.meta.ai/docs/api-reference/voice/realtime): JSON handshake with the key, binary PCM frames,
    {"type": "endStream"}, then transcript events until the server closes with 1000."""
    URL = "wss://api.meta.ai/v1/asr/realtime"

    def __init__(self, on_partial=None):
        self.on_partial = on_partial
        self.finals, self.partial, self.bytes = [], "", 0
        self.ws = self.reader = None
        self.error = None

    async def start(self, keywords=(), languages=("English", "Hindi")):
        import websockets
        if not KEY:
            raise RuntimeError("MODEL_API_KEY not set in .env")
        self.ws = await websockets.connect(self.URL, max_size=None, open_timeout=10)
        await self.ws.send(json.dumps({
            "authorization": {"accessToken": f"Bearer {KEY}"}, "audioEncoding": "PCM_16KHZ", "model": ASR,
            "mode": "PUSH_TO_TALK", "languageBias": list(languages), "keywords": list(keywords)[:100]}))
        ack = json.loads(await asyncio.wait_for(self.ws.recv(), 10))
        if "sessionId" not in ack:
            raise RuntimeError(f"realtime handshake failed: {str(ack)[:200]}")
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m.get("type") == "transcript":
                    if m.get("final"):
                        self.finals.append(m.get("transcript", ""))
                        self.partial = ""
                    else:
                        self.partial = m.get("transcript", "")
                    if self.on_partial:
                        await self.on_partial(" ".join(self.finals + [self.partial]).strip())
                elif m.get("type") == "error":
                    self.error = m.get("message")
        except Exception as e:  # closed early etc.
            self.error = self.error or str(e)

    async def send(self, pcm: bytes):
        self.bytes += len(pcm)
        await self.ws.send(pcm)

    async def finish(self, timeout=8):
        t0 = time.perf_counter()
        try:
            await self.ws.send(json.dumps({"type": "endStream"}))
            await asyncio.wait_for(self.reader, timeout)
        except Exception:
            pass
        finally:
            await self.ws.close()
        secs = self.bytes / 2 / 16000
        _log_spend("asr", secs / 3600 * PRICE_ASR_HOUR, audio_seconds=round(secs, 1))
        text = " ".join(self.finals + ([self.partial] if self.partial else [])).strip()
        return text, round((time.perf_counter() - t0) * 1000)


async def transcribe_wav(wav_bytes, keywords=(), languages=("English", "Hindi")):
    if not KEY:
        raise RuntimeError("MODEL_API_KEY not set in .env")
    req = {"mode": "PUSH_TO_TALK", "model": ASR, "audioEncoding": "WAV",
           "languageBias": list(languages), "keywords": list(keywords)[:100]}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/asr/transcribe", headers={"Authorization": f"Bearer {KEY}"},
                         files={"request": (None, json.dumps(req), "application/json"),
                                "audio": ("speech.wav", wav_bytes, "audio/wav")})
    r.raise_for_status()
    out = r.json()
    secs = out.get("audioDurationMs", 0) / 1000
    _log_spend("asr", secs / 3600 * PRICE_ASR_HOUR, audio_seconds=round(secs, 1))
    return out, round((time.perf_counter() - t0) * 1000)
