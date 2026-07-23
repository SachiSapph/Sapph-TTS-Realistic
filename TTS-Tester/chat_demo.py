"""
Chat+voice test harness for Sapph-TTS: type a message, get an LLM reply
(Gemini), hear it spoken through the TTS-Handler engine in a selected voice
and emotion. Exists to test the TTS pipeline with realistic, varied
conversational text instead of hand-picked one-off lines. PERSONA_PROMPT
below is a generic assistant persona written for this demo only, not
modeled on any specific character, swap it for your own.

Run via run.bat, or manually:
    uvicorn chat_demo:app --host 127.0.0.1 --port 3001
Then open http://127.0.0.1:3001 and set your Gemini API key from the
settings (gear icon), saved locally to settings.local.json, never
committed (see .gitignore).
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "settings.local.json"

# TTS-Handler is a sibling project (the engine), not a subpackage of this
# tester, add it to sys.path rather than merging the two together, so
# each half stays independently usable.
TTS_HANDLER_ROOT = PROJECT_ROOT.parent / "TTS-Handler"
sys.path.insert(0, str(TTS_HANDLER_ROOT))

from emotion_vectors.presets import PRESETS  # noqa: E402
from inference import enhance  # noqa: E402
from inference.generate import TTSEngine  # noqa: E402
from voices import list_voices  # noqa: E402

# The free tier gives each model its own separate, fairly small daily quota
# (e.g. 20 requests/day for gemini-2.5-flash-lite), normal chat testing
# burns through that fast. Tried in order; a 429/503 on one moves to the
# next rather than failing the whole request. Fastest/newest first, since
# newer models tend to have fresher (less-competed-for) quota.
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]

PERSONA_PROMPT = (
    "You are a helpful voice assistant demonstrating a text-to-speech "
    "engine. Keep replies short and conversational (1-3 sentences), like a "
    "real chat, not an essay. Match the requested emotional tone in your "
    "wording, not just your punctuation. If a reaction like a laugh, sigh, "
    "gasp, or grumble fits, write it phonetically as part of the sentence, "
    "never as a stage direction like '*laughs*' or 'sighs', since the "
    "voice engine will pronounce those as literal words instead of making "
    "the sound. Good phonetic spellings the voice engine handles well: "
    "laugh -> 'Hahaha!' or 'Heheh.'; sigh -> 'Haaah...' or 'Phew...'; "
    "surprise/fear -> 'Eek!' or 'Ah!'; disgust/annoyance -> 'Ugh.', "
    "'Eww...', 'Pfft.', or 'Hmph.'. For high-energy "
    "tones (excited, angry, afraid, playful), break your reply into short, "
    "separate sentences with real full stops rather than one long run-on "
    "sentence, the voice engine adds a natural pause between sentences, "
    "which reads as more clipped/urgent. For calmer tones (sad, neutral), "
    "fewer, longer sentences read as more measured."
)

AUTO_EMOTION_NAMES = list(PRESETS.keys())


class EmotionSegment(BaseModel):
    text: str
    emotion: str


class AutoReply(BaseModel):
    segments: list[EmotionSegment]


app = FastAPI(title="Sapph-TTS Chat Demo")

gemini_client: genai.Client | None = None
engine = TTSEngine(config_path="GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml")
# Deliberately in-memory only, never written to disk: this is a test
# harness, not a product with a real privacy policy, so nothing about a
# chat session should outlive the server process. Restarting the server
# (or /clear) is the only way to reset it.
conversation_history: list[dict] = []
_last_audio: bytes = b""


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return {}


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def init_gemini_client(api_key: str | None) -> None:
    global gemini_client
    gemini_client = genai.Client(api_key=api_key) if api_key else None


@app.on_event("startup")
def startup():
    settings = load_settings()
    init_gemini_client(settings.get("gemini_api_key"))
    engine.load()
    # enhance.load() intentionally not called, see the comment in /chat
    # for why VoiceFixer is off by default now.
    # Pay GPT-SoVITS's one-time CUDA warm-up cost (~5s) now, not on the
    # user's first real message.
    engine.generate("Warming up.", emotion="neutral")


class ChatRequest(BaseModel):
    message: str
    emotion: str = "neutral"
    voice: str = "kokoro_female"


class ChatResponse(BaseModel):
    reply_text: str
    emotion_used: str
    audio_url: str


class SettingsRequest(BaseModel):
    gemini_api_key: str | None = None
    default_emotion: str = "neutral"
    default_voice: str = "kokoro_female"


@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "chat_demo_static" / "index.html")


@app.get("/emotions")
def list_emotions():
    return {"emotions": ["auto"] + list(PRESETS.keys())}


@app.get("/voices")
def get_voices():
    return {"voices": list(list_voices().keys())}


class TestVoiceRequest(BaseModel):
    voice: str
    emotion: str = "neutral"


TEST_VOICE_LINE = "Hi there, this is a quick test of this voice and tone."


@app.post("/test_voice")
def test_voice(req: TestVoiceRequest):
    """Generates a short canned line in the given voice/emotion, no LLM call
    involved, so a voice (default or freshly-uploaded) can be previewed
    without needing a Gemini API key configured yet."""
    try:
        audio_bytes = engine.generate(TEST_VOICE_LINE, emotion=req.emotion, voice=req.voice)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/settings")
def get_settings():
    settings = load_settings()
    key = settings.get("gemini_api_key", "")
    return {
        "gemini_api_key_masked": ("..." + key[-4:]) if key else "",
        "gemini_api_key_set": bool(key),
        "default_emotion": settings.get("default_emotion", "auto"),
        "default_voice": settings.get("default_voice", "kokoro_female"),
    }


@app.post("/settings")
def update_settings(req: SettingsRequest):
    settings = load_settings()
    if req.gemini_api_key:
        settings["gemini_api_key"] = req.gemini_api_key
    settings["default_emotion"] = req.default_emotion
    settings["default_voice"] = req.default_voice
    save_settings(settings)
    init_gemini_client(settings.get("gemini_api_key"))
    return {"status": "saved"}


def _call_gemini_with_fallback(contents: str, config: types.GenerateContentConfig | None = None):
    """Tries each model in GEMINI_MODELS in order; a quota/overload error
    (429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE) moves to the next model
    instead of failing the request. Raises the last error if all are down.
    """
    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            return gemini_client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except errors.ClientError as e:
            if e.code in (429, 503):
                last_error = e
                continue
            raise
        except errors.ServerError as e:
            last_error = e
            continue
    raise RuntimeError(
        f"All {len(GEMINI_MODELS)} Gemini models are rate-limited or unavailable "
        f"right now. Last error: {last_error}"
    )


def _emotion_label(segments: list[tuple[str, str]]) -> str:
    """e.g. [('Yes!', 'happy'), ('...I am wiped though.', 'exhausted')] ->
    'happy+exhausted'. A single-segment reply just returns that one name."""
    seen: list[str] = []
    for _, emotion in segments:
        if emotion not in seen:
            seen.append(emotion)
    return "+".join(seen)


def _generate_reply(emotion: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (reply_text, segments). segments is an ordered list of
    (text, emotion) pairs, more than one only when auto mode's own judgment
    calls for a tone shift within the reply, see PERSONA_PROMPT's
    instructions below for the restraint this is meant to have."""
    history_lines = [f"{t['role']}: {t['text']}" for t in conversation_history[-10:]]

    if emotion == "auto":
        prompt = "\n".join(
            [
                PERSONA_PROMPT,
                "Also decide how to split your reply into one or more "
                "emotionally-tagged segments for speech. Most replies "
                "should be a SINGLE segment in a single tone, most short "
                "conversational replies carry one consistent mood the "
                "whole way through. Only use multiple segments when your "
                "reply genuinely shifts tone or blends distinct feelings "
                "across its own sentences (e.g. relief giving way to "
                "exhaustion, excitement undercut by nervousness), that "
                "should be occasional and true to the content, not a "
                "default or a gimmick used every reply. Concatenating "
                "every segment's text in order, joined by a single space, "
                "must exactly reproduce your full reply, don't drop or "
                "duplicate any of it. Each segment picks exactly one tone "
                f"from: {', '.join(AUTO_EMOTION_NAMES)}.",
                *history_lines,
            ]
        )
        response = _call_gemini_with_fallback(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AutoReply,
            ),
        )
        parsed: AutoReply = response.parsed
        segments = [
            (seg.text.strip(), seg.emotion if seg.emotion in PRESETS else "neutral")
            for seg in parsed.segments
            if seg.text.strip()
        ]
        if not segments:
            segments = [("...", "neutral")]
        reply_text = " ".join(text for text, _ in segments)
        return reply_text, segments

    prompt = "\n".join(
        [PERSONA_PROMPT, f"(Reply in a {emotion} tone.)", *history_lines]
    )
    response = _call_gemini_with_fallback(prompt)
    reply_text = response.text.strip()
    return reply_text, [(reply_text, emotion)]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if gemini_client is None:
        raise HTTPException(
            status_code=400,
            detail="No Gemini API key configured yet, set one in Settings (gear icon).",
        )

    conversation_history.append({"role": "user", "text": req.message})

    try:
        reply_text, segments = _generate_reply(req.emotion)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {e}")

    conversation_history.append({"role": "assistant", "text": reply_text})
    emotion_used = _emotion_label(segments)

    try:
        audio_bytes = engine.generate_multi(segments, voice=req.voice)
        # VoiceFixer is intentionally NOT applied here, measured directly:
        # once the reference clip itself is properly denoised, raw output
        # has a cleaner noise floor (315x peak-to-floor ratio) than
        # VoiceFixer-processed output (179x). It was adding artifacts, not
        # removing them, for this source. Keep inference/enhance.py around
        # in case a future lower-quality source needs it, but don't call it
        # by default.
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    global _last_audio
    _last_audio = audio_bytes
    return ChatResponse(reply_text=reply_text, emotion_used=emotion_used, audio_url="/last_audio")


@app.get("/last_audio")
def last_audio():
    if not _last_audio:
        raise HTTPException(status_code=404, detail="No audio generated yet")
    return Response(content=_last_audio, media_type="audio/wav")


@app.post("/clear")
def clear_history():
    conversation_history.clear()
    return {"status": "cleared"}
