"""
Standalone HTTP API for the TTS engine: POST /generate {text, emotion} ->
WAV audio bytes. Meant as a drop-in tier for any app's own TTS provider
chain — no assumptions about the caller's architecture.

Run:
    uvicorn server:app --host 0.0.0.0 --port 3100
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from inference.generate import TTSEngine

app = FastAPI(title="Sapph-TTS Handler (GPT-SoVITS)")

# Runs zero-shot against the stock base model the config already points to.
# Set gpt_weights_path/sovits_weights_path if you have a fine-tuned checkpoint.
engine = TTSEngine(config_path="GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml")


class GenerateRequest(BaseModel):
    text: str
    emotion: str = "neutral"


@app.on_event("startup")
def startup():
    engine.load()


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        audio_bytes = engine.generate(req.text, req.emotion)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": engine._pipeline is not None}
