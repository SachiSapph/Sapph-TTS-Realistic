# TTS-Tester

A chat+voice demo for exercising [`../TTS-Handler`](../TTS-Handler): type a
message, get an LLM reply (Google Gemini), hear it spoken back in whichever
voice and emotion you pick from the two dropdowns. This is what makes the
engine's behavioral side visible: pick `auto` and the LLM chooses the
emotional tone per reply itself based on the conversation, switching
naturally instead of staying locked to one setting, or pin a specific
emotion by hand. Built to test the engine against realistic, varied
conversational text instead of hand-picked one-off lines.

This is a **demo/testing harness**, not a product, swap `PERSONA_PROMPT` in
`chat_demo.py` for your own character if you want it to speak as something
else.

## Requirements

- [TTS-Handler](../TTS-Handler) set up and working (see its own README), this
  imports it directly as a sibling folder, no separate install needed there
- A free [Google Gemini API key](https://aistudio.google.com/apikey)
- Python deps: `pip install -r requirements.txt` (from a venv that also has
  TTS-Handler's dependencies installed, they can share one `.venv`)

## Running it

**Windows:** double-click `run.bat`, it starts the server, waits for the
model to finish loading, and opens your browser to it automatically.

**Manually:**
```bash
uvicorn chat_demo:app --host 127.0.0.1 --port 3001
```
Then open `http://127.0.0.1:3001`.

On first run, click the gear icon and paste in your Gemini API key, it's
saved locally to `settings.local.json` (gitignored, never committed, never
leaves your machine).

## Notes

- Gemini's free tier has small per-model daily quotas. `chat_demo.py` tries
  several models in order and falls back automatically if one is rate-limited
  or unavailable, see `GEMINI_MODELS`.
- Speech enhancement (VoiceFixer) exists in `TTS-Handler/inference/enhance.py`
  but isn't called by default, it measured *worse* noise floor than raw
  output for a clean 48kHz reference clip. Only worth enabling if your
  reference voice is itself low-quality/narrowband.
