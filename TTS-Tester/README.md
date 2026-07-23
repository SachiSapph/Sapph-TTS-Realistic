# TTS-Tester

A chat+voice demo for exercising [`../TTS-Handler`](../TTS-Handler): type a
message, get an LLM reply (Google Gemini), hear it spoken back in whichever
voice and emotion you pick from the two dropdowns. This is what makes the
engine's behavioral side visible: pick `auto` and the LLM chooses the
emotional tone per reply itself based on the conversation, switching
naturally instead of staying locked to one setting, or pin a specific
emotion by hand. Built to test the engine against realistic, varied
conversational text instead of hand-picked one-off lines.

This is a **demo/testing harness**, not a product. `PERSONA_PROMPT` in
`chat_demo.py` is a generic assistant persona written for this demo, not
modeled on any specific character, swap it for your own if you want it to
speak as something else.

## Requirements

- [TTS-Handler](../TTS-Handler) set up and working (see its own README), this
  imports it directly as a sibling folder, no separate install needed there
- A free [Google Gemini API key](https://aistudio.google.com/apikey) (only
  needed for the chat feature, not for testing voices)
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

Click the gear icon to open Settings:
- Paste in a Gemini API key (a link to get a free one is right above the
  field), saved locally to `settings.local.json`, gitignored, never
  committed, never leaves your machine.
- **Test a voice**: pick any voice, default or one you dropped into
  `TTS-Handler/voices/`, and hit Play sample to hear it immediately. No API
  key needed for this, it talks straight to TTS-Handler, no LLM involved.
- Set a default voice and tone for new chats.

## Voice and tone controls

The two dropdowns in the header pick the voice and emotional tone used for
the *next* chat reply. Set the emotion dropdown to `auto` and the LLM
chooses the tone itself per reply based on the conversation; pin a specific
one to keep every reply in that tone regardless of content.

## Notes

- Gemini's free tier has small per-model daily quotas. `chat_demo.py` tries
  several models in order and falls back automatically if one is rate-limited
  or unavailable, see `GEMINI_MODELS`.
- Speech enhancement (VoiceFixer) exists in `TTS-Handler/inference/enhance.py`
  but isn't called by default, it measured *worse* noise floor than raw
  output for a clean 48kHz reference clip. Only worth enabling if your
  reference voice is itself low-quality/narrowband.
