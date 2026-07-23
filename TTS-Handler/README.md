# TTS-Handler

The TTS engine. A local, self-hosted text-to-speech system built on
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT license) — **zero-shot
voice cloning**: give it a short (3-10s) reference clip of a voice plus its
transcript, and it synthesizes new lines in that voice, no training required.
An optional fine-tuning path exists too (GPT-SoVITS's own scripts), for a
stronger match than zero-shot alone gives.

Emotion is driven by **reference-clip prosody and sampling parameters**, not
a fixed emotion vector — [`emotion_vectors/presets.py`](emotion_vectors/presets.py)
maps a name (`happy`, `sad`, `angry`, `afraid`, `disgusted`, `excited`,
`playful`, `neutral`) to a reference clip + its transcript + sampling params
(temperature, speed, etc.), grounded in real acoustic-correlates-of-emotion
research rather than arbitrary guesses. See the comments in that file for the
reasoning per emotion.

This repo ships **no voice audio**. Bring your own reference clip — see
[Adding a voice](#adding-a-voice) below. Only use a voice you actually have
the rights to clone (your own voice, or someone else's with their explicit
consent — cloning a voice without consent isn't something this project
supports).

## Requirements

- Python 3.10–3.12 (GPT-SoVITS's own constraint — not 3.13+)
- A CUDA-capable GPU is strongly recommended (CPU inference works but is slow)
- `ffmpeg` on your PATH

## Setup

1. Clone GPT-SoVITS into this folder:
   ```bash
   git clone https://github.com/RVC-Boss/GPT-SoVITS.git
   ```
   Follow its own README to download the pretrained base model weights
   (required even for zero-shot use, before any fine-tuning).

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   pip install -r GPT-SoVITS/requirements.txt
   ```
   Install `torch`/`torchaudio` matching your CUDA driver **first**, separately
   — see the comment at the top of `requirements.txt`.

3. Add a voice (see below), then point `emotion_vectors/presets.py`'s `_REF`
   and `_TEXT` at it.

4. Smoke-test everything end to end:
   ```bash
   python scripts/test_presets.py
   ```
   Generates one line per emotion preset into `test_audio/`.

## Adding a voice

Drop a clean, natural, unspliced reference clip (3–10 seconds, one single
utterance — don't concatenate different sentences to hit the minimum length)
into `voices/<your_voice_name>/`, e.g.:

```
voices/
└── my_voice/
    └── reference.wav
```

Then update `_REF` (path to the clip) and `_TEXT` (its exact transcript) in
`emotion_vectors/presets.py`. Every preset shares that one reference clip by
default; for genuinely distinct per-emotion prosody, record separate short
clips actually performed in each emotion and give each preset its own
`ref_audio_path`/`prompt_text` instead of sharing `_REF`.

## Using it

Directly in Python:
```python
from inference.generate import TTSEngine

engine = TTSEngine(config_path="GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml")
engine.load()
audio_bytes = engine.generate("Hello there!", emotion="happy")  # WAV bytes
```

Or as an HTTP API:
```bash
uvicorn server:app --host 0.0.0.0 --port 3100
```
```
POST /generate {"text": "Hello there!", "emotion": "happy"} -> WAV audio bytes
GET  /health    -> {"status": "ok", "engine_loaded": true}
```

See [`../TTS-Tester`](../TTS-Tester) for a full chat+voice demo built on top
of this engine.

## Notes on pronunciation

GPT-SoVITS's English frontend spells out unknown words 3 letters or shorter
one letter at a time, and falls back to a neural guess for longer unknown
words — both can mangle onomatopoeia ("eek", "hmph", "pfft"). Fix pronunciation
for specific words by adding a line to
`GPT-SoVITS/GPT_SoVITS/text/engdict-hot.rep` (format: `WORD PH1 PH2 ...`,
space-separated ARPAbet phones) — it's read fresh on every load and overrides
both the dictionary and the guesser.
