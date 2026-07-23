# TTS-Handler

The engine behind [Sapph-TTS](..). A local, self-hosted text-to-speech
system built on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT
license): **zero-shot voice cloning**, give it a short reference clip of a
voice plus its transcript, and it synthesizes new lines in that voice, no
training required. An optional fine-tuning path exists too (GPT-SoVITS's
own scripts), for a stronger match than zero-shot alone gives.

Emotion is driven by **reference-clip prosody plus sampling parameters**,
not a fixed emotion vector. [`emotion_vectors/presets.py`](emotion_vectors/presets.py)
maps a name (`happy`, `sad`, `angry`, `afraid`, `disgusted`, `excited`,
`playful`, `exhausted`, `neutral`) to sampling params (temperature, speed,
gain, a shimmer post-effect for fear) chosen from real acoustic-correlates-
of-emotion research, not arbitrary guesses. See the comments in that file
for the reasoning behind each one.

A single reply can also shift tone across its own sentences instead of
staying in one emotion the whole way through, see
[Multiple emotions in one reply](#multiple-emotions-in-one-reply) below.

## Voices

Three default voices ship with this repo, synthesized with
[Kokoro](https://github.com/hexgrad/kokoro) (Apache 2.0, fully
copyright-free, no real person's voice involved): `kokoro_female`,
`kokoro_male`, `kokoro_alt`. List what's available and generate with a
specific one:

```python
from voices import list_voices
print(list(list_voices().keys()))

engine.generate("Hello there!", emotion="happy", voice="kokoro_male")
```

## Requirements

- Python 3.10-3.12 (GPT-SoVITS's own constraint, not 3.13+)
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
   Install `torch`/`torchaudio` matching your CUDA driver **first**, separately.
   See the comment at the top of `requirements.txt`.

3. Smoke-test everything end to end:
   ```bash
   python scripts/test_presets.py
   ```
   Generates one line per emotion preset (using the default voice) into `test_audio/`.

## Adding a voice

Drop an audio file (wav/mp3/m4a/flac/ogg) into `voices/<your_voice_name>/`:

```
voices/
└── my_voice/
    └── reference.wav
```

That's it. On first use it's auto-transcribed with `faster-whisper` and the
transcript cached to `prompt.txt` next to the audio, so there's no manual
labeling step. Use a clean, natural, unspliced clip (a few seconds of one
continuous utterance works best; don't concatenate different sentences
together). Only add a voice you actually have the rights to use: your own
voice, a permissively-licensed source, or someone else's with their
explicit consent.

If auto-transcription gets something wrong, just edit the generated
`prompt.txt` by hand, it's read from disk, not regenerated once it exists.

## Using it

Directly in Python:
```python
from inference.generate import TTSEngine

engine = TTSEngine(config_path="GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml")
engine.load()
audio_bytes = engine.generate("Hello there!", emotion="happy", voice="kokoro_female")
```

Or as an HTTP API:
```bash
uvicorn server:app --host 0.0.0.0 --port 3100
```
```
POST /generate       {"text": "Hello there!", "emotion": "happy", "voice": "kokoro_female"} -> WAV audio bytes
POST /generate_multi {"segments": [{"text": "...", "emotion": "..."}, ...], "voice": "kokoro_female"} -> WAV audio bytes
GET  /voices         -> {"voices": ["kokoro_alt", "kokoro_female", "kokoro_male"]}
GET  /health         -> {"status": "ok", "engine_loaded": true}
```

See [`../TTS-Tester`](../TTS-Tester) for a full chat+voice demo built on top
of this engine.

## Multiple emotions in one reply

A reply doesn't have to stay in one emotion for its whole length. Real
speech often shifts tone across a sentence, relief giving way to
exhaustion, excitement undercut by nervousness, and `TTSEngine.generate_multi()`
supports that directly: give it an ordered list of `(text, emotion)`
segments, same voice throughout, and it generates each one with its own
preset and concatenates them with a short natural pause in between.

```python
audio_bytes = engine.generate_multi(
    [
        ("We actually won!", "happy"),
        ("...though I am completely exhausted.", "exhausted"),
    ],
    voice="kokoro_female",
)
```

This is for a tone shift **across sentences**, not for blending several
emotions into a single instant, GPT-SoVITS's sampling knobs (temperature,
speed, gain) aren't meaningfully additive that way, mixing them doesn't
produce a believable blend, just an unpredictable one. A single segment
is just a plain `generate()` call under the hood, no overhead for the
common case.

Whether to actually use more than one segment is a judgment call, most
replies genuinely are one consistent tone, and should stay that way. See
[`../TTS-Tester/chat_demo.py`](../TTS-Tester/chat_demo.py)'s `auto` mode
for a working example of an LLM deciding this per reply: it's instructed
to default to a single segment and only split when the reply's own content
genuinely shifts tone, not as a routine gimmick.

## Notes on pronunciation

GPT-SoVITS's English frontend spells out unknown words 3 letters or shorter
one letter at a time, and falls back to a neural guess for longer unknown
words, both can mangle onomatopoeia ("eek", "hmph", "pfft"). Fix pronunciation
for specific words by adding a line to
`GPT-SoVITS/GPT_SoVITS/text/engdict-hot.rep` (format: `WORD PH1 PH2 ...`,
space-separated ARPAbet phones), it's read fresh on every load and overrides
both the dictionary and the guesser.
