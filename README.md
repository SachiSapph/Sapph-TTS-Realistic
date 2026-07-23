# Sapph-TTS

**v1.0.1**: early days, expect rough edges. Bug reports and pull requests welcome.

A realistic, local text-to-speech engine with actual emotional range and a
behavioral system on top, not just a flat voice reading text out loud. Built
on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for zero-shot voice
cloning, plus a layer of research-grounded emotion presets and a demo chat
app that lets a voice switch tone naturally mid-conversation instead of
staying locked to one setting.

Runs entirely on your own machine. No API key for the voice itself, no
per-character or per-minute billing, no cloud dependency, no ToS on the
voice you use. Compared to hosted options like Fish Audio or ElevenLabs,
the tradeoff is simple: you do the setup work once, and in exchange you get
emotion control that's actually tuned and explainable (grounded in real
acoustic-correlates-of-emotion research, documented in the code, not a
black-box "style" slider) instead of a handful of opaque presets behind a
paywall.

## Table of contents

- [What's inside](#whats-inside)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Running the engine (TTS-Handler)](#running-the-engine-tts-handler)
- [Running the chat demo (TTS-Tester)](#running-the-chat-demo-tts-tester)
- [Voices](#voices)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What's inside

Two clearly separated parts, each independently usable:

- **[TTS-Handler](TTS-Handler)**: the engine. Text in, spoken audio out,
  as a Python library or a small HTTP API.
- **[TTS-Tester](TTS-Tester)**: a chat demo built on top of it. Type a
  message, an LLM replies, TTS-Handler speaks it back in whatever emotion
  fits, chosen automatically or picked by hand. Also where you test voices.

## System requirements

- **OS**: developed and tested on Windows. GPT-SoVITS itself supports
  Linux and Mac too, but the setup steps below are Windows-specific
  (`.bat` launcher, `winget`); on Linux/Mac you'll need to adapt those
  parts yourself (the Python side is unchanged).
- **Python**: 3.10, 3.11, or 3.12. Not 3.13+, GPT-SoVITS doesn't support it
  yet. This project was built and verified against 3.11.
- **GPU**: an NVIDIA CUDA-capable GPU is strongly recommended.
  - **Inference** (just generating speech, no training): community reports
    put the practical minimum around **6 GB VRAM** (e.g. an RTX 2060 or
    better). Halve that further with fp16 if your GPU supports it, though
    this project pins fp32 by default, see [Troubleshooting](#troubleshooting)
    for why.
  - **Fine-tuning** (training a custom voice on GPT-SoVITS's own scripts,
    not required for zero-shot cloning): closer to **12 GB VRAM**
    recommended (e.g. RTX 3060 or better).
  - **CPU-only** works but is slow, expect inference to take noticeably
    longer per line. Fine for trying it out, not for real-time chat.
  - If your GPU is newer than what your CUDA/PyTorch install supports (e.g.
    an RTX 50-series/Blackwell card), see the PyTorch install step below,
    you need a matching CUDA build or it won't detect the GPU at all.
- **Disk space**: GPT-SoVITS's pretrained base model weights are about
  **4.6 GB** on their own, plus several more GB for the Python virtual
  environment (PyTorch and CUDA libraries are large). Budget at least 15 GB
  free to be comfortable.
- **ffmpeg**: needs to be on your PATH. Used for audio format handling.

## Installation

Every command below is meant to be run from a terminal (PowerShell or
Command Prompt on Windows), in the folder where you want this project to
live.

### 1. Install prerequisites

```bash
winget install --id Python.Python.3.11
winget install Gyan.FFmpeg
```
If you already have Python 3.10-3.12 or ffmpeg, skip the corresponding
command. After installing, **open a fresh terminal window** so the updated
PATH actually takes effect, an already-open terminal won't see it.

### 2. Clone this repo

```bash
git clone https://github.com/SachiSapph/Sapph-TTS-Realistic.git
cd Sapph-TTS-Realistic
```

### 3. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```
(On Linux/Mac: `source .venv/bin/activate`.) You should see `(.venv)` at
the start of your prompt afterward. Every command from here on assumes
this virtual environment is active.

### 4. Install PyTorch matching your GPU, first and separately

Don't skip this or let a later `pip install -r requirements.txt` silently
pull in a CPU-only build. Find the right command for your CUDA version at
the [PyTorch install matrix](https://pytorch.org/get-started/locally/), or
use this known-working pin (also what GPT-SoVITS's own upstream docs use):

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```
`cu128` works for most current NVIDIA GPUs, including 50-series/Blackwell
cards (RTX 5070 and similar need cu128 or newer specifically, an older
CUDA build won't detect them). Avoid `torchaudio` 2.9+ for now, see
[Troubleshooting](#troubleshooting) if you hit a `torchcodec` DLL error.

Verify it actually sees your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
This should print `True` and your GPU's name. If it prints `False`, stop
and fix this before continuing, everything downstream will silently fall
back to (slow) CPU inference otherwise.

### 5. Install TTS-Handler's dependencies

```bash
pip install -r TTS-Handler/requirements.txt
```

### 6. Get GPT-SoVITS itself

```bash
cd TTS-Handler
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```
Then follow GPT-SoVITS's own README to download its pretrained base model
weights (the ~4.6 GB mentioned above), required even for zero-shot use
before any fine-tuning. As of writing, the easiest path is:
```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download lj1995/GPT-SoVITS --local-dir GPT_SoVITS/pretrained_models
```
Check GPT-SoVITS's own README if that command has changed, upstream repos
move. Once done, go back to the repo root:
```bash
cd ../..
```

### 7. A couple of known install snags

These aren't hypothetical, they were hit and fixed while building this
project. Check for them now rather than debugging a mystery crash later:

- **`jieba_fast` fails to build** (needs a C/C++ compiler you probably
  don't have installed): if `pip install -r GPT-SoVITS/requirements.txt`
  errors out on it and you don't want to install Visual Studio Build
  Tools just for this, create a tiny shim instead. It only needs to be a
  pure-Python re-export of the regular `jieba` package (already a
  dependency, and API-compatible), Chinese text support is functionally
  unaffected for English-only use:
  ```bash
  mkdir .venv\Lib\site-packages\jieba_fast
  ```
  Then create `.venv\Lib\site-packages\jieba_fast\__init__.py` containing
  `from jieba import *` and `.venv\Lib\site-packages\jieba_fast\posseg.py`
  containing `from jieba.posseg import *`.
- **Chinese characters in console output crash with `UnicodeEncodeError`**:
  GPT-SoVITS prints some of its own progress messages in Chinese
  regardless of what language you're synthesizing. Windows' default
  console codepage can't display them. Fix, set these two environment
  variables before running anything:
  ```bash
  set PYTHONUTF8=1
  set PYTHONIOENCODING=utf-8
  ```
  (PowerShell: `$env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"`.)
  `run.bat` (see below) already sets these for you.

### 8. Two config tweaks worth making

GPT-SoVITS's default config isn't tuned for the best available quality out
of the box. Open `TTS-Handler/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml`
and, under the `custom:` section:

- Set `version: v2ProPlus` (not the default `v2`). `v2Pro`/`v2ProPlus` add
  a dedicated speaker-verification component that noticeably improves
  how closely the cloned voice matches your reference clip. The weights
  for it are already included in the pretrained model download from step
  6, no extra download needed.
- Set `is_half: false` (all instances of it in the file). Half-precision
  (fp16) caused audible static/noise artifacts in testing that weren't
  present in the source reference clip, a known class of vocoder
  numerical-instability issue. Slower inference, but correct audio.

### 9. Smoke-test everything

```bash
python TTS-Handler/scripts/test_presets.py
```
Generates one line per emotion preset (using the default bundled voice)
into `TTS-Handler/test_audio/`. If this produces clean-sounding WAV files,
the install is good.

## Running the engine (TTS-Handler)

Directly in Python:
```python
from inference.generate import TTSEngine

engine = TTSEngine(config_path="GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml")
engine.load()
audio_bytes = engine.generate("Hello there!", emotion="happy", voice="kokoro_female")
```
(Run from inside `TTS-Handler/`, or add it to your `sys.path`, see
`TTS-Tester/chat_demo.py` for an example of importing it as a sibling
folder.)

Or as an HTTP API:
```bash
cd TTS-Handler
uvicorn server:app --host 0.0.0.0 --port 3100
```
```
POST /generate       {"text": "Hello there!", "emotion": "happy", "voice": "kokoro_female"} -> WAV audio bytes
POST /generate_multi {"segments": [...], "voice": "kokoro_female"} -> WAV audio bytes, one reply that shifts tone across its own sentences
GET  /voices         -> {"voices": ["kokoro_alt", "kokoro_female", "kokoro_male"]}
GET  /health         -> {"status": "ok", "engine_loaded": true}
```
Full details, including all available emotion presets and how the
multi-emotion segments work, in [TTS-Handler's README](TTS-Handler/README.md).

## Running the chat demo (TTS-Tester)

First, install its dependencies too:
```bash
pip install -r TTS-Tester/requirements.txt
```

**Windows:** double-click `TTS-Tester/run.bat`, it starts the server, waits
for the model to finish loading (can take up to a minute), and opens your
browser to it automatically.

**Manually:**
```bash
cd TTS-Tester
uvicorn chat_demo:app --host 127.0.0.1 --port 3001
```
Then open `http://127.0.0.1:3001` in your browser.

Click the gear icon to open Settings:
- Paste in a free [Gemini API key](https://aistudio.google.com/apikey)
  (link is right there in the panel too) to enable the chat feature.
- **Test a voice**: pick any voice from the dropdown and hit Play sample
  to hear it immediately, no API key needed, no chat message needed.
- Set a default voice and emotional tone for new chats.

The two dropdowns in the header pick voice and tone for the *next* reply.
Set tone to `auto` and the LLM chooses it itself per reply based on the
conversation; pin a specific one to keep every reply in that tone.

## Voices

Three default voices ship with the repo (synthetic, Apache-2.0 licensed via
[Kokoro](https://github.com/hexgrad/kokoro), fully copyright-free): one
female, one male, one alternate. Pick between them from the chat demo's
dropdown, or pass `voice=` directly to the engine.

Want your own voice instead? Two ways: drop an audio file straight into
`TTS-Handler/voices/` (`voices/my_voice.wav`), or give it its own subfolder
(`voices/my_voice/reference.wav`). Either way it's picked up automatically,
transcript auto-generated too, no manual labeling step.

- **Supported formats**: wav, mp3, m4a, flac, ogg. Anything not natively
  readable (m4a, notably) is transcoded to WAV automatically.
- **Clip length**: GPT-SoVITS itself requires **3-10 seconds**, a hard
  limit of its pretrained model, not something this project can widen.
  You don't have to hit that window yourself though: drop in **up to about
  a minute** (an ordinary voice memo is fine) and a natural speech pause
  inside the 3-10s window is found and extracted automatically. Only a
  clip with no such pause anywhere, or over a minute long, gets skipped
  (with a clear reason logged to the console).

Full details in [TTS-Handler's README](TTS-Handler/README.md#adding-a-voice).
Only clone a voice you actually have the rights to use.

## Troubleshooting

- **`torchcodec` DLL load error** (something like `Could not load this
  library: libtorchcodec_core8.dll`): caused by `torchaudio` 2.9+ pairing
  with a `torchcodec` build that doesn't load correctly on some Windows
  setups. Fix, pin back to the versions in step 4 above (`torch==2.8.0`,
  `torchaudio==2.8.0`), which still has the legacy audio loader and
  doesn't need `torchcodec` at all.
- **`UnicodeEncodeError` mid-generation**: see the `PYTHONUTF8`/
  `PYTHONIOENCODING` fix in step 7 above.
- **A voice you added isn't showing up in the voice list**: check the
  server console for a `Skipping voice '...'` warning explaining why.
  GPT-SoVITS requires the actual clip used to be 3-10 seconds; anything up
  to about a minute long gets auto-trimmed to a natural pause inside that
  window automatically, so this usually only happens for a clip with no
  pause anywhere in range, or one over a minute long. See
  [Voices](#voices) above.
- **Static or noise in generated audio that isn't in your source
  recording**: check `is_half` in `tts_infer.yaml`, see step 8 above.
- **`git push` fails with a permission error** to a repo you do own: your
  machine's cached Git credentials are probably signed in as a different
  GitHub account. On Windows with Git Credential Manager:
  `git credential-manager github logout <wrong-account-name>`, then push
  again to get a fresh sign-in prompt.
- **`torch.cuda.is_available()` prints `False`** on a GPU that should
  work: almost always a CUDA/PyTorch version mismatch (see step 4), a
  newer GPU generation needs a CUDA build at least as new as what that
  generation requires.

## License

This project's own code (everything outside the `GPT-SoVITS/` vendor
folder, which you clone in yourself per the setup instructions above) is
[MIT licensed](LICENSE), copyright Sapph. GPT-SoVITS itself is also
MIT-licensed, under its own copyright.
