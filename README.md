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

## What's inside

Two clearly separated parts, each independently usable:

- **[TTS-Handler](TTS-Handler)**: the engine. Text in, spoken audio out,
  as a Python library or a small HTTP API.
- **[TTS-Tester](TTS-Tester)**: a chat demo built on top of it. Type a
  message, an LLM replies, TTS-Handler speaks it back in whatever emotion
  fits, chosen automatically or picked by hand.

## Voices

Three default voices ship with the repo (synthetic, Apache-2.0 licensed via
[Kokoro](https://github.com/hexgrad/kokoro), fully copyright-free): one
female, one male, one alternate. Pick between them from the chat demo's
dropdown, or pass `voice=` directly to the engine.

Want your own voice instead? Drop an audio file into `TTS-Handler/voices/<name>/`
and it's picked up automatically, transcript included: no manual labeling
step. See [TTS-Handler's README](TTS-Handler/README.md#adding-a-voice) for
details. Only clone a voice you actually have the rights to use.

Test any voice, default or your own, straight from TTS-Tester's Settings
panel: pick it from the dropdown and hit Play sample, no chat message or
API key needed.

## License

GPT-SoVITS itself is MIT-licensed. This project's own code (everything
outside the `GPT-SoVITS/` vendor folder, which you clone in yourself per the
setup instructions) has no license file yet, treat it as all-rights-reserved
unless the repo owner adds one.
