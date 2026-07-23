# Sapph-TTS

A local, self-hosted text-to-speech system built on
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — zero-shot voice
cloning from a short reference clip, with named emotion presets grounded in
real speech-emotion research rather than arbitrary parameter tweaking. Runs
entirely on your own machine: no API key, no per-character billing, no
third-party ToS on the voice itself.

Two clearly separated parts:

- **[TTS-Handler](TTS-Handler)** — the actual TTS engine. Use this on its own
  if you just want text-to-speech, either as a Python library (`TTSEngine`)
  or a small HTTP API (`POST /generate {text, emotion} -> audio`).
- **[TTS-Tester](TTS-Tester)** — a chat+voice demo built on top of
  TTS-Handler, for trying it out conversationally (LLM chat via Gemini,
  spoken back through the engine) rather than one line at a time.

Start with `TTS-Handler`'s own README for setup — `TTS-Tester` is optional
and depends on it.

## Bring your own voice

This repo ships **no voice audio**. GPT-SoVITS clones whatever reference
clip you give it, so you provide that yourself — see
[`TTS-Handler/README.md`](TTS-Handler/README.md#adding-a-voice). Only clone a
voice you actually have the rights to use.

## License

GPT-SoVITS itself is MIT-licensed. This project's own code (everything
outside the `GPT-SoVITS/` vendor folder, which you clone in yourself per the
setup instructions) has no license file yet — treat it as all-rights-reserved
unless the repo owner adds one.
