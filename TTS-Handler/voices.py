"""
Voice registry: auto-discovers voices from the voices/ folder. Each voice is
a subfolder with one reference audio file (wav/mp3/m4a/flac/ogg) and,
optionally, a prompt.txt with its exact transcript. Adding a voice really is
just adding an audio file to a new subfolder here: if prompt.txt is missing,
it's auto-transcribed once with faster-whisper (already a GPT-SoVITS
dependency) and cached to prompt.txt next to the audio, no manual
transcription step. To skip auto-transcription (e.g. you already know the
exact wording), just write that prompt.txt file yourself before first use.

GPT-SoVITS requires the reference clip to be 3-10 seconds long, and raises
an opaque OSError deep inside generation if it isn't, only when that voice
is actually used. Checked here at discovery time instead, so a too-short or
too-long clip is skipped with a clear reason up front rather than showing
up as selectable and then failing later.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
VOICES_DIR = PROJECT_ROOT / "voices"

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

# GPT-SoVITS's own hard requirement (TTS_infer_pack/TTS.py), not a value we chose.
MIN_REF_AUDIO_SECONDS = 3.0
MAX_REF_AUDIO_SECONDS = 10.0

_whisper_model = None  # lazy-loaded, only needed the first time a voice has no prompt.txt yet


@dataclass(frozen=True)
class Voice:
    name: str
    ref_audio_path: str  # relative to PROJECT_ROOT, same convention as EmotionPreset
    prompt_text: str


def _transcribe(audio_path: Path) -> str:
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("base.en", compute_type="int8")
    logger.info("Auto-transcribing %s ...", audio_path.name)
    segments, _ = _whisper_model.transcribe(str(audio_path), vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise RuntimeError(f"Whisper produced no transcript for {audio_path}")
    logger.info("Transcribed %s -> %r", audio_path.name, text[:80])
    return text


def list_voices() -> dict[str, Voice]:
    voices: dict[str, Voice] = {}
    if not VOICES_DIR.exists():
        return voices

    for folder in sorted(VOICES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        audio_files = [f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
        if not audio_files:
            continue

        audio_path = audio_files[0]

        duration = sf.info(str(audio_path)).duration
        if not (MIN_REF_AUDIO_SECONDS <= duration <= MAX_REF_AUDIO_SECONDS):
            logger.warning(
                "Skipping voice '%s': %s is %.1fs long, GPT-SoVITS requires "
                "%.0f-%.0f seconds. Trim it (or pad short clips with "
                "trailing silence) and it will show up automatically.",
                folder.name, audio_path.name, duration,
                MIN_REF_AUDIO_SECONDS, MAX_REF_AUDIO_SECONDS,
            )
            continue

        transcript_path = folder / "prompt.txt"
        if transcript_path.exists():
            # utf-8-sig strips a leading BOM if present (e.g. from Notepad or
            # PowerShell's Set-Content -Encoding utf8) while still reading
            # plain UTF-8 files with no BOM correctly.
            prompt_text = transcript_path.read_text(encoding="utf-8-sig").strip()
        else:
            prompt_text = _transcribe(audio_path)
            transcript_path.write_text(prompt_text, encoding="utf-8")

        voices[folder.name] = Voice(
            name=folder.name,
            ref_audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
            prompt_text=prompt_text,
        )

    return voices


def get_voice(name: str) -> Voice:
    voices = list_voices()
    if name not in voices:
        raise KeyError(f"Unknown voice '{name}'. Available: {list(voices.keys())}")
    return voices[name]
