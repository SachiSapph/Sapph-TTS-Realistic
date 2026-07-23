"""
Voice registry: auto-discovers voices from the voices/ folder, two ways:

  voices/my_voice.mp3           <- loose file dropped straight in, named "my_voice"
  voices/my_voice/reference.mp3 <- or its own subfolder, named after the folder

Either way, a matching prompt.txt (subfolder form) or my_voice.prompt.txt
(loose-file form, sitting next to the audio) supplies the exact transcript
if you already know it. If it's missing, it's auto-transcribed once with
faster-whisper (already a GPT-SoVITS dependency) and cached there, no
manual transcription step either way.

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


def _discover_candidates() -> list[tuple[str, Path, Path]]:
    """Returns (name, audio_path, transcript_path) for every voice found,
    subfolders and loose top-level files alike."""
    candidates = []
    for entry in sorted(VOICES_DIR.iterdir()):
        if entry.is_dir():
            audio_files = [f for f in entry.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
            if not audio_files:
                continue
            candidates.append((entry.name, audio_files[0], entry / "prompt.txt"))
        elif entry.suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append((entry.stem, entry, entry.with_suffix(".prompt.txt")))
    return candidates


def list_voices() -> dict[str, Voice]:
    voices: dict[str, Voice] = {}
    if not VOICES_DIR.exists():
        return voices

    for name, audio_path, transcript_path in _discover_candidates():
        duration = sf.info(str(audio_path)).duration
        if not (MIN_REF_AUDIO_SECONDS <= duration <= MAX_REF_AUDIO_SECONDS):
            logger.warning(
                "Skipping voice '%s': %s is %.1fs long, GPT-SoVITS requires "
                "%.0f-%.0f seconds. Trim it (or pad short clips with "
                "trailing silence) and it will show up automatically.",
                name, audio_path.name, duration,
                MIN_REF_AUDIO_SECONDS, MAX_REF_AUDIO_SECONDS,
            )
            continue

        if transcript_path.exists():
            # utf-8-sig strips a leading BOM if present (e.g. from Notepad or
            # PowerShell's Set-Content -Encoding utf8) while still reading
            # plain UTF-8 files with no BOM correctly.
            prompt_text = transcript_path.read_text(encoding="utf-8-sig").strip()
        else:
            prompt_text = _transcribe(audio_path)
            transcript_path.write_text(prompt_text, encoding="utf-8")

        voices[name] = Voice(
            name=name,
            ref_audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
            prompt_text=prompt_text,
        )

    return voices


def get_voice(name: str) -> Voice:
    voices = list_voices()
    if name not in voices:
        raise KeyError(f"Unknown voice '{name}'. Available: {list(voices.keys())}")
    return voices[name]
