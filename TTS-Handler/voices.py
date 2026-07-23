"""
Voice registry: auto-discovers voices from the voices/ folder. Each voice is
a subfolder with one reference audio file (wav/mp3/m4a/flac) and, optionally,
a prompt.txt with its exact transcript. Adding a voice really is just adding
an audio file to a new subfolder here: if prompt.txt is missing, it's
auto-transcribed once with faster-whisper (already a GPT-SoVITS dependency)
and cached to prompt.txt next to the audio, no manual transcription step.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VOICES_DIR = PROJECT_ROOT / "voices"

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac")

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
    segments, _ = _whisper_model.transcribe(str(audio_path))
    return " ".join(segment.text.strip() for segment in segments).strip()


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
        transcript_path = folder / "prompt.txt"
        if transcript_path.exists():
            prompt_text = transcript_path.read_text(encoding="utf-8").strip()
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
