"""
Voice registry: auto-discovers voices from the voices/ folder, two ways:

  voices/my_voice.mp3           <- loose file dropped straight in, named "my_voice"
  voices/my_voice/reference.mp3 <- or its own subfolder, named after the folder

Either way, a matching prompt.txt (subfolder form) or my_voice.prompt.txt
(loose-file form, sitting next to the audio) supplies the exact transcript
if you already know it. If it's missing, it's auto-transcribed once with
faster-whisper (already a GPT-SoVITS dependency) and cached there, no
manual transcription step either way.

GPT-SoVITS requires the reference clip to be 3-10 seconds long (this is a
hard limit of its pretrained speaker-conditioning model, not a value this
project chose or can widen), and raises an opaque OSError deep inside
generation if it isn't, only when that voice is actually used. Checked
here at discovery time instead. A clip up to a minute long that's too long
gets auto-trimmed: a natural speech pause inside the 3-10s window is found
(via Whisper's own segment boundaries, never a mid-word cut) and that span
is extracted to a cached file, so dropping in an ordinary voice memo or
recording just works without manually editing it first. Only a clip with
no such pause (or over a minute, or under 3 seconds even after) is skipped
with a clear reason.
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

# Longer than this, don't even try auto-trimming, full transcription plus a
# segment search on an arbitrarily long file isn't worth attempting.
MAX_RAW_CLIP_SECONDS = 60.0

AUTOTRIM_SUFFIX = ".autotrim.wav"

_whisper_model = None  # lazy-loaded, only needed the first time transcription/trimming is needed


@dataclass(frozen=True)
class Voice:
    name: str
    ref_audio_path: str  # relative to PROJECT_ROOT, same convention as EmotionPreset
    prompt_text: str


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("base.en", compute_type="int8")
    return _whisper_model


def _transcribe(audio_path: Path) -> str:
    logger.info("Auto-transcribing %s ...", audio_path.name)
    segments, _ = _get_whisper_model().transcribe(str(audio_path), vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise RuntimeError(f"Whisper produced no transcript for {audio_path}")
    logger.info("Transcribed %s -> %r", audio_path.name, text[:80])
    return text


def _find_natural_segment(audio_path: Path) -> tuple[float, float, str] | None:
    """Looks for a contiguous run of real speech, using Whisper's own
    segment boundaries so any cut lands on a natural pause, never mid-word,
    whose total span fits inside GPT-SoVITS's 3-10 second window. Returns
    (start_seconds, end_seconds, transcript_for_that_span), or None if no
    such run exists (e.g. one long uninterrupted sentence with no pause
    anywhere inside the window)."""
    segments = list(_get_whisper_model().transcribe(str(audio_path), vad_filter=True)[0])
    for i in range(len(segments)):
        start = segments[i].start
        text_parts = []
        for seg in segments[i:]:
            text_parts.append(seg.text.strip())
            span = seg.end - start
            if span > MAX_REF_AUDIO_SECONDS:
                break
            if span >= MIN_REF_AUDIO_SECONDS:
                return start, seg.end, " ".join(text_parts).strip()
    return None


def _extract_segment(audio_path: Path, start: float, end: float, out_path: Path) -> None:
    data, sr = sf.read(str(audio_path))
    sf.write(str(out_path), data[int(start * sr):int(end * sr)], sr)


def _discover_candidates() -> list[tuple[str, list[Path], Path]]:
    """Returns (name, audio_candidates, transcript_path) for every voice
    found, subfolders and loose top-level files alike. audio_candidates is
    every audio file present (sorted, so this is deterministic), a folder
    (or a loose file with a previously cached auto-trim) can have more than
    one, and list_voices() below tries each in order rather than assuming
    the first one found is the usable one. Files this module generated
    itself (*.autotrim.wav) never show up as their own separate loose-file
    voice, only as a candidate for the original they were trimmed from."""
    candidates = []
    for entry in sorted(VOICES_DIR.iterdir()):
        if entry.is_dir():
            audio_files = sorted(f for f in entry.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
            if not audio_files:
                continue
            candidates.append((entry.name, audio_files, entry / "prompt.txt"))
        elif entry.name.endswith(AUTOTRIM_SUFFIX):
            continue
        elif entry.suffix.lower() in AUDIO_EXTENSIONS:
            audio_candidates = [entry]
            cached_trim = VOICES_DIR / f"{entry.stem}{AUTOTRIM_SUFFIX}"
            if cached_trim.exists():
                audio_candidates.insert(0, cached_trim)  # already in range, try it first
            candidates.append((entry.stem, audio_candidates, entry.with_suffix(".prompt.txt")))
    return candidates


def list_voices() -> dict[str, Voice]:
    voices: dict[str, Voice] = {}
    if not VOICES_DIR.exists():
        return voices

    for name, audio_candidates, transcript_path in _discover_candidates():
        audio_path = None
        prompt_text = None
        rejected = []

        for candidate in audio_candidates:
            duration = sf.info(str(candidate)).duration
            if MIN_REF_AUDIO_SECONDS <= duration <= MAX_REF_AUDIO_SECONDS:
                audio_path = candidate
                break
            rejected.append((candidate, duration))

        if audio_path is None:
            for candidate, duration in rejected:
                if not (MAX_REF_AUDIO_SECONDS < duration <= MAX_RAW_CLIP_SECONDS):
                    continue
                segment = _find_natural_segment(candidate)
                if segment is None:
                    continue
                start, end, segment_text = segment
                autotrim_path = candidate.with_name(f"{candidate.stem}{AUTOTRIM_SUFFIX}")
                _extract_segment(candidate, start, end, autotrim_path)
                logger.info(
                    "Auto-trimmed voice '%s': extracted %.1fs-%.1fs from %s",
                    name, start, end, candidate.name,
                )
                audio_path = autotrim_path
                prompt_text = segment_text
                transcript_path.write_text(prompt_text, encoding="utf-8")
                break

        if audio_path is None:
            reasons = ", ".join(f"{c.name} ({d:.1f}s)" for c, d in rejected)
            logger.warning(
                "Skipping voice '%s': no audio file is in, or could be "
                "auto-trimmed into, GPT-SoVITS's required %.0f-%.0f second "
                "range (found: %s). Trim one manually to a natural pause "
                "inside that window and it will show up automatically.",
                name, MIN_REF_AUDIO_SECONDS, MAX_REF_AUDIO_SECONDS, reasons,
            )
            continue

        if prompt_text is None:
            if transcript_path.exists():
                # utf-8-sig strips a leading BOM if present (e.g. from
                # Notepad or PowerShell's Set-Content -Encoding utf8) while
                # still reading plain UTF-8 files with no BOM correctly.
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
