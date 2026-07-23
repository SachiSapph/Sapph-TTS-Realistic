"""
Voice registry: auto-discovers voices from the voices/ folder, two ways:

  voices/my_voice.mp3           <- loose file dropped straight in, named "my_voice"
  voices/my_voice/reference.mp3 <- or its own subfolder, named after the folder

Either way, a matching prompt.txt (subfolder form) or my_voice.prompt.txt
(loose-file form, sitting next to the audio) supplies the exact transcript
if you already know it. If it's missing, it's auto-transcribed once with
faster-whisper's multilingual model (already a GPT-SoVITS dependency, so
non-English clips are detected and phonemized correctly rather than forced
through an English-only transcript) and cached there alongside the
detected language (language.txt / my_voice.language.txt), no manual
transcription step either way. A hand-supplied prompt.txt with no matching
language file is treated as English.

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

soundfile/libsndfile (used for every duration check and for GPT-SoVITS's
own torchaudio-based loader on this project's pinned legacy backend) can't
actually open every format its name suggests it might: notably m4a/AAC.
Anything outside NATIVE_SOUNDFILE_EXTENSIONS is transcoded to a cached WAV
via ffmpeg (pydub, already a dependency) before anything else touches it.

One bad voice (unreadable file, corrupt audio, anything unexpected) must
never take down every OTHER voice, list_voices() is called on every page
load, so a single crash here would empty the entire voice list. Every
per-voice step is wrapped accordingly.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
VOICES_DIR = PROJECT_ROOT / "voices"

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

# The subset soundfile/libsndfile can actually open directly (verified via
# soundfile.available_formats()). Anything else in AUDIO_EXTENSIONS above
# gets transcoded to WAV first, see _ensure_readable().
NATIVE_SOUNDFILE_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

# GPT-SoVITS's own hard requirement (TTS_infer_pack/TTS.py), not a value we chose.
MIN_REF_AUDIO_SECONDS = 3.0
MAX_REF_AUDIO_SECONDS = 10.0

# Longer than this, don't even try auto-trimming, full transcription plus a
# segment search on an arbitrarily long file isn't worth attempting.
MAX_RAW_CLIP_SECONDS = 60.0

AUTOTRIM_SUFFIX = ".autotrim.wav"
CONVERTED_SUFFIX = ".converted.wav"
# Files this module generates itself, never treated as their own separate
# loose-file voice, only ever as a candidate for the original they came from.
GENERATED_SUFFIXES = (AUTOTRIM_SUFFIX, CONVERTED_SUFFIX)

_whisper_model = None  # lazy-loaded, only needed the first time transcription/trimming is needed


@dataclass(frozen=True)
class Voice:
    name: str
    ref_audio_path: str  # relative to PROJECT_ROOT, same convention as EmotionPreset
    prompt_text: str
    prompt_lang: str = "en"


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        # Multilingual, not "base.en" — a reference clip isn't guaranteed to
        # be English, and the .en variant can't detect that, it just forces
        # a plausible-but-wrong English transcript on top of any language.
        _whisper_model = WhisperModel("base", compute_type="int8")
    return _whisper_model


def _ensure_readable(audio_path: Path) -> Path:
    """Transcodes to a cached WAV if this format isn't one soundfile can
    open directly. A no-op (returns the same path) otherwise."""
    if audio_path.suffix.lower() in NATIVE_SOUNDFILE_EXTENSIONS:
        return audio_path

    converted_path = audio_path.with_name(f"{audio_path.stem}{CONVERTED_SUFFIX}")
    if not converted_path.exists():
        from pydub import AudioSegment

        logger.info("Transcoding %s to WAV (soundfile can't read this format directly) ...", audio_path.name)
        AudioSegment.from_file(str(audio_path)).export(str(converted_path), format="wav")
    return converted_path


def _transcribe(audio_path: Path) -> tuple[str, str]:
    """Returns (text, detected_language) — GPT-SoVITS needs the reference
    clip's actual spoken language to phonemize prompt_text correctly, which
    isn't necessarily English."""
    logger.info("Auto-transcribing %s ...", audio_path.name)
    segments, info = _get_whisper_model().transcribe(str(audio_path), vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise RuntimeError(f"Whisper produced no transcript for {audio_path}")
    logger.info("Transcribed %s (lang=%s) -> %r", audio_path.name, info.language, text[:80])
    return text, info.language


def _find_natural_segment(audio_path: Path) -> tuple[float, float, str, str] | None:
    """Looks for a contiguous run of real speech, using Whisper's own
    segment boundaries so any cut lands on a natural pause, never mid-word,
    whose total span fits inside GPT-SoVITS's 3-10 second window. Returns
    (start_seconds, end_seconds, transcript_for_that_span, detected_language),
    or None if no such run exists (e.g. one long uninterrupted sentence with
    no pause anywhere inside the window)."""
    raw_segments, info = _get_whisper_model().transcribe(str(audio_path), vad_filter=True)
    segments = list(raw_segments)
    for i in range(len(segments)):
        start = segments[i].start
        text_parts = []
        for seg in segments[i:]:
            text_parts.append(seg.text.strip())
            span = seg.end - start
            if span > MAX_REF_AUDIO_SECONDS:
                break
            if span >= MIN_REF_AUDIO_SECONDS:
                return start, seg.end, " ".join(text_parts).strip(), info.language
    return None


def _extract_segment(audio_path: Path, start: float, end: float, out_path: Path) -> None:
    data, sr = sf.read(str(audio_path))
    sf.write(str(out_path), data[int(start * sr):int(end * sr)], sr)


def _discover_candidates() -> list[tuple[str, list[Path], Path, Path]]:
    """Returns (name, audio_candidates, transcript_path, language_path) for
    every voice found, subfolders and loose top-level files alike.
    audio_candidates is every audio file present (sorted, so this is
    deterministic), a folder (or a loose file with a previously cached
    auto-trim) can have more than one, and list_voices() below tries each in
    order rather than assuming the first one found is the usable one. Files
    this module generated itself (*.autotrim.wav, *.converted.wav) never
    show up as their own separate loose-file voice, only as a candidate for
    the original they came from."""
    candidates = []
    for entry in sorted(VOICES_DIR.iterdir()):
        if entry.is_dir():
            audio_files = sorted(f for f in entry.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
            if not audio_files:
                continue
            candidates.append((entry.name, audio_files, entry / "prompt.txt", entry / "language.txt"))
        elif entry.name.endswith(GENERATED_SUFFIXES):
            continue
        elif entry.suffix.lower() in AUDIO_EXTENSIONS:
            audio_candidates = [entry]
            cached_trim = VOICES_DIR / f"{entry.stem}{AUTOTRIM_SUFFIX}"
            if cached_trim.exists():
                audio_candidates.insert(0, cached_trim)  # already in range, try it first
            candidates.append((entry.stem, audio_candidates, entry.with_suffix(".prompt.txt"), entry.with_suffix(".language.txt")))
    return candidates


def list_voices() -> dict[str, Voice]:
    voices: dict[str, Voice] = {}
    if not VOICES_DIR.exists():
        return voices

    for name, audio_candidates, transcript_path, language_path in _discover_candidates():
        try:
            voice = _resolve_voice(name, audio_candidates, transcript_path, language_path)
        except Exception as e:
            # A single broken voice (unreadable file, transcription
            # failure, anything unexpected) must never take down every
            # OTHER voice, list_voices() runs on essentially every request.
            logger.warning("Skipping voice '%s': %s", name, e)
            continue
        if voice is not None:
            voices[name] = voice

    return voices


def _resolve_voice(name: str, audio_candidates: list[Path], transcript_path: Path, language_path: Path) -> Voice | None:
    """Picks the first candidate already in range, or the first one that
    can be auto-trimmed into range, for one voice. Returns None (already
    logged) if nothing usable was found; raises if something unexpected
    went wrong, list_voices() above turns that into a skip too, just with
    the real exception in the log instead of a generic reason."""
    audio_path = None
    prompt_text = None
    prompt_lang = None
    rejected = []  # (raw_candidate, readable_candidate, duration)

    for raw_candidate in audio_candidates:
        try:
            candidate = _ensure_readable(raw_candidate)
            duration = sf.info(str(candidate)).duration
        except Exception as e:
            logger.warning("Voice '%s': couldn't read %s (%s), trying any other candidate.", name, raw_candidate.name, e)
            continue
        if MIN_REF_AUDIO_SECONDS <= duration <= MAX_REF_AUDIO_SECONDS:
            audio_path = candidate
            break
        rejected.append((raw_candidate, candidate, duration))

    if audio_path is None:
        for raw_candidate, candidate, duration in rejected:
            if not (MAX_REF_AUDIO_SECONDS < duration <= MAX_RAW_CLIP_SECONDS):
                continue
            segment = _find_natural_segment(candidate)
            if segment is None:
                continue
            start, end, segment_text, segment_lang = segment
            # Cache path keyed on the ORIGINAL file's name, not the
            # (possibly transcoded) readable candidate's, so a later scan's
            # cache lookup in _discover_candidates still finds it.
            autotrim_path = raw_candidate.with_name(f"{raw_candidate.stem}{AUTOTRIM_SUFFIX}")
            _extract_segment(candidate, start, end, autotrim_path)
            logger.info("Auto-trimmed voice '%s': extracted %.1fs-%.1fs from %s", name, start, end, raw_candidate.name)
            audio_path = autotrim_path
            prompt_text = segment_text
            prompt_lang = segment_lang
            transcript_path.write_text(prompt_text, encoding="utf-8")
            language_path.write_text(prompt_lang, encoding="utf-8")
            break

    if audio_path is None:
        if rejected:
            reasons = ", ".join(f"{r.name} ({d:.1f}s)" for r, _, d in rejected)
            logger.warning(
                "Skipping voice '%s': no audio file is in, or could be "
                "auto-trimmed into, GPT-SoVITS's required %.0f-%.0f second "
                "range (found: %s). Trim one manually to a natural pause "
                "inside that window and it will show up automatically.",
                name, MIN_REF_AUDIO_SECONDS, MAX_REF_AUDIO_SECONDS, reasons,
            )
        else:
            logger.warning("Skipping voice '%s': no audio file could be read.", name)
        return None

    if prompt_text is None:
        if transcript_path.exists():
            # utf-8-sig strips a leading BOM if present (e.g. from Notepad
            # or PowerShell's Set-Content -Encoding utf8) while still
            # reading plain UTF-8 files with no BOM correctly.
            prompt_text = transcript_path.read_text(encoding="utf-8-sig").strip()
            # A transcript supplied by hand (no matching language.txt) is
            # assumed English; auto-transcribed ones always write one below.
            prompt_lang = language_path.read_text(encoding="utf-8-sig").strip() if language_path.exists() else "en"
        else:
            prompt_text, prompt_lang = _transcribe(audio_path)
            transcript_path.write_text(prompt_text, encoding="utf-8")
            language_path.write_text(prompt_lang, encoding="utf-8")

    return Voice(
        name=name,
        ref_audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
        prompt_text=prompt_text,
        prompt_lang=prompt_lang,
    )


def get_voice(name: str) -> Voice:
    voices = list_voices()
    if name not in voices:
        raise KeyError(f"Unknown voice '{name}'. Available: {list(voices.keys())}")
    return voices[name]
