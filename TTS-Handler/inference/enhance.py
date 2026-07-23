"""
Post-processing speech enhancement (VoiceFixer, MIT license).

GPT-SoVITS's output inherits the reference clip's bandwidth, a low-quality
reference clip (e.g. old/narrowband recordings) caps the output well below
modern recording quality, giving it a narrow, "old telephone/studio"
character regardless of the emotion or voice identity. VoiceFixer restores
high-frequency content and upsamples to 44.1kHz, useful if your reference
voice is a lower-quality recording. Not called by default (see the caller):
tested directly against a clean 48kHz reference clip and measured a WORSE
noise floor after VoiceFixer than the raw output, so don't assume it always
helps, re-measure for your own source before enabling it.

VoiceFixer's own `restore()` only takes file paths; `restore_inmem()` is the
lower-level array-in/array-out call it's built on (confirmed by reading its
source directly), used here to avoid needing temp files.
"""

from io import BytesIO

import librosa
import soundfile as sf
from voicefixer import VoiceFixer

_SAMPLE_RATE = 44100

_voicefixer: VoiceFixer | None = None


def load():
    """Load the enhancement model. Call once at startup, not per-request."""
    global _voicefixer
    _voicefixer = VoiceFixer()


def enhance(audio_bytes: bytes) -> bytes:
    """Restore bandwidth/clarity on generated TTS audio. Returns WAV bytes."""
    if _voicefixer is None:
        raise RuntimeError("Call load() before enhance().")

    wav, _ = librosa.load(BytesIO(audio_bytes), sr=_SAMPLE_RATE)
    restored = _voicefixer.restore_inmem(wav, cuda=True, mode=0)

    out_buffer = BytesIO()
    sf.write(out_buffer, restored.squeeze(0), _SAMPLE_RATE, format="WAV")
    return out_buffer.getvalue()
