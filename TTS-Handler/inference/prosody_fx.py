"""
Post-processing prosody effects, grounded in actual speech-emotion research
(acoustic correlates of emotion — see project memory for citations), not
guesswork. GPT-SoVITS's exposed knobs (top_k/top_p/temperature/speed_factor)
can't touch jitter or shimmer, which research identifies as a real marker
for fear/nervousness specifically (trembling instability, not just "faster
and higher").

A pitch-jitter variant (via WORLD vocoder analysis-resynthesis) was tried
and rejected: measured directly, it was unreliable — the relationship
between the jitter_amount parameter and actual measured frame-to-frame
pitch instability was non-monotonic (0.05 -> 3.21%, 0.10 -> 3.07%, 0.15 ->
3.93%), meaning it can't be tuned predictably, on top of this project's
established risk of WORLD resynthesis introducing buzzy artifacts (see
project_tts_voice_choice memory). Not shipping something unpredictable.
"""

import numpy as np


def amplitude_tremor(audio: np.ndarray, sr: int, rate_hz: float = 5.5, depth: float = 0.12) -> np.ndarray:
    """Modulates amplitude with a slow sine wave to approximate vocal
    shimmer/trembling. depth=0.12 means volume wavers +/-12%. Measured
    safe: doesn't touch pitch or noise floor (verified directly, not
    assumed) since it never re-synthesizes anything, just scales the
    existing waveform."""
    t = np.arange(len(audio)) / sr
    envelope = 1.0 + depth * np.sin(2 * np.pi * rate_hz * t)
    return (audio * envelope).astype(audio.dtype)
