"""
Named emotion presets for GPT-SoVITS.

Default reference: `_REF`/`_TEXT` below point at a shipped default voice
(voices/kokoro_female), so this works out of the box. Pass `voice=` to
TTSEngine.generate() to use a different voice from voices/ instead; the
preset still supplies the emotion's sampling params either way. Only one
emotional tone exists in a single reference clip, so every preset here
shares the SAME reference clip and leans on sampling-parameter variation
instead. Real emotional tone ultimately needs real performed recordings
per emotion, not sampling tricks on one neutral take.

Parameter choices below are grounded in actual acoustic-correlates-of-
emotion research, not arbitrary guesses. Anger and happiness share a "high
pitch plus irregular pitch" signature in the literature (not just
"faster"), sadness is mainly reduced pitch *variance* rather than a
descending contour, and fear's real marker is trembling instability
(jitter/shimmer) rather than simple speed/pitch, approximated here with a
measured-safe amplitude-tremor post-effect (inference/prosody_fx.py)
rather than WORLD-vocoder pitch jitter, which was tried and rejected after
direct measurement showed it was unpredictable (non-monotonic effect
strength) and carries a real risk of introducing buzzy artifacts.

None of this replaces real performed-emotion recordings. It's a more
scientifically-grounded version of the same sampling-parameter approach,
not a replacement for the real fix.
"""

from dataclasses import dataclass

_REF = "voices/kokoro_female/reference.wav"
_TEXT = "Hey there, I am glad you decided to give this a try today."


@dataclass(frozen=True)
class EmotionPreset:
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str = "en"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    speed_factor: float = 1.0
    # Post-processing effect name (see inference/prosody_fx.py), or None.
    post_effect: str | None = None
    gain_db: float = 0.0


PRESETS: dict[str, EmotionPreset] = {
    "neutral": EmotionPreset(ref_audio_path=_REF, prompt_text=_TEXT),
    # Happy/angry share a "high + irregular pitch" signature in the
    # research: higher temperature increases GPT-SoVITS's own sampling
    # variance, which is the closest available proxy for pitch
    # irregularity (not just raising a static pitch level).
    "happy": EmotionPreset(
        ref_audio_path=_REF, prompt_text=_TEXT, temperature=1.25, speed_factor=1.15
    ),
    "angry": EmotionPreset(
        ref_audio_path=_REF,
        prompt_text=_TEXT,
        temperature=1.2,
        speed_factor=1.2,
        gain_db=2.5,  # research: anger correlates with increased volume, not just rate
    ),
    # Sadness: research says reduced pitch *variance* (flatter, more
    # monotone) matters more than a falling contour. Low temperature
    # reduces sampling variance, which is the right direction for this.
    "sad": EmotionPreset(
        ref_audio_path=_REF, prompt_text=_TEXT, temperature=0.7, speed_factor=0.8
    ),
    # Fear's real marker is trembling/instability (jitter), not simple
    # speed or pitch. Approximated with the amplitude_tremor post-effect.
    "afraid": EmotionPreset(
        ref_audio_path=_REF,
        prompt_text=_TEXT,
        temperature=1.15,
        speed_factor=1.05,
        post_effect="amplitude_tremor",
    ),
    "disgusted": EmotionPreset(
        ref_audio_path=_REF, prompt_text=_TEXT, temperature=0.85, speed_factor=0.9
    ),
    "excited": EmotionPreset(
        ref_audio_path=_REF, prompt_text=_TEXT, temperature=1.35, speed_factor=1.3
    ),
    "playful": EmotionPreset(
        ref_audio_path=_REF, prompt_text=_TEXT, temperature=1.2, speed_factor=1.1
    ),
}


def get_preset(name: str) -> EmotionPreset:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown emotion preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]
