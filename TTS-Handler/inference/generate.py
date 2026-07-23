"""
Inference wrapper around GPT-SoVITS.

Verified against the real repo (RVC-Boss/GPT-SoVITS, api_v2.py): the
pipeline is `TTS_Config(config_path)` -> `TTS(config)`, run via
`tts_pipeline.run(req)` where `req` is a dict of the params below. It
returns a generator of `(sample_rate, numpy_array)` chunks, not raw bytes,
so this wrapper takes the first chunk (non-streaming) and encodes it to WAV.
"""

import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

from emotion_vectors.presets import EmotionPreset, get_preset
from inference import prosody_fx
from voices import get_voice

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT_SOVITS_ROOT = PROJECT_ROOT / "GPT-SoVITS"

# This project's own pronunciation fixes for words GPT-SoVITS's text
# frontend mangles (onomatopoeia especially), tracked here since
# GPT-SoVITS/ itself is a gitignored vendor clone: a fresh clone of it
# wouldn't have these, and editing that file directly wouldn't survive a
# fresh clone either. Synced into the real hot-dict file on every load().
PRONUNCIATIONS_PATH = PROJECT_ROOT / "pronunciations.rep"


def _sync_pronunciations():
    """Merges this project's own pronunciation fixes into GPT-SoVITS's own
    engdict-hot.rep (read fresh on every load, overrides both its
    dictionary and its neural guesser), adding only entries not already
    present so this is safe to call on every startup."""
    if not PRONUNCIATIONS_PATH.exists():
        return

    hot_dict_path = GPT_SOVITS_ROOT / "GPT_SoVITS" / "text" / "engdict-hot.rep"
    existing_words = set()
    if hot_dict_path.exists():
        for line in hot_dict_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_words.add(line.split(" ", 1)[0])

    missing_lines = [
        line for line in PRONUNCIATIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split(" ", 1)[0] not in existing_words
    ]
    if not missing_lines:
        return

    with hot_dict_path.open("a", encoding="utf-8") as f:
        if hot_dict_path.exists() and hot_dict_path.stat().st_size > 0:
            f.write("\n")
        f.write("\n".join(missing_lines) + "\n")


class TTSEngine:
    def __init__(
        self,
        config_path: str,
        gpt_weights_path: str | None = None,
        sovits_weights_path: str | None = None,
    ):
        """
        config_path: path to GPT-SoVITS's tts_infer.yaml (device/precision settings),
            relative to this project's root
        gpt_weights_path / sovits_weights_path: fine-tuned checkpoint paths, relative
            to this project's root. Leave as None to use the config's own stock
            (non-fine-tuned) weights for zero-shot cloning.
        """
        self.config_path = str(PROJECT_ROOT / config_path)
        self.gpt_weights_path = (
            str(PROJECT_ROOT / gpt_weights_path) if gpt_weights_path else None
        )
        self.sovits_weights_path = (
            str(PROJECT_ROOT / sovits_weights_path) if sovits_weights_path else None
        )
        self._pipeline = None  # loaded lazily in load()

    def load(self):
        """Load the model onto GPU. Call once at server startup, not per-request.

        GPT-SoVITS's own config resolves several of its asset paths as bare
        relative strings against the process cwd, so cwd is temporarily
        pointed at GPT-SoVITS's own repo root for the duration of this call
        only, then restored, so embedding this engine inside a larger app
        doesn't leave that app's own relative-path file access broken for
        the rest of the process's life.
        """
        _sync_pronunciations()

        original_cwd = os.getcwd()
        os.chdir(GPT_SOVITS_ROOT)
        sys.path.append(str(GPT_SOVITS_ROOT))
        sys.path.append(str(GPT_SOVITS_ROOT / "GPT_SoVITS"))
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

            config = TTS_Config(self.config_path)
            if self.gpt_weights_path:
                config.t2s_weights_path = self.gpt_weights_path
            if self.sovits_weights_path:
                config.vits_weights_path = self.sovits_weights_path
            self._pipeline = TTS(config)
        finally:
            os.chdir(original_cwd)

    def generate(
        self,
        text: str,
        emotion: EmotionPreset | str = "neutral",
        voice: str | None = None,
        text_lang: str = "en",
    ) -> bytes:
        """
        text: what to say
        emotion: either an EmotionPreset or a preset name (see
            emotion_vectors/presets.py), supplies the sampling params that
            carry the target tone (and a default reference clip, used only
            when voice is None)
        voice: name of a voice from voices/ (see voices.py), supplies the
            reference audio clip and its transcript. Leave as None to use
            emotion's own reference clip instead.
        Returns: WAV audio bytes
        """
        if isinstance(emotion, str):
            emotion = get_preset(emotion)

        if self._pipeline is None:
            raise RuntimeError("Call .load() before generate().")

        if voice is not None:
            selected_voice = get_voice(voice)
            ref_audio_path = selected_voice.ref_audio_path
            prompt_text = selected_voice.prompt_text
        else:
            ref_audio_path = emotion.ref_audio_path
            prompt_text = emotion.prompt_text

        req = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": str(PROJECT_ROOT / ref_audio_path),
            "prompt_text": prompt_text,
            "prompt_lang": emotion.prompt_lang,
            "top_k": emotion.top_k,
            "top_p": emotion.top_p,
            "temperature": emotion.temperature,
            "speed_factor": emotion.speed_factor,
        }
        generator = self._pipeline.run(req)
        sample_rate, audio_data = next(generator)

        # GPT-SoVITS returns raw int16 PCM, not normalized float, post-
        # effects need to work in proper [-1, 1] float space, or gain/
        # tremor arithmetic silently corrupts the scale (confirmed directly:
        # applying gain to int16 data produced peak=1.0/rms=0.92, i.e.
        # hard-clipped noise, not "louder audio").
        if emotion.post_effect or emotion.gain_db:
            audio_data = audio_data.astype(np.float32) / 32768.0

            if emotion.post_effect == "amplitude_tremor":
                audio_data = prosody_fx.amplitude_tremor(audio_data, sample_rate)
            if emotion.gain_db:
                audio_data = audio_data * (10 ** (emotion.gain_db / 20))
            audio_data = np.clip(audio_data, -1.0, 1.0)

        buffer = BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        return buffer.getvalue()

    def generate_multi(
        self,
        segments: list[tuple[str, str]],
        voice: str | None = None,
        text_lang: str = "en",
    ) -> bytes:
        """
        Generates a single reply that shifts emotional tone across its own
        sentences, e.g. relief giving way to exhaustion, rather than staying
        in one tone the whole way through. Not for blending multiple
        emotions into a single instant, GPT-SoVITS's sampling knobs
        (temperature/speed/gain) aren't meaningfully additive that way, so
        this generates each segment with its own emotion and concatenates
        them in order, with a short natural pause between segments.

        segments: ordered list of (text, emotion_name) pairs, same voice
            throughout. A single segment is just a plain generate() call.
        Returns: WAV audio bytes
        """
        if not segments:
            raise ValueError("generate_multi needs at least one (text, emotion) segment")

        if len(segments) == 1:
            text, emotion = segments[0]
            return self.generate(text, emotion=emotion, voice=voice, text_lang=text_lang)

        chunks = []
        sample_rate = None
        for text, emotion in segments:
            wav_bytes = self.generate(text, emotion=emotion, voice=voice, text_lang=text_lang)
            data, sr = sf.read(BytesIO(wav_bytes), dtype="float32")
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise RuntimeError(f"Segment sample rate mismatch: {sr} != {sample_rate}")
            chunks.append(data)

        # A brief silence between segments reads as a natural pause between
        # sentences rather than an abrupt jump-cut in tone.
        gap = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
        combined = chunks[0]
        for chunk in chunks[1:]:
            combined = np.concatenate([combined, gap, chunk])

        buffer = BytesIO()
        sf.write(buffer, combined, sample_rate, format="WAV")
        return buffer.getvalue()
