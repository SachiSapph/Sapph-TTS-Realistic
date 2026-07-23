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
        """Load the model onto GPU. Call once at server startup, not per-request."""
        # GPT-SoVITS's config resolves its own relative paths (bert_base_path,
        # t2s_weights_path, etc.) against its own repo root, mirroring what
        # its own api_v2.py does at import time.
        os.chdir(GPT_SOVITS_ROOT)
        sys.path.append(str(GPT_SOVITS_ROOT))
        sys.path.append(str(GPT_SOVITS_ROOT / "GPT_SoVITS"))

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        config = TTS_Config(self.config_path)
        if self.gpt_weights_path:
            config.t2s_weights_path = self.gpt_weights_path
        if self.sovits_weights_path:
            config.vits_weights_path = self.sovits_weights_path
        self._pipeline = TTS(config)

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
