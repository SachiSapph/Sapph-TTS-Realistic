"""
Tests the actual TTSEngine wrapper (inference/generate.py) end to end,
zero-shot against the stock base model, across every emotion preset
(emotion_vectors/presets.py). Outputs land in test_audio/ regardless of
where this script lives.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from emotion_vectors.presets import PRESETS  # noqa: E402
from inference.generate import TTSEngine  # noqa: E402

engine = TTSEngine(config_path="GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml")
engine.load()

for preset_name in PRESETS:
    audio_bytes = engine.generate(
        "Hey there! Thanks for testing the pipeline with me today.",
        emotion=preset_name,
    )
    out_path = PROJECT_ROOT / "test_audio" / f"wrapper_test_{preset_name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    print(f"{preset_name} -> {out_path} ({len(audio_bytes)} bytes)")
