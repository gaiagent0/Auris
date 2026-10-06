"""A Silero VAD QA-integráció tesztjei (core.qa.vad_speech_spans).

A fallback-utak mindig futnak (a reader-venvben a sherpa-onnx nincs
telepítve). A valódi VAD-utat csak akkor futtatjuk, ha a sherpa-onnx
importálható ÉS a hub modellfájlja megvan — így a teszt más gépen is átfut.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import qa

HUB_VAD = qa.SILERO_VAD_DEFAULT
HUB_VOICE = os.path.join(
    r"D:\VoiceAI", "voices", "supertonic", "F1",
    "02-Az_öreg_kőház_udvarán_csende__serial.wav",
)


def _sherpa_available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return os.path.isfile(HUB_VAD)


def _tone(path, seconds=1.0, amp=0.2, sr=24000):
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (np.sin(2 * np.pi * 220 * t) * amp).astype("float32"), sr)
    return path


class VadFallbackTest(unittest.TestCase):
    def test_vad_off_keeps_legacy_result_and_reports_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _tone(os.path.join(tmp, "t.wav"), 0.5)
            with patch.dict(os.environ, {"VOICEAI_QA_VAD": "off"}):
                result = qa.analyze_audio(path)
        self.assertIsNone(result["vad"])
        for key in ("flags", "longest_pause_sec", "leading_silence_sec",
                    "trailing_silence_sec", "sec_per_char"):
            self.assertIn(key, result)

    def test_missing_model_falls_back_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _tone(os.path.join(tmp, "t.wav"), 0.5)
            env = {"VOICEAI_SILERO_VAD": os.path.join(tmp, "nincs_ilyen.onnx")}
            with patch.dict(os.environ, env):
                result = qa.analyze_audio(path)
        self.assertIsNone(result["vad"])
        self.assertIn("flags", result)

    def test_vad_never_raises_on_empty_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.wav")
            sf.write(path, np.zeros(0, dtype="float32"), 24000)
            result = qa.analyze_audio(path)
        self.assertIn("silent", result["flags"])


@unittest.skipUnless(_sherpa_available(), "sherpa-onnx vagy a Silero modell nincs meg")
class SileroVadPathTest(unittest.TestCase):
    def test_spans_on_real_speech_and_gap_detection(self):
        audio, sr = sf.read(HUB_VOICE, dtype="float32")
        spans = qa.vad_speech_spans(audio, sr)
        self.assertTrue(spans, "a valódi beszéden nem talált beszéd-szakaszt")

        x16 = qa._resample_to_16k(audio, sr)[0]
        gap = np.zeros(int(2.0 * 16000), dtype=np.float32)
        mix = np.concatenate([x16[:int(2.0 * 16000)], gap,
                              x16[int(2.2 * 16000):int(4.2 * 16000)]])
        spans2 = qa.vad_speech_spans(mix, 16000)
        pauses = [spans2[i + 1][0] - spans2[i][1] for i in range(len(spans2) - 1)]
        self.assertTrue(any(abs(p - 2.0) < 0.3 for p in pauses),
                        f"2,0 s szünet nem detektálható: {pauses}")

    def test_analyze_audio_uses_vad_and_keeps_flags_consistent(self):
        audio, sr = sf.read(HUB_VOICE, dtype="float32")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f1.wav")
            sf.write(path, audio, sr)
            result = qa.analyze_audio(path, "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban.")
        self.assertEqual(result["vad"], "silero")
        self.assertLess(result["longest_pause_sec"], 1.5)
        self.assertNotIn("silent", result["flags"])


if __name__ == "__main__":
    unittest.main()
