"""Sherpa-onnx csomagolású Supertonic 3 int8 — opcionális második motor az Aurisban.

Kérdés (2026-10-06): a sherpa-onnx csomagolású Supertonic int8 hogyan áll
az Auris SupertonicEngine-éhez képest (batch4 RTF 0,3348, serial 0,7624).

Eredmény (M-41, 2026-10-06):
  - betöltés: ~1,7 s
  - RTF serial 0,1497 → 400,7 perc hang/óra (sherpa-onnx batch path)
  - Auris serial 0,7624-hez képest 5,1×, Auris batch4 0,3348-hoz képest 2,2× gyorsabb
  - WER és CER (M-39-es protokoll ASR checkerje): 0,00% / 0,00%

Korlát 2026-10-06:
  - sherpa csomagolás: 44 100 Hz kimenet (Auris cache 24 000 Hz), egyetlen beépített hang (voice.bin)
  - Az Auris Settings oldala tükrözi a 44,1 kHz kimenetet és az 1 hang limitjét.

Futtatása:
  D:\VoiceAI\envs\sherpa-arm64\Scripts\python.exe hub\scripts\sherpa_supertonic_probe.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from core.local_engines import LocalEngineBase, resample
from core.tts_engine import SAMPLE_RATE, _audio_duration, _write_audio_atomic


class SherpaSupertonicEngine(LocalEngineBase):
    engine_name = "supertonic_sherpa"
    cache_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.tts: Optional[Any] = None
        self.sample_rate: int = 44100

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "label": "Supertonic 3 (sherpa-onnx int8, CPU, 44,1 kHz)",
            "voice_clone": False,
            "voice_design": False,
            "speed": True,
            "device": "CPU (natív ARM64)",
            "hungarian": "hivatalos",
            "license": "OpenRAIL-M (modell), MIT (kód)",
        }

    def model_location(self) -> str:
        return SherpaSupertonicEngine.model_dir().as_posix()

    def model_present(self) -> bool:
        d = SherpaSupertonicEngine.model_dir()
        return (d / "tts.json").is_file() and (d / "voice.bin").is_file()

    def _load_model(self) -> None:
        import sherpa_onnx

        d = SherpaSupertonicEngine.model_dir()
        cfg = sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=str(d / "duration_predictor.int8.onnx"),
            text_encoder=str(d / "text_encoder.int8.onnx"),
            vector_estimator=str(d / "vector_estimator.int8.onnx"),
            vocoder=str(d / "vocoder.int8.onnx"),
            tts_json=str(d / "tts.json"),
            unicode_indexer=str(d / "unicode_indexer.bin"),
            voice_style=str(d / "voice.bin"),
        )
        mc = sherpa_onnx.OfflineTtsModelConfig(
            supertonic=cfg, num_threads=4, debug=0, provider="cpu"
        )
        t0 = time.perf_counter()
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=mc))
        t_load = time.perf_counter() - t0
        self.sample_rate = getattr(self.tts, "sample_rate", 44100)
        self._detail = f"CPU · sherpa-onnx int8 · betöltés {t_load:.1f}s"

    def _release(self) -> None:
        self.tts = None

    def _synthesize(self, text: str, instruct, ref_audio, ref_text, speed, language) -> tuple[np.ndarray, int]:
        if self.tts is None:
            raise RuntimeError("Supertonic sherpa engine nem lett betöltve")
        audio = self.tts.generate(text, sid=0, speed=float(speed or 1.0))
        wav = np.asarray(audio.samples, dtype=np.float32)
        if wav.size:
            peak = float(np.max(np.abs(wav)))
            if peak > 1.0:
                wav = wav / peak * 0.98
        return resample_to_cache(wav, int(self.sample_rate)), SAMPLE_RATE


def resample_to_cache(wav: np.ndarray, sr: int) -> np.ndarray:
    audio = resample(wav, sr, SAMPLE_RATE)
    if audio.size and float(np.max(np.abs(audio))) > 1.0:
        audio = audio / float(np.max(np.abs(audio))) * 0.98
    return audio


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr, subtype="PCM_16")


# ════════════════════════════════════════════════════════════════════════════
# Mérés: három magyar mondat, RTF + kimeneti wav + json
# ════════════════════════════════════════════════════════════════════════════

SUPERTONIC_TEXTS_FOR_MEASUREMENT = [
    "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban.",
    "A tengerparton reggel még köd lebegett a fák között.",
    "Ebben a könyvben egy magyar nyelvű mesét találhatsz.",
]


def measure_sherpa_supertonic(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=str(model_dir / "duration_predictor.int8.onnx"),
        text_encoder=str(model_dir / "text_encoder.int8.onnx"),
        vector_estimator=str(model_dir / "vector_estimator.int8.onnx"),
        vocoder=str(model_dir / "vocoder.int8.onnx"),
        tts_json=str(model_dir / "tts.json"),
        unicode_indexer=str(model_dir / "unicode_indexer.bin"),
        voice_style=str(model_dir / "voice.bin"),
    )
    mc = sherpa_onnx.OfflineTtsModelConfig(
        supertonic=cfg, num_threads=4, debug=0, provider="cpu"
    )
    t0 = time.perf_counter()
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=mc))
    t_load = time.perf_counter() - t0
    report: dict[str, Any] = {
        "engine": "supertonic_sherpa (sherpa-onnx supertonic int8)",
        "model_dir": model_dir.as_posix(),
        "load_s": round(t_load, 3),
        "sample_rate": tts.sample_rate,
        "sentences": [],
    }
    total_wall = total_audio = 0.0
    for i, text in enumerate(SUPERTONIC_TEXTS_FOR_MEASUREMENT, 1):
        t0 = time.perf_counter()
        audio = tts.generate(text, sid=0, speed=1.0)
        wall = time.perf_counter() - t0
        wav = np.asarray(audio.samples, dtype=np.float32)
        dur = len(wav) / int(tts.sample_rate)
        total_wall += wall
        total_audio += dur
        wav_24k = resample_to_cache(wav, int(tts.sample_rate))
        wav_path = out_dir / f"sherpa_supertonic_{i:02d}.wav"
        write_wav(str(wav_path), wav_24k, SAMPLE_RATE)
        report["sentences"].append({
            "text": text,
            "wall_s": round(wall, 3),
            "audio_s": round(dur, 3),
            "rtf": round(wall / dur, 4),
            "wav": wav_path.as_posix(),
        })
    report["total_wall_s"] = round(total_wall, 3)
    report["total_audio_s"] = round(total_audio, 3)
    report["rtf"] = round(total_wall / total_audio, 4) if total_audio else float("nan")
    report["per_hour_audio_min"] = round(60.0 / report["rtf"], 1) if report["rtf"] else float("nan")
    import json

    with open(out_dir / "cases_sherpa_supertonic.json", "w", encoding="utf-8") as f:
        json.dump({s["wav"]: s["text"] for s in report["sentences"]}, f, ensure_ascii=False, indent=1)
    with open(out_dir / "report_sherpa_supertonic.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report


if __name__ == "__main__":
    model_dir = SherpaSupertonicEngine.model_dir()
    print(f"modell könyvtár: {model_dir}")
    if not model_dir.is_dir():
        print("HIBA: a sherpa Supertonic csomag nem található.")
        raise SystemExit(1)
    out = Path(os.environ.get("VOICEAI_OUT", r"D:\VoiceAI\out\sherpa-supertonic"))
    rpt = measure_sherpa_supertonic(model_dir, out)
    print("betöltés:", rpt["load_s"], "s")
    for s in rpt["sentences"]:
        print(
            f"#{s['text'][:8]}... wall={s['wall_s']:.2f}s hang={s['audio_s']:.2f}s "
            f"RTF={s['rtf']:.4f} ({s['wav']})"
        )
    print(
        f"Összesen: {rpt['total_wall_s']:.2f}s számítás / "
        f"{rpt['total_audio_s']:.2f}s hang = RTF {rpt['rtf']:.4f} = "
        f"{rpt['per_hour_audio_min']:.1f} perc hang / óra számítás"
    )
    print("kimenet:", out)
