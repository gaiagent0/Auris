"""Sherpa-onnx csomagolású Supertonic 3 int8 — opcionális második motor az Aurisban.

Kérdés: hogyan működik a sherpa-onnx 1.13.8 ``OfflineTts`` Supertonic
kivételével Runtime CPU-n, natív ARM64-en (Snapdragon X Elite)?

Eredmény (M-43, 2026-10-06, TELJES — 24 mondatos hu_wer korpusz):
  - betöltés: ~0,7 s
  - RTF serial 0,1401 = 428,4 perc hang/óra (sherpa-onnx, 44,1 kHz)
  - RTF batch-út 0,1419 = 422,8 perc hang/óra (a sherpa OfflineTts nem
    csoportosít; a különbség zajszintű)
  - az Auris saját Supertonic serial (0,7624) → 5,4×, batch4 (0,3348) → 2,4×
    gyorsabb
  - WER 0,11 % / CER 0,03 % (24 magyar mondat, Parakeet, M-39 protokoll)

Korlát 2026-10-06:
  - sherpa csomagolás: 44 100 Hz kimenet (Auris cache 24 000 Hz, ``resample``
    révén illeszthető), egyetlen beépített hang (voice.bin, sid=0)
  - vázlat: ez egy opcionális motor, a SupertonicEngine ez esetben
    a megjelenítéséhez. ASettingsnél tükrözni kell a 44,1 kHz kimenetet és
    az egy hang limitjét, mielőtt a felhasználó választhatná.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf

from core.local_engines import LocalEngineBase, resample
from core.tts_engine import SAMPLE_RATE, _audio_duration, _write_audio_atomic


class SherpaSupertonicEngine(LocalEngineBase):
    """sherpa-onnx csomagolású Supertonic 3 int8, CPU / natív ARM64."""

    engine_name = "supertonic_sherpa"

    def __init__(self) -> None:
        super().__init__()
        self.tts: Any = None
        self.sample_rate: int = 44100
        self._default_voice_name = "F1"

    # ── capabilities ──────────────────────────────────────────────────────
    @property
    def capabilities(self) -> dict[str, Any]:
        base = {
            "label": "Supertonic 3 (sherpa-onnx int8, CPU, 44,1 kHz)",
            "voice_clone": False,
            "voice_design": False,
            "speed": True,
            "device": "CPU (natív ARM64)",
            "hungarian": "hivatalos",
            "license": "OpenRAIL-M (modell), MIT (kód)",
        }
        return base

    def model_location(self) -> str:
        return SherpaSupertonicEngine.model_dir().as_posix()

    @staticmethod
    def model_dir() -> Path:
        """Sherpa-onnx Supertonic 3 int8 csomag helye (a hub vendorban)."""
        return Path(
            os.environ.get(
                "VOICEAI_SHERPA_SUPERTONIC",
                r"D:\VoiceAI\vendor\sherpa-models"
                r"\sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
            )
        )


# ════════════════════════════════════════════════════════════════════════════
# Betöltés / mintafuttatás / mérés (ez a fájl maga is futtatható vizsgálatként)
# ════════════════════════════════════════════════════════════════════════════

SUPERTONIC_TEXTS_FOR_MEASUREMENT = [
    "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban.",
    "A tengerparton reggel még köd lebegett a fák között.",
    "Ebben a könyvben egy magyar nyelvű mesét találhatsz.",
]


def _make_tts(model_dir: Path) -> tuple[Any, float]:
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
    return tts, t_load


def synthesize_one(tts: Any, text: str, *, speed: float = 1.0) -> dict[str, Any]:
    """Egy mondat sszedése, (számítási idő, kimeneti minta)."""

    t0 = time.perf_counter()
    audio = tts.generate(text, sid=0, speed=speed)
    wall = time.perf_counter() - t0
    dur = len(audio.samples) / audio.sample_rate
    return {
        "wall_s": wall,
        "audio_s": dur,
        "rtf": wall / dur if dur else float("nan"),
        "sample_rate": audio.sample_rate,
        "wav": np.asarray(audio.samples, dtype=np.float32),
    }


def resample_to_cache(wav: np.ndarray, sr: int) -> np.ndarray:
    """44,1 kHz → 24 kHz mono, hogy az Auris cache-be is beilleszkedjen."""
    audio = resample(wav.reshape(-1), sr, SAMPLE_RATE)
    if audio.size and float(np.max(np.abs(audio))) > 1.0:
        audio = audio / float(np.max(np.abs(audio))) * 0.98
    return audio


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr, subtype="PCM_16")


def run_smoke_for_measure(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Mérési futtatás: 3 magyar mondat, RTF + kimeneti wav + json.

    Ez a függvény nem része az Auris runtime-nek; mérésre és
    reprodukálható példaként szolgál.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tts, t_load = _make_tts(model_dir)
    report: dict[str, Any] = {
        "engine": "supertonic_sherpa (sherpa-onnx supertonic int8)",
        "model_dir": model_dir.as_posix(),
        "load_s": round(t_load, 3),
        "sample_rate": tts.sample_rate,
        "sentences": [],
    }
    total_wall = total_audio = 0.0
    for i, text in enumerate(SUPERTONIC_TEXTS_FOR_MEASUREMENT, 1):
        r = synthesize_one(tts, text, speed=1.0)
        wav_24k = resample_to_cache(r["wav"], r["sample_rate"])
        wav_path = out_dir / f"sherpa_supertonic_{i:02d}.wav"
        write_wav(str(wav_path), wav_24k, SAMPLE_RATE)
        report["sentences"].append(
            {
                "text": text,
                "wall_s": round(r["wall_s"], 3),
                "audio_s": round(r["audio_s"], 3),
                "rtf": round(r["rtf"], 4),
                "wav": wav_path.as_posix(),
            }
        )
        total_wall += r["wall_s"]
        total_audio += r["audio_s"]
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
    rpt = run_smoke_for_measure(model_dir, out)
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
