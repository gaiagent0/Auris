"""A 13 magyar hang katalógusa — **fix generálási paraméterekkel**.

| motor | hangok | nem |
|---|---|---|
| Supertonic 3 (sherpa-onnx, int8) | F1–F5, M1–M5 | 5 nő + 5 férfi |
| Piper (hu_HU, VITS) | anna, berta, imre | 2 nő + 1 férfi |

Elv: a hang **paraméterei hozzá vannak rendelve a hanghoz**, nem a sorhoz.
A TTS mindig ezekkel a fix értékekkel szólal meg (``speed``,
``num_steps``, ``silence_scale`` / ``length_scale``, ``noise_scale``,
``noise_w``), az időzítést pedig utólag, ``atempo``-val igazítjuk. Így a hang
karaktere soronként nem változhat meg — ez a korábbi lánc egyik igazolt hibája
volt (soronként 0,7–1,7× TTS-sebesség).

Két szereplő megkülönböztetése ugyanazzal az alaphanggal: **fix**
``pitch`` eltolás (félhang) a szereplő-rekordban, ``rubberband`` szűrővel.
Ez is szereplőhöz kötött, tehát állandó.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio


@dataclass
class Voice:
    vid: str                       # belső azonosító (pl. "SUP_F1")
    label: str                     # rövid címke (pl. "F1", "anna")
    engine: str                    # "supertonic" | "piper"
    gender: str                    # "F" | "M"
    params: dict = field(default_factory=dict)
    sr: int = 44100
    pitch: float = 0.0             # fix hangszín-eltolás félhangban

    def to_dict(self) -> dict:
        return {"vid": self.vid, "label": self.label, "engine": self.engine,
                "gender": self.gender, "params": self.params, "sr": self.sr,
                "pitch": self.pitch}


#: a hangkészlet: nemenként a preferencia-sorrend (= a katalógus sorrendje)
CATALOG: list[Voice] = [
    Voice("SUP_F1", "F1", "supertonic", "F", {"sid": 0, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_F2", "F2", "supertonic", "F", {"sid": 1, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_F3", "F3", "supertonic", "F", {"sid": 2, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_F4", "F4", "supertonic", "F", {"sid": 3, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_F5", "F5", "supertonic", "F", {"sid": 4, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("PIP_anna", "anna", "piper", "F", {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}, 22050),
    Voice("PIP_berta", "berta", "piper", "F", {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}, 22050),
    Voice("SUP_M1", "M1", "supertonic", "M", {"sid": 5, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_M2", "M2", "supertonic", "M", {"sid": 6, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_M3", "M3", "supertonic", "M", {"sid": 7, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_M4", "M4", "supertonic", "M", {"sid": 8, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("SUP_M5", "M5", "supertonic", "M", {"sid": 9, "speed": 1.0, "num_steps": 8, "silence_scale": 0.20}),
    Voice("PIP_imre", "imre", "piper", "M", {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}, 22050),
]

BY_ID: dict[str, Voice] = {v.vid: v for v in CATALOG}


def pool(gender: str) -> list[Voice]:
    return [v for v in CATALOG if v.gender == gender]


def _auto_threads(value=None) -> int:
    """CPU-szálak: a beállított érték, különben a magok fele (2…8)."""
    if value not in (None, ""):
        try:
            return max(1, min(16, int(value)))
        except (TypeError, ValueError):
            pass
    try:
        import os
        cores = os.cpu_count() or 4
    except Exception:                            # pragma: no cover — platformfüggő
        cores = 4
    return max(2, min(8, cores // 2))


def _with_steps(voice: Voice, steps) -> Voice:
    """Supertonic lépésszám felülírása (a hang többi paramétere marad)."""
    if voice.engine != "supertonic" or steps in (None, ""):
        return voice
    try:
        n = int(steps)
    except (TypeError, ValueError):
        return voice
    if not 4 <= n <= 32 or n == int(voice.params.get("num_steps", 8)):
        return voice
    return Voice(voice.vid, voice.label, voice.engine, voice.gender,
                 {**voice.params, "num_steps": n}, voice.sr, voice.pitch)


# ── motorok ──────────────────────────────────────────────────────────────
class SupertonicEngine:
    """Supertonic 3 (sherpa-onnx int8). Fix paraméterek: sid, speed, steps."""

    name = "supertonic"

    def __init__(self, model_dir: str | Path, num_threads: int = 4):
        self.dir = Path(model_dir)
        need = ["duration_predictor.int8.onnx", "text_encoder.int8.onnx",
                "vector_estimator.int8.onnx", "vocoder.int8.onnx", "tts.json",
                "unicode_indexer.bin", "voice.bin"]
        missing = [n for n in need if not (self.dir / n).is_file()]
        if missing:
            raise FileNotFoundError(f"hiányzó Supertonic fájlok: {self.dir} {missing}")
        self._tts = None
        self._threads = num_threads

    def _load(self):
        if self._tts is None:
            import sherpa_onnx

            d = self.dir
            c = sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"),
                tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"),
                voice_style=str(d / "voice.bin"))
            mc = sherpa_onnx.OfflineTtsModelConfig(supertonic=c, num_threads=self._threads,
                                                   debug=0, provider="cpu")
            self._tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=mc))
        return self._tts

    def synth(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        import sherpa_onnx

        tts = self._load()
        p = voice.params
        cfg = sherpa_onnx.GenerationConfig()
        cfg.sid = int(p.get("sid", 0))
        cfg.speed = float(p.get("speed", 1.0))
        cfg.num_steps = int(p.get("num_steps", 8))
        cfg.silence_scale = float(p.get("silence_scale", 0.2))
        a = tts.generate(text, cfg)
        return np.asarray(a.samples, dtype=np.float32), int(a.sample_rate)


class PiperEngine:
    """Piper hu_HU (VITS, onnxruntime) espeak-ng fonémázással.

    A Piper wheel nem része a projektnek, ezért közvetlenül az ONNX-modellt
    futtatjuk: az espeak-ng adja az IPA fonémákat, a modell JSON-jából
    származó ``phoneme_id_map`` pedig a fonéma-azonosítókat. Így nincs
    telepítési függőség, és a paraméterek (``length_scale``, ``noise_scale``,
    ``noise_w``) pontosan azok, amiket a modell konfigurációja definiál.
    """

    name = "piper"

    def __init__(self, model_dir: str | Path, espeak: str | Path | None = None):
        self.dir = Path(model_dir)
        self.espeak = str(espeak) if espeak else "espeak-ng"
        self._cache: dict[str, tuple] = {}
        if not self.dir.is_dir():
            raise FileNotFoundError(f"nincs Piper modellkönyvtár: {self.dir}")

    def _voice_model(self, label: str):
        if label in self._cache:
            return self._cache[label]
        onnx = self.dir / f"hu_HU-{label}-medium.onnx"
        cfg = onnx.with_suffix(".onnx.json")
        if not onnx.is_file() or not cfg.is_file():
            raise FileNotFoundError(f"nincs Piper hang: {onnx}")
        meta = json.loads(cfg.read_text(encoding="utf-8"))
        pid = meta["phoneme_id_map"]
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self._cache[label] = (sess, meta, pid,
                              sorted(pid, key=len, reverse=True),
                              int(meta["audio"]["sample_rate"]))
        return self._cache[label]

    def phonemize(self, text: str) -> str:
        # A szöveget **stdin-en** adjuk át: ha argv-n menne, az esetleg
        # "-"-szal vagy "--"-szal kezdődő sor mondatot az espeak option-ként
        # értelmezi, és csendben üres IPA-t ad vissza (mért hiba:
        # "- Add ide a keffijédet!" -> "").
        p = subprocess.run([self.espeak, "-q", "--ipa=3", "-v", "hu"],
                           input=text.encode("utf-8"), capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"espeak-ng hiba: {p.stderr.decode('utf-8', 'replace')[-200:]}")
        return " ".join(p.stdout.decode("utf-8", "replace").split())

    def synth(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        sess, meta, pid, keys, sr = self._voice_model(voice.label)
        ipa = self.phonemize(text)
        ids: list[int] = []
        i = 0
        while i < len(ipa):
            for k in keys:
                if ipa.startswith(k, i):
                    ids.extend(pid[k])
                    i += len(k)
                    break
            else:
                i += 1
        if not ids:
            raise RuntimeError(f"üres fonéma-sorozat: {text!r} -> {ipa!r}")
        seq = [1]
        for x in ids:
            seq += [x, 0]
        seq.append(2)
        inf = meta.get("inference", {})
        p = voice.params
        scales = np.array([float(p.get("noise_scale", inf.get("noise_scale", 0.667))),
                           float(p.get("length_scale", inf.get("length_scale", 1.0))),
                           float(p.get("noise_w", inf.get("noise_w", 0.8)))],
                          dtype=np.float32)
        a = np.array([seq], dtype=np.int64)
        y = sess.run(None, {"input": a,
                            "input_lengths": np.array([a.shape[1]], dtype=np.int64),
                            "scales": scales})[0]
        return np.asarray(y).reshape(-1).astype(np.float32), sr


class VoiceBank:
    """A projektben elérhető hangok és a motorjaik.

    A projekt ``tuning`` szakasza két gépi terhelést szabályozó kulcsot ismer
    (mindkettő elhagyható — akkor a laptophoz méretezett alapérték él):
      * ``synth_threads`` — CPU-szálak száma (alap: a magok fele, 2…8);
      * ``synth_steps``   — Supertonic lépésszám (alap 8; 6 = gyorsabb,
        12 = szebb, lassabb).
    """

    def __init__(self, project, out_sr: int | None = None, verbose: bool = True):
        self.project = project
        self.out_sr = int(out_sr or project.tts_sr)
        self.engines: dict[str, object] = {}
        self.available: dict[str, Voice] = {}
        self.unavailable: dict[str, str] = {}
        st = project.model("supertonic")
        pi = project.model("piper")
        es = project.model("espeak")
        tuning = project.tuning or {}
        steps = tuning.get("synth_steps")
        threads = _auto_threads(tuning.get("synth_threads"))
        for v in CATALOG:
            try:
                if v.engine == "supertonic":
                    if "supertonic" not in self.engines:
                        self.engines["supertonic"] = SupertonicEngine(st, num_threads=threads)
                else:
                    if "piper" not in self.engines:
                        self.engines["piper"] = PiperEngine(pi, es)
                self.available[v.vid] = _with_steps(v, steps)
            except Exception as e:
                self.unavailable[v.vid] = str(e)
        if verbose:
            print(f"[hang] elérhető {len(self.available)}/{len(CATALOG)} hang "
                  f"({sum(1 for v in self.available.values() if v.gender == 'F')} nő, "
                  f"{sum(1 for v in self.available.values() if v.gender == 'M')} férfi)")
            for vid, err in self.unavailable.items():
                print(f"[hang] ! {vid} nem elérhető: {err}")

    def pool(self, gender: str) -> list[Voice]:
        return [v for v in CATALOG
                if v.gender == gender and v.vid in self.available]

    def synth(self, text: str, voice: Voice, pitch: float = 0.0
              ) -> tuple[np.ndarray, int]:
        """Szintézis a hang fix paramétereivel (+ fix szereplő-hangszín)."""
        eng = self.engines[voice.engine]
        x, sr = eng.synth(text, voice)      # type: ignore[attr-defined]
        if len(x) == 0:
            raise RuntimeError(f"üres TTS-kimenet ({voice.vid}): {text[:60]!r}")
        x = audio.trim_silence(x, sr)       # a modell csendje ne egye az időt
        if pitch:
            x = audio.pitch_shift(x, sr, pitch)
        if sr != self.out_sr:
            x = audio._resample_poly(x, sr, self.out_sr)
            sr = self.out_sr
        return x, sr
