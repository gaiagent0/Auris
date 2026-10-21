"""XTTSv2-Streaming-ONNX — a gépen lévő, valódi hangklónozó motor az Aurisban.

Ez az a motor, amelyik tényleg klónoz: a felöltött referenciahangból
`get_conditioning_latents`-tel kinyeri a hang ujjlenyomatát, és arra szintetizál.
A Supertonic (beépített hangok) és a sherpa-Supertonic ezzel szemben NEM klónoz,
ezért náluk a Hangstúdió referencia-szakasza rejtve marad.

Mért adatok (M-41/M-49, Snapdragon X Elite, int8 GPT):
  - betöltés: ~1 s
  - conditioning: ~0,9 s (referenciánként egyszer, utána gyorsítótárazva)
  - szintetizálás: 2,27 s hang 3,9 s alatt (0,58x valós idejű, 4 szál)
  - minőség: a minta-klon könyv CER 2,73% / WER 11,94% (Parakeet ellenőrzés)

A modell nem a repóban van (2,1 GB), hanem a gép modell-tárházában; az
útvonal a `VOICEAI_XTTS_DIR` felülírással állítható. A spacy ARM64-en nincs
meg, ezért a `models/xtts-stubs/` alól töltjük be — a magyar út nem használja.

Hiányzó modell esetén a betöltés magyar hibaüzenetet ad a letöltési
forrással (`huggingface.co/pltobing/XTTSv2-Streaming-ONNX`).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.local_engines import LocalEngineBase, _setting
from core.tts_engine import SAMPLE_RATE

# A modell könyvtár tartalma (a hub registry models.json-ából).
XTTS_FILES = (
    "gpt_model_int8.onnx", "conditioning_encoder.onnx", "hifigan_vocoder.onnx",
    "speaker_encoder.onnx", "vocab.json", "mel_stats.npy",
)

# A spacy-stub helye: ARM64-en nincs wheel, a magyar út nem használja.
STUBS_DIRNAME = "xtts-stubs"


class XttsOnnxEngine(LocalEngineBase):
    """XTTSv2 ONNX, valódi referenciahang-klónozás CPU-n (ARM64)."""

    engine_name = "xtts_onnx"
    cache_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.tts: Optional[Any] = None
        # A conditioning latensek gyorsítótára: referenciafájl -> (cond, spk).
        # Egy könyv minden mondata ugyanazt a referenciát használja, így a
        # ~0,9 s-os kinyerés csak az első mondatnál fut le.
        self._latents: dict[str, tuple] = {}
        self._latents_lock = threading.Lock()

    # ── képességek ──────────────────────────────────────────────────────────
    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "label": "XTTSv2-ONNX int8 (CPU, klónozás, 24 kHz)",
            "voice_clone": True,
            "voice_design": False,
            "speed": True,
            "device": "CPU (natív ARM64)",
            "hungarian": "jó (mért CER 2,73%)",
            "license": "CPML (modell), MIT (kód)",
        }

    # ── modell ──────────────────────────────────────────────────────────────
    @staticmethod
    def model_dir() -> Path:
        override = os.environ.get("VOICEAI_XTTS_DIR", "").strip()
        if override:
            return Path(override)
        return Path(r"D:\<projekt>\vendor\XTTSv2-Streaming-ONNX\xtts_onnx")

    @staticmethod
    def _stubs_dir() -> Path:
        from core.local_engines import MODELS_DIR

        return MODELS_DIR / STUBS_DIRNAME

    def model_location(self) -> str:
        return self.model_dir().as_posix()

    def model_present(self) -> bool:
        d = self.model_dir()
        return all((d / name).is_file() for name in XTTS_FILES)

    def _load_model(self) -> None:
        d = self.model_dir()
        if not self.model_present():
            raise RuntimeError(
                "A XTTSv2-ONNX modell nincs meg ("
                + d.as_posix()
                + "). Töltsd le a hiteles forrásból: huggingface.co/"
                "pltobing/XTTSv2-Streaming-ONNX — vagy állítsd be a "
                "VOICEAI_XTTS_DIR környezeti változót a modell könyvtárára."
            )
        # A pipeline a saját moduljait a modellkönyvtárból importálja, a
        # spacy-t pedig a stub-könyvtárból vesszük (ARM64-en nincs wheel).
        for path in (str(self._stubs_dir()), str(d.parent)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from xtts_streaming_pipeline import StreamingTTSPipeline

        # A pipeline importkor `basicConfig(INFO)`-ot hív, és mondatonként
        # száz INFO sort ír a `data_processing_pipeline` loggeren. Az import
        # UTÁN állítjuk le, különben a basicConfig visszakapcsolja.
        import logging

        for name in ("data_processing_pipeline", "root"):
            logging.getLogger(name).setLevel(logging.WARNING)

        threads = self._threads()
        t0 = time.perf_counter()
        self.tts = StreamingTTSPipeline(
            model_dir=str(d),
            vocab_path=str(d / "vocab.json"),
            mel_norms_path=str(d / "mel_stats.npy"),
            use_int8_gpt=self._int8(),
            num_threads_gpt=threads,
        )
        t_load = time.perf_counter() - t0
        self._detail = (
            f"CPU · XTTSv2 ONNX {'int8' if self._int8() else 'fp32'} · "
            f"{threads} szál · betöltés {t_load:.1f}s"
        )

    def _release(self) -> None:
        self.tts = None
        with self._latents_lock:
            self._latents.clear()

    # ── beállítások ─────────────────────────────────────────────────────────
    @staticmethod
    def _int8() -> bool:
        try:
            return bool(_setting("xtts_int8", True))
        except Exception:
            return True

    @staticmethod
    def _threads() -> int:
        try:
            return max(1, min(8, int(_setting("xtts_threads", 4))))
        except (TypeError, ValueError):
            return 4

    def settings_identity(self) -> str:
        # Az int8 és az fp32 más hangot ad ugyanarra a szövegre, ezért a
        # gyorsítótárnak külön kell lennie a kettőnek.
        return f"int8={self._int8()}|thr={self._threads()}"

    # ── klónozás ────────────────────────────────────────────────────────────
    def voice_identity(self, instruct, ref_audio, ref_text, language) -> str:
        # Klónozó motor: a referencia hang dönti el a hangot, ezért a fájl
        # tartalmának hash-e a cache-kulcs (ne az útvonal, mert az változhat).
        if not ref_audio:
            return "clone=<nincs referencia>"
        try:
            import hashlib

            digest = hashlib.md5()
            with open(ref_audio, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return f"clone={digest.hexdigest()[:16]}"
        except OSError:
            return f"clone={os.path.basename(str(ref_audio))}"

    def _conditioning(self, ref_wav: str):
        """A referencia hang ujjlenyomata, fájlonként egyszer számolva."""
        with self._latents_lock:
            cached = self._latents.get(ref_wav)
        if cached is not None:
            return cached
        if self.tts is None:
            raise RuntimeError("A XTTS motor nincs betöltve.")
        cond, spk = self.tts.get_conditioning_latents(ref_wav)
        with self._latents_lock:
            # A memóriát korlátozzuk: a legrégebbi kilencset ejtjük.
            if len(self._latents) >= 10:
                self._latents.pop(next(iter(self._latents)), None)
            self._latents[ref_wav] = (cond, spk)
        return cond, spk

    def _synthesize(self, text, instruct, ref_audio, ref_text, speed, language):
        if self.tts is None:
            raise RuntimeError("A XTTS motor nincs betöltve.")
        if not ref_audio:
            raise RuntimeError(
                "A XTTSv2-ONNX csak hangklónozással működik: tölts fel "
                "referenciahangot a Hangstúdióban (Narrátor vagy Szereplők), "
                "vagy válassz klónozás nélküli motort (Supertonic 3)."
            )
        cond, spk = self._conditioning(str(ref_audio))
        lang = str(language or "").lower()[:2] or "hu"
        chunks = list(self.tts.inference_stream(
            text=text, language=lang, gpt_cond_latent=cond,
            speaker_embedding=spk, stream_chunk_size=4,
            speed=max(0.7, min(1.6, float(speed or 1.0))),
        ))
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, np.float32)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 1.0:
                audio = audio / peak * 0.98
        # A XTTS 24 kHz-en ad, ami megegyezik az Auris cache-sel — nem kell
        # újramintavételezni.
        return audio, SAMPLE_RATE
