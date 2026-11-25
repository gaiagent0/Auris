"""Beszélő-embedding (CAM++) — a szinkron **egyetlen** identitás-bizonyítéka.

Miért embedding és nem diarizációs címke: a diarizátor a *szegmenseket*
címkézi, és a címkéi rövid, átfedő vagy zajos részeken megbízhatatlanok
(mérés: E05-ben a szomszédos, akusztikusan azonos sorpárok 72%-a más-más
magyar hangot kapott a diarizáció-átfedés alapján). Az embedding ezzel
szemben minden egyes megszólalásról **közvetlen** mérést ad, és ugyanazzal
a metrikával össze is hasonlítható.

Modell: 3D-Speaker CAM++ (Apache-2.0), 16 kHz bemenet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio

_model = None
_extractor = None


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class SpeakerEmbedder:
    """CAM++ embedding-kinyerő (egyszer betöltve, újrahasználva)."""

    def __init__(self, model_path: str | Path, num_threads: int = 2):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"hiányzó beszélő-embedding modell: {self.model_path}\n"
                f"  (CAM++ onnx, pl. 3D-Speaker / modelscope)")
        self._ext = None
        self._threads = num_threads

    def _load(self):
        if self._ext is None:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path), num_threads=self._threads, provider="cpu")
            self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._ext

    def embed(self, x16k: np.ndarray) -> np.ndarray | None:
        """Egy beszédminta (16 kHz mono) embeddingje, L2-normalizálva."""
        ext = self._load()
        st = ext.create_stream()
        st.accept_waveform(audio.SR16, np.ascontiguousarray(x16k, dtype=np.float32))
        st.input_finished()
        if not ext.is_ready(st):
            return None
        v = np.asarray(ext.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def embed_windows(self, x, sr: int, spans: list[tuple[float, float]]
                      ) -> list[np.ndarray | None]:
        """Több időablak embeddingje ugyanabból a sávból (CPU-n sorban)."""
        x16 = audio.to_16k(x, sr)
        out: list[np.ndarray | None] = []
        for a, b in spans:
            core = audio.speech_core(x16, audio.SR16, a, b)
            out.append(None if core is None else self.embed(core))
        return out


def centroid(vecs: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray | None:
    """Súlyozott (beszédidő-arányos) centroid."""
    use = [(v, 1.0 if not weights else float(weights[i]))
           for i, v in enumerate(vecs) if v is not None]
    if not use:
        return None
    w = np.asarray([u[1] for u in use], dtype=np.float32)
    M = np.stack([u[0] for u in use])
    c = (M * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-6)
    return norm(c)
