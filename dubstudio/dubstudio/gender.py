"""Nem (férfi/nő) meghatározása tanított osztályozóval — nem F0-ból.

Az F0-alapú tippelést azért dobtuk el, mert a férfi és a női beszéd
F0-tartománya átfed, a kiabálás/suttogás/zene és a szűk sávú kodek pedig
tovább rontja. Az osztályozó **megnyilatkozás-szintű** mérést ad, amit
klaszter szinten aggregálunk.

Modell: ``alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech``
(HF-cache-ből, internet nélkül is működik, ha már le van töltve).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import audio

MODEL_ID = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

_model = None
_extractor = None
_female_idx = 0


def _torchvision_shim() -> None:
    """A venv torchvision shimje nem tartalmazza az io/transforms almodulokat,
    ezért a transformers importja elhasal; wav2vec2 következtetéshez üres
    helyőrző modulok elegendőek."""
    import importlib.abc
    import importlib.machinery
    import types

    try:
        import torchvision  # noqa: F401
    except Exception:
        return
    if any(m.startswith("torchvision.io") for m in set(sys.modules)):
        return

    class _Any:
        def __getattr__(self, n):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

        def __mro_entries__(self, bases):
            return (object,)

    def _ga(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Any()

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("torchvision."):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = _ga
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())


def _load(model_id: str = MODEL_ID):
    global _model, _extractor, _female_idx
    if _model is not None:
        return _model, _extractor
    _torchvision_shim()
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    _extractor = AutoFeatureExtractor.from_pretrained(model_id)
    _model = AutoModelForAudioClassification.from_pretrained(model_id).eval()
    labels = {int(k): str(v).lower() for k, v in _model.config.id2label.items()}
    _female_idx = next((k for k, v in labels.items() if v.startswith("f")), 0)
    print(f"[nem] {model_id} betöltve (női index: {_female_idx})", flush=True)
    return _model, _extractor


class GenderClassifier:
    """Megnyilatkozás-szintű P(nő) mérés, kötegelt CPU-következtetéssel."""

    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 8,
                 min_s: float = 0.6, max_s: float = 6.0):
        self.model_id = model_id
        self.batch_size = batch_size
        self.min_s = min_s
        self.max_s = max_s

    def p_female(self, clips: list[np.ndarray], sr: int) -> list[float]:
        """clips: mono float32 tömbök (16 kHz-re mintavételezve)."""
        import torch

        model, ext = _load(self.model_id)
        out: list[float] = []
        for i in range(0, len(clips), self.batch_size):
            batch = [audio.to_16k(c, sr)[: int(self.max_s * audio.SR16)]
                     for c in clips[i:i + self.batch_size]]
            inputs = ext(batch, sampling_rate=audio.SR16, return_tensors="pt",
                         padding=True, return_attention_mask=True)
            with torch.inference_mode():
                logits = model(**inputs).logits
            out.extend(float(v) for v in logits.softmax(dim=-1)[:, _female_idx])
        return out

    def measure_windows(self, x, sr: int,
                        spans: list[tuple[float, float]]) -> list[float | None]:
        """P(nő) több időablakra; a túl rövid ablak None."""
        x16 = audio.to_16k(x, sr)
        idx, clips = [], []
        for k, (a, b) in enumerate(spans):
            core = audio.speech_core(x16, audio.SR16, a, b, min_s=self.min_s)
            if core is None:
                continue
            idx.append(k)
            clips.append(core)
        res: list[float | None] = [None] * len(spans)
        if clips:
            for k, p in zip(idx, self.p_female(clips, audio.SR16)):
                res[k] = p
        return res


def label(p: float | None, strong_f: float = 0.85, strong_m: float = 0.15) -> str | None:
    if p is None:
        return None
    if p >= strong_f:
        return "F"
    if p <= strong_m:
        return "M"
    return None
