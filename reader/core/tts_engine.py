"""
Shared TTS infrastructure for the Snapdragon X Elite build.

Text normalization, audio caching, segment rendering helpers and the export
pool shared by every ONNX engine (Supertonic 3, Sherpa-Supertonic, XTTSv2-ONNX,
Piper). The OmniVoice/PyTorch engine class was removed together with its CUDA
helpers; the engines themselves live in core/local_engines.py,
core/supertonic_sherpa_engine.py and core/xtts_onnx_engine.py.
"""

import logging
import os
import re
import threading
import time

import numpy as np
import soundfile as sf

from core.hungarian_numbers import looks_hungarian, normalize_hungarian

log = logging.getLogger(__name__)

from core.paths import user_path

AUDIO_CACHE_DIR = str(user_path('audio_cache'))
SAMPLE_RATE = 24_000
_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")
DEFAULT_TTS_NUM_STEP = 16
ALLOWED_TTS_NUM_STEPS = (8, 16, 24, 32)
# Merge consecutive same-voice short lines before synthesis so each engine call
# does real work (only engines that honour coalescing use it).
# 0 disables coalescing. ~600–900 chars ≈ several sentences / ~15–25s speech.
DEFAULT_TTS_COALESCE_CHARS = 0
AUDIO_CACHE_FORMAT_VERSION = 2


def _audio_duration(path: str) -> float:
    """Duration from the WAV header; cache hits must not decode the audio."""
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def _write_audio_atomic(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a cache WAV without exposing a partial file to another worker."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp.wav"
    try:
        sf.write(tmp, audio, sample_rate)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _tts_num_step_from_settings() -> int:
    try:
        from core.settings import get

        value = int(get("tts_num_step", DEFAULT_TTS_NUM_STEP) or DEFAULT_TTS_NUM_STEP)
    except Exception:
        value = DEFAULT_TTS_NUM_STEP
    if value in ALLOWED_TTS_NUM_STEPS:
        return value
    # Snap to nearest allowed step count.
    return min(ALLOWED_TTS_NUM_STEPS, key=lambda s: abs(s - value))


def _map_tn_lang(language: str | None, text: str) -> str:
    """Map book language to a TN language code (en/zh/ja/other)."""
    code = (language or "").strip().lower()
    if code in {"en", "english"}:
        return "en"
    if code in {"zh", "zh-cn", "zh-tw", "chinese", "mandarin"}:
        return "zh"
    if code in {"ja", "japanese"}:
        return "ja"
    if code and code not in {"none", "auto"}:
        return code
    # Script-based auto-detect when language is missing.
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    return "en"


def _apply_with_bracket_protection(text: str, fn) -> str:
    """Run ``fn`` on text while leaving ``[...]`` control spans untouched."""
    parts: list[str] = []
    last = 0
    for m in _BRACKET_TAG_RE.finditer(text):
        if m.start() > last:
            parts.append(fn(text[last:m.start()]))
        parts.append(m.group())
        last = m.end()
    if last < len(text):
        parts.append(fn(text[last:]))
    return "".join(parts) if parts else fn(text)


def _num2words_fallback(text: str, language: str | None) -> str:
    """Best-effort integer expansion when full TN is unavailable."""
    if looks_hungarian(language, text):
        return _apply_with_bracket_protection(text, normalize_hungarian)

    try:
        from num2words import num2words
    except ImportError:
        return text

    lang = _map_tn_lang(language, text)
    if lang in {"en", "zh", "ja"}:
        n2w_lang = "en" if lang != "zh" else "zh"
    else:
        n2w_lang = lang

    def _repl(match: re.Match) -> str:
        try:
            return num2words(int(match.group()), lang=n2w_lang)
        except Exception:
            try:
                return num2words(int(match.group()), lang="en")
            except Exception:
                return match.group()

    return _apply_with_bracket_protection(text, lambda s: re.sub(r"\d+", _repl, s))


_WETEXT_CACHE: dict[str, object] = {}


def _wetext_normalize(text: str, language: str | None) -> str | None:
    """Windows-friendly WeText runtime (no pynini). Returns None if unavailable."""
    try:
        from wetext import Normalizer
    except ImportError:
        return None

    lang = _map_tn_lang(language, text)
    if lang not in {"en", "zh", "ja"}:
        return None

    try:
        normalizer = _WETEXT_CACHE.get(lang)
        if normalizer is None:
            normalizer = Normalizer(lang=lang, operator="tn")
            _WETEXT_CACHE[lang] = normalizer
        return _apply_with_bracket_protection(text, normalizer.normalize)
    except Exception as exc:
        log.warning("wetext normalization failed (%s); trying next fallback.", type(exc).__name__)
        return None


def apply_text_normalization(text: str, language: str | None = None) -> str:
    """Normalize numbers/dates for TTS.

    Order:
    1. ``wetext`` pure-Python runtime (recommended on Windows; no pynini)
    2. ``num2words`` integer fallback
    """
    if not text or not text.strip():
        return text

    # Hungarian needs language-specific cardinal, ordinal, date and suffix
    # handling. Generic normalizers do not preserve these forms reliably.
    if looks_hungarian(language, text):
        return _apply_with_bracket_protection(text, normalize_hungarian)

    # 1) wetext (pynini-free; works on Windows pip).
    wetext_out = _wetext_normalize(text, language)
    if wetext_out is not None:
        return wetext_out

    # 2) Integer-only fallback.
    return _num2words_fallback(text, language)


def hungarian_normalizer_key(text: str, language: str | None, normalize_text: bool) -> str:
    """Cache-key suffix that changes when Hungarian normalization changes.

    Without it, a normalizer fix would never reach already cached audio.
    Other languages keep their historical keys.
    """
    if not normalize_text:
        return ""
    lang = str(language or "").lower()[:2]
    if lang == "hu" or (not lang and looks_hungarian(text)):
        from core import hungarian_numbers

        return f"|hn={getattr(hungarian_numbers, 'NORMALIZER_VERSION', 1)}"
    return ""


class TTSExportPool:
    """Single-resident export wrapper for the Snapdragon ONNX engines.

    Parallel model replicas were a CUDA/VRAM feature (two OmniVoice copies on
    a 20 GB card). The ONNX engines are CPU- and memory-bound on this machine,
    so a full export always runs on the one resident engine.
    """

    def __init__(self, primary, requested_workers: int = 0):
        self.primary = primary
        self.requested_workers = 1
        self.engines = [primary]
        self._replicas: list = []

    @property
    def worker_count(self) -> int:
        return 1

    def start(self) -> int:
        return 1

    def can_parallelize(self, items: list[dict]) -> bool:
        return False

    def generate_many(
        self,
        items: list[dict],
        *,
        num_step: int,
        on_item=None,
        on_status=None,
    ) -> list[dict]:
        return self.primary.generate_many(
            items,
            num_step=num_step,
            on_item=on_item,
            on_status=on_status,
        )

    def close(self) -> None:
        self.engines = [self.primary]
        self._replicas.clear()
