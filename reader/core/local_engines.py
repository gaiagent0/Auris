"""Additional local TTS engines: Piper, Supertonic 3 and MOSS-TTS.

Every engine exposes the same interface as ``TTSEngine``/``HiggsTTSEngine``
(status, load/unload, generate, generate_many, generate_preview), writes
24 kHz mono WAV files into the shared audio cache, and reports its
capabilities so the UI can hide controls an engine cannot honour.

Engines without voice cloning (Piper, Supertonic) choose one of their preset
voices from the voice description ("male, …" / "female, …") or from an
explicit ``voice:<name>`` token in it. A reference recording is ignored by
them and never becomes part of their cache key.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from core.tts_engine import (
    AUDIO_CACHE_DIR,
    SAMPLE_RATE,
    _audio_duration,
    _write_audio_atomic,
    apply_text_normalization,
    hungarian_normalizer_key,
)

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
from core.paths import user_path

MODELS_DIR = user_path('models')
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

# Capability matrix shown in Settings and used to adapt Voice Studio/reader.
ENGINE_INFO: dict[str, dict] = {
    "omnivoice": {
        "label": "OmniVoice", "voice_clone": True, "voice_design": True,
        "speed": True, "device": "GPU/CPU", "hungarian": "hivatalos",
        "license": "kód Apache-2.0, súlyok CC-BY-NC", "takes": True,
    },
    "higgs": {
        "label": "Higgs TTS 3 — 4B", "voice_clone": True, "voice_design": False,
        "speed": True, "device": "GPU", "hungarian": "hivatalos",
        "license": "Boson kutatási licenc", "takes": True,
    },
    "moss_tts": {
        "label": "MOSS-TTS 1.5 (4B)", "voice_clone": True, "voice_design": False,
        "speed": False, "device": "GPU (~14 GB VRAM)", "hungarian": "hivatalos",
        "license": "Apache-2.0", "takes": True,
    },
    "supertonic": {
        "label": "Supertonic 3 (CPU, előre beállított hangok)", "voice_clone": False,
        "voice_design": False, "speed": True, "device": "CPU", "hungarian": "hivatalos",
        "license": "OpenRAIL-M (modell), MIT (kód)",
    },
    "piper": {
        "label": "Piper (CPU, Anna/Berta/Imre)", "voice_clone": False,
        "voice_design": False, "speed": True, "device": "CPU", "hungarian": "hivatalos",
        "license": "GPL-3.0 (piper-tts), CC0 (hangok)",
    },
}

_TAG_RE = re.compile(r"\[[a-z][a-z0-9_\- ]{0,40}\]", re.IGNORECASE)
_VOICE_TOKEN_RE = re.compile(r"(?:^|,)\s*voice:\s*([\w\-]+)", re.IGNORECASE)
_QUOTES = str.maketrans({"„": '"', "”": '"', "“": '"', "»": '"', "«": '"', "‚": "'", "’": "'", "‘": "'"})


def _setting(key: str, default):
    try:
        from core import settings

        value = settings.get(key, default)
        return default if value is None else value
    except Exception:
        return default


def _gender_of(instruct: str | None) -> str | None:
    text = str(instruct or "").lower()
    if re.search(r"\bfemale\b|\bnő\b", text):
        return "female"
    if re.search(r"\bmale\b|\bférfi\b", text):
        return "male"
    return None


def explicit_voice(instruct: str | None) -> str | None:
    match = _VOICE_TOKEN_RE.search(str(instruct or ""))
    return match.group(1) if match else None


def choose_preset_voice(instruct: str | None, voices: dict[str, str], default: str) -> str:
    """Map a voice description to one preset voice, deterministically."""
    wanted = explicit_voice(instruct)
    if wanted:
        for name in voices:
            if name.lower() == wanted.lower():
                return name
    gender = _gender_of(instruct)
    candidates = [name for name, g in voices.items() if gender and g == gender]
    if not candidates:
        return default if default in voices else next(iter(voices))
    digest = int(hashlib.md5(str(instruct or "").encode("utf-8")).hexdigest()[:8], 16)
    return sorted(candidates)[digest % len(candidates)]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # soundfile layout is (frames, channels); some runtimes return (C, T).
        audio = audio.mean(axis=1) if audio.shape[1] <= 8 else audio.mean(axis=0)
    return audio.reshape(-1)


def resample(audio: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    audio = _to_mono(audio)
    if int(sr) == int(target) or audio.size == 0:
        return audio
    try:
        import soxr

        return soxr.resample(audio, sr, target).astype(np.float32)
    except Exception:
        pass
    # soxr has no win_arm64 wheel, and torchaudio has none either. SciPy ships
    # an ARM64 wheel everywhere, so it is the portable last resort.
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sr), int(target))
        return resample_poly(audio, target // divisor, sr // divisor).astype(np.float32)
    except Exception:
        import torch
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(audio), int(sr), int(target)).numpy().astype(np.float32)


def prepare_text(text: str, language: str | None, normalize_text: bool) -> str:
    """Strip OmniVoice expression tags and apply Auris normalization."""
    cleaned = _TAG_RE.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if normalize_text:
        cleaned = apply_text_normalization(cleaned, language)
    return cleaned


def hf_download(repo: str, target: Path, *, revision: str | None = None,
                allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo, revision=revision, local_dir=str(target), allow_patterns=allow_patterns,
    )
    return target


class LocalEngineBase:
    """Lifecycle, cache and batching shared by the additional engines."""

    engine_name = ""
    cache_version = 1

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()
        self._load_start_lock = threading.Lock()
        self._load_mutex = threading.Lock()
        self._cancel_load = threading.Event()
        self._generating = threading.Event()
        self._loading = False
        self._ready = False
        self._error: str | None = None
        self._detail = ""

    # ── capabilities ────────────────────────────────────────────────────────
    @property
    def capabilities(self) -> dict:
        return dict(ENGINE_INFO.get(self.engine_name, {}))

    def model_location(self) -> str:
        return ""

    def model_present(self) -> bool:
        return True

    # ── lifecycle ───────────────────────────────────────────────────────────
    def status(self) -> dict:
        base = {"engine": self.engine_name, "capabilities": self.capabilities,
                "model": self.model_location()}
        if self._error:
            return {**base, "state": "error", "message": self._error}
        if self._ready:
            return {**base, "state": "ready", "generating": self._generating.is_set(),
                    "accel": {"effective": self._detail or "cpu", "message": self._detail}}
        if self._loading:
            return {**base, "state": "loading", "message": self._detail}
        return {**base, "state": "not_loaded", "model_path": self.model_location(),
                "model_exists": self.model_present()}

    def load_async(self) -> None:
        with self._load_start_lock:
            if self._ready or self._loading:
                return
            self._loading = True
            self._cancel_load.clear()
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self) -> None:
        self._cancel_load.clear()
        self._load()
        if not self._ready:
            raise RuntimeError(self._error or f"{self.engine_name} failed to load")

    def _load(self) -> None:
        with self._load_mutex:
            if self._ready:
                self._loading = False
                return
            self._loading = True
            self._error = None
            try:
                self._load_model()
                if self._cancel_load.is_set():
                    self._release()
                    return
                self._ready = True
                log.info("%s ready (%s)", self.engine_name, self._detail or "cpu")
            except Exception as exc:
                self._error = str(exc)
                self._release()
                log.error("Failed to load %s: %s", self.engine_name, exc)
            finally:
                self._loading = False

    def _load_model(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _release(self) -> None:
        self.model = None

    def unload(self) -> None:
        self._cancel_load.set()
        with self._lock:
            self._release()
            self._ready = False
        gc.collect()

    def reload(self) -> None:
        self.unload()
        self._error = None
        self.load_async()

    def wait_until_unloaded(self, timeout: float = 600) -> bool:
        return True

    def cancel(self) -> bool:
        return False

    def set_dedicated_cuda_stream(self, enabled: bool) -> None:
        return

    def invalidate_voice_prompt(self, ref_audio=None, ref_text=None) -> None:
        return

    def _get_voice_clone_prompt(self, ref_audio, ref_text):
        return None

    # ── voices and cache identity ───────────────────────────────────────────
    def voice_identity(self, instruct, ref_audio, ref_text, language) -> str:
        """What determines the voice; unused inputs must not enter the key."""
        return f"{instruct}|{ref_audio}|{ref_text}"

    def settings_identity(self) -> str:
        return ""

    def cache_key(self, text, instruct, ref_audio, speed, ref_text=None,
                  language=None, normalize_text=False, num_step=0) -> str:
        speed = float(speed or 1.0) if self.capabilities.get("speed") else 1.0
        payload = (
            f"{self.engine_name}-v{self.cache_version}|{text}|"
            f"{self.voice_identity(instruct, ref_audio, ref_text, language)}|"
            f"{speed:.3f}|{language or ''}|nt={int(bool(normalize_text))}|"
            f"{self.settings_identity()}"
        )
        payload += hungarian_normalizer_key(text, language, bool(normalize_text))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def cache_path(key: str) -> str:
        return os.path.join(AUDIO_CACHE_DIR, f"{key}.wav")

    # ── synthesis ───────────────────────────────────────────────────────────
    def _synthesize(self, text, instruct, ref_audio, ref_text, speed, language):
        """Return (audio, sample_rate) for already prepared text."""
        raise NotImplementedError  # pragma: no cover

    def generate(self, text: str, instruct: str | None = None, ref_audio: str | None = None,
                 ref_text: str | None = None, speed: float = 1.0, num_step: int | None = None,
                 language: str | None = None, normalize_text: bool | None = None,
                 take: int = 0) -> dict:
        if normalize_text is None:
            normalize_text = bool(_setting("normalize_text", True))
        key = self.cache_key(text, instruct, ref_audio, speed, ref_text=ref_text,
                             language=language, normalize_text=bool(normalize_text))
        if take:
            key = hashlib.md5(f"{key}|take={int(take)}".encode("utf-8")).hexdigest()
        self._take = int(take or 0)
        path = self.cache_path(key)
        if os.path.exists(path):
            return {"audio_path": path, "duration_sec": _audio_duration(path),
                    "cache_hit": True, "cache_key": key}
        if not self._ready:
            raise RuntimeError(
                f"{self.capabilities.get('label', self.engine_name)} nincs betöltve. "
                + (self._error or "")
            )
        prepared = prepare_text(text, language, bool(normalize_text))
        if not prepared:
            audio = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
        else:
            audio = self._synthesize_verified(
                prepared, instruct, ref_audio, ref_text, float(speed or 1.0), language)
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
        _write_audio_atomic(path, audio, SAMPLE_RATE)
        return {"audio_path": path, "duration_sec": len(audio) / SAMPLE_RATE,
                "cache_hit": False, "cache_key": key}

    def _synthesize_verified(self, prepared, instruct, ref_audio, ref_text, speed, language):
        """Synthesise, then reject and retry a take with a broken internal pause.

        The qa.loudness path flags these too (``long_pause``), but only after the
        audio is cached, so the reader hears them first. Retrying here costs one
        extra synthesis and keeps the defect out of the cache entirely.

        The threshold scales with the text: a one-sentence excerpt can legitimately
        contain a long pause, while a 20-sentence paragraph cannot hide nineteen
        seconds of codec noise behind one.
        """
        attempts = max(1, min(3, int(_setting("silence_retry_attempts", 2))))
        sentences = max(1, len(split_sentences(prepared)))
        limit = max(SILENCE_MIN_SEC, SILENCE_SEC_PER_SENTENCE * sentences)
        for attempt in range(attempts):
            self._generating.set()
            try:
                with self._lock:
                    raw, sr = self._synthesize(prepared, instruct, ref_audio, ref_text,
                                               speed, language)
            finally:
                self._generating.clear()
            audio = resample(raw, sr)
            if audio.size == 0:
                raise RuntimeError(f"{self.engine_name} returned empty audio")
            peak = float(np.max(np.abs(audio)))
            if peak > 1.0:
                audio = audio / peak * 0.98
            # resample() returns the cache rate, so the gap check uses it too.
            gaps = find_long_silences(audio, sample_rate=SAMPLE_RATE,
                                      min_silence_sec=limit)
            if not gaps:
                return audio
            if attempt + 1 < attempts:
                log.warning(
                    "%s produced %d internal silence(s) over %.1f s (longest %.2f s); "
                    "regenerating (attempt %d/%d)", self.engine_name, len(gaps), limit,
                    max(end - start for start, end in gaps), attempt + 2, attempts)
                # A different seed is what makes the retry produce other audio.
                self._take = getattr(self, "_take", 0) + 1
        log.error(
            "%s kept a broken take after %d attempts: internal silence(s) at %s",
            self.engine_name, attempts, gaps)
        return audio

    def generate_many(self, items: list[dict], num_step: int | None = None,
                      batch_size: int | None = None, on_item=None, on_status=None) -> list[dict]:
        results = []
        total = len(items)
        for index, item in enumerate(items):
            if on_status is not None:
                try:
                    on_status(f"{self.engine_name} {index + 1}/{total}…")
                except Exception:
                    pass
            result = self.generate(
                text=item["text"], instruct=item.get("instruct"),
                ref_audio=item.get("ref_audio"), ref_text=item.get("ref_text"),
                speed=float(item.get("speed") or 1.0), language=item.get("language"),
                normalize_text=item.get("normalize_text"),
            )
            results.append(result)
            if on_item is not None:
                on_item(index, result)
        return results

    def generate_preview(self, instruct: str, sample_text: str, ref_audio: str | None = None,
                         ref_text: str | None = None, language: str | None = None,
                         normalize_text: bool | None = None) -> dict:
        return self.generate(sample_text, instruct=instruct, ref_audio=ref_audio,
                             ref_text=ref_text, language=language, normalize_text=normalize_text)


# ════════════════════════════════════════════════════════════════════════════
# Piper
# ════════════════════════════════════════════════════════════════════════════

PIPER_VOICES = {"anna": "female", "berta": "female", "imre": "male"}
PIPER_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "hu/hu_HU/{name}/medium/hu_HU-{name}-medium{ext}"
)


class PiperEngine(LocalEngineBase):
    engine_name = "piper"

    def __init__(self):
        super().__init__()
        self._voices: dict[str, object] = {}

    @property
    def folder(self) -> Path:
        return Path(_setting("piper_models_dir", "") or (MODELS_DIR / "piper"))

    def model_location(self) -> str:
        return str(self.folder)

    def _voice_file(self, name: str) -> Path:
        return self.folder / f"hu_HU-{name}-medium.onnx"

    def model_present(self) -> bool:
        return all(self._voice_file(name).is_file() for name in PIPER_VOICES)

    def _download(self) -> None:
        import urllib.request

        self.folder.mkdir(parents=True, exist_ok=True)
        for name in PIPER_VOICES:
            for ext in (".onnx", ".onnx.json"):
                target = self.folder / f"hu_HU-{name}-medium{ext}"
                if target.is_file() and target.stat().st_size > 0:
                    continue
                self._detail = f"Piper hang letöltése: {name}{ext}"
                tmp = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(PIPER_URL.format(name=name, ext=ext), tmp)
                os.replace(tmp, target)

    def _load_model(self) -> None:
        try:
            from piper import PiperVoice  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "A Piper nincs telepítve. Beállítások → Beszédmotorok → Piper telepítése, "
                "vagy: reader\\.venv\\Scripts\\python.exe -m pip install piper-tts==1.8.0"
            ) from exc
        if not self.model_present():
            self._download()
        from piper import PiperVoice

        self._voices = {name: PiperVoice.load(str(self._voice_file(name))) for name in PIPER_VOICES}
        self.model = self._voices
        self._detail = "CPU · ONNX Runtime"

    def _release(self) -> None:
        self._voices = {}
        self.model = None

    def voice_identity(self, instruct, ref_audio, ref_text, language) -> str:
        return "voice=" + choose_preset_voice(instruct, PIPER_VOICES, self._default_voice())

    def _default_voice(self) -> str:
        return str(_setting("piper_voice", "anna") or "anna")

    def _synthesize(self, text, instruct, ref_audio, ref_text, speed, language):
        from piper import SynthesisConfig

        name = choose_preset_voice(instruct, PIPER_VOICES, self._default_voice())
        voice = self._voices[name]
        config = SynthesisConfig(length_scale=1.0 / max(0.5, min(2.0, speed)))
        chunks = [chunk.audio_float_array for chunk in voice.synthesize(text, syn_config=config)]
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return audio, int(voice.config.sample_rate)


# ════════════════════════════════════════════════════════════════════════════
# Supertonic 3 — preset voices only, no voice cloning
# ════════════════════════════════════════════════════════════════════════════

SUPERTONIC_REPO = "supertone-oss-archive/supertonic-3"
SUPERTONIC_REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
SUPERTONIC_VOICES = {**{f"F{i}": "female" for i in range(1, 6)},
                     **{f"M{i}": "male" for i in range(1, 6)}}
# Quantization variants. "int8" uses the smaller graphs from the sherpa-onnx
# export; they take the same inputs and produce the same outputs, so the engine
# only has to point at different file names. The voice styles are shared, which
# is why the INT8 folder ships none.
SUPERTONIC_VARIANT_DIRS = {"fp32": "supertonic-3", "int8": "supertonic-3-int8"}
# Mirrors helper.variant_files(). It is duplicated here only so that the
# presence check works without importing the vendored helper (which needs the
# vendor directory on sys.path); a test asserts the two never drift apart.
SUPERTONIC_FILES = {
    "fp32": ("duration_predictor.onnx", "text_encoder.onnx",
             "vector_estimator.onnx", "vocoder.onnx", "unicode_indexer.json"),
    "int8": ("duration_predictor.int8.onnx", "text_encoder.int8.onnx",
             "vector_estimator.int8.onnx", "vocoder.int8.onnx", "unicode_indexer.bin"),
}


def _supertonic_files(variant: str) -> tuple[str, ...]:
    return SUPERTONIC_FILES.get(variant, SUPERTONIC_FILES["fp32"])


class SupertonicEngine(LocalEngineBase):
    engine_name = "supertonic"

    def __init__(self):
        super().__init__()
        self._styles: dict[str, object] = {}

    @property
    def variant(self) -> str:
        chosen = str(_setting("supertonic_variant", "fp32") or "fp32").lower()
        return chosen if chosen in SUPERTONIC_VARIANT_DIRS else "fp32"

    @property
    def folder(self) -> Path:
        override = _setting("supertonic_models_dir", "")
        if override:
            return Path(override)
        return MODELS_DIR / SUPERTONIC_VARIANT_DIRS[self.variant]

    @property
    def voice_dir(self) -> Path:
        """Voice styles are identical in both variants, so keep one location."""
        override = _setting("supertonic_models_dir", "")
        if override:
            return Path(override) / "voice_styles"
        return MODELS_DIR / SUPERTONIC_VARIANT_DIRS["fp32"] / "voice_styles"

    def model_location(self) -> str:
        return str(self.folder)

    def model_present(self) -> bool:
        names = _supertonic_files(self.variant)
        onnx_dir = self.folder / "onnx"
        return (
            (onnx_dir / "tts.json").is_file()
            and all((onnx_dir / name).is_file() for name in names)
            and (self.voice_dir / "F1.json").is_file()
        )

    def _load_model(self) -> None:
        if not self.model_present():
            self._detail = "Supertonic 3 letöltése (~0,4 GB)…"
            # The fp32 release also carries the voice styles every variant uses.
            hf_download(SUPERTONIC_REPO, MODELS_DIR / SUPERTONIC_VARIANT_DIRS["fp32"],
                        revision=SUPERTONIC_REVISION,
                        allow_patterns=["onnx/*", "voice_styles/*", "LICENSE*", "README*"])
        sys.path.insert(0, str(VENDOR_DIR / "supertonic"))
        try:
            import helper as supertonic_helper
        finally:
            sys.path.pop(0)
        self._helper = supertonic_helper
        self.model = supertonic_helper.load_text_to_speech(
            str(self.folder / "onnx"), use_gpu=False, variant=self.variant)
        self._styles = {}
        self._detail = f"CPU · ONNX Runtime · {self.variant}"

    def _release(self) -> None:
        self._styles = {}
        self.model = None

    def _default_voice(self) -> str:
        return str(_setting("supertonic_voice", "F1") or "F1")

    def voice_identity(self, instruct, ref_audio, ref_text, language) -> str:
        # No cloning: a reference recording never changes the output.
        return "style=" + choose_preset_voice(instruct, SUPERTONIC_VOICES, self._default_voice())

    def settings_identity(self) -> str:
        # The variant belongs here: fp32 and int8 produce different audio for
        # the same text, so a cache shared between them would serve stale takes.
        return f"variant={self.variant}|steps={self._steps()}"

    @staticmethod
    def _steps() -> int:
        try:
            return max(4, min(32, int(_setting("supertonic_steps", 10))))
        except (TypeError, ValueError):
            return 10

    def _style(self, name: str):
        if name not in self._styles:
            self._styles[name] = self._helper.load_voice_style(
                [str(self.voice_dir / f"{name}.json")]
            )
        return self._styles[name]

    def _synthesize(self, text, instruct, ref_audio, ref_text, speed, language):
        name = choose_preset_voice(instruct, SUPERTONIC_VOICES, self._default_voice())
        lang = str(language or "").lower()[:2] or "na"
        text = text.translate(_QUOTES)
        wav, duration = self.model(text, lang, self._style(name), self._steps(),
                                   max(0.7, min(1.6, speed)))
        sr = int(self.model.sample_rate)
        audio = np.asarray(wav)[0, : int(sr * float(np.asarray(duration).reshape(-1)[0]))]
        return audio, sr


# ════════════════════════════════════════════════════════════════════════════
# MOSS-TTS-Nano — CPU voice cloning
# ════════════════════════════════════════════════════════════════════════════

NANO_REFERENCE_TEXT = (
    "Jó napot kívánok. Ez egy nyugodt, tiszta hangminta, természetes magyar hangsúllyal."
)
# A pause is only a defect when it is out of proportion to the text. Hungarian
# narration pauses 0.6-1.2 s between sentences, so a fixed 1 s threshold flags
# perfectly normal reading; measured speech duty is roughly 5 s per sentence.
SILENCE_MIN_SEC = 1.0
SILENCE_SEC_PER_SENTENCE = 1.2

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])[\s ]+|(?<=[.!?…])$")


def find_long_silences(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                        min_silence_sec: float = 1.0,
                        silence_db: float = -45.0) -> list[tuple[float, float]]:
    """Locate silences longer than ``min_silence_sec`` inside the speech.

    Returns (start_sec, end_sec) pairs. A leading or trailing pause is not
    reported: only a gap that interrupts the reading is a defect. Used to catch
    the codec-noise blocks some engines emit, which the loudness check counts as
    silence but which still hiss in an exported file.
    """
    mono = audio.mean(axis=1) if np.ndim(audio) > 1 else np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return []
    frame = max(1, int(0.05 * sample_rate))
    count = mono.size // frame
    if count < 4:
        return []
    frames = mono[:count * frame].reshape(count, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    silent = rms < 10 ** (silence_db / 20)
    speech = np.flatnonzero(~silent)
    if not speech.size:
        return []
    found: list[tuple[float, float]] = []
    run = 0
    for index in range(speech[0], speech[-1] + 1):
        if silent[index]:
            run += 1
            continue
        if run * 0.05 >= min_silence_sec:
            start = (index - run) * 0.05
            found.append((round(start, 2), round(index * 0.05, 2)))
        run = 0
    return found


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping the punctuation with its own sentence."""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(str(text or ""))]
    return [part for part in parts if part]


def fallback_key(instruct) -> str:
    gender = _gender_of(instruct) or "female"
    return f"{gender}|{choose_preset_voice(instruct, SUPERTONIC_VOICES, 'F1')}"


def fallback_reference(instruct) -> str:
    """MOSS engines only clone; without a recording they clone a preset voice.

    The Hungarian reference sentence is spoken by Supertonic (matching the
    gender of the description), or Piper when Supertonic is unavailable, and
    cached in the audio cache.
    """
    folder = Path(AUDIO_CACHE_DIR) / "voice_refs"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(fallback_key(instruct).encode("utf-8")).hexdigest()[:16]
    path = folder / f"moss_ref_{key}.wav"
    if path.is_file():
        return str(path)
    errors = []
    for factory in (SupertonicEngine, PiperEngine):
        helper = factory()
        try:
            helper.load_sync()
            raw, sr = helper._synthesize(
                prepare_text(NANO_REFERENCE_TEXT, "hu", False), instruct, None, None, 1.0, "hu"
            )
            _write_audio_atomic(str(path), resample(raw, sr), SAMPLE_RATE)
            return str(path)
        except Exception as exc:
            errors.append(f"{factory.engine_name}: {exc}")
        finally:
            helper.unload()
    raise RuntimeError(
        "Ez a motor csak hangklónozással működik. Tölts fel referenciahangot a "
        "Hangstúdióban. (Automatikus mintahang nem készült: " + "; ".join(errors) + ")"
    )


# ════════════════════════════════════════════════════════════════════════════

MOSS_TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MOSS_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
# The worker runs the repositories' remote code; pin the reviewed revisions so
# a later upstream push cannot change the executed code.
MOSS_TTS_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
MOSS_CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"


class MossTTSEngine(LocalEngineBase):
    """Runs the 4B model in ``moss_worker.py`` so VRAM is freed by exiting."""

    engine_name = "moss_tts"

    def __init__(self):
        super().__init__()
        self._worker = None
        self._stdin_lock = threading.Lock()

    def model_location(self) -> str:
        return MOSS_TTS_REPO

    def _load_model(self) -> None:
        import subprocess

        worker_path = Path(__file__).with_name("moss_worker.py")
        self._detail = "MOSS-TTS betöltése (első indításkor ~17,6 GB letöltés)…"
        from core.desktop_support import python_command
        self._worker = subprocess.Popen(
            [*python_command(), "-u", str(worker_path)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8",
        )
        response = self._rpc({
            "model": MOSS_TTS_REPO, "revision": MOSS_TTS_REVISION,
            "codec": MOSS_CODEC_REPO, "codec_revision": MOSS_CODEC_REVISION,
        })
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "MOSS-TTS worker failed to start")
        self.model = self._worker
        self._sample_rate = int(response.get("sample_rate", 48000))
        self._detail = f"{response.get('device', 'cuda')} · {response.get('dtype', 'bf16')}"

    def _rpc(self, payload: dict) -> dict:
        import json

        worker = self._worker
        if worker is None or worker.stdin is None or worker.stdout is None:
            raise RuntimeError("A MOSS-TTS worker nem fut.")
        with self._stdin_lock:
            worker.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            worker.stdin.flush()
        while True:
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError(f"A MOSS-TTS worker leállt (kód {worker.poll()}).")
            line = line.strip()
            if line.startswith("{") and '"ok"' in line:
                try:
                    return json.loads(line)
                except ValueError:
                    pass
            log.info("MOSS worker: %s", line)

    def _release(self) -> None:
        worker, self._worker = self._worker, None
        self.model = None
        if worker is not None and worker.poll() is None:
            try:
                if worker.stdin:
                    worker.stdin.write('{"command": "shutdown"}\n')
                    worker.stdin.flush()
                worker.wait(timeout=10)
            except Exception:
                worker.terminate()

    def cancel(self) -> bool:
        worker = self._worker
        if worker is None or worker.poll() is not None or not self._generating.is_set():
            return False
        worker.terminate()
        self._worker = None
        self._ready = False
        self.load_async()
        return True

    def voice_identity(self, instruct, ref_audio, ref_text, language) -> str:
        if ref_audio:
            try:
                st = os.stat(ref_audio)
                ident = f"{os.path.abspath(ref_audio)}|{st.st_mtime_ns}|{st.st_size}"
            except OSError:
                ident = str(ref_audio)
            return f"ref={ident}|text={ref_text or ''}"
        return "fallback=" + fallback_key(instruct)

    def settings_identity(self) -> str:
        return (f"model={self.model_location()}|t={_setting('moss_temperature', 1.7)}|"
                f"p={_setting('moss_top_p', 0.8)}|k={_setting('moss_top_k', 25)}|"
                f"seed={_setting('moss_seed', 1234)}")

    def _synthesize(self, text, instruct, ref_audio, ref_text, speed, language):
        reference = ref_audio if ref_audio and os.path.exists(ref_audio) else \
            fallback_reference(instruct)
        handle, output = tempfile.mkstemp(suffix=".wav", prefix="auris-moss-")
        os.close(handle)
        try:
            response = self._rpc({
                "command": "generate", "text": text, "language": _moss_language(language),
                "reference_audio": reference,
                "reference_text": (ref_text or None) if ref_audio else NANO_REFERENCE_TEXT,
                "output_path": output,
                "temperature": float(_setting("moss_temperature", 1.7)),
                "top_p": float(_setting("moss_top_p", 0.8)),
                "top_k": int(_setting("moss_top_k", 25)),
                "seed": int(_setting("moss_seed", 1234)) + 7919 * getattr(self, "_take", 0),
            })
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "MOSS-TTS generation failed")
            audio, sr = sf.read(output, dtype="float32")
            return audio, int(sr)
        finally:
            try:
                os.remove(output)
            except OSError:
                pass


_MOSS_LANGUAGES = {
    "hu": "Hungarian", "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "pl": "Polish", "cs": "Czech",
    "nl": "Dutch", "fi": "Finnish", "ro": "Romanian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean",
}


def _moss_language(language: str | None) -> str | None:
    return _MOSS_LANGUAGES.get(str(language or "").lower()[:2])


ENGINE_CLASSES = {
    "piper": PiperEngine,
    "supertonic": SupertonicEngine,
    "moss_tts": MossTTSEngine,
}
