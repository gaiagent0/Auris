"""Hangkezelés: I/O, resample, **időnyújtás hangszín-őrzéssel**, fix hangszín-eltolás.

Az időzítés elve: a TTS **mindig a hang fix paramétereivel** szólal meg, a
sor hosszát pedig utólag, ``atempo``-val igazítjuk. Így a hang karaktere
soronként nem változik (a korábbi lánc soronként 0,7–1,7× közötti TTS-sebességet
adott át, ami a hang színét is elmozdította).
"""
from __future__ import annotations

import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # pragma: no cover
        pass

SR16 = 16_000
FFMPEG = "ffmpeg"


def _resample_poly(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if sr == target:
        return x.astype(np.float32, copy=False)
    g = gcd(int(sr), int(target))
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def load_mono(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """WAV beolvasása mono float32-ként, opcionális újramintavételezéssel."""
    import soundfile as sf

    a, native = sf.read(str(path), dtype="float32", always_2d=True)
    x = a.mean(axis=1) if a.ndim > 1 else a
    x = np.ascontiguousarray(x, dtype=np.float32)
    if sr is not None:
        x = _resample_poly(x, native, int(sr))
        native = sr
    return x, int(native)


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    return _resample_poly(np.ascontiguousarray(x, dtype=np.float32), int(sr), SR16)


def rms_db(x: np.ndarray) -> float:
    # float32-ben számolunk: egy 40 perces stereo sáv float64-re másolása
    # 1,4 GiB-ot kérne — szűkös memóriánál (pl. betöltött LLM-mel) ettől
    # a keverés megakadt; a dB-pontosság float32-ben is elég
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(float(np.mean(np.square(x)))) + 1e-12))


def speech_core(x: np.ndarray, sr: int, a: float, b: float,
                min_s: float = 0.45, max_s: float = 4.0) -> np.ndarray | None:
    """Egy időablak beszédes magja (a csend és a zene levágva).

    A beszélő-embedding ezen a részen készül: így a zene, a csend és a
    szomszédos sor áthallása nem rontja az azonosítást.
    """
    i0, i1 = int(max(0.0, a) * sr), int(max(0.0, b) * sr)
    seg = x[i0:i1]
    if len(seg) < int(0.15 * sr):
        return None
    hop = max(1, int(0.02 * sr))
    n = len(seg) // hop
    if n < 2:
        return None
    r = seg[: n * hop].reshape(n, hop)
    rms = np.sqrt((r ** 2).mean(axis=1))
    thr = max(0.006, float(np.percentile(rms, 90)) * 0.25)
    on = np.flatnonzero(rms >= thr)
    if len(on) == 0:
        return None
    core = seg[int(on[0]) * hop: int(on[-1] + 1) * hop]
    if len(core) < int(min_s * sr):
        return None
    if len(core) > int(max_s * sr):
        w = int(max_s * sr)
        k = int(np.argmax(np.convolve(rms, np.ones(max(1, w // hop)) / max(1, w // hop),
                                      mode="valid"))) if n >= w // hop else 0
        core = seg[k * hop: k * hop + w]
    return np.ascontiguousarray(core, dtype=np.float32)


def _ff_filter(x: np.ndarray, sr: int, filt: str) -> np.ndarray:
    """Egy ffmpeg audio-szűrő futtatása pipe-on (nincs temp fájl)."""
    cmd = [FFMPEG, "-v", "error", "-f", "f32le", "-ar", str(int(sr)), "-ac", "1",
           "-i", "pipe:0", "-filter:a", filt, "-f", "f32le", "-ac", "1", "pipe:1"]
    p = subprocess.run(cmd, input=np.asarray(x, dtype=np.float32).tobytes(),
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg szűrő hiba ({filt}): "
                           f"{p.stderr.decode('utf-8', 'replace')[-400:]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def fit_duration(x: np.ndarray, sr: int, factor: float,
                 lo: float = 0.70, hi: float = 1.50) -> tuple[np.ndarray, float]:
    """A hang hosszának igazítása ``factor`` arányban, hangszín-őrzéssel.

    ``factor`` > 1 gyorsít (rövidebb lesz). A ``[lo, hi]`` tartományon kívül
    nem nyújtunk — inkább hagyjuk kicsit túllógni, és a keverő vágja le:
    a nagy nyújtás hallhatóan torzítaná a hangot.
    """
    f = float(min(max(factor, lo), hi))
    if abs(f - 1.0) <= 0.02:
        return x, 1.0
    return _ff_filter(x, sr, f"atempo={f:.4f}"), f


def trim_tail(x: np.ndarray, sr: int, dur: float,
              fade: float = 0.02) -> np.ndarray:
    """A hang **kemény** levágása ``dur`` másodpercre, rövid kifuttatással.

    A TTS-farok (a szintézis utáni csend) levágása biztonságos; a keverő ezzel
    garantálja, hogy a régi sor ne lógjon az új sor alá. Ha a vágás beszédbe
    esne, a 20 ms-os fade kattanás nélkül zárja le.
    """
    n = max(0, min(len(x), int(round(dur * sr))))
    y = x[:n].copy()
    f = min(n, max(1, int(round(fade * sr))))
    if f > 1:
        y[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return y


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Fix hangszín-eltolás félhangban (rubberband), az időtartam megtartásával."""
    if abs(semitones) < 0.01:
        return x
    ratio = float(2.0 ** (semitones / 12.0))
    return _ff_filter(x, sr, f"rubberband=pitch={ratio:.6f}")


def trim_silence(x: np.ndarray, sr: int, head: float = 0.04, tail: float = 0.08,
                 rel_thr: float = 0.02) -> np.ndarray:
    """A TTS kimenet elejéről/végéről leváágott csend eltávolítása.

    A Supertonic modell a szöveg elé és mögé is tesz csendet (mért példa:
    0,47 s + 0,66 s egy 2,3 s-es mondatnál). Ha ezt nem vágjuk le, az
    időillesztés ereje nagy részét erre a csendre költjük, és a beszéd
    indokolatlanul gyorsul/lassul.
    """
    if len(x) < int(0.05 * sr):
        return x
    hop = max(1, int(0.010 * sr))
    n = len(x) // hop
    if n < 3:
        return x
    r = np.abs(x[: n * hop].reshape(n, hop)).max(axis=1)
    thr = max(1e-4, float(r.max()) * rel_thr)
    on = np.flatnonzero(r >= thr)
    if on.size == 0:
        return x
    a = max(0, int(on[0] * hop - head * sr))
    b = min(len(x), int((on[-1] + 1) * hop + tail * sr))
    return np.ascontiguousarray(x[a:b], dtype=np.float32)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), int(sr))
