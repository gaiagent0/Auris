"""Keverés: a magyar dialógus rákerül a zene+effekt (M&E) sávra.

Két javítás a korábbi lánchoz képest:
  * **ducking**: ahol magyar beszéd van, a zene/effekt halkan visszahúzódik —
    így a szöveg érthető marad anélkül, hogy a dialógust túl hangosra kellene
    venni (a szinkronstúdiók bevett gyakorlata);
  * **adaptív ducking**: a fix −4 dB nem elég ott, ahol a zenei ágy maga is
    kiemelt (a sorozat-intro tipikusan +6…15 dB-lel a sávátlag felett van) —
    a beszéd alatt az ágy soha nem marad a teljes sáv átlagánál hangosabb;
  * a dialógus szintje a **mért beszédszinthez** igazodik, ezért a halkan
    beszélt sorok sem tűnnek el a zene alatt.

A ducking 10 ms-os kereteken, blokkonként történik (nem mintánként), így egy
40 perces epizód is másodpercek alatt keverhető.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio
from .plan import Entry

DIALOG_OVER_BED_DB = 4.0     # a dialógus ennyivel legyen a zene fölött
DUCK_DB = 4.0                # ennyit húzunk a zenén, ahol beszéd van
DUCK_MAX_DB = 18.0           # az adaptív ducking mélységi korlátja
ATTACK_S = 0.06
RELEASE_S = 0.35
PEAK_CEIL = 0.98
FRAME_S = 0.010


def _frame_env(nf: int, sr: int, clips: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """A dialógus burkológörbéje keretfelbontáson (0..1), támadás/elengedés."""
    hop = max(1, int(FRAME_S * sr))
    env = np.zeros(nf, dtype=np.float32)
    for start, x in clips:
        i0 = int(start * sr) // hop
        if i0 >= nf:
            continue
        k = min(len(x) // hop, nf - i0)
        if k <= 0:
            continue
        blocks = np.abs(x[: k * hop]).reshape(k, hop).max(axis=1)
        env[i0:i0 + k] = np.maximum(env[i0:i0 + k], blocks)
    pos = env[env > 0]
    thr = max(1e-4, float(np.percentile(pos, 60)) * 3.0) if pos.size else 1e-4
    e = np.clip(env / thr, 0.0, 1.0)

    out = np.empty_like(e)
    a_re = float(np.exp(-FRAME_S / RELEASE_S))
    prev = 0.0
    for i in range(len(e)):
        v = float(e[i])
        if v >= prev:
            prev = v                 # támadás: azonnali (nincs kattanás, 10 ms keret)
        else:
            prev *= a_re             # elengedés: exponenciális (~0,35 s)
        out[i] = prev
    return out


def _frame_rms(track: np.ndarray, hop: int, nf: int,
               block: int = 500) -> np.ndarray:
    """A zenei ágy keret-szintű RMS-e (blokkonként, hogy ne legyen nagy temp).

    Egy 40 perces sztereó sáv ~0,8 GB float32 — a teljes mátrixot egyszerre
    négyzetre emelni memóriacsúcs lenne, ezért 5 s-os blokkokban mérünk.
    """
    out = np.empty(max(nf, 0), dtype=np.float32)
    for i in range(0, nf, block):
        stop = min(nf, i + block)
        blk = track[i * hop:stop * hop]
        out[i:stop] = np.sqrt(np.mean(np.square(
            blk.reshape(stop - i, hop, -1)), axis=(1, 2)) + 1e-12)
    return out


def _duck_curve(env: np.ndarray, f_rms: np.ndarray, duck: float,
                allowed: float) -> tuple[np.ndarray, np.ndarray]:
    """A zenei ágy erősítésgörbéje.

    ``env`` a beszéd burkológörbéje (0..1), ``f_rms`` az ágy keret-RMS-e,
    ``allowed`` a megengedett legnagyobb ágyszint beszéd alatt (a teljes sáv
    átlagos RMS-e). A fix ducking mellé ez a plafon lép: a hangos
    (intro) szakaszok a dialógus alatt az átlagszintig húzódnak vissza.
    """
    target = np.minimum(duck, allowed / np.maximum(f_rms, 1e-9))
    target = np.clip(target, 10.0 ** (-DUCK_MAX_DB / 20.0), 1.0)
    return 1.0 + (target - 1.0) * env, target


def build_track(entries: list[Entry], tts_dir: Path, me_path: Path,
                out_wav: Path, verbose: bool = True,
                duck_db: float = DUCK_DB,
                over_bed_db: float = DIALOG_OVER_BED_DB) -> dict:
    """A végleges magyar hangsáv elkészítése."""
    import soundfile as sf

    raw, sr = sf.read(str(me_path), dtype="float32", always_2d=True)
    stereo = raw.shape[1] > 1
    bed = raw if stereo else raw.reshape(-1, 1)

    clips: list[tuple[float, np.ndarray]] = []
    levels: list[float] = []
    live = sorted((e for e in entries if not e.skip), key=lambda e: e.start)
    for k, e in enumerate(live):
        p = tts_dir / f"{e.i:05d}.wav"
        if not p.exists():
            continue
        x, _ = audio.load_mono(p, sr=sr)
        if len(x) == 0:
            continue                    # üres/hiányos szintézis: ne torzítson
        # biztonsági vágás: a sor soha ne lógjon át a következő sor alá (a
        # terv már nem enged átfedést, de a korábbi cache-elt WAV hosszabb lehet)
        if k + 1 < len(live):
            limit = live[k + 1].start - e.start - 0.005
            if limit <= 0:
                continue
            if len(x) / sr > limit:
                x = audio.trim_tail(x, sr, limit, fade=0.02)
        clips.append((e.start, x))
        levels.append(audio.rms_db(x))
    if not clips:
        raise RuntimeError("egyetlen szintetizált sor sincs — nincs mit keverni")

    bed_lvl = audio.rms_db(bed)
    dialog_lvl = float(np.median(levels))
    gain = 10.0 ** ((bed_lvl + over_bed_db - dialog_lvl) / 20.0)

    track = np.array(bed, dtype=np.float32, copy=True)
    hop = max(1, int(FRAME_S * sr))
    nf = len(track) // hop
    env = _frame_env(nf, sr, clips)
    duck = 10.0 ** (-duck_db / 20.0)
    # adaptív ducking: az ágy a dialógus alatt ne legyen hangosabb a teljes
    # sáv átlagánál (az intro zenéje tipikusan +6…15 dB-lel kiemelt)
    allowed = 10.0 ** (bed_lvl / 20.0)
    f_rms = _frame_rms(track, hop, nf)
    g, g_target = _duck_curve(env, f_rms, duck, allowed)
    if nf:
        view = track[: nf * hop].reshape(nf, hop, -1)
        view *= g[:, None, None]

    fade = max(1, int(0.012 * sr))
    used = 0
    for start, x in clips:
        i0 = int(start * sr)
        if i0 >= len(track):
            continue
        seg = x[: len(track) - i0].copy() * gain
        if len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        track[i0:i0 + len(seg)] += seg[:, None]
        used += 1

    peak = float(np.abs(track).max())
    clipped = peak > PEAK_CEIL
    if clipped:
        track *= PEAK_CEIL / peak

    write = track.mean(axis=1) if stereo else track[:, 0]
    audio.write_wav(out_wav, write, sr)
    info = {"sr": sr, "clips": used, "bed_db": round(bed_lvl, 1),
            "dialog_db": round(dialog_lvl, 1),
            "gain_db": round(20 * np.log10(gain), 1),
            "clipped": bool(clipped),
            "duck_max_db": round(20 * np.log10(float(g_target.min())), 1)
            if nf else 0.0,
            "duck_frames": int(np.sum(g_target < duck * 0.999)) if nf else 0,
            "rms_db": round(audio.rms_db(write), 1),
            "duration": round(len(write) / sr, 1)}
    if verbose:
        print(f"[keverés] {used} sor, M&E {bed_lvl:.1f} dB, dialógus {dialog_lvl:.1f} dB, "
              f"erősítés {info['gain_db']:+.1f} dB, ducking -{duck_db:.0f} dB "
              f"(mély ducking {info['duck_max_db']:.0f} dB, "
              f"{info['duck_frames']} keret)"
              f"{' (csúcs-korlátozás)' if clipped else ''} -> {out_wav.name} "
              f"({info['duration'] / 60:.1f} perc)")
    return info
