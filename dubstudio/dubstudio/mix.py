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

**Hangerő-mastering** (a lánc végén): a zene+effekt ágy előbb egy fix
referenciaszintre normalizálódik (``ME_TARGET_LUFS``), a kész sáv pedig az
EBU R128 integrált hangosságát tekintve a ``TARGET_LUFS``-ra kerül, előretekintő
limiterrel a ``TARGET_PEAK_DB`` csúcsplafon alatt. Enélkül a keverés a
forrásfilm ágyának szintjét örökli (mért eset: −34,9 LUFS, −13 dBFS csúcs — a
beszéd a −40 dB-es ágyhoz igazodott, majd 10 dB-t vágtak rajta).

Két emberi fogaskerék (a projekt ``tuning`` szakaszából):
  * ``bed_db`` — a zene+effekt ágy **összesített** halkítása (0…−30 dB);
    arra kell, amikor az eredeti beszéd a szétválasztás után is átszűrődik
    a zene alatt (dokumentumfilm, előadás);
  * ``bed_speech_duck_db`` — az ágy halkítása **ott, ahol az eredeti (angol)
    beszéd szól**; a szétválasztás maradékát ez nyomja el, nem a zene alatt
    végig. Az eredeti énekbeszéd-sáv (``vocals.wav``) burkológörbéjét méri.
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
FRAME_S = 0.010
#: a kész sáv cél-hangossága (EBU R128 integrált) és csúcsplafonja
TARGET_LUFS = -16.0
TARGET_PEAK_DB = -1.5
#: a zene+effekt ágy normalizálási célja: ehhez képest áll a dialógus szintje,
#: így a forrásfilm eredeti keverése nem visz el több dB-t
ME_TARGET_LUFS = -27.0
ME_MAX_GAIN_DB = 24.0         # a normalizálás korlátja (néma/rossz ágy ellen)
LIMIT_FRAME_S = 0.005
LIMIT_RELEASE_S = 0.060
LIMIT_LOOKAHEAD = 3           # keret (15 ms) — a gyorsítók ne csússzanak át
MASTER_ITER = 5               # emelés + limiter ismétlése a cél-hangerőig
MASTER_TOL_LU = 0.2


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
    return _smooth(np.clip(env / thr, 0.0, 1.0))


def _smooth(e: np.ndarray) -> np.ndarray:
    """Támadás/elengedés a keret-burkológörbén (kattanás nélkül)."""
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


def _speech_env(path: Path, sr: int, nf: int, hop: int,
                block: int = 2000) -> np.ndarray:
    """Az **eredeti** beszédsáv keret-burkológörbéje (0..1).

    A ``vocals.wav`` 40 percnél ~0,4 GB float32 lenne egyben, ezért blokkonként
    olvassuk. Ahol a szétválasztás beszédet talált, ott 1-hez közeli érték jön
    — a keverés ezzel húzza le az ágyat az angol beszéd alatt.
    """
    import soundfile as sf

    env = np.zeros(max(nf, 0), dtype=np.float32)
    idx = 0
    with sf.SoundFile(str(path)) as fh:
        while idx < nf:
            chunk = fh.read(hop * block, dtype="float32", always_2d=True)
            if len(chunk) < hop:
                break
            mono = chunk.mean(axis=1)
            k = len(mono) // hop
            blocks = np.abs(mono[: k * hop]).reshape(k, hop).max(axis=1)
            stop = min(nf, idx + k)
            env[idx:stop] = blocks[: stop - idx]
            idx = stop
    pos = env[env > 0]
    thr = max(1e-4, float(np.percentile(pos, 60)) * 0.6) if pos.size else 1e-4
    return _smooth(np.clip(env / thr, 0.0, 1.0))


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


def _limit_peaks(track: np.ndarray, sr: int, ceiling: float
                 ) -> tuple[int, int]:
    """Előretekintő csúcslimiter a kész sávra (helyben módosít).

    A cél-hangerőre emelés után a beszéd tranziensei a plafon fölé kerülnek; a
    hard vágás torzítana, ezért 5 ms-os kereteken, 15 ms előretekintéssel
    húzzuk vissza a csúcsokat, 60 ms-os elengedéssel (nincs pumpálás).

    Visszatérés: (csökkentett keretek száma, a limiter után is plafon fölé eső
    minták száma). Utóbbi csaknem mindig 0; ha nem, a végén hard vágunk.
    """
    hop = max(1, int(LIMIT_FRAME_S * sr))
    n = (len(track) // hop) * hop
    if n < hop * 2:
        return 0, int(np.sum(np.abs(track) > ceiling))
    mono = track.ndim == 1
    view = (track[:n].reshape(-1, hop) if mono
            else track[:n].reshape(-1, hop, track.shape[1]))
    peak = np.abs(view).max(axis=tuple(range(1, view.ndim))).astype(np.float32)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    g = need.copy()
    for j in range(1, LIMIT_LOOKAHEAD + 1):
        g[:-j] = np.minimum(g[:-j], need[j:])
    rel = float(np.exp(-LIMIT_FRAME_S / LIMIT_RELEASE_S))
    prev = 1.0
    for i in range(len(g)):
        v = float(g[i])
        prev = v if v < prev else 1.0 - (1.0 - prev) * rel
        if prev > v:                 # az elengedés sosem léphet a szükséges
            prev = v                 # csillapítás fölé (ott csúcs maradna)
        g[i] = prev
    reduced = int(np.sum(g < 0.9999))
    view *= g[:, None] if mono else g[:, None, None]
    over = int(np.sum(np.abs(track) > ceiling))
    if over:
        np.clip(track, -ceiling, ceiling, out=track)
    return reduced, over


def _master(track: np.ndarray, sr: int, target_db: float = TARGET_LUFS,
            peak_db: float = TARGET_PEAK_DB) -> dict:
    """Cél-hangerőre emelés + csúcsplafon, **ismételve** a konvergenciáig.

    Egyetlen emelés nem elég: a beszéd csúcs/helyzet aránya nagy, ezért a
    limiter az első menetben több dB-t visszavesz, és a sáv a cél alatt marad
    (mért példa: +5,2 dB emelés → −1,5 dBFS plafon → −19,0 LUFS). A ciklus az
    újramért hiányt emeli rá, amíg a limiter már csak a csúcsokat fogja — így
    a kész sáv tényleg a célon landol, nem 3 dB-lel alatta.
    """
    ceiling = 10.0 ** (peak_db / 20.0)
    makeup = 0.0
    reduced = over = 0
    lufs = audio.integrated_lufs(track, sr)
    for _ in range(MASTER_ITER):
        step = target_db - lufs
        if abs(step) <= MASTER_TOL_LU:
            break
        track *= np.float32(10.0 ** (step / 20.0))
        makeup += step
        reduced, over = _limit_peaks(track, sr, ceiling)
        lufs = audio.integrated_lufs(track, sr)
    if over:                          # maradék túllövés: néma mikro-halkítás
        track *= np.float32(ceiling / max(float(np.abs(track).max()), 1e-9))
        over = 0
    return {"makeup_db": float(makeup), "limited_frames": int(reduced),
            "clipped": bool(over), "lufs": float(lufs)}


def master(track: np.ndarray, sr: int,
           target_lufs: float = TARGET_LUFS,
           peak_db: float = TARGET_PEAK_DB) -> dict:
    """Cél-hangerőre masterelés (a szinkronlánc és a narráció közös útja)."""
    return _master(np.asarray(track, dtype=np.float32), sr, target_lufs, peak_db)


def build_track(entries: list[Entry], tts_dir: Path, me_path: Path,
                out_wav: Path, verbose: bool = True,
                duck_db: float = DUCK_DB,
                over_bed_db: float = DIALOG_OVER_BED_DB,
                bed_db: float = 0.0,
                speech_duck_db: float = 0.0,
                orig_vocals: Path | None = None) -> dict:
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

    # ── M&E szintnormalizálás ───────────────────────────────────────────
    # a forrásfilm ágya epizódonként más szinten jön (E01: −40,5 dB RMS); a
    # dialógus hozzá igazítása ezt örökölné, ezért az ágyat előbb egy fix
    # referenciaszintre visszük, és a dialógus ehhez képest áll
    bed_lufs = audio.integrated_lufs(bed, sr)
    me_gain_db = 0.0
    if bed_lufs > -70.0:
        me_gain_db = float(np.clip(ME_TARGET_LUFS - bed_lufs,
                                   -ME_MAX_GAIN_DB, ME_MAX_GAIN_DB))
    bed_lvl = audio.rms_db(bed) + me_gain_db
    dialog_lvl = float(np.median(levels))
    gain = 10.0 ** ((bed_lvl + over_bed_db - dialog_lvl) / 20.0)

    # az ágy: M&E-normalizálás + a kézi halkítás együtt (egy szorzó)
    bed_gain = 10.0 ** ((me_gain_db + bed_db) / 20.0)
    track = np.asarray(bed, dtype=np.float32) * np.float32(bed_gain)
    hop = max(1, int(FRAME_S * sr))
    nf = len(track) // hop
    env = _frame_env(nf, sr, clips)
    duck = 10.0 ** (-duck_db / 20.0)
    # adaptív ducking: az ágy a dialógus alatt ne legyen hangosabb a teljes
    # sáv átlagánál (az intro zenéje tipikusan +6…15 dB-lel kiemelt)
    allowed = 10.0 ** ((bed_lvl + bed_db) / 20.0)
    f_rms = _frame_rms(track, hop, nf)
    g, g_target = _duck_curve(env, f_rms, duck, allowed)
    # az eredeti (angol) beszéd maradéka a szétválasztott ágyban: ahol a forrás
    # énekbeszéd-sávja beszél, ott az ágyat külön is lehúzzuk
    speech_env = None
    if speech_duck_db > 0 and orig_vocals is not None \
            and Path(orig_vocals).is_file() and nf:
        speech_env = _speech_env(Path(orig_vocals), sr, nf, hop)
        g = g * (10.0 ** (-speech_duck_db / 20.0 * speech_env))
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

    # ── cél-hangerő + csúcsplafon (EBU R128) ────────────────────────────
    # a kész sáv integrált hangosságát a célra állítjuk: enélkül a keverés a
    # forrásfilm ágyának szintjét örökli (mért: −34,9 LUFS, −13 dBFS csúcs).
    # A mastering a **kiírt** jelen dolgozik (a sztereó ágy mono leszűrése
    # ~3 dB-t vesz el), különben a fájl a cél alá kerül.
    write = track.mean(axis=1) if stereo else track[:, 0]
    master = _master(write, sr)
    audio.write_wav(out_wav, write, sr)
    info = {"sr": sr, "clips": used,
            "me_lufs": round(bed_lufs, 1),
            "me_gain_db": round(float(me_gain_db), 1),
            "bed_lvl_db": round(bed_lvl, 1),
            "dialog_db": round(dialog_lvl, 1),
            "gain_db": round(20 * np.log10(gain), 1),
            "bed_db": round(float(bed_db), 1),
            "lufs": round(audio.integrated_lufs(write, sr), 1),
            "lufs_target": TARGET_LUFS,
            "makeup_db": round(master["makeup_db"], 1),
            "peak_db": round(audio.peak_db(write), 1),
            "peak_target_db": TARGET_PEAK_DB,
            "limited": bool(master["limited_frames"]),
            "limited_frames": master["limited_frames"],
            "clipped": master["clipped"],
            "duck_max_db": round(20 * np.log10(float(g_target.min())), 1)
            if nf else 0.0,
            "duck_frames": int(np.sum(g_target < duck * 0.999)) if nf else 0,
            "speech_duck_db": round(float(speech_duck_db), 1),
            "speech_frames": (int(np.sum(speech_env > 0.5))
                              if speech_env is not None else 0),
            "rms_db": round(audio.rms_db(write), 1),
            "duration": round(len(write) / sr, 1)}
    if verbose:
        print(f"[keverés] {used} sor, M&E {info['me_lufs']:.1f} LUFS "
              f"({info['me_gain_db']:+.1f} dB normalizálás) -> "
              f"{bed_lvl:.1f} dB RMS, dialógus {dialog_lvl:.1f} dB, "
              f"erősítés {info['gain_db']:+.1f} dB, ducking -{duck_db:.0f} dB "
              f"(mély ducking {info['duck_max_db']:.0f} dB, "
              f"{info['duck_frames']} keret)"
              + (f", háttér {bed_db:+.0f} dB" if bed_db else "")
              + (f", eredeti beszéd elnyomása -{speech_duck_db:.0f} dB "
                 f"({info['speech_frames']} keret)"
                 if info["speech_frames"] else "")
              + f", mastering {info['makeup_db']:+.1f} dB -> "
              + f"{info['lufs']:.1f} LUFS / {info['peak_db']:.1f} dBFS"
              + (f" (limiter {info['limited_frames']} keret)"
                 if info["limited_frames"] else "")
              + f" -> {out_wav.name} ({info['duration'] / 60:.1f} perc)")
    return info
