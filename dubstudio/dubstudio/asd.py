"""Aktív beszélő pontozás tiszta matematikája (numpy) — tesztelhető mag.

Két pontozó él egymás mellett:

* ``corr_sig`` / ``score_candidates`` — a száj-sáv mozgásának korrelációja a
  hang-burkológéppel (tanított modell nélkül). **Mérés:** 15 hosszú soron a
  jel a zajszinten van, ezért ez csak negatív ellenőrzésre jó.
* ``decide_light`` / ``speak_spans`` — a tanított Light-ASD
  (``dubstudio/lightasd.py``) valószínűségeiből dönt. Ez a projekt
  elsődleges útja.

A ``work/visual_asd.py`` innen kapja a jel-feldolgozást: így a döntési logika
(idióta-biztos-e a korreláció? mikor „beszél” egy arc?) unit tesztekkel
ellenőrizhető, képek és ffmpeg nélkül.

A kulcs-gondolat: rövid ablakon (12 fps × 2 s ≈ 24 minta) a korreláció
maximuma **véletlenből is ~0,5** lehet, ezért a nyers r helyett a ciklikus
eltolásos null-eloszláshoz viszonyított z-érték dönt.
"""

from __future__ import annotations

import numpy as np

MIN_SAMPLES = 8      # ennél rövidebb jelen nem korrelálunk
MAX_LAG = 3          # ±0,25 s 12 fps-en: a száj a hang **előtt** nyílik
Z_SPEAK = 2.5        # a ciklikus eltolásos null-eloszláshoz képest
R_MIN = 0.50         # abszolút hatásméret-küszöb
N_MIN = 20           # ennél rövidebb ablakon (12 fps-en 1,7 s) nem döntünk
MARGIN = 0.10        # a véletlen szint fölé ennyivel kell kerülni
PEAK_MIN = 4.0       # ennyi alatt a száj-sáv gyakorlatilag nem mozog (0–255 skála)
ACTIVE_MIN = 0.15    # ennyi másodperc tényleges szájmozgás kell a döntéshez

# --- Light-ASD döntés (tanított modell valószínűségein) -------------------
# A küszöbök a 15 kézzel címzett sor méréséből (docs/asd.md): a látható
# beszélők p-je 0,41–0,94, a hallgató/statiszta arcoké 0,00–0,15. A 0,65 túl
# szigorú volt (a valódi beszélőket „nem eldönthető”-be tolta), a 0,45 a mért
# üres sávba esik: a hallgatók fölött marad, a legkisebb mért beszélő alatt.
SPEAK_P = 0.45       # ennyi fölött mondjuk, hogy ez az arc beszél
OFFSCREEN_P = 0.35   # ennyi alatt biztos, hogy nem ez az arc beszél
MARGIN_P = 0.15      # a másodikhoz képest ennyivel kell vezetni (kétbeszélős kép)
FACE_MIN_PX = 48     # ennél kisebb arc kivágása 112×112-re mosott: nem döntünk rá


def mouth_spans(sig: np.ndarray, fps: float, peak_min: float = 2.5,
                frac: float = 0.35, min_s: float = 0.12
                ) -> list[tuple[float, float]]:
    """A szájaktív intervallumok (másodperc) egy képkockánkénti mozgásjelből.

    A küszöb a jel csúcsának ``frac``-a, de legalább ``peak_min``; a
    ``min_s``-nél rövidebb villanás zaj, azt eldobjuk.
    """
    if sig is None or len(sig) == 0:
        return []
    thr = max(float(peak_min), float(np.max(sig)) * float(frac))
    spans: list[tuple[float, float]] = []
    run: list[int] = []
    for i, v in enumerate(sig):
        if v >= thr:
            run.append(i)
        elif run:
            spans.append((run[0] / fps, (run[-1] + 1) / fps))
            run = []
    if run:
        spans.append((run[0] / fps, (run[-1] + 1) / fps))
    return [s for s in spans if s[1] - s[0] >= min_s]


def _peak_r(u: np.ndarray, v: np.ndarray, max_lag: int) -> float:
    """A legjobb ±max_lag eltolással vett korreláció."""
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = u[lag:], v[: len(v) - lag]
        else:
            x, y = u[: len(u) + lag], v[-lag:]
        m = min(len(x), len(y))
        if m < MIN_SAMPLES:
            continue
        c = float(np.corrcoef(x[:m], y[:m])[0, 1])
        if np.isfinite(c) and c > best:
            best = c
    return best if best > -1.0 else 0.0


def chance_r(n: int) -> float:
    """Mekkora r érhető el **pusztán véletlenből** n mintán (lag-kereséssel).

    Empirikus mérés (véletlen párok, ±3 képkockás maximum): 24 mintán
    (2 s, 12 fps) a legjobb korreláció eléri a **0,80**-at, 60 mintán 0,41-et.
    Vagyis 2 másodperces ablakon a döntés a zajszinten van — ezért van a
    szabály: ``r >= max(R_MIN, chance_r(n) + MARGIN)``.
    """
    return float(min(0.9, 0.80 * (24.0 / max(n, 24)) ** 0.8))


def r_needed(n: int) -> float:
    """A döntéshez szükséges korreláció az adott ablakméreten."""
    return float(max(R_MIN, chance_r(n) + MARGIN))


def corr_sig(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG
             ) -> tuple[float, float]:
    """(r, z): a szájjel és a hang szótag-modulációjának illeszkedése.

    Mindkét jelet differenciáljuk (a száj szótagonként nyílik, a hangsáv
    burkológépe frázis-szinten lassú). Az eltolást **keressük** (±0,25 s),
    mert a száj jellemzően a hang előtt nyílik — zéró eltolással a valódi
    beszéd is negatív korrelációt ad (mért példa: r = −0,22).

    A szignifikanciát a jel **ciklikusan eltolt** változatai adják (ugyanazzal
    a lag-kereséssel), így a z a „mennyivel jobb a valódi illesztés a
    véletlen késésnél” kérdésre válaszol. Lapos jelre (0, 0).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < MIN_SAMPLES or len(b) < MIN_SAMPLES:
        return 0.0, 0.0
    da, db = np.diff(a), np.diff(b)
    if da.std() < 1e-9 or db.std() < 1e-9:
        return 0.0, 0.0
    n = min(len(da), len(db))
    da, db = da[:n], db[:n]
    r = _peak_r(da, db, max_lag)
    null = np.asarray([_peak_r(da, np.roll(db, k), max_lag)
                       for k in range(max_lag + 2, n - max_lag - 1)])
    null = null[np.isfinite(null)]
    if len(null) < 6 or null.std() < 1e-9:
        return float(r), 0.0
    return float(r), float((r - null.mean()) / null.std())


def speak_spans(probs: np.ndarray, fps: float = 25.0, thr: float = 0.5,
                min_s: float = 0.30) -> list[tuple[float, float]]:
    """Beszélő intervallumok (másodperc) a képkockánkénti valószínűségekből."""
    if probs is None or len(probs) == 0:
        return []
    spans: list[tuple[float, float]] = []
    run: list[int] = []
    for i, v in enumerate(probs):
        if float(v) >= thr:
            run.append(i)
        elif run:
            spans.append((run[0] / fps, (run[-1] + 1) / fps))
            run = []
    if run:
        spans.append((run[0] / fps, (run[-1] + 1) / fps))
    return [s for s in spans if s[1] - s[0] >= min_s]


def decide_light(cands: list[dict], speak_p: float = SPEAK_P,
                 offscreen_p: float = OFFSCREEN_P, margin_p: float = MARGIN_P,
                 face_min: int = FACE_MIN_PX, n_min: int = 25
                 ) -> tuple[list[dict], int | None, str]:
    """Beszélő arc kiválasztása a Light-ASD valószínűségeiből.

    Négy kimenet lehetséges, és a bizonytalant **nem** kényszerítjük döntésre:

    * ``arc #k beszél`` — a legjobb valószínűség ≥ 0,45 ÉS ≥ 0,15-tel vezet;
    * ``nincs beszélő arc a képen`` — minden arc ≤ 0,35 (képen kívüli beszélő);
    * ``nem eldönthető`` — a kettő között (0,35–0,65, vagy kicsi a különbség);
    * ``csak túl kicsi arc`` — 48 px alatti arcra a háló bemenete mosott.
    """
    if not cands:
        return [], None, "nincs arc a képen"
    usable = [c for c in cands
              if int(c.get("size", 0)) >= face_min and int(c.get("n", 0)) >= n_min]
    usable.sort(key=lambda c: -float(c.get("prob", 0.0)))
    if not usable:
        return usable, None, "csak túl kicsi (vagy túl rövid) arc — nem eldönthető"
    best = usable[0]
    second = usable[1] if len(usable) > 1 else None
    p = float(best.get("prob", 0.0))
    margin = p - float(second.get("prob", 0.0)) if second else 1.0
    if p >= speak_p and margin >= margin_p:
        return usable, int(best["face"]), (
            f"arc #{best['face']} beszél (p {p:.2f}"
            + (f", előny {margin:+.2f}" if second else "") + ")")
    if p <= offscreen_p:
        return usable, None, (f"nincs beszélő arc a képen (legjobb p {p:.2f} ≤ "
                              f"{offscreen_p:.2f}) → képen kívüli beszélő")
    return usable, None, (f"nem eldönthető (p {p:.2f}"
                          + (f", előny {margin:+.2f}" if second else "") + ")")


# A korábbi korreláció-alapú döntés (``score_candidates``) **törölve**.
# Mért eredmény: 15 hosszú soron 0 találat (a legjobb r 0,16–0,37, mind a
# véletlen szintjén), miközben a Light-ASD ugyanazokon a sorokon 0,68–0,95
# valószínűséggel nevezte meg a beszélőt. A korreláció így **mérés** marad
# (``corr_sig``: r, z a riportban), de nem dönt — a döntés a ``decide_light``.
