"""Szintézis **zárolt hangparaméterekkel**, soronkénti gyorsítótárral.

Minden sor a szereplőjéhez rendelt hang **fix paramétereivel** szólal meg
(sebesség, lépésszám, szünet, hangszín-eltolás). A sor hosszát utólag
igazítjuk ``atempo``-val, ami a hangszínt nem változtatja meg — így a hang
karaktere a teljes epizódban (és évadban) állandó marad.

A gyorsítótár kulcsa a **hang + paraméterek + szöveg** lenyomata: ha bármelyik
változik, a sor újraszintetizálódik; ha nem, a meglévő WAV-ot használjuk.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import audio, subs
from .config import Project
from .plan import Entry
from .voices import Voice, VoiceBank

# Az illesztés elve (mért: a természetes TTS-hossz medián 0,74× az ablaké):
#   * **sosem lassítunk 10%-nál többet** — a 0,70-es tempó hallhatóan elhúzott
#     beszédet ad, ezért a rövidebb sorok inkább kicsit hamarabb véződnek;
#   * **gyorsítani csak ha kell** (a sor túllógna az ablakán), legfeljebb 1,5×-ig.
FIT_LO, FIT_HI = 0.90, 1.50
TOLERANCE = 0.25          # ennyivel lóghat túl a sor a saját ablakán
#: ennyi valódi csendet kaphat a sor a következő sor elôtt, ha a magyar hang
#: hosszabb az ablaknál — így nem kell agresszívan összenyomni a beszédet
BORROW_MAX = 0.60


def _room(e: Entry, next_start: dict[int, float]) -> float:
    """A sor rendelkezésre álló ablaka (csend-ráhagyással, a következő sorig)."""
    cap = e.dur + BORROW_MAX
    nxt = next_start.get(id(e))
    if nxt is not None:
        cap = min(cap, nxt - e.start - 0.02)
    return max(e.dur, cap)


def _stored_fit(stored: str) -> float | None:
    """A gyorsítótár-bejegyzésben tárolt, ténylegesen illesztett hossz."""
    try:
        return float(str(stored).split("|", 1)[1])
    except (IndexError, ValueError):
        return None


def cache_key(text: str, voice: Voice, pitch: float,
              dur: float | None = None) -> str:
    """A gyorsítótár kulcsa. A **célablak** is benne van: ha az időzítés
    változik, a korábban más hosszra igazított WAV nem használható tovább
    (különben a régi, hosszabb farok visszamarad és átlóg a következő sorra).
    """
    params = json.dumps(voice.params, sort_keys=True)
    target = f"dur{dur:.3f}" if dur is not None else "dur-na"
    h = hashlib.sha1(f"{voice.vid}|{pitch:.3f}|{params}|{text}|{target}|"
                     f"fit{FIT_LO:.2f}-{FIT_HI:.2f}".encode("utf-8"))
    return h.hexdigest()[:16]


class Synthesizer:
    def __init__(self, project: Project, bank: VoiceBank):
        self.project = project
        self.bank = bank

    def run(self, entries: list[Entry], out_dir: Path, limit: int = 0,
            verbose: bool = True) -> dict:
        voices = {v.vid: v for v in self.bank.available.values()}
        todo = [e for e in entries if not e.skip]
        todo = todo[:limit] if limit else todo
        # a csend-ráhagyás felső korlátja: a következő sor kezdete
        live = sorted((e for e in entries if not e.skip), key=lambda e: e.start)
        next_start: dict[int, float] = {}
        for a, b in zip(live, live[1:]):
            next_start[id(a)] = b.start
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "_meta.json"
        meta: dict[str, str] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        t0 = time.time()
        stats = {"done": 0, "cached": 0, "empty": 0, "over": 0, "rewritten": 0,
                 "borrowed": 0, "failed": [], "max_over": 0.0, "fits": []}
        # korlátozott futásnál (--limit) a limiten túli sorok WAV-ja a KORÁBBI
        # beállításokkal készült: ha a gyorsítótár-kulcs már nem egyezik (pl.
        # megváltozott a szereplő hangszíne), az elavult sor kimarad a próba
        # keveréséből — a teljes futás úgyis újragenerálja
        stale = 0
        if limit:
            fresh = {e.i for e in todo}
            for e in entries:
                if e.skip or e.i in fresh:
                    continue
                voice = voices.get(e.voice)
                out = out_dir / f"{e.i:05d}.wav"
                if voice is None or not out.exists():
                    continue
                if meta.get(str(e.i)) != cache_key(subs.prepare_hu(e.hu), voice,
                                                   e.pitch, e.dur):
                    e.skip = True
                    stale += 1
            if stale and verbose:
                print(f"[szintézis] {stale} sor a limiten túl elavult (más beállítás) "
                      f"— kimarad a próba keveréséből", flush=True)
        for e in todo:
            out = out_dir / f"{e.i:05d}.wav"
            if e.voice not in voices:
                raise RuntimeError(f"#{e.i}: ismeretlen hang ({e.voice}) — "
                                   f"a szereplő-rekord hibás")
            voice = voices[e.voice]
            text = subs.prepare_hu(e.hu)
            if not text.strip():
                stats["empty"] += 1
                continue
            key = cache_key(text, voice, e.pitch, e.dur)
            stored = str(meta.get(str(e.i)) or "")
            if out.exists() and stored.split("|", 1)[0] == key:
                stats["cached"] += 1
                fitted = _stored_fit(stored)
                if fitted and fitted > e.dur + 1e-3:
                    target = min(fitted, _room(e, next_start))
                    if target > e.dur + 1e-3:
                        e.end = round(e.start + target, 3)
                        stats["borrowed"] += 1
            else:
                if out.exists():
                    stats["rewritten"] += 1
                wav = None
                for attempt in (1, 2):       # egyszeri újrapróbálás (modell-hiba)
                    try:
                        wav, sr = self.bank.synth(text, voice, e.pitch)
                        break
                    except Exception as ex:
                        err = f"#{e.i} {voice.vid} {attempt}. próba: {ex}"
                        if verbose:
                            print(f"[szintézis] ! {err}", flush=True)
                if wav is None or len(wav) == 0:
                    # nem írunk ki üres WAV-ot: a keverő kihagyja a hiányzó sort,
                    # és a hiba a riportban is látszik
                    stats["failed"].append({"i": e.i, "uid": e.uid,
                                             "voice": e.voice, "text": text[:60]})
                    continue
                target = max(0.3, e.dur)
                dur = len(wav) / sr
                # a magyar hang csak akkor kaphat több helyet, ha hosszabb az
                # ablaknál; rövidebb sort **nem** lassítunk, csak a
                # összenyomást csökkentjük a következő sor elôtti csenddel
                if dur > target:
                    target = min(dur, _room(e, next_start))
                factor = dur / target
                wav, used = audio.fit_duration(wav, sr, factor, FIT_LO, FIT_HI)
                if len(wav) / sr > target + 0.005:
                    # az atempo felső korlátja miatt még túllógna: kemény vágás
                    wav = audio.trim_tail(wav, sr, target, fade=0.02)
                stats["fits"].append(round(used, 3))
                audio.write_wav(out, wav, sr)
                fitted = len(wav) / sr
                if fitted > e.dur + 1e-3:
                    stats["borrowed"] += 1
                e.end = round(e.start + max(e.dur, fitted), 3)
                meta[str(e.i)] = f"{key}|{fitted:.3f}"
                stats["done"] += 1
                if stats["done"] % 25 == 0:
                    meta_path.write_text(json.dumps(meta), encoding="utf-8")
                    print(f"  {stats['done']}/{len(todo)} sor, "
                          f"{time.time() - t0:.0f}s", flush=True)

        # illeszkedés mérése (a QA kapuhoz)
        for e in todo:
            p = out_dir / f"{e.i:05d}.wav"
            if not p.exists():
                continue
            x, sr = audio.load_mono(p)
            over = len(x) / sr - e.dur
            if over > TOLERANCE:
                stats["over"] += 1
                stats["max_over"] = max(stats["max_over"], over)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        fits = stats["fits"] or [1.0]
        if stats["failed"]:
            print(f"[szintézis] ! {len(stats['failed'])} sor szintézise megmaradt "
                  f"(nincs hang rájuk): "
                  + ", ".join(f"#{f['i']}" for f in stats["failed"][:12]), flush=True)
        if verbose:
            print(f"[szintézis] {stats['done']} új, {stats['cached']} gyorsítótárból, "
                  f"{stats['empty']} üres; időillesztés medián x{float(np.median(fits)):.2f} "
                  f"(min x{min(fits):.2f}, max x{max(fits):.2f}); "
                  f"{stats['over']} sor lóg túl ({stats['max_over']:+.2f}s)" +
                  (f"; {stats['borrowed']} sor csend-ráhagyással"
                   if stats["borrowed"] else ""))
        if stats["rewritten"]:
            print(f"[szintézis] {stats['rewritten']} sor újraszintetizálva "
                  f"(hang/paraméter/szöveg változott)")
        return stats
