"""Szó-idő ellenőrzés: a héber CTC illesztés eredményének független vizsgálata.

Az illesztő (``speaker_audit._align_hebrew``) a felirat szövegét kényszeríti a
hangra, ezért **önmagában konzisztens** marad akkor is, ha a szóhatárok
elcsúsznak. Ez a modul olyan jeleket mér, amelyek nem az illesztő saját
döntéséből származnak:

1. **Szerkezet** — a szó az illesztő ablakán (felirat-sáv ± 0,35 s) belül van-e,
   nő-e a sorrend, van-e átfedés. A felirat-sáv önmagában rendszeresen szűkebb
   a beszédnél, ezért a sávon belüli arány csak tájékoztató.
2. **Energia** — a szó ablaka alatt van-e beszédenergia a saját felirat-sávjához
   mérve (20 ms-os RMS-keretek), tehát nem csendben „úszik-e” a szó.
3. **Tempó** — a szó hossza és betűszáma alapján héber beszédre hihető
   karakter/másodperc-e; a nyújtott vagy összenyomott szavak modellfüggetlenül
   kiszűrhetők.
4. **Akusztikus támasz** — az illesztő saját utólagos valószínűségei: ha a
   medián nagyon alacsony, a szóidők gyengén támasztottak, és nem szabad rájuk
   vágást építeni (ezt eddig senki nem nézte).

Az eredmény nem javítja az illesztést és nem minősíti a fordítást: méri, hogy a
szóidők használhatók-e, és megnevezi a gyanús szavakat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PIPELINE = "dubstudio.words v1"

DEFAULTS = {
    "frame_s": 0.02,          # RMS-keret
    "hop_s": 0.01,
    "window_tol_s": 0.36,     # az illesztő ablaka: felirat-sáv +/- 0,35 s
    "cue_tol_s": 0.08,        # ennyivel lóghat ki "szépen" a szó a sávból
    "min_word_s": 0.05,       # ennél rövidebb szó (egy betűs) informális
    "long_word_s": 3.0,       # ennél hosszabb szó gyanús
    "cue_energy_ratio": 0.30,  # szó/felirat-sáv RMS arány alatt csend
    "floor_pct": 20.0,        # globális csend-küszöb (percentilis)
    "cps_fast": 26.0,         # ennél több héber betű/másodperc gyanús
    "cps_slow": 3.0,          # ennél kevesebb betű/másodperc gyanús
    "min_cps_chars": 2,       # ennyi betű alatt nincs tempó-ellenőrzés
    "min_cps_span_s": 0.12,   # ennél rövidebb szóból nem számolunk tempót
    "support_posterior": 0.10,  # e szint alatt az illesztő alig támaszt
}

_HEB_LO, _HEB_HI = 0x05D0, 0x05EA


def _letters(text: str) -> str:
    """Csak a héber betűk (a niqqud és az írásjelek kimaradnak)."""
    return "".join(ch for ch in (text or "") if _HEB_LO <= ord(ch) <= _HEB_HI)


def frame_rms(signal: np.ndarray, sr: int, frame_s: float = 0.02,
              hop_s: float = 0.01) -> tuple[np.ndarray, float]:
    """Keretenkénti RMS és a keretközép ideje (másodperc)."""
    frame = max(1, int(round(frame_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    if signal.size < frame:
        return np.zeros(1, dtype=np.float32), 0.0
    n = 1 + (signal.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = signal[idx].astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return rms, float(hop) / float(sr)


def _overflow(w: dict, seg: dict | None, tol: float = 0.0) -> float:
    """Mennyivel lóg ki a szó a sávból, a megadott tűrés levonása után."""
    if not seg:
        return 0.0
    a, b = float(w.get("start", 0.0)), float(w.get("end", 0.0))
    return max(0.0, float(seg["start"]) - a - tol, b - float(seg["end"]) - tol)


def check_spans(words: list[dict], segments: list[dict],
                rms: np.ndarray | None = None, hop_s: float = 0.01,
                tuning: dict | None = None) -> dict:
    """Szerkezeti, energia-, tempó- és támasz-ellenőrzés (tiszta függvény)."""
    t = {**DEFAULTS, **(tuning or {})}
    by_id = {int(s.get("id", i)): s for i, s in enumerate(segments or [])}
    floor = None
    if rms is not None and rms.size:
        floor = max(float(np.percentile(rms, t["floor_pct"])), 1e-6)

    def window_rms(a: float, b: float) -> float | None:
        if rms is None or not rms.size:
            return None
        lo = max(0, int(round(a / hop_s)))
        hi = min(rms.size, int(round(b / hop_s)))
        if hi <= lo:
            return None
        return float(np.mean(rms[lo:hi]))

    outside_window, outside_cue, silent = [], [], []
    short, long_, overlap, disorder, fast, slow = [], [], [], [], [], []
    overflows: list[float] = []
    prev_end, prev_cue = None, None
    for w in words or []:
        a, b = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        cue = w.get("source_cue")
        cue = int(cue) if cue is not None else None
        span = b - a
        seg = by_id.get(cue) if cue is not None else None
        if span < t["min_word_s"]:
            short.append(w)
        if span > t["long_word_s"]:
            long_.append(w)
        if seg is not None:
            over = _overflow(w, seg)
            overflows.append(over)
            if over > t["cue_tol_s"]:
                outside_cue.append({**w, "overflow": round(over, 3)})
            # Valódi hiba: a szó az illesztő ablakán is kívül esik.
            if _overflow(w, seg, t["window_tol_s"]) > 0:
                outside_window.append({**w, "overflow": round(over, 3)})
            got = window_rms(a, b)
            level = window_rms(float(seg["start"]), float(seg["end"]))
            if got is not None and level and got < level * t["cue_energy_ratio"]:
                silent.append({**w, "rms": round(got, 5),
                               "cue_rms": round(level, 5),
                               "ratio": round(got / level, 3)})
        if cue == prev_cue and prev_end is not None:
            if a < prev_end - t["cue_tol_s"]:
                overlap.append(w)
            if a + t["cue_tol_s"] < prev_end - t["cue_tol_s"]:
                disorder.append(w)
        prev_end, prev_cue = b, cue
        chars = len(_letters(w.get("word", "")))
        if chars >= t["min_cps_chars"] and span >= t["min_cps_span_s"]:
            cps = chars / span
            if cps > t["cps_fast"]:
                fast.append({**w, "cps": round(cps, 1)})
            elif cps < t["cps_slow"]:
                slow.append({**w, "cps": round(cps, 1)})

    n = len(words or [])
    scores = [float(w.get("score") or 0.0) for w in (words or []) if w.get("score") is not None]
    support = None
    if scores:
        arr = np.asarray(scores, dtype=np.float32)
        support = {
            "n": int(arr.size),
            "median": round(float(np.median(arr)), 4),
            "mean": round(float(arr.mean()), 4),
            "share_below": round(float(np.mean(arr < t["support_posterior"])), 3),
            "weak": bool(float(np.median(arr)) < t["support_posterior"]),
        }

    def rate(bad: list) -> float:
        return (n - len(bad)) / n if n else 0.0

    return {
        "n_words": n,
        "outside_window": outside_window,
        "outside_cue": outside_cue,
        "silent": silent,
        "too_short": short,
        "too_long": long_,
        "overlap": overlap,
        "disorder": disorder,
        "too_fast": fast,
        "too_slow": slow,
        "floor_rms": None if floor is None else round(floor, 5),
        "cue_overflow_s": {
            "mean": round(float(np.mean(overflows)), 3) if overflows else 0.0,
            "max": round(float(np.max(overflows)), 3) if overflows else 0.0,
        },
        "support": support,
        "rates": {
            "inside_window": rate(outside_window),
            "inside_cue": rate(outside_cue),
            "has_energy": rate(silent),
            "plausible_length": (n - len(long_)) / n if n else 0.0,
            "plausible_tempo": rate(fast + slow),
            "no_overlap": rate(overlap),
        },
    }


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    from . import audio

    signal, sr = sf.read(str(path), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if int(sr) != 16000:
        signal = audio.to_16k(signal, int(sr))
    return signal, 16000


def verify(audio_path: str | Path, words: list[dict], segments: list[dict],
           tuning: dict | None = None) -> dict:
    """Teljes ellenőrzés: szerkezet + energia + tempó + akusztikus támasz."""
    t = {**DEFAULTS, **(tuning or {})}
    signal, sr = load_audio(audio_path)
    rms, hop = frame_rms(signal, sr, t["frame_s"], t["hop_s"])
    result = check_spans(words, segments, rms=rms, hop_s=hop, tuning=t)
    rates = result["rates"]
    verdict = "pass"
    if (rates["inside_window"] < 0.98 or rates["no_overlap"] < 0.98
            or rates["has_energy"] < 0.80):
        verdict = "fail"
    elif (rates["has_energy"] < 0.95 or rates["plausible_tempo"] < 0.95
          or rates["plausible_length"] < 0.95
          or (result["support"] or {}).get("weak")):
        verdict = "warn"
    result["verdict"] = verdict
    result["pipeline"] = PIPELINE
    return result


def report(result: dict) -> str:
    r = result.get("rates") or {}
    ov = result.get("cue_overflow_s") or {}
    sup = result.get("support") or {}
    lines = [f"szavak: {result.get('n_words', 0)}"]
    lines.append(f"illesztő ablakán belül: {100 * r.get('inside_window', 0):.1f}%")
    lines.append(f"felirat-sávon belül:    {100 * r.get('inside_cue', 0):.1f}% "
                 f"(átlagos kilógás {ov.get('mean', 0):.2f}s, max {ov.get('max', 0):.2f}s)")
    lines.append(f"van beszédenergia:      {100 * r.get('has_energy', 0):.1f}%")
    lines.append(f"hihető hosszúság:      {100 * r.get('plausible_length', 0):.1f}%")
    lines.append(f"hihető tempó:          {100 * r.get('plausible_tempo', 0):.1f}%")
    lines.append(f"átfedés nélkül:         {100 * r.get('no_overlap', 0):.1f}%")
    if sup:
        lines.append(f"akusztikus támasz: medián {sup['median']:.3f} posterior, "
                     f"{100 * sup['share_below']:.0f}% a {DEFAULTS['support_posterior']:.2f} alatt"
                     + ("  — GYENGE" if sup.get("weak") else ""))
    for key, label in (("outside_window", "ablakon kívül"), ("silent", "csendben"),
                       ("too_fast", "túl gyors"), ("too_slow", "túl lassú"),
                       ("too_short", "túl rövid"), ("too_long", "túl hosszú"),
                       ("overlap", "átfedő"), ("disorder", "rossz sorrend")):
        items = result.get(key) or []
        if items:
            head = ", ".join(f"{w.get('start', 0):.2f}s" for w in items[:6])
            lines.append(f"{label}: {len(items)} ({head})")
    lines.append(f"ítélet: {result.get('verdict')}")
    return "\n".join(lines)


def verify_audit(project, tag: str, audit: dict, write: bool = True,
                 tuning: dict | None = None) -> dict:
    """Egy meglévő audit-riport szóidőit ellenőrzi (a kivágott WAV-ból)."""
    tag = tag.upper()
    start = float(audit.get("start") or 0.0)
    duration = float(audit.get("duration") or 0.0)
    out_dir = project.ep_dir(tag) / "source_audit"
    lang = str(audit.get("language") or "")
    candidates = [out_dir / f"source_{start:.0f}_{duration:.0f}.wav"]
    if lang:
        candidates.insert(0, out_dir / f"source_{lang}_{start:.0f}_{duration:.0f}.wav")
    wav = next((p for p in candidates if p.is_file()), None)
    if wav is None:
        raise FileNotFoundError(f"Nincs kivágott hang a kurzushoz: {candidates[0]}")
    result = verify(wav, audit.get("words") or [], audit.get("subtitle_segments") or [],
                    tuning=tuning)
    result.update({"episode": tag, "start": start, "duration": duration,
                   "audio": str(wav)})
    if write:
        path = out_dir / f"word_check_{start:.0f}_{duration:.0f}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   default=lambda o: list(o)), encoding="utf-8")
        result["path"] = str(path)
    return result
