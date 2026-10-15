"""Klónreferencia-mintavétel bármilyen hangfájlból.

A klónozás mindent átvesz a referenciából: a szobazajt, a zihálást, a
hosszú szüneteket és a hangerőt is. Ezért nem mindegy, melyik 5-15 másodperc
kerül be. A kézi kiválasztás lassú és ismételhetetlen; ez a modul a
teljes hangfájlt átnézi, és **objektív szempontok szerint** a legjobb
ablakot adja vissza, tisztítva és minősítve.

## Mitől jó egy klónreferencia

| Szempont | Miért |
|---|---|
| folyamatos beszéd, minimális szünet | a motor a szüneteket is „megtanulja" |
| magas jel-zaj arány | a szobazaj a klónba is átkerül |
| stabil hangerő | a hangmagasság-követés instabillá válik |
| nincs klip | torz hang, amit a modell is átvesz |
| nincs zene/háttérzene | a VAD beszédnek nézheti, a klón pedig magáévá teszi |
| nem a fájl elején/végén | ott az intro/outro, bemutatkozás, elköszönés |

## A pontozás

Minden ablak 0-1 közé skálázott részpontot kap, majd súlyozott átlaguk
adja a végpontot (0-100). A részpontok a `score_window` visszatérésében
külön is megvannak, hogy a felhasználó lássa, *miért* lett az a szakasz
a legjobb — nem egy fekete doboz.

## Függőségek

Csak `numpy` + `soundfile`, plusz ffmpeg a nem-WAV bemenethez. A Silero VAD
(`core.qa.vad_speech_spans`) **opcionális**: ha a sherpa-onnx vagy a modell
nem elérhető, RMS-alapú beszédkeresésre vált, és a jelentésben jelzi.
A tisztítás a `core.reference_audio.clean_reference` láncot használja.

## Használat

    from core import reference_sampler
    report = reference_sampler.sample_reference("eloszo.mp3", "ref.wav")

Parancssorról:

    python -m core.reference_sampler eloszo.mp3 --out ref.wav --json jelentes.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

from core import reference_audio
from core.qa import vad_speech_spans

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000          # megegyezik a tisztítólánccal és az XTTS elvárásával
TARGET_SEC = 12.0            # ideális referenciahossz
MIN_SEC = 5.0                # ennél rövidebb nem ad stabil hangképet
MAX_SEC = 15.0               # ennél hosszabbat a hangszín encoder nem használ ki
_FRAME = 0.05                # RMS-keret (megyezeg a qa.py-vel)

# Súlyok: a besűrűség és a klip a legfontosabb, mert azok rontják a legjobban
WEIGHTS = {
    "speech_density": 3.0,
    "no_clipping": 2.5,
    "snr": 2.0,
    "level_stability": 1.5,
    "speech_band": 1.5,
    "no_long_pause": 1.0,
    "not_at_edge": 0.5,
}

GRADE_THRESHOLD = (("kiváló", 80.0), ("jó", 65.0), ("elfogadható", 50.0))


# ────────────────────────────────────────────────────────── betöltés és VAD


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 jellel tér vissza bármilyen soundfile/ffmpeg olvasható formátumból."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(sr)
    except Exception:
        # A mp3/m4a/opus-t a libsndfile nem mindig tudja: ffmpeg-re bízzuk.
        if not _has_ffmpeg():
            raise
        raw = _ffmpeg_decode(path)
        if raw is None:
            raise
        return raw, SAMPLE_RATE


def _has_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def _ffmpeg_decode(path: str) -> np.ndarray | None:
    """Dekódolás 24 kHz mono float32 WAV-köztesbe, majd visszaolvasás."""
    import tempfile
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp = handle.name
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", tmp],
            check=True, capture_output=True, timeout=600)
        audio, sr = sf.read(tmp, dtype="float32")
        return audio.astype(np.float32)
    except Exception as exc:
        log.warning("ffmpeg dekódolás nem sikerült (%s): %s", path, exc)
        return None
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def speech_spans(audio: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], str]:
    """Beszéd-időintervallumok (mp) és a módszer neve.

    A Silero VAD pontosabb a zajjal szemben, de **képes teljesen kihagyni**
    a halk beszédet: a modell belső energiaküszöbe miatt egy alacsony
    hangerőjű szakaszt nem beszédnek lát. Ezért a két detektálást
    **összevonjuk**: a Silero szűkít, az RMS kiterjeszt. Mérve: a
    2026-10-10-es eseten a Silero egyedül a 12 mp-es tiszta szakaszból
    0 mp-t talált meg, az RMS-együttessel mind a 12 mp-et.

    Soha nem dob hibát: a mintavételnek működnie kell VAD nélkül is.
    """
    silero = vad_speech_spans(audio, sr) or []
    rms = _rms_spans(audio, sr)
    if silero and rms:
        return _merge_spans(_union_spans(silero, rms), 0.25), "silero+rms"
    if silero:
        return _merge_spans(silero, 0.25), "silero"
    return _merge_spans(rms, 0.25), "rms"


def _union_spans(a: list[tuple[float, float]],
                 b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Két intervallum-halmaz egyesítése, rendezve."""
    return _merge_spans(sorted(list(a) + list(b)), 0.0)


def _merge_spans(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _rms_spans(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """RMS-alapú beszédkeresés: a -45 dB feletti, összefüggő szakaszok."""
    frames = _frame_rms(audio, sr)
    if not len(frames):
        return []
    threshold = float(np.percentile(frames, 90)) * 0.05
    threshold = max(threshold, 10 ** (-45 / 20))
    active = frames > threshold
    spans = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start * _FRAME, i * _FRAME))
            start = None
    if start is not None:
        spans.append((start * _FRAME, len(active) * _FRAME))
    # a zajra állított küszöb sok rövid ugrást ad: a rövid szüneteket összefűzzük
    return _merge_spans(spans, 0.3)


def _frame_rms(audio: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * _FRAME))
    usable = len(audio) - (len(audio) % n)
    if usable <= 0:
        return np.zeros(0, np.float32)
    frames = audio[:usable].reshape(-1, n).astype(np.float64)
    return np.sqrt((frames ** 2).mean(axis=1)).astype(np.float32)


# ────────────────────────────────────────────────────────────── pontozás


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def score_window(audio: np.ndarray, sr: int, start: float, end: float,
                 spans: list[tuple[float, float]], total: float) -> dict:
    """Egy jelöltablak részpontjai és végpontja (0-100).

    Minden részpont 0-1; a végpont a `WEIGHTS` szerinti súlyozott átlag.
    """
    a = audio[int(start * sr):int(end * sr)]
    duration = len(a) / sr if sr else 0.0
    if duration <= 0 or not len(a):
        return {"score": 0.0, "parts": {}, "duration": 0.0}

    # 1) beszéd-sűrűség: mennyi idő beszéd az ablakban
    inside = sum(max(0.0, min(end, e) - max(start, s)) for s, e in spans)
    density = _clip01(inside / duration) if duration else 0.0

    # 2) klip: a csúcsérték közelítse a teljes skálát
    peak = float(np.max(np.abs(a)))
    clipping = int(np.count_nonzero(np.abs(a) >= 0.99))
    if clipping or peak >= 0.995:
        no_clipping = 0.0
    elif peak >= 0.95:
        no_clipping = 0.5
    else:
        no_clipping = 1.0

    # 3) jel-zaj arány: beszéd RMS a legcsendesebb részhez képest.
    # A log10-nak pozitív hányados kell: teljes csendnél speech_rms=0, és
    # a 0/0 math domain error-t adna.
    frames = _frame_rms(a, sr)
    if len(frames) >= 4:
        speech_rms = float(np.percentile(frames, 90))
        noise_rms = float(np.percentile(frames, 10))
        if speech_rms > 1e-9:
            snr_db = 20 * math.log10(speech_rms / max(noise_rms, 1e-9))
        else:
            snr_db = -120.0            # teljes csend
    else:
        snr_db = 0.0
    snr = _clip01((snr_db - 15.0) / 25.0)

    # 4) hangerő-stabiliság: a keretek szórása dB-ben.
    # Feltétel, hogy legyen beszéd-szint: tiszta csendnél minden keret
    # ~0, és a szórás nulla — az „stabil" ott nem érdem, hanem hiány.
    _SPEECH_FLOOR = 10 ** (-50 / 20)
    if len(frames) >= 4 and speech_rms > _SPEECH_FLOOR:
        voiced = frames[frames > speech_rms * 0.1]
        if len(voiced) >= 3:
            db = 20 * np.log10(np.maximum(voiced, 1e-9))
            stability = _clip01(1.0 - float(np.std(db)) / 12.0)
        else:
            stability = 0.0
    else:
        stability = 0.0

    # 5) beszéd-sáv: a 300-3400 Hz-es sáv aránya. Zene/robotzene alacsony.
    speech_band = _speech_band_ratio(a, sr)

    # 6) nincs hosszú belső szünet
    inside_frames = [
        bool(any(s < start + i * _FRAME < e for s, e in spans))
        for i in range(len(frames))
    ]
    longest = _longest_false_run(inside_frames)
    no_long_pause = _clip01(1.0 - (longest * _FRAME) / 1.0)

    # 7) nem a fájl szélén (intro/outro kihagyása)
    if total > 0:
        edge = min(start, total - end) / max(1.0, total * 0.15)
        not_at_edge = _clip01(edge)
    else:
        not_at_edge = 1.0

    parts = {
        "speech_density": density,
        "no_clipping": no_clipping,
        "snr": snr,
        "level_stability": stability,
        "speech_band": speech_band,
        "no_long_pause": no_long_pause,
        "not_at_edge": not_at_edge,
    }
    weights = sum(WEIGHTS.values())
    score = 100.0 * sum(parts[k] * WEIGHTS[k] for k in parts) / weights
    parts = {k: round(v, 3) for k, v in parts.items()}
    return {"score": round(score, 1), "parts": parts,
            "duration": round(duration, 2), "peak": round(peak, 4),
            "snr_db": round(snr_db, 1)}


def _speech_band_ratio(audio: np.ndarray, sr: int) -> float:
    """A 300-3400 Hz-es sáv energiájának aránya a teljeshez.

    A beszéd java ebben a sávban van; a zene szélesebb energiaspektrumot
    hagy, a gépzaj pedig alacsony frekvenciákon koncentrálódik.
    """
    n = len(audio)
    if n < 512:
        return 0.0
    chunk = min(n, 1 << 15)
    windowed = audio[:chunk] * np.hanning(chunk)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(chunk, 1.0 / sr)
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    band = float(spectrum[(freqs >= 300) & (freqs <= 3400)].sum())
    return _clip01(band / total)


def _longest_false_run(flags: list[bool]) -> int:
    longest = run = 0
    for value in flags:
        run = run + 1 if not value else 0
        longest = max(longest, run)
    return longest


def grade(score: float) -> str:
    for label, limit in GRADE_THRESHOLD:
        if score >= limit:
            return label
    return "gyenge"


# ────────────────────────────────────────────────────────── a fő folyamat


def sample_reference(src: str, dst: str, *, min_sec: float = MIN_SEC,
                     max_sec: float = MAX_SEC, target_sec: float = TARGET_SEC,
                     clean: bool = True, top_n: int = 5) -> dict:
    """Megkeresi és kiírja a legjobb klónreferencia-szakaszt.

    Visszatérési érték: jelentés-dict, benne a kiválasztott ablak kezdete és
    vége, a részpontok, a fokozat és a tisztítás eredménye. Ha a ``dst``
    nem készült el, a ``sampled`` False — ilyenkor az ``error`` mondja meg,
    miért (pl. „nincs beszéd a felvételben").
    """
    report: dict = {
        "source": os.path.abspath(src), "output": os.path.abspath(dst),
        "sampled": False, "error": "", "vad": "", "candidates": 0,
        "start_sec": None, "end_sec": None, "score": 0.0, "grade": "gyenge",
        "parts": {}, "cleaned": None, "after": {}, "top": [],
    }
    if not os.path.isfile(src):
        report["error"] = "A bemeneti fájl nem található."
        return report

    try:
        audio, sr = load_audio(src)
    except Exception as exc:
        report["error"] = f"A hang nem olvasható: {exc}"
        return report
    if not len(audio):
        report["error"] = "Üres hangfájl."
        return report

    total = len(audio) / sr
    report["source_seconds"] = round(total, 2)
    report["source_sample_rate"] = int(sr)

    spans, method = speech_spans(audio, sr)
    report["vad"] = method
    if not spans:
        report["error"] = "Nem találtam beszédet a felvételen."
        return report

    span_seconds = sum(e - s for s, e in spans)
    report["speech_seconds"] = round(span_seconds, 2)
    if span_seconds < min_sec:
        report["error"] = (f"A beszéd összesen {span_seconds:.1f} mp, "
                           f"kevesebb mint a szükséges {min_sec:.0f} mp.")
        return report

    # Jelöltek: minden beszédkezdettől indítunk, és az ablakot a beszédhez
    # igazítjuk. Ha a teljes beszéd rövidebb mint a célhossz, AZT vesszük:
    # egy 8 mp-es felvételből is lehet 6 mp-es referenciát csinálni.
    available = max(e - s for s, e in spans)
    wanted = float(min(max(target_sec, min_sec), max_sec))
    if available < wanted:
        wanted = max(min_sec, available)
    candidates = _candidate_windows(spans, wanted, total)
    report["candidates"] = len(candidates)
    if not candidates:
        report["error"] = "Nem találtam megfelelő hosszúságú beszédszakaszt."
        return report

    scored = [
        dict(start=round(s, 3), end=round(e, 3), **score_window(audio, sr, s, e, spans, total))
        for s, e in candidates
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    best = scored[0]
    report["top"] = scored[:top_n]
    report.update({
        "start_sec": best["start"], "end_sec": best["end"],
        "score": best["score"], "parts": best["parts"], "peak": best.get("peak"),
        "snr_db": best.get("snr_db"),
    })
    report["grade"] = grade(best["score"])

    # Kimetszés a teljes jelből (nem a tisztítottból: a tisztítás a kivágott
    # darabon fut, hogy a szűrők ne a fájl szélén csengjenek ki).
    segment = audio[int(best["start"] * sr):int(best["end"] * sr)]
    if not len(segment):
        report["error"] = "A kijelölt szakasz üres."
        return report
    tmp = dst + ".raw.wav"
    try:
        sf.write(tmp, segment.astype(np.float32), sr, subtype="PCM_16")
        if clean:
            outcome = reference_audio.clean_reference(tmp, dst)
            report["cleaned"] = outcome["cleaned"]
            report["clean_message"] = outcome["message"]
            if not outcome["cleaned"]:
                # A tisztítás csak javítás: ha nem sikerült, a nyers marad.
                os.replace(tmp, dst)
                report["cleaned"] = False
        else:
            os.replace(tmp, dst)
            report["cleaned"] = False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    if not os.path.isfile(dst):
        report["error"] = "A kimeneti fájl nem készült el."
        return report

    report["after"] = _measure(dst)
    report["sampled"] = True
    report["recommendation"] = _recommendation(report)
    return report


def _candidate_windows(spans: list[tuple[float, float]], wanted: float,
                       total: float) -> list[tuple[float, float]]:
    """Jelöltablakok: minden beszédkezdettől, az ablak a beszédhez igazítva.

    Az ablakot mindig a következő beszédkezdéshez igazítjuk, és a végét a
    benne foglalt beszéd végéig húzzuk — így a szünet nem esik az ablak
    közepére, ahol a pontozás rosszabbul jár érte.
    """
    out: list[tuple[float, float]] = []
    for i, (start, _) in enumerate(spans):
        end = min(start + wanted, total)
        # húzzuk meg a végét az utolsó beszéd-szakasz végéig, ha az hamarabb van
        covered = [e for s, e in spans if s < end]
        if covered:
            last = max(covered)
            end = max(end, min(last + 0.2, total))
        end = min(end, total)
        if end - start < wanted * 0.75:
            continue
        out.append((start, end))
        # második jelölt: az ablak közepe is lehet jó beszédkezdés
        mid = start + wanted * 0.5
        if any(abs(mid - s) < 0.5 for s, _ in spans[i + 1:]):
            out.append((mid, min(mid + wanted, total)))
    # ha egyetlen jelölt sincs (pl. minden beszéd rövid, de együtt elég),
    # adjuk meg az egész beszédet egyetlen ablakként
    if not out and spans:
        whole = (spans[0][0], min(spans[-1][1], total))
        if whole[1] - whole[0] >= 1.0:
            out.append(whole)
    return sorted(set(out))


def _measure(path: str) -> dict:
    """A kész fájl mérhető tulajdonságai a minőségi jelentéshez."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception as exc:
        return {"error": str(exc)}
    if not len(audio):
        return {"error": "üres"}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    clipped = int(np.count_nonzero(np.abs(audio) >= 0.99))
    frames = _frame_rms(audio, sr)
    noise = float(np.percentile(frames, 10)) if len(frames) >= 4 else 0.0
    snr = 20 * math.log10(rms / max(noise, 1e-9)) if noise > 0 else 0.0
    return {
        "seconds": round(len(audio) / sr, 2),
        "sample_rate": int(sr),
        "channels": 1,
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-9)), 2),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-9)), 2),
        "clipped_samples": clipped,
        "snr_db": round(snr, 1),
    }


def _recommendation(report: dict) -> str:
    """Emberi nyelvű tanács a jelentés alapján."""
    if not report.get("sampled"):
        return "Nem készült referencia — lásd az „error” mezőt."
    score = report.get("score", 0.0)
    after = report.get("after", {})
    bits = [f"A {report['end_sec'] - report['start_sec']:.1f} mp-es szakasz "
            f"{report['start_sec']:.1f} mp-től a legjobb a {report['candidates']} "
            f"jelölt közül ({score:.0f} pont, {report['grade']})."]
    if after.get("clipped_samples"):
        bits.append("Klip: a tisztítás után is maradt csúcsérték — érdemes másik hangot választani.")
    if report.get("parts", {}).get("snr", 1.0) < 0.4:
        bits.append("Alacsony jel-zaj arány: zajos környezet, a klón átveheti.")
    if report.get("parts", {}).get("speech_density", 1.0) < 0.7:
        bits.append("Sok szünet van a szakaszban: hallgasd meg, mielőtt használod.")
    if score >= 80:
        bits.append("Hallgasd meg a kimenetet, majd használhatod.")
    else:
        bits.append("Használat előtt hallgasd meg; gyengébb a hangkép.")
    return " ".join(bits)


# ────────────────────────────────────────────────────────────── CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Klónreferencia-mintavétel bármilyen hangfájlból")
    ap.add_argument("input", help="bemeneti hang (wav/mp3/m4a/...)")
    ap.add_argument("--out", default="", help="kimeneti WAV (alap: <bemenet>_ref.wav)")
    ap.add_argument("--min-sec", type=float, default=MIN_SEC)
    ap.add_argument("--max-sec", type=float, default=MAX_SEC)
    ap.add_argument("--target-sec", type=float, default=TARGET_SEC)
    ap.add_argument("--no-clean", action="store_true", help="kihagyja a tisztítóláncot")
    ap.add_argument("--json", default="", help="a teljes jelentés JSON-fájlba")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or (os.path.splitext(args.input)[0] + "_ref.wav")
    report = sample_reference(
        args.input, out, min_sec=args.min_sec, max_sec=args.max_sec,
        target_sec=args.target_sec, clean=not args.no_clean)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    if args.quiet:
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["sampled"] else 1

    if not report["sampled"]:
        print(f"NEM SIKERÜLT: {report['error']}")
        return 1
    after = report.get("after", {})
    print(f"Kiválasztott szakasz : {report['start_sec']:.1f} - {report['end_sec']:.1f} mp "
          f"({after.get('seconds', 0):.1f} mp a tisztítás után)")
    print(f"VAD                  : {report['vad']}   jelöltek: {report['candidates']}")
    print(f"Pontszám             : {report['score']:.1f} / 100  ({report['grade']})")
    parts = report.get("parts", {})
    if parts:
        print("Részpontok           : " + ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1])))
    print(f"Kimenet              : {after.get('peak_dbfs', 0):.1f} dBFS csúcs, "
          f"klip: {after.get('clipped_samples', 0)}, "
          f"SNR: {after.get('snr_db', 0):.1f} dB, {after.get('sample_rate', 0)} Hz")
    print(f"Fájl                 : {report['output']}")
    print(f"\n{report.get('recommendation', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
