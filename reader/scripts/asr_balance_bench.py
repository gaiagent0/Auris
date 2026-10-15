"""Minőség–sebesség egyensúly: a választható ASR-módok mérése ezen a gépen (M-53).

A cél az „elfogadható minőség és elkészítési sebesség egyensúlya". Ez a program
hármat csinál, mock nélkül, a valódi gépen:

  1. ÉLŐ MÓDSZERVÁLTÁS: kiírja a Beállításokban mentett `asr_backend` értéket a
     valódi settings fájlba, és ellenőrzi, hogy `qa.asr_backend_for("hu")` és a
     `qa.Transcriber.for_language("hu")` tényleg más-más módot ad vissza.
     Az eredeti érték visszaállításra kerül.
  2. MÉRÉS: a három választható mód (parakeet / whisper / hybrid) ugyanazon a
     négy valódi generált hangoskönyv-szeleten fut, azonos hivatkozó szöveggel.
     Mért: CER, WER, eltelt idő, RTF.
  3. VETÍTÉS: a teljes könyv hosszára (qa_report.json) kiszámolja, mennyi idő
     lenne minden móddal a könyv végighallgatása.

A hibrid a `qa_api._evaluate` szabályát követi: Parakeet megy elöl, Whisper csak
akkor hallgatja meg a szegmenst, ha a CER eléri a `qa.DEFAULT_CER_WARN`
küszöböt, és a jobb olvasat számít.

Futtatás (auris venv, a reader könyvtárból):
    .venv/Scripts/python.exe scripts/asr_balance_bench.py [--seconds 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HUB = r"D:\VoiceAI"
TXT_DIR = os.path.join(HUB, "out", "minta-klon", "teljes")
WAV_DIR = os.path.join(HUB, "out", "minta-klon", "hangoskonyv_teljes")
REPORT = os.path.join(WAV_DIR, "qa_report.json")
OUT = os.path.join(WAV_DIR, "asr_balance_bench.json")

# (fejezetszám, a fejezet hányad részénél vágott szelet) — négy különböző fejezet.
# Mindig a FEJEZET ELEJÉRŐL vágunk: a hivatkozó szöveget arányosan szeleteljük,
# és ez csak a nullpontos indulással csal pontosra (a M-50 protokoll is ilyen).
# Fejezetközi felépítésű szeletekre az arányos metszés szétúszik (mérve: CER 0,84).
SEGMENTS = [(0, 0.0), (4, 0.0), (8, 0.0), (13, 0.0)]

# a `scripts/` könyvtárból futtatva is a reader gyökeréből importáljunk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chapter_files(index: int) -> tuple[str, str]:
    txt = os.path.join(TXT_DIR, f"{index:02d}_main-{index}_szoveg_tisztitott.txt"
                       if index else "00_main_szoveg_tisztitott.txt")
    wav = os.path.join(WAV_DIR, f"{index:02d}_main{'-' + str(index) if index else ''}_eloszo01.wav")
    return txt, wav


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return " ".join(p.strip() for p in f.read().split("\n\n") if p.strip())


def book_seconds() -> float:
    try:
        with open(REPORT, encoding="utf-8") as f:
            data = json.load(f)
        chapters = data["chapters"] if isinstance(data, dict) else data
        return sum(float(c.get("seconds") or 0) for c in chapters)
    except Exception:
        return 0.0


def live_switch_check(qa) -> dict:
    """A Beállítások kapcsolója -> ténylegesen másik módot ad vissza?"""
    from core import settings

    original = str(settings.get("asr_backend", "auto") or "auto")
    results: dict[str, dict] = {}
    try:
        for value in ("auto", "parakeet", "whisper", "hybrid"):
            settings.save({"asr_backend": value})
            effective = qa.asr_backend_for("hu")
            cls = type(qa.Transcriber.for_language("hu")).__name__
            results[value] = {"effective": effective, "transcriber": cls}
            print(f"  beállítás={value:<9} -> effektív mód={effective:<9} "
                  f"osztály={cls}", flush=True)
    finally:
        settings.save({"asr_backend": original})
    print(f"  (eredeti érték visszaállítva: {original})", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    from core import qa
    import soundfile as sf

    print("=== 1. Élő módszerváltás (valódi settings fájl) ===", flush=True)
    switch = live_switch_check(qa)

    print("=== 2. Mérés: parakeet / whisper / hybrid ugyanazon a hangon ===",
          flush=True)
    parakeet = qa.ParakeetTranscriber()
    whisper = qa.Transcriber.whisper_for("hu")
    confirm_above = float(qa.DEFAULT_CER_WARN)

    tmpdir = os.path.join(HUB, "out", "minta-klon")
    os.makedirs(tmpdir, exist_ok=True)
    warm = os.path.join(tmpdir, "_bench_warmup.wav")

    # Első hívás: a modellbetöltés is ebben az időben van — külön mérve.
    first_txt, first_wav = chapter_files(SEGMENTS[0][0])
    audio0, sr0 = sf.read(first_wav, dtype="float32", always_2d=True)
    audio0 = audio0.mean(axis=1)
    sf.write(warm, audio0[: sr0], sr0)
    t0 = time.perf_counter()
    parakeet.transcribe(warm, "hu")
    parakeet_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    whisper.transcribe(warm, "hu")
    whisper_load = time.perf_counter() - t0
    print(f"  betöltés: parakeet {parakeet_load:.1f} s, whisper {whisper_load:.1f} s",
          flush=True)
    os.remove(warm)

    rows: list[dict] = []
    for index, frac in SEGMENTS:
        txt_path, wav_path = chapter_files(index)
        if not os.path.exists(wav_path):
            print(f"  [{index}] hiányzó WAV, kihagyva", flush=True)
            continue
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        duration_full = len(audio) / sr
        start = int(frac * max(len(audio) - int(args.seconds * sr), 0))
        chunk = audio[start: start + int(args.seconds * sr)]
        duration = len(chunk) / sr
        tmp = os.path.join(tmpdir, "_bench_seg.wav")
        sf.write(tmp, chunk, sr)

        text_full = load_text(txt_path)
        lo = int(len(text_full) * start / max(len(audio), 1))
        hi = int(len(text_full) * (start + len(chunk)) / max(len(audio), 1))
        text = text_full[lo:hi]

        t0 = time.perf_counter()
        heard_p = parakeet.transcribe(tmp, "hu")
        tp = time.perf_counter() - t0
        score_p = qa.score_transcript(text, heard_p["text"], "hu")

        t0 = time.perf_counter()
        heard_w = whisper.transcribe(tmp, "hu")
        tw = time.perf_counter() - t0
        score_w = qa.score_transcript(text, heard_w["text"], "hu")

        triggered = score_p["cer"] >= confirm_above
        if triggered and score_w["cer"] < score_p["cer"]:
            hybrid_score, hybrid_from = score_w, "whisper"
        else:
            hybrid_score, hybrid_from = score_p, "parakeet"
        rows.append({
            "index": index, "sec": round(duration, 1),
            "parakeet": {"sec": round(tp, 1), "cer": round(score_p["cer"], 4),
                         "wer": round(score_p["wer"], 4)},
            "whisper": {"sec": round(tw, 1), "cer": round(score_w["cer"], 4),
                        "wer": round(score_w["wer"], 4)},
            "hybrid": {"sec": round(tp + (tw if triggered else 0.0), 1),
                       "cer": round(hybrid_score["cer"], 4),
                       "wer": round(hybrid_score["wer"], 4),
                       "confirm": triggered, "from": hybrid_from},
        })
        print(f"  [{index}] {duration:.0f} s | parakeet CER {score_p['cer']:.4f} "
              f"({tp:.1f} s) | whisper CER {score_w['cer']:.4f} ({tw:.1f} s) "
              f"| hibrid megerősítés: {'IGEN' if triggered else 'nem'}", flush=True)
        os.remove(tmp)

    if not rows:
        print("Nincs mérhető szelet.")
        return 1

    print("=== 3. Összesítés és teljes könyves vetítés ===", flush=True)
    audio_total = sum(r["sec"] for r in rows)
    seconds = book_seconds() or 30180.8
    summary = {}
    for method in ("parakeet", "whisper", "hybrid"):
        infer = sum(r[method]["sec"] for r in rows)
        wers = [r[method]["wer"] for r in rows]
        cers = [r[method]["cer"] for r in rows]
        load = (parakeet_load + (whisper_load if method != "parakeet" else 0.0)
                if method == "hybrid" else
                parakeet_load if method == "parakeet" else whisper_load)
        if method == "hybrid":
            used_whisper = any(r["hybrid"]["confirm"] for r in rows)
            load = parakeet_load + (whisper_load if used_whisper else 0.0)
        rtf = infer / audio_total
        summary[method] = {
            "wer": round(sum(wers) / len(wers), 4),
            "cer": round(sum(cers) / len(cers), 4),
            "measured_infer_sec": round(infer, 1),
            "load_sec": round(load, 1),
            "rtf_infer": round(rtf, 4),
            "book_minutes": round((load + rtf * seconds) / 60.0, 1),
            "confirms": sum(1 for r in rows if r["hybrid"]["confirm"]),
        }
    for method, s in summary.items():
        print(f"  {method:<9} WER {s['wer']*100:5.2f} %  CER {s['cer']*100:5.2f} %  "
              f"RTF {s['rtf_infer']:.3f}  teljes könyv: {s['book_minutes']} perc "
              f"(betöltés {s['load_sec']} s)", flush=True)

    effective_auto = switch.get("auto", {}).get("effective")
    result = {
        "protocol": {"segments": [s[0] for s in SEGMENTS], "seconds": args.seconds,
                     "confirm_above": confirm_above,
                     "book_seconds": round(seconds, 1)},
        "switch": switch,
        "effective_auto": effective_auto,
        "loads": {"parakeet_sec": round(parakeet_load, 1),
                  "whisper_sec": round(whisper_load, 1)},
        "rows": rows,
        "summary": summary,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"jelentés: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
