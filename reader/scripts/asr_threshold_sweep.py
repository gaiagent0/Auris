"""Hibrid megerősítő küszöb megválasztása: mondat-szintű sweep (M-54).

A `qa_api._evaluate` mondatonként (szegmensenként) dönt: ha a Parakeet CER-je
eléri a `confirm_above` küszöböt, a Whisper újrahallgatja a szegmenst. Hol
érdemes húzni ezt a küszöböt 0,04 és 0,12 között?

Protokoll:
  1. Három valódi fejezet (4, 7, 11) teljes Parakeet-átirata 180 s-os
     ablakokban, szóidőbélyegekkel.
  2. `alignment.align_words` a forrásszövegre — így minden mondatnak
     pontos időtartama van a hangban (a fejezet eleji arányos metszés
     helyett, ami szétúszott — lásd M-53 megjegyzését).
  3. Fejezetenként 20, egyenletesen kiválasztott mondat: felvágás,
     és MINDKÉT móddal külön-külön feliratkoztatva (Parakeet + Whisper),
     mért CER/WER és idő.
  4. Sweep: küszöb 0,04–0,12 (0,01 lépésköz) — megerősítések aránya,
     hibrid minőség, hibrid idő, teljes könyves vetítés.

A hibrid szabály megegyezik a gyártással: megerősítés csak a küszöb felett,
és csak akkor számít a Whisper, ha kisebb a CER-je.

Futtatás (auris venv, a reader könyvtárból):
    .venv/Scripts/python.exe scripts/asr_threshold_sweep.py [--per-chapter 20]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HUB = r"D:\<projekt>"
TXT_DIR = os.path.join(HUB, "out", "minta-klon", "teljes")
WAV_DIR = os.path.join(HUB, "out", "minta-klon", "hangoskonyv_teljes")
OUT = os.path.join(WAV_DIR, "asr_threshold_sweep.json")
REPORT = os.path.join(WAV_DIR, "qa_report.json")

CHAPTERS = [4, 7, 11]          # 22,2 + 26,2 + 16,7 perc — gyors, mégis vegyes
WINDOW = 180.0                 # a Parakeet ONNX encoder ennyit bír el egyben
THRESHOLDS = [round(0.04 + i * 0.01, 2) for i in range(9)]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chapter_files(index: int) -> tuple[str, str]:
    txt = os.path.join(TXT_DIR, f"{index:02d}_main-{index}_szoveg_tisztitott.txt")
    wav = os.path.join(WAV_DIR, f"{index:02d}_main-{index}_eloszo01.wav")
    return txt, wav


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return "\n".join(p.strip() for p in f.read().split("\n\n") if p.strip())


_SENT_END = re.compile(r'[.!?…][\"»”’)\]]*$')


def split_sentences(text: str) -> list[dict]:
    """Mondatai karakterkoordinátákkal (a szó-tokenek indexe kell az illesztéshez)."""
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    sentences, start, words = [], None, 0
    for pos, (begin, end, tok) in enumerate(tokens):
        if start is None:
            start = begin
        words += 1
        if _SENT_END.search(tok) or pos == len(tokens) - 1:
            sentences.append({"start": start, "end": end, "words": words,
                              "text": text[start:end].strip()})
            start, words = None, 0
    return [s for s in sentences if 4 <= s["words"] <= 40]


def chapter_words(parakeet, wav_path: str, tmpdir: str, index: int):
    """-> (szók, időtartam, mono audió, mintavételi gyakoriság)"""
    """Teljes fejezet Parakeet-átirata ablakokban, időbélyegekkel tolva."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    duration = len(audio) / sr
    win = int(WINDOW * sr)
    words: list[dict] = []
    for widx, start in enumerate(range(0, len(audio), win)):
        chunk = audio[start: start + win]
        if len(chunk) < sr:
            continue
        tmp = os.path.join(tmpdir, f"_sweep_{index}_{widx}.wav")
        sf.write(tmp, chunk, sr)
        offset = start / sr
        for word in parakeet.transcribe(tmp, "hu")["words"]:
            if word.get("start") is not None:
                words.append({"word": word["word"],
                              "start": float(word["start"]) + offset,
                              "end": (float(word["end"]) + offset
                                      if word.get("end") is not None else None)})
        os.remove(tmp)
    return words, duration, audio, sr


def book_seconds() -> float:
    try:
        with open(REPORT, encoding="utf-8") as f:
            data = json.load(f)
        chapters = data["chapters"] if isinstance(data, dict) else data
        return sum(float(c.get("seconds") or 0) for c in chapters)
    except Exception:
        return 30180.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chapter", type=int, default=20)
    args = ap.parse_args()

    from core import alignment, qa
    import soundfile as sf

    parakeet = qa.ParakeetTranscriber()
    whisper = qa.Transcriber.whisper_for("hu")
    tmpdir = os.path.join(HUB, "out", "minta-klon")
    rows: list[dict] = []

    for index in CHAPTERS:
        txt_path, wav_path = chapter_files(index)
        if not os.path.exists(wav_path):
            print(f"[{index}] hiányzó WAV", flush=True)
            continue
        text = load_text(txt_path)
        t0 = time.perf_counter()
        words, duration, audio, sr = chapter_words(parakeet, wav_path, tmpdir, index)
        print(f"[{index}] Parakeet teljes átirat {duration/60:.1f} perc hangra "
              f"{time.perf_counter()-t0:.0f} s alatt, {len(words)} szó", flush=True)

        t0 = time.perf_counter()
        timings = alignment.align_words(text, words, duration, "hu")
        print(f"[{index}] illesztés a forrásszövegre {time.perf_counter()-t0:.0f} s "
              f"({len(timings)} szó-idő)", flush=True)

        # mondatok -> szó-token index tartomány -> időtartomány a hangban
        token_starts = [m.start() for m in re.finditer(r"\S+", text)]
        token_index = {begin: i for i, begin in enumerate(token_starts)}
        sentences = split_sentences(text)
        prepared = []
        for sent in sentences:
            first = token_index.get(sent["start"])
            if first is None or first >= len(timings):
                continue
            last = min(first + sent["words"] - 1, len(timings) - 1)
            t_start = max(timings[first][0] - 0.05, 0.0)
            t_end = min(timings[last][1] + 0.15, duration)
            # óvintézkedés: ésszerű mondat-hossz (2 és 60 s közt); a hibás
            # idővonal itt bukna ki, nem az ONNX encoderben (M-54)
            if not (2.0 <= t_end - t_start <= 60.0):
                continue
            prepared.append((sent, t_start, t_end))
        if not prepared:
            continue
        step = max(len(prepared) // args.per_chapter, 1)
        chosen = prepared[:: step][: args.per_chapter]
        print(f"[{index}] {len(prepared)} mondatból {len(chosen)} mérve", flush=True)

        for sent, t_start, t_end in chosen:
            chunk = audio[int(t_start * sr): int(t_end * sr)]
            tmp = os.path.join(tmpdir, "_sweep_sent.wav")
            sf.write(tmp, chunk, sr)
            t0 = time.perf_counter()
            heard_p = parakeet.transcribe(tmp, "hu")
            tp = time.perf_counter() - t0
            score_p = qa.score_transcript(sent["text"], heard_p["text"], "hu")
            t0 = time.perf_counter()
            heard_w = whisper.transcribe(tmp, "hu")
            tw = time.perf_counter() - t0
            score_w = qa.score_transcript(sent["text"], heard_w["text"], "hu")
            os.remove(tmp)
            rows.append({
                "chapter": index, "text": sent["text"][:80],
                "sec": round(t_end - t_start, 2),
                "cer_p": round(score_p["cer"], 4), "wer_p": round(score_p["wer"], 4),
                "cer_w": round(score_w["cer"], 4), "wer_w": round(score_w["wer"], 4),
                "tp": round(tp, 2), "tw": round(tw, 2),
            })
            print(f"  [{index}] {t_end-t_start:5.1f} s | P CER {score_p['cer']:.3f} "
                  f"({tp:.1f} s) | W CER {score_w['cer']:.3f} ({tw:.1f} s) | "
                  f"{sent['text'][:50]}", flush=True)

    if not rows:
        print("Nincs mérhető mondat.")
        return 1

    audio_total = sum(r["sec"] for r in rows)
    book = book_seconds()
    sweep = []
    for threshold in THRESHOLDS:
        triggered = [r for r in rows if r["cer_p"] >= threshold]
        improved = [r for r in triggered if r["cer_w"] < r["cer_p"]]
        kept = [r for r in triggered if r["cer_w"] >= r["cer_p"]]
        # gyártási szabály: megerősítésnél a jobb olvasat számít
        wers, cers = [], []
        for r in rows:
            use_w = r["cer_p"] >= threshold and r["cer_w"] < r["cer_p"]
            wers.append(r["wer_w"] if use_w else r["wer_p"])
            cers.append(r["cer_w"] if use_w else r["cer_p"])
        time_total = sum(r["tp"] for r in rows) + sum(r["tw"] for r in triggered)
        sweep.append({
            "threshold": threshold,
            "triggered": len(triggered),
            "triggered_pct": round(100 * len(triggered) / len(rows), 1),
            "improved": len(improved), "wasted": len(kept),
            "wer": round(sum(wers) / len(wers), 4),
            "cer": round(sum(cers) / len(cers), 4),
            "rtf": round(time_total / audio_total, 4),
            "book_minutes": round(time_total / audio_total * book / 60.0, 1),
        })
    base_p = {"wer": round(sum(r["wer_p"] for r in rows) / len(rows), 4),
              "cer": round(sum(r["cer_p"] for r in rows) / len(rows), 4),
              "book_minutes": round(sum(r["tp"] for r in rows) / audio_total * book / 60.0, 1)}
    base_w = {"wer": round(sum(r["wer_w"] for r in rows) / len(rows), 4),
              "cer": round(sum(r["cer_w"] for r in rows) / len(rows), 4),
              "book_minutes": round(sum(r["tw"] for r in rows) / audio_total * book / 60.0, 1)}

    print("\n=== Sweep (mondat-szint, a gyártási szabállyal) ===", flush=True)
    print(f"csak Parakeet : WER {base_p['wer']*100:5.2f} %  "
          f"teljes könyv {base_p['book_minutes']} perc", flush=True)
    print(f"csak Whisper  : WER {base_w['wer']*100:5.2f} %  "
          f"teljes könyv {base_w['book_minutes']} perc", flush=True)
    print(f"{'küszöb':>7} {'megerősít':>10} {'javít':>6} {'pazarolt':>9} "
          f"{'WER':>8} {'konyv perc':>11}", flush=True)
    for s in sweep:
        print(f"{s['threshold']:>7.2f} {s['triggered_pct']:>9.1f}% "
              f"{s['improved']:>6} {s['wasted']:>9} {s['wer']*100:>7.2f}% "
              f"{s['book_minutes']:>11}", flush=True)

    result = {"chapters": CHAPTERS, "sentences": len(rows),
              "audio_sec": round(audio_total, 1), "book_seconds": round(book, 1),
              "parakeet_only": base_p, "whisper_only": base_w,
              "sweep": sweep, "rows": rows}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\njelentés: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
