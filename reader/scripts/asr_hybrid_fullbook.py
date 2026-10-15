"""Hibrid QC a teljes hangoskönyvön — mind az 15 fejezet, minden mondat (M-55).

A M-54 mondat-szintű protokollját futtatja a teljes könyvön, a gyártási
megerősítő szabállyal (``confirm_above`` = ``qa_cer_warn`` = 0,08):

  1. Fejezetenként teljes Parakeet-átirat 180 s-os ablakokban, szóidőbélyegekkel
     (a Parakeet ONNX encoder ennyit bír el egyben; M-53).
  2. ``alignment.align_words`` a forrásszövegre — minden mondatnak pontos
     időtartama lesz a hangban (nem arányos metszés).
  3. Minden mondat: Parakeet feliratkoztatás; ha CER ≥ küszöb, a Whisper
     újrahallgatja, és a jobb (kisebb CER-ű) olvasat számít — ugyanaz a szabály,
     amit a gyártási ``qa_api._evaluate`` használ.
  4. Végső WER/CER: mikro-átlag (szó-/karakterhibák / összes szó/karakter) a
     megtartott olvasatokból — ez a „megerősítés utáni" szám. A makro-átlag
     (mondatonkénti WER-ek átlaga) is megvan, a M-54-es számokkal való
     összehasonlíthatóságért.

Checkpoint fejezetenként; ``--resume`` folytatja a meglévő JSON-ból.

Futtatás (auris venv, a reader könyvtárból):
    .venv/Scripts/python.exe scripts/asr_hybrid_fullbook.py
    .venv/Scripts/python.exe scripts/asr_hybrid_fullbook.py --chapters 11 --max-sentences 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HUB = r"D:\VoiceAI"
TXT_DIR = os.path.join(HUB, "out", "minta-klon", "teljes")
WAV_DIR = os.path.join(HUB, "out", "minta-klon", "hangoskonyv_teljes")
OUT = os.path.join(WAV_DIR, "qa_hybrid_fullbook.json")
TMP_DIR = os.path.join(HUB, "out", "minta-klon", "_hybrid_tmp")

ALL_CHAPTERS = list(range(15))
WINDOW = 180.0                 # Parakeet ONNX ablak (M-53/M-54)
DEFAULT_THRESHOLD = 0.08       # gyártási megerősítő küszöb (qa_cer_warn)
SENT_END = re.compile(r'[.!?…]["»”’)\]]*$')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chapter_files(index: int) -> tuple[str, str]:
    """A fejezet forrásszövege és generált hangja (a 0 = előszó eltérő néven)."""
    txt = os.path.join(TXT_DIR,
                       "00_main_szoveg_tisztitott.txt" if index == 0
                       else f"{index:02d}_main-{index}_szoveg_tisztitott.txt")
    wav = os.path.join(WAV_DIR,
                       "00_main_eloszo01.wav" if index == 0
                       else f"{index:02d}_main-{index}_eloszo01.wav")
    return txt, wav


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return "\n".join(p.strip() for p in f.read().split("\n\n") if p.strip())


def split_sentences(text: str) -> list[dict]:
    """Mondatai karakterkoordinátákkal (a szó-tokenek indexe kell az illesztéshez)."""
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    sentences, start, words = [], None, 0
    for pos, (begin, end, tok) in enumerate(tokens):
        if start is None:
            start = begin
        words += 1
        if SENT_END.search(tok) or pos == len(tokens) - 1:
            sentences.append({"start": start, "end": end, "words": words,
                              "text": text[start:end].strip()})
            start, words = None, 0
    return [s for s in sentences if 1 <= s["words"] <= 60]


def chapter_parakeet_words(parakeet, wav_path: str, index: int):
    """Teljes fejezet Parakeet-átirata ablakokban, időbélyegekkel tolva.

    A hang ablakonként kerül lemezről, hogy a 87 perces 0. fejezet se húzzon
    fel ~1 GB-ot a memóriába egy 4 órás felügyeletlen futás alatt.
    """
    import soundfile as sf

    info = sf.info(wav_path)
    sr = info.samplerate
    duration = info.frames / sr
    words: list[dict] = []
    win = int(WINDOW * sr)
    for widx, frame_start in enumerate(range(0, info.frames, win)):
        chunk, _ = sf.read(wav_path, start=frame_start,
                           stop=min(frame_start + win, info.frames),
                           dtype="float32", always_2d=True)
        chunk = chunk.mean(axis=1)
        if len(chunk) < sr:  # 1 s-nál rövidebb farok kihagyása
            continue
        tmp = os.path.join(TMP_DIR, f"_chapter_{index}_{widx}.wav")
        sf.write(tmp, chunk, sr)
        offset = frame_start / sr
        for word in parakeet.transcribe(tmp, "hu")["words"]:
            if word.get("start") is not None:
                words.append({"word": word["word"],
                              "start": float(word["start"]) + offset,
                              "end": (float(word["end"]) + offset
                                      if word.get("end") is not None else None)})
        os.remove(tmp)
    return words, duration, sr


def cut_sentence(wav_path: str, sr: int, t_start: float, t_end: float, dest: str):
    import soundfile as sf

    chunk, _ = sf.read(wav_path, start=int(t_start * sr), stop=int(t_end * sr),
                       dtype="float32", always_2d=True)
    sf.write(dest, chunk.mean(axis=1), sr)


def counts(reference: str, heard: str) -> tuple[int, int, int]:
    """(szó, szóhiba, karakterhiba) a comparable_text-en — mikro-átlaghoz."""
    from core import qa

    ref = qa.comparable_text(reference, "hu")
    hyp = qa.comparable_text(heard, "hu")
    ref_words, hyp_words = ref.split(), hyp.split()
    return (len(ref_words),
            qa._levenshtein(ref_words, hyp_words),
            qa._levenshtein(ref.replace(" ", ""), hyp.replace(" ", "")))


def run_chapter(index: int, parakeet, whisper, threshold: float, max_sentences: int):
    from core import alignment, qa

    txt_path, wav_path = chapter_files(index)
    if not os.path.exists(wav_path):
        print(f"[{index}] HIÁNYZÓ WAV: {wav_path}", flush=True)
        return None
    text = load_text(txt_path)
    t0 = time.perf_counter()
    words, duration, sr = chapter_parakeet_words(parakeet, wav_path, index)
    t_full = time.perf_counter() - t0
    print(f"[{index}] Parakeet teljes átirat {duration/60:.1f} perc hangra "
          f"{t_full:.0f} s alatt, {len(words)} szó", flush=True)

    t0 = time.perf_counter()
    timings = alignment.align_words(text, words, duration, "hu")
    print(f"[{index}] illesztés a forrásszövegre {time.perf_counter()-t0:.0f} s "
          f"({len(timings)} szó-idő)", flush=True)

    # mondatok -> szó-token index tartomány -> időtartomány a hangban
    token_starts = [m.start() for m in re.finditer(r"\S+", text)]
    token_index = {begin: i for i, begin in enumerate(token_starts)}
    sentences = split_sentences(text)
    prepared, skipped = [], 0
    for sent in sentences:
        first = token_index.get(sent["start"])
        if first is None or first >= len(timings):
            skipped += 1
            continue
        last = min(first + sent["words"] - 1, len(timings) - 1)
        t_start = max(timings[first][0] - 0.05, 0.0)
        t_end = min(timings[last][1] + 0.15, duration)
        # óvintézkedés: ésszerű mondat-hossz (2 és 60 s közt); a hibás
        # idővonal itt bukna ki, nem az ONNX encoderben (M-54)
        if not (2.0 <= t_end - t_start <= 60.0):
            skipped += 1
            continue
        prepared.append((sent, t_start, t_end))
    if max_sentences:
        prepared = prepared[:max_sentences]
    print(f"[{index}] {len(sentences)} mondatból {len(prepared)} mérve "
          f"({skipped} kihagyva: nincs illesztés vagy ésszerűtlen hossz)", flush=True)

    rows: list[dict] = []
    stats = {"ref_words": 0, "word_edits_before": 0, "word_edits_after": 0,
             "ref_chars": 0, "char_edits_before": 0, "char_edits_after": 0,
             "triggered": 0, "improved": 0, "sentences": 0,
             "audio_sec": 0.0, "t_parakeet_sent": 0.0, "t_whisper": 0.0}
    tmp = os.path.join(TMP_DIR, "_sentence.wav")
    for n, (sent, t_start, t_end) in enumerate(prepared, 1):
        cut_sentence(wav_path, sr, t_start, t_end, tmp)
        t0 = time.perf_counter()
        heard_p = parakeet.transcribe(tmp, "hu")
        tp = time.perf_counter() - t0
        score_p = qa.score_transcript(sent["text"], heard_p["text"], "hu")
        cer_w = wer_w = None
        tw = 0.0
        triggered = score_p["cer"] >= threshold
        if triggered:
            t0 = time.perf_counter()
            heard_w = whisper.transcribe(tmp, "hu")
            tw = time.perf_counter() - t0
            score_w = qa.score_transcript(sent["text"], heard_w["text"], "hu")
            cer_w, wer_w = score_w["cer"], score_w["wer"]
        # gyártási szabály: megerősítésnél csak a jobb (kisebb CER-ű) olvasat számít
        use_w = triggered and cer_w < score_p["cer"]
        kept = heard_w["text"] if use_w else heard_p["text"]
        rw, wp_before, cp_before = counts(sent["text"], heard_p["text"])
        _, wp_after, cp_after = counts(sent["text"], kept)
        stats["ref_words"] += rw
        stats["word_edits_before"] += wp_before
        stats["word_edits_after"] += wp_after
        stats["ref_chars"] += len(qa.comparable_text(sent["text"], "hu").replace(" ", ""))
        stats["char_edits_before"] += cp_before
        stats["char_edits_after"] += cp_after
        stats["triggered"] += triggered
        stats["improved"] += bool(use_w)
        stats["sentences"] += 1
        stats["audio_sec"] += t_end - t_start
        stats["t_parakeet_sent"] += tp
        stats["t_whisper"] += tw
        rows.append({"chapter": index, "text": sent["text"][:70],
                     "sec": round(t_end - t_start, 2),
                     "wer_p": score_p["wer"], "cer_p": score_p["cer"],
                     "wer_w": wer_w, "cer_w": cer_w,
                     "triggered": triggered, "used_whisper": bool(use_w),
                     "tp": round(tp, 2), "tw": round(tw, 2)})
        if n % 25 == 0 or n == len(prepared):
            print(f"  [{index}] {n}/{len(prepared)} mondat | "
                  f"megerősítve {stats['triggered']} | javítva {stats['improved']} | "
                  f"utolsó: P CER {score_p['cer']:.3f}"
                  + (f" → W CER {cer_w:.3f}" if triggered else ""), flush=True)
    os.remove(tmp)
    stats["t_full_pass"] = round(t_full, 1)
    return {"index": index, "duration_sec": round(duration, 1),
            "sentences_total": len(sentences), "skipped": skipped,
            "stats": stats, "rows": rows}


def micro(rows_stats: list[dict], after: bool) -> float:
    edits = sum(s["word_edits_after" if after else "word_edits_before"] for s in rows_stats)
    ref = sum(s["ref_words"] for s in rows_stats)
    return round(edits / ref, 4) if ref else 0.0


def micro_cer(rows_stats: list[dict], after: bool) -> float:
    edits = sum(s["char_edits_after" if after else "char_edits_before"] for s in rows_stats)
    ref = sum(s["ref_chars"] for s in rows_stats)
    return round(edits / ref, 4) if ref else 0.0


def macro(rows: list[dict], after: bool) -> float:
    values = [(r["wer_w"] if after and r["used_whisper"] else r["wer_p"]) for r in rows]
    return round(sum(values) / len(values), 4) if values else 0.0


def save(chapters_done: dict, threshold: float, out: str) -> dict:
    all_rows = [r for c in chapters_done.values() for r in c["rows"]]
    stats = [c["stats"] for c in chapters_done.values()]
    summary = {
        "threshold": threshold,
        "chapters_done": sorted(int(k) for k in chapters_done),
        "sentences": sum(s["sentences"] for s in stats),
        "audio_sec": round(sum(s["audio_sec"] for s in stats), 1),
        "triggered": sum(s["triggered"] for s in stats),
        "improved": sum(s["improved"] for s in stats),
        "wer_before_micro": micro(stats, False),
        "wer_after_micro": micro(stats, True),
        "wer_before_macro": macro(all_rows, False),
        "wer_after_macro": macro(all_rows, True),
        "cer_before_micro": micro_cer(stats, False),
        "cer_after_micro": micro_cer(stats, True),
        "asr_seconds": round(sum(s["t_parakeet_sent"] + s["t_whisper"] for s in stats), 1),
    }
    result = {"protocol": "M-55: M-54 mondat-szintű hibrid QC a teljes 15 fejezeten",
              "summary": summary,
              "chapters": {k: {kk: vv for kk, vv in c.items() if kk != "rows"}
                           for k, c in chapters_done.items()},
              "rows": all_rows}
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", default="", help="pl. 0,4,7 — üresen mind az 15")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--max-sentences", type=int, default=0,
                    help="csak az első N mondat fejezetenként (füstteszt)")
    ap.add_argument("--resume", action="store_true",
                    help="a meglévő JSON-ban már megvolt fejezetek kihagyása")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from core import qa
    import soundfile as sf  # noqa: F401  (a parancs korai hibát ad, ha hiányzik)

    os.makedirs(TMP_DIR, exist_ok=True)
    chapters = ([int(x) for x in args.chapters.split(",")] if args.chapters
                else list(ALL_CHAPTERS))
    done: dict = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            previous = json.load(f)
        done = previous.get("chapters", {})
        # A checkpoint a mondat-sorokat külön, felüli listában tárolja; a
        # folytatáskor vissza kell őket tenni a fejezetekhez, különben az
        # új fejezet utáni mentés `KeyError: 'rows'`-ban bukik.
        rows_by_chapter: dict = {}
        for row in previous.get("rows", []):
            rows_by_chapter.setdefault(str(row["chapter"]), []).append(row)
        for key, chapter in done.items():
            chapter.setdefault("rows", rows_by_chapter.get(key, []))
        print(f"újraindítás: {len(done)} kész fejezet kihagyva "
              f"({sorted(int(k) for k in done)})", flush=True)

    parakeet = qa.ParakeetTranscriber()
    whisper = qa.Transcriber.whisper_for("hu")
    print(f"ASR: Parakeet {qa.PARAKEET_MODEL} + Whisper {whisper.model_id}; "
          f"megerősítő küszöb CER ≥ {args.threshold}", flush=True)
    print(f"kimenet: {args.out}", flush=True)

    t_book0 = time.perf_counter()
    for index in chapters:
        if str(index) in done:
            continue
        t0 = time.perf_counter()
        chapter = run_chapter(index, parakeet, whisper, args.threshold,
                              args.max_sentences)
        if chapter is None:
            continue
        done[str(index)] = chapter
        save(done, args.threshold, args.out)
        s = chapter["stats"]
        before = (s["word_edits_before"] / s["ref_words"]) if s["ref_words"] else 0
        after = (s["word_edits_after"] / s["ref_words"]) if s["ref_words"] else 0
        print(f"[{index}] KÉSZ {time.perf_counter()-t0:.0f} s | WER {before*100:.2f} % "
              f"→ megerősítés után {after*100:.2f} % | checkpoint mentve", flush=True)

    summary = save(done, args.threshold, args.out)
    book = time.perf_counter() - t_book0
    print("\n=== VÉGEREDMÉNY (mind a feldolgozott fejezet) ===", flush=True)
    print(f"fejezetek       : {summary['chapters_done']}", flush=True)
    print(f"mondatok        : {summary['sentences']} "
          f"({summary['audio_sec']/60:.1f} perc hang)", flush=True)
    print(f"megerősítés     : {summary['triggered']} mondat "
          f"({100*summary['triggered']/max(summary['sentences'],1):.1f} %), "
          f"ebből {summary['improved']} javított a Whisper-olvasat", flush=True)
    print(f"WER (mikro)     : {summary['wer_before_micro']*100:.2f} % → "
          f"{summary['wer_after_micro']*100:.2f} %", flush=True)
    print(f"WER (makro)     : {summary['wer_before_macro']*100:.2f} % → "
          f"{summary['wer_after_macro']*100:.2f} %", flush=True)
    print(f"CER (mikro)     : {summary['cer_before_micro']*100:.2f} % → "
          f"{summary['cer_after_micro']*100:.2f} %", flush=True)
    print(f"ASR-idő         : {summary['asr_seconds']/60:.1f} perc "
          f"(mért mondatokra), teljes falidő {book/60:.1f} perc", flush=True)
    print(f"jelentés        : {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
