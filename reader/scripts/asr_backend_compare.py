"""Melyik ASR ellenőrzi jobban a generált hangoskönyvet?

Ugyanazon a fejezet-hangon és ugyanazon a forrásszövegen futtat több
beszédfelismerőt, és CER/WER-rel pontozza mindegyiket. Ebből látszik, hogy az
eltérés a TTS-felolvasásból ered-e (akkor minden felismerő ugyanazt a hibát
találja), vagy egy adott ASR zajából (akkor az eltérő modellek mást hallanak).

Háttér: a teljes könyves futás Parakeettel készül (gyors, ONNX, `onnx-asr`).
A Whisper-út a `core/qa.py` `Transcriber` osztályán át érhető el (transformers +
torch, CPU).

Futtatás (a repo venvjéből):
    .venv/Scripts/python.exe scripts/asr_backend_compare.py \
        --index 4 --seconds 180 --backends parakeet,whisper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qa import HUNGARIAN_ASR_MODEL, score_transcript  # noqa: E402

HUB = r"D:\<projekt>"
TXT_DIR = os.path.join(HUB, "out", "minta-klon", "teljes")
WAV_DIR = os.path.join(HUB, "out", "minta-klon", "hangoskonyv_teljes")


def chapter_files(index: int) -> tuple[str, str]:
    txt = os.path.join(TXT_DIR, f"{index:02d}_main-{index}_szoveg_tisztitott.txt"
                       if index else "00_main_szoveg_tisztitott.txt")
    wav = os.path.join(WAV_DIR, f"{index:02d}_main{'-' + str(index) if index else ''}_eloszo01.wav")
    return txt, wav


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return "\n".join(p.strip() for p in f.read().split("\n\n") if p.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=180, help="0 = teljes fejezet")
    ap.add_argument("--backends", default="parakeet,whisper")
    ap.add_argument("--whisper-model", default=HUNGARIAN_ASR_MODEL)
    ap.add_argument("--window", type=float, default=180, help="ASR-ablak (mp) — Parakeethez")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import soundfile as sf
    from core import qa

    txt_path, wav_path = chapter_files(args.index)
    if not os.path.exists(wav_path):
        print(f"nincs WAV: {wav_path}")
        return 1

    info = sf.info(wav_path)
    duration = info.duration
    text = load_text(txt_path)
    if args.seconds:
        text = text[: int(len(text) * min(args.seconds, duration) / duration)]

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if args.seconds:
        audio = audio[: int(args.seconds * sr)]
    windowed = len(audio) > args.window * sr

    # Az időtartam-rövidítésnél a WAV-ot külön fájlba írjuk ki, hogy minden
    # backend ugyanazt a hangot hallja (a Transcriber mindig fájlt kér).
    work_wav = wav_path
    if args.seconds:
        work_wav = os.path.join(HUB, "out", "minta-klon", f"_cmp_{args.index}.wav")
        sf.write(work_wav, audio, sr)

    results = []
    try:
        for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
            if backend == "parakeet":
                tr = qa.ParakeetTranscriber()
                model_id = tr.model_id
            else:
                model = args.whisper_model if backend == "whisper" else backend
                tr = qa.Transcriber(model)
                model_id = model

            t0 = time.perf_counter()
            if backend == "parakeet" and windowed:
                win = int(args.window * sr)
                parts = []
                for start in range(0, len(audio), win):
                    chunk = audio[start:start + win]
                    if len(chunk) < sr:
                        continue
                    tmp = os.path.join(HUB, "out", "minta-klon", f"_cmp_chunk_{args.index}.wav")
                    sf.write(tmp, chunk, sr)
                    parts.append(tr.transcribe(tmp, "hu")["text"])
                    os.remove(tmp)
                heard = " ".join(parts)
            else:
                heard = tr.transcribe(work_wav, "hu")["text"]
            elapsed = time.perf_counter() - t0

            checked = min(duration, args.seconds or duration)
            sc = score_transcript(text, heard, "hu")
            entry = {
                "backend": model_id,
                "cer": sc["cer"],
                "wer": sc["wer"],
                "missing_words": sc["missing_words"],
                "checked_sec": round(checked, 1),
                "asr_sec": round(elapsed, 1),
                "rtf": round(elapsed / max(checked, 1e-9), 3),
                "heard": heard,
                "heard_excerpt": heard[:200],
            }
            results.append(entry)
            print(json.dumps(entry, ensure_ascii=False, indent=1), flush=True)
    finally:
        if args.seconds and os.path.exists(work_wav):
            os.remove(work_wav)

    print("\nÖsszehasonlítás (alacsonyabb a jobb):")
    for r in results:
        print(f"  {r['backend']:<45} CER {r['cer']:6.2%}  WER {r['wer']:6.2%}  "
              f"RTF {r['rtf']:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"index": args.index, "results": results}, f,
                      ensure_ascii=False, indent=1)
        print(f"jelentés: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
