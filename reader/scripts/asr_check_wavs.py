"""ASR-ellenőrzés tetszőleges WAV-készletre (WER/CER).

Használat (a repo venvjéből):
    .venv/Scripts/python.exe scripts/asr_check_wavs.py --wav-dir <dir> --json <f.json>

A json a mapping: {"fajl.wav": "az elvárt szöveg"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qa import Transcriber, score_transcript  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="{fajl.wav: elvárt szöveg}")
    ap.add_argument("--language", default="hu")
    ap.add_argument("--no-parakeet", action="store_true")
    args = ap.parse_args()

    cases = json.load(open(args.json, encoding="utf-8"))
    tr = Transcriber.for_language(None if args.no_parakeet else args.language)

    results = []
    for name, expected in cases.items():
        t0 = time.perf_counter()
        result = tr.transcribe(name, args.language)
        dt = time.perf_counter() - t0
        # A Transcriber dictet ad vissza ({"text", "words"}); a score csak a szöveget kéri.
        heard = result["text"] if isinstance(result, dict) else str(result)
        sc = score_transcript(expected, heard, args.language)
        results.append(sc)
        print(f"{os.path.basename(name)}  WER {sc['wer']:6.2f}%  CER {sc['cer']:6.2f}%")
        print(f"   elvárt: {expected}")
        print(f"   hallott: {sc['heard']}")
        if sc.get("missing_words"):
            print(f"   hiányzik: {sc['missing_words']}")
        print()

    n = len(results)
    if n:
        avg_wer = sum(r["wer"] for r in results) / n
        avg_cer = sum(r["cer"] for r in results) / n
        print(f"átlag: WER {avg_wer:.2f}%  CER {avg_cer:.2f}%  ({n} mondat)")
        worst = max(results, key=lambda r: r["wer"])
        print(f"legrosszabb: WER {worst['wer']:.2f}% — {worst['expected'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())