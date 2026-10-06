"""Hallgatható minták a serial vs batchelt összehasonlításhoz.

A gyorsítást (batch=4) a mérés igazolta, de a hangot is ellenőrizni kell.
Ez a script a magyar WER-korpuszt (`core.hu_wer.CORPUS`) a kiválasztott
hangokon **kétszer** szintetizálja — egyszer serial (`supertonic_batch=1`),
egyszer batchelt (`supertonic_batch=4`) —, és ember által hallgatható,
sorszámozott fájlokat ír ki:

    exports/test-samples/F1/01-simple__serial.wav
    exports/test-samples/F1/01-simple__batch.wav
    ...

A hangcache kiürül a mért mondatokra, különben a második kör csak
gyorsítótárazott felvételt adna vissza, és a gyorsulás láthatatlan lenne.

Használat:
    .venv/Scripts/python.exe scripts/export_test_samples.py
    .venv/Scripts/python.exe scripts/export_test_samples.py --voices F1 M3 --mp3
    .venv/Scripts/python.exe scripts/export_test_samples.py --variant int8

Az fp32 minták az ``<out>/<hang>/``, az INT8 minták az ``<out>/int8/<hang>/``
mappába kerülnek, így a két variáns összehasonlítható anélkül, hogy bármelyik
felülírná a másikat. A hangcache kulcsa is tartalmazza a ``variant=`` részt,
tehát a két variáns nem szolgálhatja ki egymás hangját.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
READER = os.path.dirname(HERE)
if READER not in sys.path:
    sys.path.insert(0, READER)

from core import settings  # noqa: E402
from core.hu_wer import CORPUS  # noqa: E402
from core.local_engines import (  # noqa: E402
    SUPERTONIC_VOICES,
    SupertonicEngine,
    choose_preset_voice,
)


def safe_name(text: str, index: int) -> str:
    keep = [c if c.isalnum() else "_" for c in text[:28]]
    return "%02d-%s" % (index, "".join(keep).strip("_"))


def to_mp3(wav_path: str, ffmpeg: str) -> bool:
    out = os.path.splitext(wav_path)[0] + ".mp3"
    try:
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", wav_path,
             "-codec:a", "libmp3lame", "-q:a", "3", out],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"  (mp3 konverzió kimaradt: {exc})")
        return False


def synthesize(engine, voice: str, batch: int, out_dir: str, make_mp3: bool, ffmpeg: str):
    settings.save({"supertonic_batch": batch})
    normalize = bool(settings.get("normalize_text", True))
    items, names, keys = [], [], []
    for index, case in enumerate(CORPUS, start=1):
        instruct = "voice:" + voice
        key = engine.cache_key(case.text, instruct, None, 1.0, language="hu",
                               normalize_text=normalize)
        path = engine.cache_path(key)
        if os.path.exists(path):
            os.remove(path)  # cold cache: the timing must be real
        items.append({"text": case.text, "language": "hu", "instruct": instruct})
        names.append(safe_name(case.text, index))
        keys.append(path)

    os.makedirs(out_dir, exist_ok=True)
    started = time.perf_counter()
    results = engine.generate_many(items, batch_size=batch)
    wall = time.perf_counter() - started

    written, audio = [], 0.0
    for name, result in zip(names, results):
        if not result or not result.get("audio_path"):
            continue
        wav_path = os.path.join(out_dir, name + "__serial.wav"
                                if batch == 1 else name + "__batch.wav")
        shutil.copyfile(result["audio_path"], wav_path)
        audio += float(result.get("duration_sec") or 0.0)
        written.append(wav_path)
        if make_mp3:
            to_mp3(wav_path, ffmpeg)
    return wall, audio, written, keys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", nargs="+", default=["F1", "M3"],
                        help="melyik beépített hangokon (F1-F5, M1-M5)")
    parser.add_argument("--out", default=os.path.join(READER, "exports", "test-samples"))
    parser.add_argument("--mp3", action="store_true", help="készíts MP3-et is (ffmpeg)")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--variant", choices=("fp32", "int8"), default="fp32",
                        help="melyik modellvariáns szólaljon meg")
    args = parser.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    if args.mp3 and not shutil.which("ffmpeg"):
        print("figyelem: ffmpeg nincs a PATH-on, csak WAV készül")

    settings.save({"supertonic_variant": args.variant})
    engine = SupertonicEngine()
    engine.load_sync()
    # The cache key carries "variant=...", so the two variants can never serve
    # each other's audio; each run also gets its own sample folder.
    print(f"variáns: {engine.variant} · settings_identity={engine.settings_identity()}")

    out_root = args.out if args.variant == "fp32" else os.path.join(
        args.out, args.variant)
    print(f"korpusz: {len(CORPUS)} mondat · hangok: {', '.join(args.voices)}\n")
    baseline = {}
    for voice in args.voices:
        chosen = choose_preset_voice("voice:" + voice, SUPERTONIC_VOICES, "F1")
        out_dir = os.path.join(out_root, chosen)
        for batch in args.batches:
            wall, audio, written, _ = synthesize(engine, chosen, batch, out_dir,
                                                 args.mp3, ffmpeg)
            tag = "serial" if batch == 1 else f"batch{batch}"
            speed = ""
            if batch == 1:
                baseline[chosen] = (wall, audio)
            elif chosen in baseline and audio:
                speed = f"  ({baseline[chosen][0] / wall:.2f}x gyorsabb)"
            print(f"{chosen:<4} {tag:<8} {wall:6.2f}s wall · {audio:6.1f}s hang · "
                  f"RTF {wall / audio if audio else 0:.4f} · {len(written)} fájl{speed}")

    settings.save({"supertonic_batch": 4})
    print(f"\nMinták: {os.path.abspath(out_root)}")
    print("Párokítás: azonos nevű __serial és __batch fájl egy mondat két felvétele.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())