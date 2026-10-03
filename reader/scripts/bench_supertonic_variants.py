r"""Supertonic 3 fp32 kontra INT8: méret, betöltés, RTF és magyar érthetőség.

    .venv\Scripts\python.exe scripts\bench_supertonic_variants.py
    .venv\Scripts\python.exe scripts\bench_supertonic_variants.py --cases 5 --skip-wer
    .venv\Scripts\python.exe scripts\bench_supertonic_variants.py --variant int8

Minden variáns **külön processzben** fut: a Windows peak working set monoton
növekszik, ezért egy processzen belül a második betöltés memóriaigénye nem
látható. Az ``--variant`` ezt a belső módot indítja, a `--json-out` pedig a
részleteket fájlba írja.

A két variáns ugyanazt a magyar korpuszt olvasza fel, így a WER közvetlenül
megmondja, hogy a kvantálás ront-e a kiejtésen. A hangcache-kulcs tartalmazza
a variánst, ezért a kettő nem szolgálhatja ki egymás hangját.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.hu_wer import CORPUS, run_bench  # noqa: E402


def peak_mem_mb() -> float:
    """A folyamat csúcsmemória-igénye MB-ban (a Windows KiB-ben adja meg)."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / (1024 * 1024)


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _drop_cached_take(engine, text: str, take: int, normalize: bool) -> None:
    """Delete one cached take so the next generate really synthesises.

    Mirrors how `generate` derives the file name: the plain cache key (which
    includes ``normalize_text``), then the take-specific hash. Recomputing it
    here beats clearing the whole cache, which would throw away the user's
    audiobook renders.
    """
    import hashlib

    base = engine.cache_key(text, None, None, 1.0, language="hu",
                            normalize_text=normalize)
    key = hashlib.md5(f"{base}|take={int(take)}".encode("utf-8")).hexdigest()
    Path(engine.cache_path(key)).unlink(missing_ok=True)


def measure(variant: str, cases, steps: int, transcribe, measure_wer: bool,
            rtf_samples: int = 3) -> dict:
    # A beállításnak a motor felépítése ELŐTT kell állnia, mert a mappa és a
    # fájlnév a variánsból dől ki.
    from core import local_engines, settings

    settings.save({"supertonic_variant": variant, "supertonic_steps": steps})
    engine = local_engines.SupertonicEngine()
    result = {
        "variant": engine.variant,
        "folder": str(engine.folder),
        "voice_dir": str(engine.voice_dir),
        "steps": steps,
    }
    result["size_mb"] = round(dir_size_mb(engine.folder), 1)
    if not engine.model_present():
        raise RuntimeError(
            f"{variant}: hiányzó modellfájlok a {engine.folder} mappában")

    before = peak_mem_mb()
    started = time.perf_counter()
    engine.load_sync()
    result["load_sec"] = round(time.perf_counter() - started, 2)
    result["peak_mem_mb"] = round(peak_mem_mb(), 1)
    result["load_mem_delta_mb"] = round(result["peak_mem_mb"] - before, 1)

    # RTF: a `take` a gyorsítótár-kulcs része, így minden futás valódi
    # generálás. A szöveg szándékosan nem a korpuszból való, hogy a WER-futás
    # ne találjon gyorsítótárazott felvételt.
    text = "Ez a mérés bemelegítő mondata, nem része a korpusznak."
    normalize = bool(settings.get("normalize_text", True))
    engine.generate(text, language="hu", take=0)  # betölti a hangstílust
    # A korábbi mérések felvételeit ne találja meg: azok gyorsítótárazva
    # vannak, és a cache-találatot nulla generálási idővel számolnánk.
    for attempt in range(1, rtf_samples + 1):
        _drop_cached_take(engine, text, attempt, normalize)
    audio_seconds = 0.0
    gen_seconds = 0.0
    for attempt in range(1, rtf_samples + 1):
        started = time.perf_counter()
        out = engine.generate(text, language="hu", take=attempt,
                              normalize_text=normalize)
        gen_seconds += time.perf_counter() - started
        if out.get("cache_hit"):
            raise RuntimeError("a RTF-mérés gyorsítótárazott felvételt számolt")
        audio_seconds += float(out["duration_sec"])
    result["rtf"] = round(gen_seconds / audio_seconds, 4)
    result["rtf_samples"] = rtf_samples
    result["rtf_audio_sec"] = round(audio_seconds, 1)

    if measure_wer:
        def synthesize(case):
            out = engine.generate(case.text, language="hu")
            if out.get("cache_hit"):
                raise RuntimeError(f"{case.id}: gyorsítótárazott hangot mértünk")
            return out["audio_path"]

        report = run_bench(cases, synthesize, transcribe)
        result["wer"] = report["wer"]
        result["cer"] = report["cer"]
        result["errors"] = report["errors"]
        result["failed"] = [
            {"id": r["id"], "error": r["error"]}
            for r in report["results"] if "error" in r
        ]
        result["worst"] = [
            {"id": r["id"], "wer": r["wer"], "missing": r["missing_words"][:4]}
            for r in report["worst"]
        ]
        result["per_case_wer"] = {
            r["id"]: r["wer"] for r in report["results"] if "wer" in r
        }
    engine.unload()
    gc.collect()
    return result


def compare(results: list[dict]) -> None:
    fp32 = next(r for r in results if r["variant"] == "fp32")
    int8 = next(r for r in results if r["variant"] == "int8")
    print("\n" + "=" * 66)
    print(f"{'mutató':<26}{'fp32':>12}{'int8':>12}{'változás':>14}")
    print("-" * 66)
    for key, label, scale in (
        ("size_mb", "modellmappa (MB)", 1),
        ("load_sec", "betöltés (s)", 1),
        ("peak_mem_mb", "csúcsmemória (MB)", 1),
        ("rtf", "RTF (kisebb = gyorsabb)", 1),
        ("wer", "WER %", 100),
    ):
        if key not in fp32 or fp32[key] is None:
            continue
        a, b = fp32[key] * scale, int8[key] * scale
        pct = (b - a) / a * 100 if a else 0.0
        print(f"{label:<26}{a:>12.2f}{b:>12.2f}{pct:>13.1f}%")

    print(f"\nRTF: fp32 {1 / fp32['rtf']:.1f}× valós idejű, "
          f"int8 {1 / int8['rtf']:.1f}× valós idejű")
    for row in results:
        for bad in row.get("failed", []):
            print(f"  HIBA {row['variant']} {bad['id']}: {bad['error']}")
    if "wer" in fp32 and fp32["wer"] and int8.get("wer") is not None:
        delta = (int8["wer"] - fp32["wer"]) * 100
        print(f"\nWER különbség: {delta:+.2f} százalékpont")
        if delta <= 0.5:
            print("Az INT8 kiejtése nem romlott érdemben.")
        elif delta <= 2.0:
            print("Az INT8 kiejtése kissé romlott; érdemes egy teljes könyvön mérni.")
        else:
            print("Az INT8 kiejtése számottevően romlott: maradj fp32-nél.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Supertonic 3 fp32 kontra INT8")
    parser.add_argument("--variant", choices=("fp32", "int8"),
                        help="egyetlen variáns; kihagyásakor mindkettő külön processzben fut")
    parser.add_argument("--cases", type=int, default=0, help="csak ennyi mondat (0 = mind)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rtf-samples", type=int, default=3,
                        help="hány generálásból számoljuk az RTF-t")
    parser.add_argument("--skip-wer", action="store_true", help="ASR nélkül, csak RTF")
    parser.add_argument("--json-out", help="ebbe a fájlba írja az eredményt (JSON)")
    args = parser.parse_args(argv)

    cases = list(CORPUS)[:args.cases] if args.cases else list(CORPUS)
    want_wer = not args.skip_wer

    if args.variant:
        transcribe = None
        if want_wer:
            from core.qa import Transcriber

            transcriber = Transcriber.for_language("hu")
            print(f"ASR: {getattr(transcriber, 'model_id', '?')}", flush=True)

            def transcribe(path):
                return transcriber.transcribe(path, "hu").get("text", "")

        result = measure(args.variant, cases, args.steps, transcribe, want_wer,
                      args.rtf_samples)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("per_case_wer", "failed", "worst")},
                         ensure_ascii=False, indent=2))
        return 0

    # Külön processz per variáns: csak így mérhető a csúcsmemória.
    outputs: dict[str, str] = {}
    for variant in ("fp32", "int8"):
        target = Path(args.json_out or "supertonic_bench.json")
        target = target.with_name(f"{target.stem}-{variant}{target.suffix or '.json'}")
        outputs[variant] = str(target)
        command = [sys.executable, str(Path(__file__)), "--variant", variant,
                   "--steps", str(args.steps), "--rtf-samples", str(args.rtf_samples),
                   "--json-out", str(target)]
        if args.cases:
            command += ["--cases", str(args.cases)]
        if args.skip_wer:
            command.append("--skip-wer")
        print(f"--- {variant} külön processzben ---", flush=True)
        completed = subprocess.run(command, cwd=str(ROOT))
        if completed.returncode != 0:
            print(f"A {variant} variáns mérése nem sikerült.")
            return completed.returncode

    results = []
    for variant, path in outputs.items():
        with open(path, encoding="utf-8") as handle:
            results.append(json.load(handle))

    from core import settings

    settings.save({"supertonic_variant": "fp32"})
    compare(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\nRészletek: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())