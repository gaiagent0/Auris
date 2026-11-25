#!/usr/bin/env python3
"""A szinkronstúdió környezetének ellenőrzése, és új dubstudio projekt indítása.

A szinkronfelület (`/szinkron`, `/felulvizsgalat`) a **dubstudio** láncot
használja, amely a repó része (`dubstudio/`, a `DUBSTUDIO_ROOT`-tal máshova is
állítható). Ez a szkript azt mondja meg, mi hiányzik ahhoz, hogy egy friss
klónon működjön: megvan-e a dubstudio kódja, a Python csomagok, az
ffmpeg/ffprobe, az espeak-ng, az `audio-separator`, a modellek és a projektfájl.

Használat (a repo gyökeréből, az Auris venv-jével):

    reader\\.venv\\Scripts\\python.exe scripts\\check_dubstudio.py
    reader\\.venv\\Scripts\\python.exe scripts\\check_dubstudio.py --json
    reader\\.venv\\Scripts\\python.exe scripts\\check_dubstudio.py --init "Film címe"

A `--init` egy `projects/<slug>.json` projektet ír a dubstudio `projects/`
mappájába, a gépen talált modellútvonalakkal. A fájl utána kézzel is
szerkeszthető (a Web UI Beállítások fülén is).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

AURIS_ROOT = Path(__file__).resolve().parents[1]

#: a dubstudio Python-függőségei: (pip-név, import-név, mihez kell)
PACKAGES = [
    ("numpy", "numpy", "tömbkezelés az egész láncban"),
    ("scipy", "scipy", "újramintavételezés, jelmérés"),
    ("soundfile", "soundfile", "WAV írás/olvasás"),
    ("onnxruntime", "onnxruntime", "ONNX futtatás (Supertonic, diarizáció)"),
    ("sherpa-onnx", "sherpa_onnx", "pyannote szegmentáló + Supertonic TTS"),
    ("torch", "torch", "nem-osztályozó és embedding-segédek"),
    ("transformers", "transformers", "nem-osztályozó modell betöltése"),
    ("audio-separator", "audio_separator", "vokál / M&E szétválasztás (intake)"),
]

BINARIES = [
    ("ffmpeg", "hangsáv kinyerés, mux, atempo"),
    ("ffprobe", "média-vizsgálat"),
    ("audio-separator", "UVR/MDX leválasztás (az intake hívja)"),
    ("espeak-ng", "Piper-hangok IPA fonémázása"),
]

#: a projekt `models` szakaszának kulcsai és a hozzájuk tartozó fájlnév
MODEL_KEYS = {
    "embedding": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "diar_embedding": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "diar_segmentation": "diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "supertonic": "supertonic-sherpa-int8/",
    "piper": "piper/",
}

GATE_DEFAULTS = {
    "voice_switch_max": 0,
    "voice_switch_ratio_max": 0.0,
    "one_voice_per_utt_max": 0,
    "gender_conflict_max": 0,
    "coverage_min": 1.0,
    "scene_conflict_max": 0,
    "main_shared_max": 0,
    "fit_over_ratio_max": 0.02,
    "fit_max_over_s": 0.6,
    "asr_cer_max": {"parakeet": 0.3, "whisper": 0.6, "default": 0.6},
    "voice_overlap_max": 2,
    "scene_minor_ratio_max": 0.01,
}

TUNING_DEFAULTS = {
    "gap_merge": 0.9,
    "gap_cont": 2.2,
    "cos_same": 0.24,
    "cos_diff": 0.12,
    "sim_same": 0.24,
    "link_cos": 0.25,
    "viterbi_pen": 0.3,
    "main_share_min": 0.04,
    "sim_gender": 0.17,
    "recall_cos": 0.5,
}


def shared_models(root: Path) -> list[tuple[str, Path]]:
    """A két modul közös modelljei a szokásos helyükön (fetch_models.py)."""
    models = root / "work" / "models"
    return [
        ("Supertonic 3 sherpa int8 (hangoskönyv + szinkron)",
         AURIS_ROOT / "reader" / "models" / "supertonic-sherpa-int8" / "tts.json"),
        ("pyannote 3.0 szegmentáló",
         models / "diar" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"),
        ("CAM++ beszélő-embedding",
         models / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"),
        ("Piper hu_HU hangok",
         models / "piper" / "hu_HU-anna-medium.onnx"),
        ("UVR/MDX leválasztó",
         models / "separate" / "UVR_MDXNET_Main.onnx"),
    ]


def dub_root() -> Path:
    """A dubstudio gyökere: `DUBSTUDIO_ROOT`, különben a repón belüli `dubstudio/`.

    Ugyanaz a feloldás, amit a `reader/core/dub_api.py` használ, hogy a
    szkript és a felület ugyanazt a könyvtárat lássa.
    """
    return Path(os.environ.get("DUBSTUDIO_ROOT")
                or (AURIS_ROOT / "dubstudio")).resolve()


def slugify(name: str) -> str:
    """Projekt-azonosító: ékezet nélküli, kisbetűs, kötőjeles (max 48)."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:48] or "film"


def _row(ok: bool, label: str, detail: str = "") -> dict:
    return {"ok": bool(ok), "label": label, "detail": detail}


def _package(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _binary(name: str) -> str:
    """A program helye a PATH-on, vagy az Auris venv Scripts/bin mappájában.

    Az `audio-separator` a venv-be települ (az intake is ott keresi), ezért a
    puszta PATH-vizsgálat téves hiányt jelezne.
    """
    found = shutil.which(name)
    if found:
        return found
    bin_dir = Path(sys.executable).parent
    for candidate in (bin_dir / name, bin_dir / f"{name}.exe"):
        if candidate.is_file():
            return str(candidate)
    return ""


def _project_models(doc: dict, base: Path) -> tuple[Path, dict]:
    """A projekt gyökere és a `models` szakasz feloldott útvonalai."""
    root = (base / str(doc.get("root") or ".")).resolve()
    models = {}
    for key, value in (doc.get("models") or {}).items():
        path = Path(str(value)).expanduser()
        models[key] = path if path.is_absolute() else (root / path).resolve()
    return root, models


def check(root: Path) -> dict:
    """A teljes ellenőrzés: dubstudio, függőségek, binárisok, projektek."""
    rows: list[dict] = []
    rows.append(_row(AURIS_ROOT.is_dir(), "Auris repo", str(AURIS_ROOT)))
    rows.append(_row(root.is_dir(), "dubstudio gyökér",
                     f"{root} (DUBSTUDIO_ROOT / a repón belüli dubstudio)"))
    pkg_dir = root / "dubstudio"
    rows.append(_row((pkg_dir / "cli.py").is_file(),
                     "dubstudio csomag", str(pkg_dir)))
    for pip_name, import_name, why in PACKAGES:
        rows.append(_row(_package(import_name), f"Python: {pip_name}", why))
    for name, why in BINARIES:
        found = _binary(name)
        rows.append(_row(bool(found), f"Program: {name}",
                         found or f"nincs a PATH-on — {why}"))
    for label, path in shared_models(root):
        rows.append(_row(path.exists(), f"Modell: {label}",
                         "megvan" if path.exists() else
                         f"hiányzik: {path} — futtasd: scripts/fetch_models.py"))

    projects: list[dict] = []
    folder = root / "projects"
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    rows.append(_row(bool(files), "Projekt (projects/*.json)",
                     f"{len(files)} projekt a {folder} mappában"))
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            projects.append({"file": path.name, "error": str(exc)})
            continue
        proj_root, models = _project_models(doc, folder)
        missing = [str(p) for p in models.values() if not Path(p).exists()]
        projects.append({
            "file": path.name, "name": doc.get("name"), "kind": doc.get("kind"),
            "root": str(proj_root),
            "data_dir_exists": (proj_root / str(doc.get("data_dir") or "data")).is_dir(),
            "models_missing": missing,
        })
        rows.append(_row(not missing, f"Modellek: {path.stem}",
                         ", ".join(missing) if missing else "minden fájl a helyén"))
    return {"root": str(root), "checks": rows, "projects": projects}


def _model_defaults(models_dir: Path, auris_models: Path) -> dict:
    """A `models` szakasz alapértékei a gépen megtalálható helyekről."""
    espeak = shutil.which("espeak-ng") or ""
    return {
        "embedding": str(models_dir / MODEL_KEYS["embedding"]),
        "diar_embedding": str(models_dir / MODEL_KEYS["diar_embedding"]),
        "diar_segmentation": str(models_dir / MODEL_KEYS["diar_segmentation"]),
        "supertonic": str(auris_models / "supertonic-sherpa-int8"),
        "piper": str(models_dir / "piper"),
        "espeak": espeak,
    }


def project_doc(name: str, kind: str, season: int, models_dir: Path,
                auris_models: Path, source_lang: str) -> dict:
    """Egy új projekt JSON-ja (a dubstudio `_movie_project_doc` sémája szerint)."""
    slug = slugify(name)
    root = f"../movies/{slug}" if kind == "movie" else f"../series/{slug}"
    return {
        "name": name,
        "season": season,
        "kind": kind,
        "root": root,
        "cache_dir": f"../work/{slug}",
        "data_dir": "data",
        "subs_dir": "subs",
        "vocals_globs": ["{tag}/vocals.wav"],
        "me_globs": ["{tag}/me.wav"],
        "source_lang": source_lang,
        "original_audio": 0,
        "tts_sr": 44100,
        "models": _model_defaults(models_dir, auris_models),
        "tuning": dict(TUNING_DEFAULTS),
        "gate": dict(GATE_DEFAULTS),
        "llm": {"enabled": False, "base": "http://127.0.0.1:11434",
                "model": "qwen3.5:9b", "batch": 10, "timeout": 180,
                "required": False},
        "asr": {"mode": "parakeet", "batch": 1, "confirm_max": 30},
        "intake": {
            "translate": {"enabled": True, "base": "http://127.0.0.1:11434/v1",
                          "model": "ministral-3:8b", "api_key": "",
                          "api_key_env": "", "batch": 20, "timeout": 300,
                          "max_tokens": 4096, "source_lang": source_lang},
            "separate": {"enabled": True, "model": "UVR_MDXNET_Main.onnx",
                         "model_dir": str(models_dir / "separate"),
                         "audio_stream": 0, "second_pass": False,
                         "sample_rate": 44100, "timeout": 3600},
        },
        "diarization": {"enabled": True, "num_speakers": 0, "threshold": 0.5,
                        "min_turn": 0.3, "merge_gap": 0.6, "island_max": 0.8,
                        "min_duration_on": 0.15, "min_duration_off": 0.25,
                        "attach_gap": 2.0, "num_threads": 4},
        "series": {"auto_process": False, "poll_s": 30},
        "media_server": {"path": "", "deliver": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A szinkronstúdió (dubstudio) környezetének ellenőrzése.")
    parser.add_argument("--json", action="store_true",
                        help="gépi (JSON) kimenet")
    parser.add_argument("--init", metavar="NÉV",
                        help="új projekt létrehozása ezzel a névvel")
    parser.add_argument("--kind", choices=("movie", "series"), default="movie",
                        help="film (E01) vagy sorozat (SxxExx) — alap: movie")
    parser.add_argument("--season", type=int, default=1,
                        help="sorozat évadja (alap: 1)")
    parser.add_argument("--source-lang", default="eng",
                        choices=("eng", "heb"),
                        help="a forrás nyelve (alap: eng)")
    parser.add_argument("--models-dir", default="",
                        help="a diarizációs/leválasztó modellek mappája "
                             "(alap: <dubstudio>/work/models)")
    parser.add_argument("--auris-models", default="",
                        help="az Auris modellek mappája "
                             "(alap: <Auris>/reader/models)")
    args = parser.parse_args(argv)

    root = dub_root()
    models_dir = Path(args.models_dir).expanduser() if args.models_dir \
        else root / "work" / "models"
    auris_models = Path(args.auris_models).expanduser() if args.auris_models \
        else AURIS_ROOT / "reader" / "models"

    created = ""
    if args.init:
        name = args.init.strip()
        if not name:
            parser.error("a --init egy projektnevet vár")
        (root / "projects").mkdir(parents=True, exist_ok=True)
        dest = root / "projects" / f"{slugify(name)}.json"
        if dest.exists():
            print(f"Már létezik ilyen projekt: {dest}", file=sys.stderr)
            return 2
        doc = project_doc(name, args.kind, args.season, models_dir,
                          auris_models, args.source_lang)
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        created = str(dest)

    report = check(root)
    report["created"] = created
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(f"Auris:     {AURIS_ROOT}")
        print(f"dubstudio: {report['root']}\n")
        for row in report["checks"]:
            mark = "OK  " if row["ok"] else "HIÁNYZIK"
            print(f"[{mark}] {row['label']}"
                  + (f" — {row['detail']}" if row["detail"] else ""))
        if created:
            print(f"\nLétrehozva: {created}")
            print("A médiafájlt tedd a projekt data/ mappájába, majd nyisd meg a"
                  " /szinkron oldalt.")
        missing = [r for r in report["checks"] if not r["ok"]]
        if missing:
            print(f"\n{len(missing)} tétel hiányzik — lásd docs/SZINKRON-TELEPITES.md")
    return 1 if [r for r in report["checks"] if not r["ok"]] else 0


if __name__ == "__main__":
    raise SystemExit(main())
