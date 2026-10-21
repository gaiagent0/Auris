"""XTTSv2-Streaming-ONNX klónmodell letöltése a repóba.

A `pltobing/XTTSv2-Streaming-ONNX` Hugging Face-repó **gated**: a letöltés
előtt a böngészőben el kell fogadni a CC BY-NC 4.0 licencet, és egy
HF_TOKEN szükséges. A modell nem része az Aurisnak (2,2 GB, nem kereskedelmi
licenc), ezért kézzel, egyszer kell letölteni.

Használat (a repó gyökeréből):

    python scripts/download_xtts.py            # letöltés vendor/ alá
    python scripts/download_xtts.py --check    # csak ellenőrzés, nincs hálózat
    python scripts/download_xtts.py --dest D:\\modell\\XTTSv2-Streaming-ONNX

Kimenet: `<dest>/xtts_onnx/` a motor által kért fájlokkal, mellette a
pipeline-modulok — pontosan ezt a szerkezetet várja az Auris.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "pltobing/XTTSv2-Streaming-ONNX"
REPO_URL = "https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX"
LICENSE = "CC BY-NC 4.0"

# A motor `XTTS_FILES` listája (core/xtts_onnx_engine.py).
REQUIRED_IN_XTTS_ONNX = (
    "gpt_model_int8.onnx",
    "conditioning_encoder.onnx",
    "hifigan_vocoder.onnx",
    "speaker_encoder.onnx",
    "vocab.json",
    "mel_stats.npy",
)

# A pipeline moduljai a repó gyökerén (az engine a szülőkönyvtárat teszi a sys.path-ra).
REQUIRED_AT_ROOT = (
    "xtts_streaming_pipeline.py",
    "xtts_tokenizer.py",
    "xtts_onnx_orchestrator.py",
    "zh_num2words.py",
)

# A bemutató hangminták nem kellenek a futáshoz.
EXCLUDE = ("audio_ref/*", "audio_synth/*")


def default_dest() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "XTTSv2-Streaming-ONNX"


def check(dest: Path) -> list[str]:
    """Hiányzó kötelező fájlok; üres lista = minden megvan."""
    missing = [name for name in REQUIRED_IN_XTTS_ONNX
               if not (dest / "xtts_onnx" / name).is_file()]
    missing += [name for name in REQUIRED_AT_ROOT if not (dest / name).is_file()]
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Letölti a gated XTTSv2-Streaming-ONNX klónozó modellt "
                    f"({REPO_ID}, {LICENSE}).",
    )
    parser.add_argument("--dest", type=Path, default=default_dest(),
                        help="Célkönyvtár (alapértelmezés: <repó>/vendor/"
                             "XTTSv2-Streaming-ONNX)")
    parser.add_argument("--token", default=None,
                        help="HF token; nélküle a ~/.cache/huggingface/token "
                             "vagy a HF_TOKEN környezeti változó kell")
    parser.add_argument("--check", action="store_true",
                        help="Csak ellenőrzi, hogy a modell megvan-e (nincs hálózat)")
    args = parser.parse_args(argv)

    dest: Path = args.dest.expanduser()
    missing = check(dest)
    if args.check:
        if missing:
            print(f"NINCS KÉSZ: {dest}")
            print("Hiányzó fájlok: " + ", ".join(missing))
            print(f"Futtasd: python scripts/download_xtts.py")
            return 1
        print(f"OK: a modell megvan ({dest})")
        return 0
    if not missing:
        print(f"Már megvan a modell ({dest}) — semmit nem töltök le.")
        print("Kényszerített újratöltéshez töröld a célmappát, vagy használd a "
              "--check kiírását hibakereséshez.")
        return 0

    print(f"XTTSv2-ONNX letöltése: {REPO_ID}")
    print(f"  cél:   {dest}")
    print(f"  licenc: {LICENSE} (gated repó)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("HIBA: nincs telepítve a huggingface_hub — futtasd a telepítés "
              "után: reader\\.venv\\Scripts\\python.exe scripts\\download_xtts.py",
              file=sys.stderr)
        return 2

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(dest),
            token=args.token,
            ignore_patterns=list(EXCLUDE),
        )
    except Exception as exc:  # noqa: BLE001 — a gated hiba magyarázata a lényeg
        name = type(exc).__name__
        if "Gated" in name or "403" in str(exc) or "401" in str(exc):
            print(
                "A repó GATED (zárt): a letöltéshez egy Hugging Face-fiók kell.\n"
                f"  1. Nyisd meg: {REPO_URL}\n"
                "  2. Jelentkezz be, és fogadd el a licenc-űrlapot "
                "(CC BY-NC 4.0 — a jóváhagyás azonnali).\n"
                "  3. Kérj egy tokent: https://huggingface.co/settings/tokens\n"
                "     (read jogú, `hf_` kezdetű), majd futtasd:\n"
                "       set HF_TOKEN=hf_...  (Linux/macOS: export HF_TOKEN=hf_...)\n"
                "       python scripts/download_xtts.py\n"
                f"Eredeti hiba: {exc}",
                file=sys.stderr,
            )
            return 3
        print(f"HIBA a letöltés közben: {exc}", file=sys.stderr)
        return 4

    missing = check(dest)
    if missing:
        print("HIBA: a letöltés után hiányos a modell: " + ", ".join(missing),
              file=sys.stderr)
        return 5
    print(f"KÉSZ: a klónozó motor megvan ({dest}).")
    print("Indítsd újra az Aurist, majd a Hangstúdióban tölts fel referenciát.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
