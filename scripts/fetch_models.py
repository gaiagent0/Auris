#!/usr/bin/env python3
"""A hangoskönyv és a szinkron közös AI-modelljeinek letöltése.

Az Auris két modulja ugyanazt a beszéd- és beszélőláncot használja. Ez a
szkript teszi egy friss klónon használhatóvá mindkettőt: a Supertonic 3
sherpa-onnx int8 motort (hangoskönyv **és** szinkronhangok), valamint a
szinkronlánc saját modelljeit (pyannote szegmentáló, CAM++ embedding,
Piper hu_HU hangok, UVR/MDX leválasztó).

A modellek nem részei a repónak; a szinkronmodelljei a dubstudio
`work/models/` mappájába kerülnek, amit a projekt `models` szakasza hivatkozik.
A szkript idempotens: ami megvan, azt kihagyja, és `--check`-kel hálózat
nélkül is ellenőrizhető.

Használat (a repo gyökeréből, az Auris venv-jével):

    reader\\.venv\\Scripts\\python.exe scripts\\fetch_models.py
    reader\\.venv\\Scripts\\python.exe scripts\\fetch_models.py --check
    reader\\.venv\\Scripts\\python.exe scripts\\fetch_models.py --only sync
    reader\\.venv\\Scripts\\python.exe scripts\\fetch_models.py --models-dir D:\\AI\\dubstudio\\work\\models

Környezet:
    DUBSTUDIO_ROOT   a dubstudio gyökere (alapból az Auris melletti `dubstudio`)
    HF_TOKEN         csak a gated Hugging Face-letöltésekhez kell (ezekhez nem)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

AURIS_ROOT = Path(__file__).resolve().parents[1]

UA = "Auris-model-fetch/1.0 (+https://github.com/gaiagent0/Auris)"

SHERPA_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
PIPER_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
             "hu/hu_HU/{voice}/medium/hu_HU-{voice}-medium{ext}?download=true")
UVR_RELEASE = ("https://github.com/TRvlvr/model_repo/releases/download/"
               "all_public_uvr_models")
SUPERTONIC_REPO = "csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"


@dataclass(frozen=True)
class Model:
    """Egy letöltendő modell vagy modellcsomag."""

    key: str
    label: str
    scope: str                    # "shared" = mindkét modul, "sync" = szinkron
    check: str                    # e fájl létezése jelenti a "kész" állapotot
    files: tuple[tuple[str, str], ...] = ()   # (célútvonal, URL)
    archive: str = ""             # tar.bz2 csomag URL-je
    hub_repo: str = ""            # Hugging Face repó (teljes mappa)
    note: str = ""


MODELS: tuple[Model, ...] = (
    Model(
        key="supertonic",
        label="Supertonic 3 (sherpa-onnx int8, 10 hang)",
        scope="shared",
        check="supertonic-sherpa-int8/tts.json",
        hub_repo=SUPERTONIC_REPO,
        note="a hangoskönyv és a szinkronhangok közös motorja (~139 MB)",
    ),
    Model(
        key="diar_segmentation",
        label="pyannote 3.0 szegmentáló (sherpa-onnx)",
        scope="sync",
        check="diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        archive=f"{SHERPA_RELEASE}/speaker-segmentation-models/"
                "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        note="beszélőfordulók a teljes énekbeszéd-sávon (~6 MB)",
    ),
    Model(
        key="embedding",
        label="3D-Speaker ERes2Net beszélő-embedding (CAM++)",
        scope="sync",
        check="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        files=((
            "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            f"{SHERPA_RELEASE}/speaker-recongition-models/"
            "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        ),),
        note="szereplő-azonosság mérése (~28 MB)",
    ),
    Model(
        key="piper",
        label="Piper hu_HU hangok (anna, berta, imre)",
        scope="sync",
        check="piper/hu_HU-anna-medium.onnx",
        files=tuple(
            (f"piper/hu_HU-{voice}-medium{ext}", PIPER_URL.format(voice=voice, ext=ext))
            for voice in ("anna", "berta", "imre")
            for ext in (".onnx", ".onnx.json")
        ),
        note="3 magyar hang, espeak-ng fonémázással (~63 MB/hang)",
    ),
    Model(
        key="separator",
        label="UVR MDX-NET Main (vokál / M&E leválasztás)",
        scope="sync",
        check="separate/UVR_MDXNET_Main.onnx",
        files=((
            "separate/UVR_MDXNET_Main.onnx",
            f"{UVR_RELEASE}/UVR_MDXNET_Main.onnx",
        ),),
        note="az intake hangsáv-szétválasztása (~50 MB, MIT)",
    ),
)


def dub_root() -> Path:
    """A dubstudio gyökere — ugyanaz a feloldás, mint a felületen."""
    return Path(os.environ.get("DUBSTUDIO_ROOT")
                or (AURIS_ROOT.parent / "dubstudio")).resolve()


def _dest(model: Model, models_dir: Path) -> Path:
    """A modell célmappája: a szinkroné a dubstudio alatt, a közösé az Aurisban."""
    if model.scope == "sync":
        return models_dir
    return AURIS_ROOT / "reader" / "models"


def _target(model: Model, models_dir: Path) -> Path:
    return _dest(model, models_dir) / model.check


def is_present(model: Model, models_dir: Path) -> bool:
    return _target(model, models_dir).exists()


def _human(size: float) -> str:
    return f"{size / 1e6:.1f} MB"


def download_file(url: str, dest: Path, label: str) -> None:
    """Egy fájl letöltése `.part` ideiglenes névre, majd atomi átnevezéssel."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(tmp, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last_step = -1
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                pct = done * 100.0 / total if total else 0.0
                step = int(pct // 10) if total else done // (16 << 20)
                if step != last_step:
                    last_step = step
                    shown = f"{_human(done)} / {_human(total)}" if total \
                        else _human(done)
                    print(f"\r    {label}: {shown}", end="", flush=True)
        print()
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def download_archive(url: str, dest_dir: Path, label: str) -> None:
    """tar.bz2 csomag letöltése és a benne lévő könyvtár kiemelése."""
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auris-model-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "model.tar.bz2"
        download_file(url, archive, label)
        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:bz2") as tar:
            try:
                tar.extractall(unpacked, filter="data")
            except TypeError:      # Python < 3.12
                tar.extractall(unpacked)
        source = unpacked / dest_dir.name
        if not source.is_dir():
            found = [p for p in unpacked.iterdir() if p.is_dir()]
            if len(found) != 1:
                raise RuntimeError(f"a csomagban nincs {dest_dir.name} mappa")
            source = found[0]
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(source), str(dest_dir))


def download_hub(repo: str, dest_dir: Path, label: str) -> None:
    """Teljes Hugging Face repó letöltése a célmappába."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:      # pragma: no cover - a venvben megvan
        raise RuntimeError(
            "a Hugging Face-letöltéshez huggingface_hub kell: "
            "reader\\.venv\\Scripts\\python.exe -m pip install huggingface_hub"
        ) from exc
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"    {label}: Hugging Face ({repo})", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest_dir),
        token=os.environ.get("HF_TOKEN") or None,
    )


def fetch(model: Model, models_dir: Path) -> None:
    """Egy modell letöltése, a meglévő fájlok kihagyásával."""
    dest = _dest(model, models_dir)
    if model.hub_repo:
        if is_present(model, models_dir):
            print(f"  = {model.label}: már megvan")
            return
        print(f"  + {model.label}")
        download_hub(model.hub_repo, dest / Path(model.check).parent, model.label)
    elif model.archive:
        print(f"  + {model.label}")
        download_archive(model.archive, dest / Path(model.check).parent,
                         model.label)
    else:
        todo = [(rel, url) for rel, url in model.files
                if not (dest / rel).is_file()]
        if not todo:
            print(f"  = {model.label}: már megvan")
            return
        print(f"  + {model.label} ({len(todo)} fájl)")
        for rel, url in todo:
            download_file(url, dest / rel, Path(rel).name)
    if not is_present(model, models_dir):
        raise RuntimeError(f"a letöltés után sem található: {_target(model, models_dir)}")
    print(f"  OK {model.label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A hangoskönyv és a szinkron közös AI-modelljeinek letöltése.")
    parser.add_argument("--check", action="store_true",
                        help="csak ellenőrzés, letöltés nélkül")
    parser.add_argument("--only", choices=("shared", "sync"), default=None,
                        help="csak a közös (shared) vagy csak a szinkron modellek")
    parser.add_argument("--models-dir", default="",
                        help="a szinkronmodellek mappája "
                             "(alap: <dubstudio>/work/models)")
    args = parser.parse_args(argv)

    models_dir = (Path(args.models_dir).expanduser().resolve()
                  if args.models_dir else dub_root() / "work" / "models")
    selected = [m for m in MODELS if not args.only or m.scope == args.only]

    print(f"Auris:    {AURIS_ROOT}")
    print(f"Modellek: {models_dir}\n")

    if args.check:
        missing = []
        for model in selected:
            ok = is_present(model, models_dir)
            print(f"[{'OK  ' if ok else 'HIÁNYZIK'}] {model.label}"
                  + ("" if ok else f" — {_target(model, models_dir)}"))
            if not ok:
                missing.append(model)
        if missing:
            print(f"\n{len(missing)} modell hiányzik — futtasd a szkriptet "
                  "a --check kapcsoló nélkül a letöltéshez.")
        return 1 if missing else 0

    errors: list[tuple[Model, str]] = []
    for model in selected:
        try:
            fetch(model, models_dir)
        except Exception as exc:      # egy hiba ne állítsa meg a többit
            errors.append((model, str(exc)))
            print(f"  ! {model.label}: {exc}")

    print("\nKész." if not errors else f"\n{len(errors)} modell nem töltődött le:")
    for model, message in errors:
        print(f"  - {model.label}: {message}")
    if not errors:
        print("A szinkron projekthez a modellek útvonalát a projekt `models` "
              "szakasza adja meg; új projektet a\n"
              f"  {AURIS_ROOT / 'scripts' / 'check_dubstudio.py'} --init \"Név\"\n"
              "parancs hoz létre a helyes útvonalakkal.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
