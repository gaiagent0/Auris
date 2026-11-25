"""Médiaszerver-modul: a forrás helye és a kész szinkron visszamásolása.

A projektbe a médiafájl a szerverről **másolva** kerül (a laptop helyi
munkapéldánya), a kész `` - HUN DUB.mkv`` pedig ugyanoda írható vissza,
lecserélve az eredetit. A hozzárendelést a ``work/media-deliver.json``
tartja (tag -> eredeti fájl), így a szinkronfutás után a CLI is tudja.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def path_for(project) -> Path:
    return project.work / "media-deliver.json"


def load(project) -> dict:
    p = path_for(project)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(project, data: dict) -> Path:
    p = path_for(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def record(project, tag: str, source: str | Path) -> dict:
    """Egy epizód forrásának rögzítése (honnan jött a média)."""
    data = load(project)
    data[str(tag).upper()] = {
        "source": str(Path(source)),
        "added_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    save(project, data)
    return data[str(tag).upper()]


def mapping(project, tag: str) -> dict | None:
    return (load(project) or {}).get(str(tag).upper())


def copy_back(project, tag: str, output: str | Path) -> dict:
    """A kész szinkron visszamásolása az eredeti médiafájl helyére.

    Az eredeti fájlról egyszeri ``.bak`` biztonsági másolat készül, az új fájl
    pedig először ``.dubpart`` néven íródik, és csak sikeres másolás után
    cseréli le az eredetit (félkész fájl ne maradjon a szerveren).
    """
    row = mapping(project, tag)
    if not row or not row.get("source"):
        raise FileNotFoundError(f"ehhez a taghez nincs médiaszerver-forrás: {tag}")
    src = Path(row["source"])
    out = Path(output)
    if not out.is_file():
        raise FileNotFoundError(f"nincs kész szinkronfájl: {out}")
    if not src.parent.is_dir():
        raise FileNotFoundError(f"a médiaszerver mappa nem érhető el: {src.parent}")
    backup = src.with_name(src.name + ".bak")
    if src.is_file() and not backup.exists():
        shutil.copy2(src, backup)
    tmp = src.with_name(src.name + ".dubpart")
    try:
        shutil.copy2(out, tmp)
        os.replace(tmp, src)
    finally:
        tmp.unlink(missing_ok=True)
    return {"target": str(src), "backup": str(backup), "output": str(out)}