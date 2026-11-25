"""Sorozat- és filmkönyvtár a WebUI-hoz.

A modul **nem** futtat pipeline-t és nem tölt modellt. Csak a projekt
médiakönyvtárát, a work-epizódokat és a kész kimeneteket nézi, hogy a
felület meg tudja mondani:

* mi van már betöltve,
* melyik epizód előkészített / kész,
* mi hiányzik a szinkronindításhoz (magyar felirat, vokál, M&E),
* sorozatnál mi a következő epizódszám.

A film (`kind="movie"`) egyetlen E01 egység; a sorozat (`kind="series"`)
epizódjai a data_dir SxxExx fájljaiból és a work/E## könyvtárakból állnak.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Project

MEDIA_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".m4v"}
_TAG_RE = re.compile(r"^E\d{2,3}$")
_SOURCE_RE = (
    re.compile(r"(?i)[Ss](\d{1,2})[Ee][\. _-]?[Ee]?(\d{1,3})"),
    re.compile(r"(?i)[Ee](\d{1,3})(?!\d)"),
    re.compile(r"(?i)epiz[óo]d[ \-.:_]*(\d{1,3})"),
)


def is_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES


def tag_from_name(name: str) -> str | None:
    """Epizódszám a fájlnévből (S05E12 / E12 / epizód 12)."""
    for pattern in _SOURCE_RE:
        match = pattern.search(name or "")
        if match:
            num = int(match.group(match.lastindex))
            return f"E{num:02d}" if 0 < num < 1000 else None
    return None


def _work_tag(path: Path) -> str | None:
    return path.name if path.is_dir() and _TAG_RE.match(path.name) else None


def subtitle_ready(project: Project, tag: str) -> bool:
    return project.subtitle(tag, "hun").is_file()


def vocals_ready(project: Project, tag: str) -> bool:
    try:
        return project.vocals(tag).is_file()
    except (FileNotFoundError, OSError):
        return False


def me_ready(project: Project, tag: str) -> bool:
    try:
        return project.me_bed(tag).is_file()
    except (FileNotFoundError, OSError):
        return False


def media_for(project: Project, tag: str) -> Path | None:
    try:
        path = project.episode_media(tag)
    except (FileNotFoundError, OSError):
        return None
    return path if path.is_file() else None


def output_for(project: Project, tag: str) -> Path | None:
    """A kész MKV a work/out alatt (a tagot szóhatárként keresi)."""
    if not project.out.is_dir():
        return None
    ep = int("".join(c for c in tag if c.isdigit()) or 0)
    if project.kind == "movie":
        hits = sorted(project.out.glob("*.mkv"))
    else:
        pats = (
            f"*S{project.season:02d}E{ep:02d}*.mkv",
            f"*E{ep:02d}*.mkv",
        )
        hits = []
        for pattern in pats:
            hits = sorted(project.out.glob(pattern))
            if hits:
                break
    return hits[-1] if hits else None


def report_passed(project: Project, tag: str) -> bool:
    path = project.work / tag / "report.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("passed"))


def missing_assets(project: Project, tag: str, media: Path | None = None) -> list[str]:
    """Emberi nyelvű hiánylista a szinkronindításhoz."""
    media = media if media is not None else media_for(project, tag)
    missing: list[str] = []
    if media is None:
        missing.append("forrásmédia")
    if not subtitle_ready(project, tag):
        missing.append("magyar felirat")
    if not vocals_ready(project, tag):
        missing.append("leválasztott énekbeszéd")
    if not me_ready(project, tag):
        missing.append("zene+effekt (M&E)")
    return missing


def episode_status(project: Project, tag: str) -> dict:
    media = media_for(project, tag)
    prepared = (project.work / tag / "utterances.json").is_file()
    dub = (project.work / tag / "dub.wav").is_file()
    output = output_for(project, tag)
    missing = missing_assets(project, tag, media)
    try:
        from .intake import status as intake_status

        intake = intake_status(project, tag)
    except Exception:                       # noqa: BLE001 — a könyvtár ne omoljon el
        intake = {"state": "unknown", "error": "", "retry_allowed": True}
    # forrásfelirat nélkül nem tud fordítás indulni: ezt külön jelezzük, hogy
    # a felhasználó tudja, mit kell a mappába tenni
    if not subtitle_ready(project, tag) and not intake.get("source"):
        missing.append("fordítható forrásfelirat")
    if output is not None and report_passed(project, tag):
        status = "dubbed"
    elif prepared:
        status = "prepared"
    elif not missing:
        status = "ready"
    else:
        status = "waiting"
    return {
        "tag": tag,
        "media": str(media) if media else None,
        "media_name": media.name if media else None,
        "prepared": prepared,
        "dub": dub,
        "output": str(output) if output else None,
        "report_passed": report_passed(project, tag),
        "ready": not missing,
        "missing": missing,
        "intake": intake,
        "status": status,
    }


def _media_tags(project: Project) -> dict[str, Path]:
    """A data_dir epizódfájljai; filmnél az első média E01."""
    try:
        files = sorted(p for p in project.data_dir.iterdir() if is_media(p))
    except OSError:
        return {}
    if project.kind == "movie":
        return {"E01": files[0]} if files else {}
    out: dict[str, Path] = {}
    for path in files:
        tag = tag_from_name(path.name)
        if tag and tag not in out:
            out[tag] = path
    return out


def library(project: Project) -> dict:
    """A WebUI könyvtárnézete (média + work + kimenet)."""
    media_tags = _media_tags(project)
    work_tags = set()
    try:
        for path in project.work.iterdir():
            tag = _work_tag(path)
            if tag:
                work_tags.add(tag)
    except OSError:
        pass
    tags = sorted(set(media_tags) | work_tags)
    if project.kind == "movie":
        tags = ["E01"]
    elif not tags:
        tags = []
    episodes = []
    for tag in tags:
        row = episode_status(project, tag)
        # A work-only epizódnál is legyen stabil hely a listában.
        row["media"] = row["media"] or (
            str(media_tags[tag]) if tag in media_tags else None)
        episodes.append(row)
    last = max((int(t[1:]) for t in tags if _TAG_RE.match(t)), default=0)
    next_tag = f"E{last + 1:02d}"
    counts = {
        "episodes": len(episodes),
        "ready": sum(1 for e in episodes if e["ready"]),
        "prepared": sum(1 for e in episodes if e["prepared"]),
        "dubbed": sum(1 for e in episodes if e["status"] == "dubbed"),
    }
    return {
        "name": project.name,
        "kind": project.kind,
        "is_movie": project.kind == "movie",
        "season": project.season,
        "next_tag": next_tag,
        "episodes": episodes,
        "counts": counts,
        "watch": {
            "enabled": bool((project.series or {}).get("auto_process")),
            "poll_s": int((project.series or {}).get("poll_s", 30) or 30),
            "auto_intake": bool((project.series or {}).get("auto_intake", True)),
        },
    }


def scan_sources(project: Project, folder: str | Path | None = None) -> list[dict]:
    """Új médiafájlok a megadott mappában (nem másol, nem módosít)."""
    base = Path(folder).expanduser() if folder else project.data_dir
    if base.is_file():
        paths = [base]
    elif base.is_dir():
        paths = sorted(p for p in base.iterdir() if is_media(p))
    else:
        return []
    known = set(_media_tags(project))
    out = []
    for path in paths:
        tag = "E01" if project.kind == "movie" else tag_from_name(path.name)
        out.append({
            "tag": tag,
            "path": str(path),
            "name": path.name,
            "known": bool(tag and tag in known),
        })
    return out
