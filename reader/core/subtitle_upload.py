"""Külső felirat feltöltése a szinkronprojektekhez.

A modul a WebUI-tól független: a feltöltött fájlt ellenőrzi, SRT-vé alakítja
(ha WebVTT), majd a projekt ``subs/<EP>_<nyelv>.srt`` helyére írja. A meglévő
feliratot csak kifejezett kérésre cseréli, és előtte időbélyeges biztonsági
másolatot készít. A magyar feliratot a dubstudio intake fordítás nélkül
használja; az angol/héber feltöltés a fordítás forrása lesz.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

#: a projekt által ismert feliratnyelvek (a magyar a kész szinkronszöveg)
ALLOWED_LANGS = ("hun", "eng", "heb")
#: bőséges felső korlát: egy teljes évados SRT is jóval ez alatt van
MAX_BYTES = 4 * 1024 * 1024
_TAG_RE = re.compile(r"^E\d{2,3}$")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")
_BLOCK_SPLIT = re.compile(r"\n\s*\n")


class SubtitleUploadError(ValueError):
    """A feltöltött felirat nem használható (üzenete a felhasználónak szól)."""


def _decode(raw: bytes) -> str:
    """UTF-8/UTF-16/Windows-1250 kódolású felirat szöveggé alakítása."""
    for encoding in ("utf-8-sig", "utf-16", "cp1250"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SubtitleUploadError("A felirat szövegkódolása nem ismerhető fel "
                              "(UTF-8 vagy UTF-16 szükséges).")


def _normalize_time(value: str) -> str:
    """VTT-időbélyeg (pont) → SRT-időbélyeg (vessző)."""
    value = value.strip().replace(".", ",")
    if not _TIME_RE.fullmatch(value):
        raise SubtitleUploadError(f"Érvénytelen időbélyeg: {value}")
    return value


def _vtt_to_srt(text: str) -> str:
    """Egyszerű WebVTT → SRT átalakítás a szinkronstúdióhoz.

    Csak a cue-blokkokat tartja meg: a ``WEBVTT`` fejlécet, a NOTE/STYLE/REGION
    blokkokat és a cue-beállításokat eldobja. A szöveget nem formázza át, mert
    a dubstudio a saját tisztítását használja.
    """
    blocks = _BLOCK_SPLIT.split(text.replace("\r\n", "\n").replace("\r", "\n"))
    out: list[str] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n")]
        if not lines:
            continue
        first = lines[0].strip().upper()
        if first.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_idx is None:
            continue
        try:
            left, right = lines[timing_idx].split("-->", 1)
        except ValueError:
            continue
        # a VTT a záró idő után cue-beállításokat is írhat (pl. align:start)
        right = right.strip().split()[0]
        body = [line for line in lines[timing_idx + 1:] if line.strip()]
        if not body:
            continue
        out.append(f"{len(out) + 1}\n{_normalize_time(left)} --> "
                   f"{_normalize_time(right)}\n" + "\n".join(body))
    return "\n\n".join(out) + ("\n" if out else "")


def validate_srt(text: str) -> int:
    """SRT-változat ellenőrzése; visszaadja a feliratsorok számát."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise SubtitleUploadError("A feltöltött felirat üres.")
    cues = 0
    for block in _BLOCK_SPLIT.split(text):
        lines = [line for line in block.split("\n") if line.strip()]
        if not any("-->" in line for line in lines):
            continue
        timing = next(line for line in lines if "-->" in line)
        left, right = timing.split("-->", 1)
        try:
            _normalize_time(left)
            _normalize_time(right.strip().split()[0])
        except SubtitleUploadError as exc:
            raise SubtitleUploadError(
                f"A felirat időzítése hibás (pl. hiányzó óra/perc): {exc}") from exc
        if not any(line.strip() for line in lines[lines.index(timing) + 1:]):
            raise SubtitleUploadError(f"A {cues + 1}. feliratsor szövege üres.")
        cues += 1
    if cues == 0:
        raise SubtitleUploadError("A fájl nem tartalmaz feliratsort (hiányzik a -->).")
    return cues


def read_upload(stream, filename: str) -> tuple[str, int]:
    """A feltöltött fájl beolvasása és normalizálása (SRT szöveg, cueszám)."""
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in (".srt", ".vtt"):
        raise SubtitleUploadError("Csak .srt vagy .vtt felirat tölthető fel.")
    try:
        raw = stream.read(MAX_BYTES + 1)
    except (AttributeError, OSError) as exc:
        raise SubtitleUploadError(f"A feltöltött fájl nem olvasható: {exc}") from exc
    if not isinstance(raw, (bytes, bytearray)):
        raw = str(raw or "").encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise SubtitleUploadError("A felirat legfeljebb 4 MB lehet.")
    text = _decode(bytes(raw))
    if suffix == ".vtt" or text.lstrip().upper().startswith("WEBVTT"):
        text = _vtt_to_srt(text)
    cues = validate_srt(text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n", cues


def save_subtitle(project, tag: str, lang: str, stream, filename: str, *,
                  replace: bool = False) -> dict:
    """Felirat mentése a projekt ``subs/`` mappájába.

    ``project`` egy ``dubstudio.config.Project``; a modul csak a projekt
    útvonalait használja, így a WebUI és bármely más hívó ugyanazt kapja.
    """
    tag = str(tag or "").strip().upper()
    if not _TAG_RE.match(tag):
        raise SubtitleUploadError("Érvénytelen epizód-azonosító (pl. E04).")
    lang = str(lang or "eng").strip().lower()
    if lang not in ALLOWED_LANGS:
        raise SubtitleUploadError("A felirat nyelve hun, eng vagy heb lehet.")
    text, cues = read_upload(stream, filename)
    target = project.subtitle(tag, lang)
    if target.exists() and not replace:
        raise FileExistsError(f"Már van {tag} {lang} felirat: {target.name}. "
                              "Cseréhez jelöld be a felülírást.")
    project.subs_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = target.with_name(f"{target.stem}.{stamp}.bak.srt")
        shutil.copy2(target, backup)
    tmp = target.with_suffix(".srt.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise SubtitleUploadError(f"A felirat nem írható: {exc}") from exc
    return {"path": str(target), "cues": cues, "lang": lang, "tag": tag,
            "bytes": target.stat().st_size, "replaced": backup is not None,
            "backup": str(backup) if backup else None}
