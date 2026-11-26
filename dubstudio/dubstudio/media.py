"""ffprobe/ffmpeg segédek és az MKV muxolás."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .config import Project

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # pragma: no cover
        pass

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: az eredeti hangsáv automatikus címkéje a nyelvkódból (ha a forrásban nincs
#: saját cím); a lejátszó így megkülönböztethető nevet mutat a magyar sáv mellett
_ORIGINAL_TITLES = {
    "eng": "Eredeti angol",
    "heb": "Eredeti héber",
    "hun": "Eredeti magyar",
}


def probe_streams(media: str | Path) -> list[dict]:
    p = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name:stream_tags=language,title",
         "-of", "json", str(media)], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe hiba: {p.stderr.decode('utf-8', 'replace')[-400:]}")
    return json.loads(p.stdout.decode("utf-8", "replace")).get("streams", [])


def audio_streams(media: str | Path) -> list[dict]:
    return [s for s in probe_streams(media) if s.get("codec_type") == "audio"]


def subtitle_streams(media: str | Path) -> list[dict]:
    return [s for s in probe_streams(media) if s.get("codec_type") == "subtitle"]


def extract_audio(media: str | Path, a_index: int, out_wav: str | Path,
                  sr: int = 48000) -> Path:
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-v", "error", "-y", "-i", str(media), "-map", f"0:a:{a_index}",
           "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", str(out)]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg hiba: {p.stderr.decode('utf-8', 'replace')[-400:]}")
    return out


def mux(project: Project, tag: str, dub_wav: Path, out_name: str | None = None,
        skip: bool = False) -> Path | None:
    """Videó + magyar szinkron + eredeti hangok + feliratok egy MKV-be."""
    if skip:
        return None
    src = project.episode_media(tag)
    ep = int("".join(c for c in tag if c.isdigit()) or 0)
    project.out.mkdir(parents=True, exist_ok=True)
    out = project.out / (out_name or f"{src.stem} - HUN DUB.mkv")
    if out.exists():
        out.unlink()

    auds = audio_streams(src)
    subs = subtitle_streams(src)
    if not auds:
        raise RuntimeError(f"nincs hangsáv a forrásban: {src}")

    # a forrás hangjai: az eredetit (source_lang) megtartjuk, a többit is
    picked_subs = []
    hun = [s for s in subs if (s.get("tags") or {}).get("language") == "hun"]
    eng = [s for s in subs if (s.get("tags") or {}).get("language") == "eng"
           and "dubtitle" not in ((s.get("tags") or {}).get("title") or "").lower()]
    if hun:
        picked_subs.append(hun[0])
    if eng:
        picked_subs.append(eng[0])

    # 0: a mi magyar szinkronunk; utána az eredeti hangsávok
    orig = [s["index"] for s in auds]
    ours_idx = None
    for s in auds:
        lang = (s.get("tags") or {}).get("language")
        if lang == project.source_lang and ours_idx is None:
            ours_idx = s["index"]
    ordered = ([ours_idx] if ours_idx is not None else []) + \
              [i for i in orig if i != ours_idx]

    cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-i", str(dub_wav),
           "-map", "0:v:0", "-map", "1:a:0"]
    for i in ordered:
        cmd += ["-map", f"0:{i}"]
    for s in picked_subs:
        cmd += ["-map", f"0:{s['index']}"]

    cmd += ["-c:v", "copy", "-c:a:0", "aac", "-b:a:0", "192k", "-ac:a:0", "2"]
    for k in range(1, len(ordered) + 1):
        cmd += [f"-c:a:{k}", "copy"]
    if picked_subs:
        cmd += ["-c:s", "srt"]
    cmd += ["-metadata:s:a:0", "language=hun", "-metadata:s:a:0", "title=Magyar szinkron"]
    for k, i in enumerate(ordered, start=1):
        st = next((s for s in auds if s["index"] == i), {})
        tags = st.get("tags") or {}
        lang = tags.get("language") or "und"
        if lang in ("", "und") and getattr(project, "source_lang", None):
            lang = project.source_lang
        cmd += [f"-metadata:s:a:{k}", f"language={lang}"]
        title = tags.get("title") or _ORIGINAL_TITLES.get(lang)
        if not title and lang != "und":
            title = "Eredeti hang"
        if title:
            cmd += [f"-metadata:s:a:{k}", f"title={title}"]
    for k, s in enumerate(picked_subs):
        lang = (s.get("tags") or {}).get("language", "und")
        cmd += [f"-metadata:s:s:{k}", f"language={lang}"]
    cmd += ["-disposition:a:0", "default"]
    for k in range(1, len(ordered) + 1):
        cmd += [f"-disposition:a:{k}", "0"]
    cmd += [str(out)]

    p = subprocess.run([str(c) for c in cmd], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg mux hiba: {p.stderr.decode('utf-8', 'replace')[-800:]}")
    print(f"MKV kész: {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def sample_clip(project: Project, tag: str, start: float, dur: float,
                dub_wav: Path, out_path: Path, a_index: int = 0) -> Path:
    """Hangminta kivágása egy elkészült mkv-ból (A/B hallgatáshoz)."""
    src = project.episode_media(tag)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(src), "-i", str(dub_wav), "-map", "1:a:0",
           "-c:a", "libmp3lame", "-b:a", "192k", "-ac", "2", str(out)]
    p = subprocess.run([str(c) for c in cmd], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-400:])
    return out


def safe_name(tag: str, project: Project) -> str:
    """A kimeneti fájl neve (a forrás nevéből, ütközés nélkül)."""
    src = project.episode_media(tag)
    m = re.search(r"S\d+E\d+", src.stem, re.I)
    return f"{src.stem[:m.end()] if m else src.stem} - HUN DUB.mkv"
