"""YouTube-forrás letöltése a szinkronprojekthez.

A modul a nyilvános YouTube-videót a legjobb elérhető (legfeljebb 1080p)
MP4 formátumban tölti le, mellé kéri az angol feliratot (kézi vagy automatikus),
majd a projekt ``data/`` és ``subs/`` mappájába helyezi. A tényleges fordítást
és szinkront nem itt, hanem a meglévő ``intake`` + ``Pipeline`` lánc végzi.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

MEDIA_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SUBTITLE_SUFFIXES = (".srt", ".vtt")
_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be", "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


class YouTubeError(RuntimeError):
    """A YouTube-forrás nem tölthető le vagy nincs fordítható felirata."""


def is_youtube_url(url: str) -> bool:
    """Csak nyilvános YouTube URL-t fogadunk el (SSRF-védelem)."""
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    if parsed.username or parsed.password:
        return False
    return (parsed.hostname or "").lower() in _YOUTUBE_HOSTS


def _lang_code(lang: str) -> str:
    return {"eng": "en", "en": "en", "heb": "he", "he": "he",
            "hun": "hu", "hu": "hu"}.get(str(lang or "eng").lower(), "en")


def _pick_subtitle(folder: Path, code: str) -> Path | None:
    """A letöltött feliratok közül a kért nyelvűt választja."""
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUBTITLE_SUFFIXES)
    if not files:
        return None
    exact = [p for p in files if f".{code}." in p.name]
    return (exact or files)[0]


def _to_srt(source: Path, target: Path) -> None:
    """A WebVTT/egyéb feliratot SRT-vé alakítja (a cél mindig UTF-8 SRT)."""
    if source.suffix.lower() == ".srt":
        shutil.copy2(source, target)
        return
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(source), "-c:s", "srt", str(target)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)
    except FileNotFoundError as exc:
        raise YouTubeError("Az ffmpeg nem érhető el a felirat átalakításához.") from exc
    except subprocess.TimeoutExpired as exc:
        raise YouTubeError("A felirat átalakítása túllépte az időkorlátot.") from exc
    if proc.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        detail = (proc.stderr or "").strip().splitlines()[-3:]
        raise YouTubeError("A felirat nem alakítható SRT-vé: " + " | ".join(detail))


class _Logger:
    """yt-dlp napló a szinkronstúdió futásnaplójába."""

    def debug(self, message):
        if str(message).startswith("[download]"):
            print(str(message), flush=True)

    def info(self, message):                       # pragma: no cover - yt-dlp
        print(str(message), flush=True)

    def warning(self, message):
        print(f"[youtube] figyelmeztetés: {message}", flush=True)

    def error(self, message):
        print(f"[youtube] hiba: {message}", flush=True)


def download(project, tag: str, url: str, *, source_lang: str = "eng") -> dict:
    """YouTube videó + angol felirat letöltése, majd a projektbe helyezése.

    A letöltés egy ideiglenes ``data/_youtube_<EP>`` mappába történik; csak a
    sikeres média és felirat kerül a projekt gyökerébe. Így egy megszakadt
    letöltés nem hagy fél fájlt az ``episode_media()`` számára.
    """
    tag = str(tag or "E01").upper()
    if not is_youtube_url(url):
        raise YouTubeError("Érvénytelen YouTube URL.")
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:                    # pragma: no cover - telepítés
        raise YouTubeError("A YouTube-letöltő nincs telepítve. "
                           "Telepítsd: python -m pip install yt-dlp") from exc

    existing = [p for p in project.data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES] \
        if project.data_dir.is_dir() else []
    if existing:
        raise YouTubeError(f"A projektben már van média: {existing[0].name}")

    stage = project.data_dir / f"_youtube_{tag}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    code = _lang_code(source_lang)
    opts = {
        # H.264-et kérünk először: a kész MKV/MP4 és a böngészős proxy
        # szempontjából ez a legjobban támogatott; AV1 csak tartalék.
        "format": "bv*[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
                  "b[height<=1080][ext=mp4][vcodec^=avc1]/"
                  "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(stage / "source.%(ext)s"),
        "noplaylist": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [code, f"{code}-orig", f"{code}-US"],
        "subtitlesformat": "srt/best",
        # a YouTube néha csak WebVTT/SRV formátumot ad; az ffmpeg-postprocessor
        # egységesen SRT-t hagy a projektben
        "postprocessors": [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "logger": _Logger(),
    }
    print(f"[youtube] letöltés: {url}", flush=True)
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:                      # noqa: BLE001 - yt-dlp hibái
        raise YouTubeError(f"A YouTube-letöltés nem sikerült: {exc}") from exc
    if not isinstance(info, dict):
        raise YouTubeError("A YouTube-válasz nem egyetlen videó.")
    media = sorted((p for p in stage.iterdir()
                    if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not media:
        raise YouTubeError("A letöltés nem adott videófájlt.")
    subtitle = _pick_subtitle(stage, code)
    if subtitle is None:
        raise YouTubeError(
            f"A videóhoz nincs {code} nyelvű felirat (sem kézi, sem automatikus). "
            "Felirat nélkül a szinkron nem indítható; tölts fel külön SRT-t.")
    project.data_dir.mkdir(parents=True, exist_ok=True)
    project.subs_dir.mkdir(parents=True, exist_ok=True)
    media_target = project.data_dir / f"source{media[0].suffix.lower()}"
    shutil.move(str(media[0]), media_target)
    sub_lang = source_lang if source_lang in ("hun", "eng", "heb") else "eng"
    sub_target = project.subtitle(tag, sub_lang)
    _to_srt(subtitle, sub_target)
    shutil.rmtree(stage, ignore_errors=True)
    return {
        "tag": tag,
        "title": str(info.get("title") or ""),
        "video_id": str(info.get("id") or ""),
        "duration": info.get("duration"),
        "media": str(media_target),
        "subtitle": str(sub_target),
        "source_lang": source_lang,
        "url": str(info.get("webpage_url") or url),
    }
