"""YouTube-forrás letöltése a szinkronprojekthez.

A modul a nyilvános YouTube-videót a legjobb elérhető (legfeljebb 1080p)
MP4 formátumban tölti le, mellé kéri az angol feliratot (kézi vagy automatikus),
majd a projekt ``data/`` és ``subs/`` mappájába helyezi. A tényleges fordítást
és szinkront nem itt, hanem a meglévő ``intake`` + ``Pipeline`` lánc végzi.

A YouTube ugyanahhoz a feliratsávhoz több (kliensenként külön) letöltési címet
ad, és ezek egy része átmenetileg ``HTTP 429``-cel elutasítható. A yt-dlp a
feliratnál csak az első címet próbálja, ezért a feliratot itt magunk kérjük le:
a jelölteket sorban, több körben, rövid várakozással — így egy rate limitelt
cím nem viszi el a teljes letöltést.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MEDIA_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SUBTITLE_SUFFIXES = (".srt", ".vtt")
_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be", "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
#: a kért nyelv kódja a YouTube-on (a projekt nyelve → YouTube nyelvkód)
_LANG_CODES = {"eng": "en", "en": "en", "heb": "he", "he": "he",
               "hun": "hu", "hu": "hu"}
#: a YouTube nyelvkódja → a projekt feliratneve (``subs/E01_eng.srt``)
_SUBTITLE_LANGS = {"en": "eng", "he": "heb", "hu": "hun"}
#: a feliratjelöltek formátum-preferenciája (a kész SRT/VTT a legjobb)
_FORMAT_ORDER = ("srt", "vtt", "json3")
#: hány körben próbáljuk a jelölteket, és mennyit várunk a körök között
_ATTEMPT_DELAYS = (0.0, 1.5, 4.0)
#: egy körben legfeljebb ennyi jelölt-címet próbálunk (a YouTube sokat ad)
_CANDIDATE_LIMIT = 24
#: a média formátum-tartaléklánca: ha az első 403-at kap (gyakori a 1080p60-as
#: H.264-nél JS-futtatókörnyezet nélkül), a következő, kompatibilisebb jön
_FORMAT_CHAIN = (
    "bv*[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
    "b[height<=1080][ext=mp4][vcodec^=avc1]/"
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
    "bv*[height<=720]+ba/b[height<=720]/b",
    "18/b",
)


def _js_runtimes() -> dict:
    """A yt-dlp JS-futtatókörnyezete, ha van a gépen (node vagy deno).

    A YouTube kinyerése JS-t vár; nélküle bizonyos formátumok (jellemzően a
    1080p60-as H.264) ``403``-at adnak letöltéskor.
    """
    if shutil.which("node"):
        return {"node": {}}
    if shutil.which("deno"):
        return {"deno": {}}
    return {}


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
    return _LANG_CODES.get(str(lang or "eng").strip().lower(), "en")


def _subtitle_lang(lang: str) -> str:
    """A projekt feliratneve (eng/heb/hun) a YouTube nyelvkódjából."""
    return _SUBTITLE_LANGS.get(_lang_code(lang), "eng")


# ── felirat: jelöltek és letöltés ────────────────────────────────────────────


def _caption_candidates(info: dict, code: str) -> list[dict]:
    """A kért nyelvű feliratsávok: a kézi felirat előbb, aztán az automatikus.

    A YouTube nyelvváltozatait is elfogadjuk (``en-orig``, ``en-US``, …), mert
    egy videónál gyakran csak ezek egyike létezik.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    out: list[dict] = []
    for bucket in (manual, auto):
        names = [n for n in (code, f"{code}-orig", f"{code}-US") if n in bucket]
        prefix = f"{str(code).lower()}-"
        names += [n for n in bucket
                  if str(n).lower().startswith(prefix) and n not in names]
        for name in names:
            out.extend(bucket.get(name) or [])
    return out


def _ordered_candidates(entries: list[dict]) -> list[tuple[str, dict]]:
    """A jelöltek formátum szerint rendezve (SRT → VTT → JSON3), cím nélkül."""
    ordered: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for ext in _FORMAT_ORDER:
        for entry in entries:
            url = str(entry.get("url") or "")
            if not url or url in seen:
                continue
            if str(entry.get("ext") or "").lower() != ext:
                continue
            seen.add(url)
            ordered.append((ext, entry))
    return ordered[:_CANDIDATE_LIMIT]


def _fetch(url: str, headers: dict | None = None, timeout: float = 30.0) -> bytes:
    """Egy felirat-URL letöltése a yt-dlp által adott fejlécekkel."""
    merged = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/131.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    }
    merged.update({str(k): str(v) for k, v in (headers or {}).items()})
    request = urllib.request.Request(url, headers=merged)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


_TS = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})")
_TAG = re.compile(r"<[^>]*>")


def _clean_line(line: str) -> str:
    """A VTT/JSON3 sorból kiveszi a beágyazott időzítést és a HTML-tageket."""
    text = _TAG.sub("", str(line or ""))
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _stamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours, rest = divmod(int(total), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((total - int(total)) * 1000))
    if millis == 1000:
        secs, millis = secs + 1, 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _srt_time(value: str) -> str:
    """VTT-időbélyeg (``00:01:02.500``) → SRT-időbélyeg (``00:01:02,500``)."""
    match = _TS.search(str(value or ""))
    if not match:
        return _stamp(0.0)
    hours = int(match.group(1) or 0)
    seconds = hours * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    return _stamp(seconds + int(match.group(4)) / 1000.0)


def _vtt_to_srt(text: str) -> str:
    """WebVTT → SRT, a YouTube gördülő (ismételt soros) feliratának tisztításával.

    Az automatikus felirat minden cue-ban megismétli az előző sor végét; ezt
    kiszűrjük, hogy a fordítás ne kapja kétszer ugyanazt a mondatot.
    """
    cues: list[dict] = []
    current: dict | None = None
    for raw in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if "-->" in line and _TS.search(line):
            start, _, rest = line.partition("-->")
            end = rest.strip().split()[0] if rest.strip() else ""
            current = {"start": _srt_time(start), "end": _srt_time(end),
                       "lines": []}
            cues.append(current)
            continue
        if current is None:
            continue
        if not line:
            current = None
            continue
        current["lines"].append(raw)

    blocks: list[str] = []
    prev_last = ""
    prev_text = ""
    for cue in cues:
        lines = [_clean_line(line) for line in cue["lines"]]
        lines = [line for line in lines if line]
        if not lines:
            continue
        body = lines[1:] if len(lines) > 1 and lines[0] == prev_last else lines
        content = " ".join(body).strip()
        prev_last = lines[-1]
        if not content or content == prev_text:
            continue
        prev_text = content
        blocks.append(f"{cue['start']} --> {cue['end']}\n{content}")
    return _number_srt(blocks)


def _json3_to_srt(text: str) -> str:
    """YouTube ``json3`` felirat → SRT (az utolsó lehetséges tartalék)."""
    try:
        data = json.loads(text)
    except ValueError:
        return ""
    blocks: list[str] = []
    for event in (data.get("events") or []):
        body = "".join(str(seg.get("utf8") or "")
                       for seg in (event.get("segs") or []))
        body = _clean_line(body)
        if not body:
            continue
        start = float(event.get("tStartMs") or 0) / 1000.0
        end = start + float(event.get("dDurationMs") or 0) / 1000.0
        if end <= start:
            end = start + 2.0
        blocks.append(f"{_stamp(start)} --> {_stamp(end)}\n{body}")
    return _number_srt(blocks)


_SENTENCE_END = re.compile(r"[.!?\u2026:;]['\"\u201d\u2019)\]]*$")


def _merge_fragments(srt: str, *, max_chars: int = 100,
                     max_gap: float = 0.45) -> str:
    """A YouTube automatikus feliratának töredékeit mondatokká vonja össze.

    Az automatikus felirat 2–5 szavas töredékekre vágja a beszédet („Google
    just turned Gemini Notebook into” / „a website builder. You drop in”),
    ezért egy-egy fordítás és egy-egy szinkronhang sem mond értelmes mondatot.
    A szomszédos sorokat addig vonjuk össze, amíg a szöveg nem ér véget
    mondatvégi írásjellel; az időzítés az első sor elejétől az utolsó végéig
    tart, így a magyar szinkron a teljes mondatra kap időkeretet.
    """
    cues: list[list] = []
    for chunk in str(srt or "").strip().split("\n\n"):
        lines = [line for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        left, _, right = lines[1].partition("-->")
        start = _seconds(left)
        end = _seconds(right)
        text = " ".join(line.strip() for line in lines[2:]).strip()
        if not text:
            continue
        cues.append([start, max(end, start), text])
    merged: list[list] = []
    for start, end, text in cues:
        if merged:
            prev = merged[-1]
            joined = f"{prev[2]} {text}"
            if (start - prev[1] <= max_gap and len(joined) <= max_chars
                    and not _SENTENCE_END.search(prev[2])):
                prev[2] = joined
                prev[1] = max(prev[1], end)
                continue
        merged.append([start, end, text])
    # a gördülő felirat sorai átfedik egymást; a szinkronhang nem beszélhet
    # egyszerre két sorral, ezért a következő mondat elejénél elvágjuk
    for index, cue in enumerate(merged[:-1]):
        following = merged[index + 1][0]
        if cue[0] < following < cue[1]:
            cue[1] = following
    return _number_srt([f"{_stamp(start)} --> {_stamp(end)}\n{text}"
                        for start, end, text in merged])


def _seconds(stamp: str) -> float:
    """SRT-időbélyeg → másodperc."""
    match = _TS.search(str(stamp or ""))
    if not match:
        return 0.0
    return (int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60
            + int(match.group(3)) + int(match.group(4)) / 1000.0)


def _number_srt(blocks: list[str]) -> str:
    """A kész blokkokat sorszámozza és SRT-szöveget ad vissza."""
    if not blocks:
        return ""
    return "\n\n".join(f"{i}\n{block}" for i, block in enumerate(blocks, 1)) + "\n"


def _convert(raw: bytes, ext: str) -> str:
    """A letöltött feliratot SRT-szöveggé alakítja (a formátumtól függően)."""
    text = raw.decode("utf-8-sig", errors="replace")
    ext = str(ext or "").lower()
    if ext == "srt":
        return text if "-->" in text else ""
    if ext == "vtt":
        return _vtt_to_srt(text)
    if ext == "json3":
        return _json3_to_srt(text)
    return ""


def _download_subtitle(entries: list[dict], code: str, *, fetch=None,
                       delays=_ATTEMPT_DELAYS, limit: int | None = None) -> str:
    """A felirat letöltése a jelöltekből, körökben újrapróbálva.

    A 429/5xx válasz nem végzetes: a következő jelölt-címet, majd a következő
    kört próbáljuk. Csak akkor hibázunk, ha minden kör minden jelöltje elbukott.
    """
    candidates = _ordered_candidates(entries)
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise YouTubeError(
            f"A videóhoz nincs {code} nyelvű felirat (sem kézi, sem "
            "automatikus). Felirat nélkül a szinkron nem indítható; tölts fel "
            "külön SRT-t a feliratpanelen.")
    fetch = fetch or _fetch
    errors: list[str] = []
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        for ext, entry in candidates:
            url = str(entry.get("url") or "")
            try:
                raw = fetch(url, entry.get("http_headers") or {})
            except urllib.error.HTTPError as exc:
                errors.append(f"{ext}: HTTP {exc.code}")
                continue
            except (urllib.error.URLError, OSError, ValueError) as exc:
                errors.append(f"{ext}: {exc}")
                continue
            srt = _convert(raw, ext)
            if srt.strip():
                if attempt:
                    print(f"[youtube] felirat a(z) {attempt + 1}. körben jött le "
                          f"({ext}).", flush=True)
                return srt
            errors.append(f"{ext}: üres felirat")
        if attempt + 1 < len(delays):
            print(f"[youtube] a felirat még nem jött le ({errors[-1] if errors else '?'}), "
                  "újrapróbálkozás…", flush=True)
    detail = ", ".join(sorted(set(errors))[-4:]) or "ismeretlen hiba"
    raise YouTubeError(
        "A felirat letöltése nem sikerült (" + detail + "). A YouTube "
        "átmenetileg korlátozhatja a feliratkéréseket (HTTP 429); próbáld újra "
        "néhány perc múlva, vagy tölts fel SRT-t a feliratpanelen.")


# ── médialetöltés ────────────────────────────────────────────────────────────


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


def _ydl_class():
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:                    # pragma: no cover - telepítés
        raise YouTubeError("A YouTube-letöltő nincs telepítve. "
                           "Telepítsd: python -m pip install yt-dlp") from exc
    return YoutubeDL


def _media_opts(stage: Path, fmt: str) -> dict:
    opts = {
        # H.264-et kérünk először: a kész MKV/MP4 és a böngészős proxy
        # szempontjából ez a legjobban támogatott; AV1 csak tartalék.
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(stage / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "logger": _Logger(),
    }
    runtimes = _js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    return opts


def _extract_info(url: str) -> dict:
    """A videó metaadatai + a feliratsávok címei letöltés nélkül."""
    YoutubeDL = _ydl_class()
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True,
                        "noplaylist": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:                      # noqa: BLE001 - yt-dlp hibái
        raise YouTubeError(f"A YouTube-videó nem olvasható be: {exc}") from exc
    if not isinstance(info, dict):
        raise YouTubeError("A YouTube-válasz nem egyetlen videó.")
    if info.get("_type") == "playlist" or info.get("entries"):
        raise YouTubeError("A megadott URL lejátszási lista, nem egyetlen videó.")
    return info


def _download_media(url: str, stage: Path) -> Path:
    """A videó letöltése az elkészített ``stage`` mappába.

    A formátumlánc első eleme a legjobb minőség; ha a YouTube 403-at ad rá
    (jellemzően a 1080p60-as H.264-re JS-futtatókörnyezet nélkül), a következő
    elemmel próbáljuk újra, a félkész fájlokat kitakarítva.
    """
    YoutubeDL = _ydl_class()
    last: Exception | None = None
    for index, fmt in enumerate(_FORMAT_CHAIN):
        try:
            with YoutubeDL(_media_opts(stage, fmt)) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as exc:                  # noqa: BLE001 - yt-dlp hibái
            last = exc
            print(f"[youtube] a letöltés nem ment ezzel: {fmt[:48]}… ({exc})",
                  flush=True)
            for leftover in stage.iterdir():
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)
            if index + 1 < len(_FORMAT_CHAIN):
                print("[youtube] újrapróbálkozás gyengébb formátummal…",
                      flush=True)
            continue
        media = sorted((p for p in stage.iterdir()
                        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES),
                       key=lambda p: p.stat().st_size, reverse=True)
        if media:
            return media[0]
        last = None
    raise YouTubeError(f"A YouTube-letöltés nem sikerült: {last}")


# ── belépési pont ────────────────────────────────────────────────────────────


def download(project, tag: str, url: str, *, source_lang: str = "eng") -> dict:
    """YouTube videó + felirat letöltése, majd a projektbe helyezése.

    A letöltés egy ideiglenes ``data/_youtube_<EP>`` mappába történik; csak a
    sikeres média és felirat kerül a projekt gyökerébe. Így egy megszakadt
    letöltés nem hagy fél fájlt az ``episode_media()`` számára.

    A felirat előbb jön le (ez a szinkron feltétele), utána a videó; így egy
    felirat nélküli videó nem foglal feleslegesen több száz megabájtot.
    """
    tag = str(tag or "E01").upper()
    if not is_youtube_url(url):
        raise YouTubeError("Érvénytelen YouTube URL.")
    existing = [p for p in project.data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES] \
        if project.data_dir.is_dir() else []
    if existing:
        raise YouTubeError(f"A projektben már van média: {existing[0].name}")

    code = _lang_code(source_lang)
    print(f"[youtube] letöltés: {url}", flush=True)
    info = _extract_info(url)
    title = str(info.get("title") or "")
    duration = info.get("duration")
    print(f"[youtube] {title or info.get('id')} ({(duration or 0) // 60} perc)",
          flush=True)
    # a nyers automatikus felirat töredékes — mondatokká összevonva megy a
    # fordítónak és a szinkronhangnak
    srt = _merge_fragments(_download_subtitle(_caption_candidates(info, code), code))

    stage = project.data_dir / f"_youtube_{tag}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        media = _download_media(url, stage)
        project.data_dir.mkdir(parents=True, exist_ok=True)
        project.subs_dir.mkdir(parents=True, exist_ok=True)
        media_target = project.data_dir / f"source{media.suffix.lower()}"
        shutil.move(str(media), media_target)
        sub_target = project.subtitle(tag, _subtitle_lang(source_lang))
        sub_target.write_text(srt, encoding="utf-8")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "tag": tag,
        "title": title,
        "video_id": str(info.get("id") or ""),
        "duration": duration,
        "media": str(media_target),
        "subtitle": str(sub_target),
        "source_lang": source_lang,
        "url": str(info.get("webpage_url") or url),
    }
