"""Böngészős felülvizsgáló lejátszó a forrás-MKV/MP4-hez.

A böngészők az MKV több hangsávját nem kezelik megbízhatóan, ezért a
szerver oldalon **egy választott hangsávot tartalmazó MP4 proxyt** készítünk.
A videósávot lehetőleg copy-val visszük át; csak akkor kódolunk újra, ha a
forrás nem H.264. A magyar szinkron külön hangsávként (`dub.wav`) jelenik meg,
így a felülvizsgáló a forrás és a kész magyar hang között válthat.

A feliratokat WebVTT-vé alakítjuk; a képfeliratok (PGS/VobSub) nem
konvertálhatók szöveggé, ezeket a manifest kihagyja.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import copy
from pathlib import Path

from .config import Project

MEDIA_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".m4v"}
_TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
_VIDEO_COPY = {"h264", "avc1"}
#: 10 bites/4:2:2/4:4:4 vagy HDR videót a böngészők nem dekódolnak: ilyenkor
#: 8 bites SDR-re kell kódolni (különben fekete kép, csak hang szól).
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
#: a proxy fájlnevének verziója — a régi, rossz kódolású proxykat elkerüli
_PROXY_VERSION = "_v2"
#: HDR (BT.2020/PQ) → SDR (BT.709) tonemap, 8 bit
_TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
            "format=yuv420p")
#: 8 bites, széles körben dekódolható H.264 kimenet
_ENCODE_SDR = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709"]

#: az aszinkron proxy-előkészítés állapota (a folyamat memóriájában)
_PREP: dict[tuple[str, str, str], dict] = {}
_PREP_LOCK = threading.Lock()
#: a drága ffprobe ne fusson másodpercenként a státusz-lekérdezéseknél
_MANIFEST_CACHE: dict[tuple, dict] = {}


class PlayerError(RuntimeError):
    """A lejátszó-proxy elkészítése nem sikerült."""


def _run(cmd: list[str], timeout: int = 300) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise PlayerError("Az ffmpeg nem érhető el (telepítsd és tedd a PATH-ra).") from exc
    except subprocess.TimeoutExpired as exc:
        raise PlayerError("Az ffmpeg túllépte az időkorlátot.") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-4:]
        raise PlayerError("Az ffmpeg hibával zárt: " + " | ".join(tail))


def probe(path: Path) -> dict:
    """Stream-metaadatok (nem tölt le semmit)."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise PlayerError("Az ffprobe nem érhető el.") from exc
    except subprocess.TimeoutExpired as exc:
        raise PlayerError("Az ffprobe túllépte az időkorlátot.") from exc
    if proc.returncode != 0:
        raise PlayerError("Az ffprobe nem tudta olvasni a médiát.")
    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError as exc:
        raise PlayerError("Az ffprobe válasza nem értelmezhető.") from exc
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise PlayerError("A fájlban nincs videósáv.")
    fmt = data.get("format") or {}
    try:
        duration = round(float(fmt.get("duration") or 0.0), 2)
    except (TypeError, ValueError):
        duration = 0.0
    audio, subs = [], []
    a_i = s_i = 0
    for stream in streams:
        kind = stream.get("codec_type")
        tags = stream.get("tags") or {}
        lang = str(tags.get("language") or "").strip()
        title = str(tags.get("title") or "").strip()
        if kind == "audio":
            audio.append({"ordinal": a_i, "index": stream.get("index"),
                          "codec": stream.get("codec_name") or "?",
                          "lang": lang, "title": title})
            a_i += 1
        elif kind == "subtitle":
            subs.append({"ordinal": s_i, "index": stream.get("index"),
                         "codec": stream.get("codec_name") or "?",
                         "lang": lang, "title": title})
            s_i += 1
    codec = (video.get("codec_name") or "").lower()
    profile = str(video.get("profile") or "")
    pix_fmt = str(video.get("pix_fmt") or "")
    transfer = str(video.get("color_transfer") or "").lower()
    ten_bit = "10" in pix_fmt or "10" in profile
    wide = "4:2:2" in profile or "4:4:4" in profile
    hdr = transfer in _HDR_TRANSFERS
    return {
        "duration": duration,
        "video": {
            "codec": codec or "?",
            "width": video.get("width"),
            "height": video.get("height"),
            "profile": profile,
            "pix_fmt": pix_fmt,
            "hdr": hdr,
            # csak akkor másolható a sáv, ha a böngésző biztosan dekódolja
            "copy": (codec in _VIDEO_COPY and not ten_bit and not wide and not hdr
                     and pix_fmt in {"yuv420p", "yuvj420p"}),
        },
        "audio": audio,
        "subtitles": subs,
    }


def _label(lang: str, title: str, fallback: str) -> str:
    names = {"hun": "magyar", "eng": "angol", "heb": "héber",
             "en": "angol", "he": "héber", "und": "ismeretlen"}
    bits = []
    if title:
        bits.append(title)
    if lang:
        bits.append(names.get(lang.lower(), lang))
    return " · ".join(bits) if bits else fallback


def manifest(project: Project, tag: str) -> dict:
    """A lejátszóhoz tartozó választható sávok (még nincs proxy)."""
    source = project.episode_media(tag)
    if not source.is_file():
        raise PlayerError(f"Nincs forrásmédia: {source}")
    dub = project.work / tag / "dub.wav"
    stamp = [str(source), source.stat().st_mtime_ns, tag.upper()]
    for lang in ("hun", "eng", "heb"):
        sub = project.subtitle(tag, lang)
        if sub.is_file():
            stamp += [str(sub), sub.stat().st_mtime_ns]
    if dub.is_file():
        stamp += [str(dub), dub.stat().st_mtime_ns]
    cache_key = tuple(stamp)
    with _PREP_LOCK:
        cached = _MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        # a proxy időközben el is készülhet: a cached jelzést mindig frissítjük
        for track in result.get("audio") or []:
            track["cached"] = _proxy_fresh(project, tag, track, result)
        return result
    info = probe(source)
    audio = []
    for stream in info["audio"]:
        label = _label(stream["lang"], stream["title"],
                       f"{stream['codec']} hangsáv {stream['ordinal'] + 1}")
        audio.append({
            "id": f"a{stream['ordinal']}",
            "label": label,
            "lang": stream["lang"],
            "codec": stream["codec"],
            "kind": "source",
            "ordinal": stream["ordinal"],
            "default": not audio,
        })
    dub = project.work / tag / "dub.wav"
    if dub.is_file():
        audio.append({
            "id": "dub",
            "label": "Magyar szinkron (kész)",
            "lang": "hun",
            "codec": "pcm_s16le",
            "kind": "dub",
            "default": False,
        })
    if not audio:
        raise PlayerError("A forrásnak nincs hangsávja.")
    subs = []
    seen = set()
    for stream in info["subtitles"]:
        if (stream["codec"] or "").lower() not in _TEXT_SUB_CODECS:
            continue
        label = _label(stream["lang"], stream["title"],
                       f"Beágyazott felirat {stream['ordinal'] + 1}")
        sid = f"s{stream['ordinal']}"
        if sid in seen:
            continue
        seen.add(sid)
        subs.append({"id": sid, "label": label, "lang": stream["lang"],
                     "kind": "embedded", "ordinal": stream["ordinal"]})
    for lang, label in (("hun", "Magyar felirat"), ("eng", "Angol felirat")):
        path = project.subtitle(tag, lang)
        if not path.is_file():
            continue
        sid = f"file_{lang}"
        if sid in seen:
            continue
        seen.add(sid)
        subs.append({"id": sid, "label": label, "lang": lang,
                     "kind": "file", "path": str(path)})
    result = {
        "tag": tag,
        "source": str(source),
        "duration": info["duration"],
        "video": info["video"],
        "audio": audio,
        "subtitles": subs,
        "dub_sample": _dub_sample(project, tag) if dub.is_file() else None,
    }
    for track in result["audio"]:
        track["cached"] = _proxy_fresh(project, tag, track, result)
    with _PREP_LOCK:
        _MANIFEST_CACHE[cache_key] = copy.deepcopy(result)
    return result


def _dub_sample(project: Project, tag: str) -> dict:
    """Rövid magyar ellenőrző minta a kész dub.wav-ból.

    Elsősorban a leghosszabb 1,5–8 másodperces megnyilatkozást választja,
    hogy a hang és a kiejtés is hallható legyen; ha nincs utterances.json,
    a film első 8 másodpercét ajánlja.
    """
    try:
        rows = json.loads((project.work / tag / "utterances.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rows = []
    candidates = []
    for row in rows if isinstance(rows, list) else []:
        try:
            start = float(row.get("start") or 0.0)
            end = float(row.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        dur = end - start
        if start >= 0 and 1.5 <= dur <= 8.0:
            candidates.append((dur, start, end))
    if candidates:
        dur, start, end = max(candidates, key=lambda item: item[0])
        return {"start": round(max(0.0, start - 0.15), 2),
                "duration": round(min(8.0, dur + 0.35), 2)}
    return {"start": 0.0, "duration": 8.0}


def sample_for(project: Project, tag: str, info: dict | None = None) -> dict:
    """Rövid magyar ellenőrző klip (nem a teljes 40 perces proxyt készíti).

    A videóból és a kész ``dub.wav``-ból 6-8 másodpercet vágunk össze, így a
    minta az első kattintásra gyorsan lejátszható; a teljes hangsáv proxya
    ettől függetlenül készül, amikor a felhasználó azt választja.
    """
    info = info or manifest(project, tag)
    source = Path(info["source"])
    dub = project.work / tag / "dub.wav"
    if not dub.is_file():
        raise PlayerError("Ehhez még nincs kész magyar hangsáv (dub.wav).")
    window = _dub_sample(project, tag)
    start = float(window.get("start") or 0.0)
    duration = max(1.0, min(10.0, float(window.get("duration") or 8.0)))
    out = _cache_dir(project, tag) / f"player_dub_sample{_PROXY_VERSION}.mp4"
    deps = [source, dub, project.work / tag / "utterances.json"]
    if out.is_file() and all(out.stat().st_mtime >= p.stat().st_mtime
                             for p in deps if p.is_file()):
        return {"path": str(out), "start": start, "duration": duration}
    tmp = out.with_suffix(".part.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(dub),
           "-map", "0:v:0", "-map", "1:a:0",
           *_video_args(info),
           "-c:a", "aac", "-b:a", "160k", "-ac", "2",
           "-movflags", "+faststart", "-sn", "-dn", str(tmp)]
    try:
        _run(cmd, timeout=180)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": str(out), "start": start, "duration": duration}


def _cache_dir(project: Project, tag: str) -> Path:
    out = project.work / tag / "review"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _proxy_path(project: Project, tag: str, audio_id: str) -> Path:
    return _cache_dir(project, tag) / f"player_{audio_id}{_PROXY_VERSION}.mp4"


def _video_args(info: dict) -> list[str]:
    """Videósáv a böngészőhöz: 8 bites SDR H.264 (a 10 bites HDR nem játszható)."""
    filters = [_TONEMAP] if info["video"].get("hdr") else ["format=yuv420p"]
    return [*_ENCODE_SDR, "-vf", ",".join(filters)]


def _proxy_fresh(project: Project, tag: str, track: dict, info: dict) -> bool:
    source = Path(info["source"])
    out = _proxy_path(project, tag, track["id"])
    deps = [source]
    if track.get("kind") == "dub":
        deps.append(project.work / tag / "dub.wav")
    try:
        return out.is_file() and all(out.stat().st_mtime >= p.stat().st_mtime
                                     for p in deps if p.is_file())
    except OSError:
        return False


def media_for(project: Project, tag: str, audio_id: str, info: dict | None = None) -> Path:
    """A választott hangsávval rendelkező, böngészőbarát MP4 proxy."""
    info = info or manifest(project, tag)
    track = next((a for a in info["audio"] if a["id"] == audio_id), None)
    if track is None:
        raise PlayerError(f"Ismeretlen hangsáv: {audio_id}")
    source = Path(info["source"])
    dub = project.work / tag / "dub.wav"
    out = _proxy_path(project, tag, audio_id)
    if _proxy_fresh(project, tag, track, info):
        return out

    video_codec = ["-c:v", "copy"] if info["video"].get("copy") else _video_args(info)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if track["kind"] == "dub":
        cmd += ["-i", str(dub), "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", f"0:a:{track['ordinal']}"]
    cmd += video_codec + ["-c:a", "aac", "-b:a", "160k", "-ac", "2",
                          "-movflags", "+faststart", "-sn", "-dn",
                          str(out)]
    tmp = out.with_suffix(".part.mp4")
    cmd[-1] = str(tmp)
    # egy 40 perces film újrakódolása néhány percig is eltarthat
    budget = int(max(1800, float(info.get("duration") or 0.0) * 6))
    try:
        _run(cmd, timeout=budget)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return out


def _prep_key(project: Project, tag: str, audio_id: str, info: dict) -> tuple[str, str, str]:
    source = Path(info["source"])
    try:
        stamp = str(source.stat().st_mtime_ns)
    except OSError:
        stamp = "0"
    return (f"{source}|{stamp}", tag.upper(), audio_id)


def _status_payload(project: Project, tag: str, audio_id: str, info: dict,
                    state: dict | None) -> dict:
    track = next((a for a in info.get("audio") or [] if a["id"] == audio_id), None)
    out = _proxy_path(project, tag, audio_id)
    fresh = bool(track) and _proxy_fresh(project, tag, track, info)
    payload = {"audio": audio_id, "state": "ready" if fresh else "missing",
               "cached": fresh, "duration": info.get("duration"),
               "path": str(out) if fresh else None}
    if fresh:
        try:
            payload["size"] = out.stat().st_size
        except OSError:
            pass
        return payload
    if state and not (state.get("state") == "ready" and not fresh):
        payload.update({key: state.get(key) for key in
                        ("state", "error", "started", "finished")})
        if state.get("state") == "preparing":
            part = out.with_suffix(".part.mp4")
            try:
                payload["part_bytes"] = part.stat().st_size
            except OSError:
                payload["part_bytes"] = 0
            started = float(state.get("started") or time.time())
            payload["elapsed"] = round(max(0.0, time.time() - started), 1)
    return payload


def proxy_status(project: Project, tag: str, audio_id: str,
                 info: dict | None = None) -> dict:
    """Egy hangsáv proxyjának állapota (ready / preparing / missing / error)."""
    info = info or manifest(project, tag)
    track = next((a for a in info.get("audio") or [] if a["id"] == audio_id), None)
    if track is None:
        raise PlayerError(f"Ismeretlen hangsáv: {audio_id}")
    key = _prep_key(project, tag, audio_id, info)
    with _PREP_LOCK:
        state = dict(_PREP.get(key) or {})
    return _status_payload(project, tag, audio_id, info, state or None)


def prepare(project: Project, tag: str, audio_id: str,
            info: dict | None = None) -> dict:
    """A proxy elkészítése háttérszálon; azonnal állapotot ad vissza."""
    info = info or manifest(project, tag)
    track = next((a for a in info.get("audio") or [] if a["id"] == audio_id), None)
    if track is None:
        raise PlayerError(f"Ismeretlen hangsáv: {audio_id}")
    key = _prep_key(project, tag, audio_id, info)
    with _PREP_LOCK:
        current = _PREP.get(key)
        if current and current.get("state") == "preparing":
            return _status_payload(project, tag, audio_id, info, dict(current))
        if _proxy_fresh(project, tag, track, info):
            return _status_payload(project, tag, audio_id, info, None)
        started = {"state": "preparing", "started": time.time(), "error": ""}
        _PREP[key] = started
    thread = threading.Thread(target=_prepare_worker,
                              args=(project, tag, audio_id, info, key),
                              name=f"dub-player-{tag}-{audio_id}", daemon=True)
    thread.start()
    return _status_payload(project, tag, audio_id, info, started)


def _prepare_worker(project: Project, tag: str, audio_id: str, info: dict,
                    key: tuple[str, str, str]) -> None:
    try:
        path = media_for(project, tag, audio_id, info)
        with _PREP_LOCK:
            _PREP[key] = {"state": "ready", "finished": time.strftime("%H:%M:%S"),
                          "path": str(path)}
    except Exception as exc:               # noqa: BLE001 — a státuszba kerül
        with _PREP_LOCK:
            _PREP[key] = {"state": "error", "error": f"{type(exc).__name__}: {exc}",
                          "finished": time.strftime("%H:%M:%S")}


def subtitle_for(project: Project, tag: str, sub_id: str, info: dict | None = None) -> Path:
    """A választott felirat WebVTT-változata."""
    info = info or manifest(project, tag)
    track = next((s for s in info["subtitles"] if s["id"] == sub_id), None)
    if track is None:
        raise PlayerError(f"Ismeretlen felirat: {sub_id}")
    out = _cache_dir(project, tag) / f"sub_{sub_id}.vtt"
    source = Path(info["source"])
    if track["kind"] == "file":
        src = Path(track["path"])
        dep = src
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(src), "-c:s", "webvtt", "-f", "webvtt", str(out)]
    else:
        dep = source
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(source), "-map", f"0:s:{track['ordinal']}",
               "-c:s", "webvtt", "-f", "webvtt", str(out)]
    if out.is_file() and out.stat().st_mtime >= dep.stat().st_mtime:
        return out
    tmp = out.with_suffix(".part.vtt")
    cmd[-1] = str(tmp)
    try:
        _run(cmd, timeout=120)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return out
