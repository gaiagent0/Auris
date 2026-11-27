"""Narráció — egyszereplős videó külön WebUI-oldala.

A film/sorozat szinkronstúdió (``dub_api``) szereplőkkel, diarizációval és
kézi felülvizsgálattal dolgozik. Egy egyszereplős videóhoz (előadás, bemutató,
saját felvétel) ez nem kell: itt csak a forrás, a hang (kész hang vagy
klónminta) és az eredmény van. A háttérfolyamat a ``dubstudio narrate``
parancsot futtatja, tehát a WebUI és a CLI ugyanazt a modult használja.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file

bp = Blueprint("narration", __name__)

DUB_ROOT = Path(
    os.environ.get("DUBSTUDIO_ROOT")
    or (Path(__file__).resolve().parents[2] / "dubstudio")
).resolve()
#: ide kerül minden narrációs munka és kimenet (a filmes work/ mellett)
NARR_ROOT = DUB_ROOT / "work" / "narration"
UPLOAD_ROOT = NARR_ROOT / "_uploads"

VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

_JOB: dict = {"state": "idle"}
_LOCK = threading.Lock()
_LOG_TAIL = 14000


def _narration():
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import narration
    return narration


def _slug(text: str, fallback: str = "narráció") -> str:
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:40] or fallback


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _log_tail(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-_LOG_TAIL:]


@bp.route("/narracio")
def narration_page():
    return render_template("narracio.html")


@bp.route("/api/narration/voices")
def voices():
    """A választható hangok és a klón lehetősége (motor nélkül, gyorsan)."""
    n = _narration()
    models = n.model_paths()
    clone = n.clone_available(models)
    return jsonify({
        "voices": [{"vid": v.vid, "label": v.label, "engine": v.engine,
                    "gender": v.gender, "clone": bool(v.clone)}
                   for v in n.BY_ID.values()],
        "clone_available": bool(clone),
        "clone_ref": str(models.get("xtts_ref") or ""),
        "default_voice": "SUP_M1",
    })


def _save_upload(kind: str) -> Path:
    """Feltöltött videó/hangminta/felirat mentése a narrációs munkamappába."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ValueError("Nincs feltöltött fájl.")
    name = Path(upload.filename).name
    suffix = Path(name).suffix.lower()
    if kind == "video" and suffix not in VIDEO_SUFFIXES:
        raise ValueError("Támogatott videó kell: " + ", ".join(sorted(VIDEO_SUFFIXES)))
    if kind == "audio" and suffix not in AUDIO_SUFFIXES:
        raise ValueError("Támogatott hangminta kell: " + ", ".join(sorted(AUDIO_SUFFIXES)))
    if kind == "srt" and suffix not in (".srt", ".vtt"):
        raise ValueError("Csak .srt vagy .vtt felirat tölthető fel.")
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = UPLOAD_ROOT / f"{stamp}-{_slug(Path(name).stem)}{suffix}"
    upload.save(target)
    if kind == "audio" and suffix != ".wav":
        # az XTTS klón WAV-ot vár: a feltöltött mintát átalakítjuk
        from dubstudio import media
        wav = target.with_suffix(".wav")
        p = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", str(target),
                            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav)],
                           capture_output=True)
        if p.returncode != 0:
            raise ValueError("A hangminta átalakítása nem sikerült: "
                             + p.stderr.decode("utf-8", "replace")[-200:])
        target.unlink(missing_ok=True)
        target = wav
    return target


@bp.route("/api/narration/upload", methods=["POST"])
def upload():
    kind = str(request.form.get("kind") or "video").lower()
    if kind not in ("video", "audio", "srt"):
        return jsonify({"error": "Ismeretlen feltöltés-típus."}), 400
    try:
        path = _save_upload(kind)
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "path": str(path), "name": path.name,
                    "kind": kind})


@bp.route("/api/narration/pick", methods=["POST"])
def pick():
    """Helyi fájlválasztó (a böngésző nem ad valódi útvonalat)."""
    body = request.get_json(silent=True) or {}
    kind = str(body.get("kind") or "video")
    if kind == "audio":
        patterns = "*.wav *.mp3 *.m4a *.flac *.ogg *.opus"
        title = "Hangminta kiválasztása"
    elif kind == "srt":
        patterns = "*.srt *.vtt"
        title = "Felirat kiválasztása"
    else:
        patterns = " ".join(f"*{s}" for s in sorted(VIDEO_SUFFIXES))
        title = "Videófájl kiválasztása"
    script = (
        "import json, tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        f"picked = filedialog.askopenfilename(title={title!r}, "
        f"filetypes=[('Támogatott', {patterns!r}), ('Minden fájl', '*.*')])\n"
        "root.destroy()\n"
        "print(json.dumps(picked or ''))\n"
    )
    try:
        p = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=900, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return jsonify({"error": "A fájlválasztó nem zárt be."}), 504
    if p.returncode != 0:
        return jsonify({"error": "A fájlválasztó nem indult el: "
                                 + (p.stderr or "").strip()[-200:]}), 500
    try:
        picked = json.loads((p.stdout or "").strip().splitlines()[-1] or '""')
    except (ValueError, IndexError):
        picked = ""
    return jsonify({"ok": True, "path": picked or ""})


def _norm_source(source: str) -> str:
    """A forrás összehasonlítható alakja (helyi út vagy YouTube URL).

    A CLI a dubstudio gyökeréből fut, ezért a riportban relatív út is állhat:
    az ilyet innen (nem a WebUI munkakönyvtárából) kell feloldani.
    """
    text = str(source or "").strip()
    if not text:
        return ""
    if text.lower().startswith(("http://", "https://")):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = DUB_ROOT / path
    try:
        return os.path.normcase(str(path.resolve()))
    except (OSError, ValueError):
        return str(path)


def _norm_path(path: str) -> str | None:
    if not path:
        return None
    return _norm_source(path)


def _reusable_work(source: str, voice: str, ref: str,
                   language: str) -> Path | None:
    """Egy korábbi, ugyanahhoz a forráshoz és hanghoz tartozó munkamappa.

    Így egy próbafutás után a teljes narráció a már kész blokkokat
    gyorsítótárból használja, és nem készül minden elölről.
    """
    want_source = _norm_source(source)
    want_voice = (voice or "SUP_M1").upper()
    want_ref = _norm_path(ref)
    best: tuple[float, Path] | None = None
    fallback: tuple[float, Path] | None = None
    try:
        candidates = [p for p in NARR_ROOT.iterdir() if p.is_dir()]
    except OSError:
        return None
    for folder in candidates:
        report = folder / "report.json"
        if not report.is_file():
            continue
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _norm_source(str(data.get("source") or "")) != want_source:
            continue
        if str(data.get("voice") or "").upper() != want_voice:
            continue
        if str(data.get("language") or "hu") != (language or "hu"):
            continue
        if _norm_path(str(data.get("clone_ref") or "")) != want_ref:
            continue
        if not (folder / "blocks").is_dir() and not (folder / "transcript.json").is_file():
            continue
        try:
            stamp = report.stat().st_mtime
        except OSError:
            continue
        item = (stamp, folder)
        if fallback is None or stamp > fallback[0]:
            fallback = item
        # a teljes (nem próba) futás a jobb választás, ha van ilyen
        if not bool(data.get("limited")) and (best is None or stamp > best[0]):
            best = item
    chosen = best or fallback
    return chosen[1] if chosen else None


def _start(source: str, voice: str, ref: str, subtitle: str, language: str,
           keep_original: bool, limit: int) -> dict:
    n = _narration()
    if not source:
        raise ValueError("Add meg a videót (YouTube URL vagy fájl).")
    if not (n.youtube.is_youtube_url(source) or Path(source).expanduser().is_file()):
        raise ValueError(f"A videó nem található: {source}")
    if ref and not Path(ref).expanduser().is_file():
        raise ValueError(f"A hangminta nem található: {ref}")
    if subtitle and not Path(subtitle).expanduser().is_file():
        raise ValueError(f"A felirat nem található: {subtitle}")
    with _LOCK:
        if _JOB.get("state") == "running":
            raise RuntimeError("Már fut egy narráció. Várd meg vagy állítsd le.")
        work = _reusable_work(source, voice, ref, language or "hu")
        reused = work is not None
        if work is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = (source.rsplit("v=", 1)[-1] if n.youtube.is_youtube_url(source)
                    else Path(source).stem)
            work = NARR_ROOT / f"{stamp}-{_slug(name)}"
        work.mkdir(parents=True, exist_ok=True)
        log_path = work / "run.log"
        cmd = [sys.executable, "-m", "dubstudio", "narrate", source,
               "--work", str(work), "--lang", language or "hu"]
        if voice:
            cmd += ["--voice", voice]
        if ref:
            cmd += ["--ref", ref]
        if subtitle:
            cmd += ["--srt", subtitle]
        if limit:
            cmd += ["--limit", str(limit)]
        if not keep_original:
            cmd += ["--replace-audio"]
        handle = open(log_path, "w", encoding="utf-8", errors="replace")
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(DUB_ROOT), stdout=handle, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except (OSError, ValueError) as exc:
            handle.close()
            raise RuntimeError(f"A folyamat nem indítható: {exc}") from exc
        _JOB.clear()
        _JOB.update({"state": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "pid": proc.pid, "work": str(work), "log": str(log_path),
                     "source": source, "voice": voice or "SUP_M1",
                     "clone": bool(ref), "proc": proc, "handle": handle,
                     "reused": reused, "returncode": None, "finished_at": None})
    return {"state": "running", "pid": proc.pid, "work": str(work),
            "log": str(log_path), "reused": reused}


@bp.route("/api/narration/start", methods=["POST"])
def start():
    body = request.get_json(silent=True) or {}
    try:
        result = _start(
            str(body.get("source") or "").strip(),
            str(body.get("voice") or "").strip(),
            str(body.get("ref") or "").strip(),
            str(body.get("subtitle") or "").strip(),
            str(body.get("language") or "hu").strip() or "hu",
            bool(body.get("keep_original", True)),
            int(body.get("limit") or 0))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, **result})


def _report(work: str | None) -> dict | None:
    if not work:
        return None
    path = Path(work) / "report.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _payload() -> dict:
    job = dict(_JOB)
    proc = job.get("proc")
    if job.get("state") == "running" and proc is not None:
        rc = proc.poll()
        if rc is not None:
            with _LOCK:
                handle = _JOB.get("handle")
                if handle:
                    try:
                        handle.close()
                    except OSError:
                        pass
                _JOB.update(state="done" if rc == 0 else "failed",
                            returncode=rc,
                            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                job = dict(_JOB)
    payload = {k: job.get(k) for k in
               ("state", "pid", "work", "log", "source", "voice", "clone",
                "reused", "started_at", "finished_at", "returncode")}
    payload["log_tail"] = _log_tail(job.get("log"))
    report = _report(job.get("work"))
    if report:
        payload["report"] = report
        out = report.get("output")
        payload["output"] = out
        if out and Path(out).is_file():
            payload["output_url"] = "/api/narration/file?path=" + str(out)
        audio = report.get("audio")
        if audio and Path(audio).is_file():
            payload["audio_url"] = "/api/narration/file?path=" + str(audio)
    return payload


@bp.route("/api/narration/status")
def status():
    return jsonify(_payload())


@bp.route("/api/narration/stop", methods=["POST"])
def stop():
    with _LOCK:
        job = dict(_JOB)
        proc = job.get("proc")
        if job.get("state") != "running" or proc is None:
            return jsonify({"error": "Nincs futó narráció."}), 409
        proc.terminate()
        _JOB.update(state="stopped", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        handle = _JOB.get("handle")
        if handle:
            try:
                handle.close()
            except OSError:
                pass
    return jsonify({"ok": True, "state": "stopped"})


@bp.route("/api/narration/file")
def file():
    raw = str(request.args.get("path") or "").strip()
    if not raw:
        return jsonify({"error": "Hiányzó útvonal."}), 400
    path = Path(raw).expanduser()
    if not path.is_file() or not _inside(path, NARR_ROOT):
        return jsonify({"error": "A fájl nem érhető el."}), 404
    as_attachment = str(request.args.get("download") or "") in ("1", "true", "yes")
    return send_file(path, as_attachment=as_attachment,
                     download_name=path.name, conditional=True)
