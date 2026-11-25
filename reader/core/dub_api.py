"""Szinkron stúdió: a dubstudio munkafájljai fölötti felülvizsgáló felület.

Ez a modul **nem** szinkronizál és nem ír produkciós fájlt. A dubstudio
``work/<EP>/source_audit/`` könyvtárában lévő forrás-auditot, a diarizált
beszélőket, a stabil ``C###`` karakterazonosságot és a hangkiosztó döntési
modell javaslatait teszi átláthatóvá, és az emberi döntéseket ugyanabba a
``review.json`` állapotba írja, amit a parancssor is használ.

A projekt útvonala a ``DUBSTUDIO_ROOT`` környezeti változóval írható felül;
alapértelmezés a repón belüli ``dubstudio`` könyvtár.
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
from urllib.parse import urlencode

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from core import subtitle_upload

bp = Blueprint("dub", __name__)

DUB_ROOT = Path(
    os.environ.get("DUBSTUDIO_ROOT")
    or (Path(__file__).resolve().parents[2] / "dubstudio")
).resolve()

_TAG_RE = re.compile(r"^[A-Z]\d{2,3}$")

#: a WebUI-ból szerkeszthető projektszakaszok (a többi csak olvasható)
_EDITABLE_SECTIONS = ("models", "tuning", "gate", "llm", "asr")
#: numerikus szakaszok: minden levél szám (vagy számokból álló dict) kell legyen
_NUMERIC_SECTIONS = ("tuning", "gate")
#: a WebUI-ból elindítható pipeline-lépések (a dubstudio CLI alhalmaza)
_RUNNABLE_STEPS = ("intake", "prepare", "cluster", "dub", "run", "asr")
#: a WebUI által közvetlenül nem indítható, de belsőleg használt lépések
_INTERNAL_STEPS = (*_RUNNABLE_STEPS, "youtube")


def _core():
    """A dubstudio importja lustán történik (numpy/sherpa nélkül is betölthető)."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import config, identity, review, speaker_audit, voice_decision
    return config, identity, review, speaker_audit, voice_decision


def _registry():
    """A szereplő-regiszter modell lustán (a hangzár-kiosztás súly-modellje)."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import registry
    return registry


def _library():
    """A sorozat/film könyvtárnézet lustán importált modulja."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import library
    return library


def _player():
    """A böngészős lejátszó-proxy lustán importált modulja."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import player
    return player


def _intake():
    """A nyers epizód intake (feliratfordítás + hangsáv-szétválasztás) modulja."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import intake
    return intake


def _youtube():
    """A YouTube-letöltő modul lustán importálva."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import youtube
    return youtube


def _newproject():
    """Az új projekt gyára (modellek, küszöbök, kapuk egy helyen)."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import newproject
    return newproject


def _project_key(name: str | None) -> str:
    """A kérésben szereplő projekt **fájlneve** (stem), vagy az első projekt.

    A felület a megjelenített nevet (``project.name``) használja, a CLI és a
    projektfájl viszont a stemet; ezért mindkettőt elfogadjuk. A név szerinti
    keresés csak néhány kis JSON-t olvas, és csak ismeretlen azonosítónál fut.
    """
    names = _project_names()
    if not names:
        return ""
    if name in names:
        return name
    wanted = str(name or "").strip().lower()
    if wanted:
        for stem in names:
            try:
                doc = json.loads((DUB_ROOT / "projects" / f"{stem}.json")
                                 .read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(doc.get("name") or "").strip().lower() == wanted:
                return stem
    return names[0]



def _cast_rows(project) -> dict[str, dict]:
    """A hangzár (cast) szereplő-rekordjai nyersen, modell-töltés nélkül.

    Itt van a **beszédidő-alapú szerep** (``role``: fő/mellék, a share-ből
    számítva) és a rangsor — amiből a kiosztás súlyozása származik.
    """
    try:
        data = json.loads(project.cast_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows: dict[str, dict] = {}
    for row in (data.get("characters") or []):
        cid = str(row.get("char_id") or "")
        if cid:
            rows[cid] = row
    return rows


def _project_names() -> list[str]:
    folder = DUB_ROOT / "projects"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json"))


def _project(name: str | None):
    config, *_ = _core()
    chosen = _project_key(name)
    if not chosen:
        abort(404, description="Nincs dubstudio projekt (projects/*.json).")
    return config.load_project(DUB_ROOT / "projects" / f"{chosen}.json")


def _tag(value: str | None) -> str:
    tag = str(value or "").strip().upper()
    if not _TAG_RE.match(tag):
        abort(400, description="Érvénytelen epizód-azonosító (pl. E06).")
    return tag


def _audit_summary(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {
        "file": path.name,
        "start": data.get("start"),
        "duration": data.get("duration"),
        "audio_stream": data.get("audio_stream"),
        "speakers": len((data.get("speaker_identity") or {}).get("speakers") or {}),
        "words": len(data.get("words") or []),
        "generated_at": (data.get("speaker_identity") or {}).get("generated_at"),
    }


@bp.route("/szinkron")
def szinkron_page():
    return render_template("szinkron.html")


@bp.route("/api/dub/projects")
def projects():
    out = []
    for name in _project_names():
        try:
            project = _project(name)
        except Exception as exc:                      # pragma: no cover - hibás json
            out.append({"id": name, "name": name, "error": str(exc)})
            continue
        episodes = sorted(p.parent.name for p in project.work.glob("*/utterances.json"))
        audits = sorted(p.parent.parent.name for p in
                        project.work.glob("*/source_audit/audit_*.json"))
        out.append({"id": name, "name": project.name, "season": project.season,
                    "kind": project.kind, "is_movie": project.kind == "movie",
                    "episodes": episodes, "audits": sorted(set(audits)),
                    "media_missing": [t for t in episodes if not _has_media(project, t)]})
    return jsonify({"root": str(DUB_ROOT), "projects": out})


# ── Sorozat/film könyvtár és növekményes epizódkezelés ───────────────────────


def _source_target(project, src: Path, tag: str) -> Path:
    """A forrás helye a projekt data_dir-jében."""
    if project.kind == "movie":
        return project.data_dir / src.name
    ep = int("".join(c for c in tag if c.isdigit()) or 0)
    return project.data_dir / f"S{project.season:02d}E{ep:02d} - {src.stem}{src.suffix.lower()}"


def _place_source(project, src: Path, tag: str) -> tuple[Path, str]:
    """Forrás ellenőrzése és hardlink/másolása a projektbe."""
    probe, err = _probe_movie(src)
    if probe is None:
        raise ValueError(err)
    target = _source_target(project, src, tag)
    if target.is_file():
        if target.resolve() == src.resolve():
            return target, "helyben"
        raise FileExistsError(f"A célfájl már létezik: {target}")
    try:
        existing = project.episode_media(tag)
    except (FileNotFoundError, OSError):
        existing = None
    if existing and Path(existing).resolve() != src.resolve():
        raise FileExistsError(f"Ehhez a taghez már van média: {existing}")
    project.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, target)
        mode = "hardlink"
    except OSError:
        try:
            shutil.copy2(src, target)
        except OSError as exc:
            raise OSError(f"A fájl elhelyezése nem sikerült: {exc}") from exc
        mode = "másolás"
    return target, mode


@bp.route("/api/dub/library")
def library_view():
    """Sorozat- vagy filmkönyvtár: mi van betöltve, mi hiányzik, mi kész."""
    project = _project(request.args.get("project"))
    key = _project_key(request.args.get("project"))
    if key:
        _ensure_watcher(key)
    return jsonify(_library().library(project))


@bp.route("/api/dub/series/scan", methods=["POST"])
def series_scan():
    """Új médiafájlok keresése a data_dir-ben vagy egy megadott mappában."""
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    folder = str(body.get("folder") or "").strip() or None
    try:
        candidates = _library().scan_sources(project, folder)
    except OSError as exc:
        return jsonify({"error": f"A mappa nem olvasható: {exc}"}), 400
    return jsonify({"candidates": candidates,
                    "new": [c for c in candidates if not c["known"]],
                    "library": _library().library(project)})


@bp.route("/api/dub/series/add", methods=["POST"])
def series_add():
    """Egy vagy több epizód/film hozzáadása a projekthez.

    Sorozatnál a tag az SxxExx / E12 mintából jön; ha a fájlnévből nem
    állapítható meg, a kérésben megadott ``episode`` használható (egy fájlnál).
    A művelet csak a médiafájlt helyezi el; a szinkronindítás külön lépés.
    """
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    raw_path = str(body.get("path") or "").strip().strip('"')
    if not raw_path:
        return jsonify({"error": "Add meg a fájl vagy a mappa útvonalát."}), 400
    src = Path(raw_path).expanduser()
    if not src.is_absolute():
        src = DUB_ROOT / src
    if src.is_file():
        files = [src]
    elif src.is_dir():
        files = sorted(p for p in src.iterdir()
                       if p.is_file() and p.suffix.lower() in _MOVIE_SUFFIXES)
    else:
        return jsonify({"error": f"A fájl/mappa nem található: {src}"}), 400
    if not files:
        return jsonify({"error": "A mappában nincs támogatott videó."}), 400
    explicit = str(body.get("episode") or "").strip().upper()
    added, skipped = [], []
    for path in files:
        if project.kind == "movie":
            tag = "E01"
        elif explicit and len(files) == 1:
            try:
                tag = _tag(explicit)
            except Exception:                       # noqa: BLE001 — magyar hiba
                return jsonify({"error": "Érvénytelen epizód-azonosító (pl. E12)."}), 400
        else:
            tag = _library().tag_from_name(path.name)
        if not tag:
            skipped.append({"path": str(path), "reason": "nincs felismerhető epizódszám"})
            continue
        try:
            target, mode = _place_source(project, path, tag)
        except FileExistsError as exc:
            skipped.append({"path": str(path), "tag": tag, "reason": str(exc)})
            continue
        except (ValueError, OSError) as exc:
            skipped.append({"path": str(path), "tag": tag, "reason": str(exc)})
            continue
        added.append({"tag": tag, "path": str(target), "mode": mode})
        # médiaszerverről jött fájl: jegyezzük fel, hová kell a kész szinkront
        # visszamásolni
        root = _media_root(project)
        if root is not None:
            try:
                path.resolve().relative_to(root)
            except ValueError:
                pass
            else:
                from dubstudio import deliver
                deliver.record(project, tag, path)
    return jsonify({"ok": bool(added), "added": added, "skipped": skipped,
                    "library": _library().library(project)})


# ── médiaszerver: fájlválasztás, listázás, visszamásolás ────────────────────

_PICK_MUTEX = threading.Lock()


def _project_doc_path(name: str) -> Path:
    key = _project_key(name)
    if not key:
        abort(404, description="Nincs dubstudio projekt.")
    return DUB_ROOT / "projects" / f"{key}.json"


def _media_root(project) -> Path | None:
    """A beállított médiaszerver-mappa (ha van és elérhető-e)."""
    raw = str((project.media_server or {}).get("path") or "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except OSError:
        return None


@bp.route("/api/dub/media/config", methods=["GET", "POST"])
def media_config():
    """A médiaszerver mappa és a visszamásolás beállítása (projekt JSON)."""
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project") if request.method == "POST"
                       else request.args.get("project"))
    if request.method == "GET":
        root = _media_root(project)
        return jsonify({"path": str((project.media_server or {}).get("path") or ""),
                        "deliver": bool((project.media_server or {}).get("deliver")),
                        "exists": bool(root and root.is_dir())})
    raw = str(body.get("path") or "").strip()
    path = Path(raw).expanduser() if raw else None
    if path is not None and not path.is_dir():
        return jsonify({"error": f"A mappa nem érhető el: {raw}"}), 400
    doc_path = _project_doc_path(body.get("project"))
    try:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return jsonify({"error": "A projektfájl nem olvasható."}), 500
    doc["media_server"] = {"path": str(path) if path is not None else "",
                           "deliver": bool(body.get("deliver"))}
    _write_project_json(doc_path, doc)
    return jsonify({"ok": True, "path": doc["media_server"]["path"],
                    "deliver": doc["media_server"]["deliver"],
                    "exists": bool(path is not None and path.is_dir())})


@bp.route("/api/dub/media/list")
def media_list():
    """A médiaszerver egy mappájának videói (alkönyvtárakkal)."""
    project = _project(request.args.get("project"))
    root = _media_root(project)
    if root is None:
        return jsonify({"error": "Nincs beállítva médiaszerver-mappa."}), 400
    raw = str(request.args.get("path") or "").strip()
    base = Path(raw).expanduser() if raw else root
    try:
        base = base.resolve()
        base.relative_to(root)
    except (OSError, ValueError):
        return jsonify({"error": "A mappa a médiaszerver gyökerén kívül van."}), 400
    if not base.is_dir():
        return jsonify({"error": f"A mappa nem érhető el: {base}"}), 400
    dirs, files = [], []
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return jsonify({"error": f"A mappa nem olvasható: {exc}"}), 400
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append({"name": entry.name, "path": str(entry)})
        elif entry.suffix.lower() in _MOVIE_SUFFIXES:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            files.append({"name": entry.name, "path": str(entry),
                          "size": size,
                          "tag": _library().tag_from_name(entry.name)})
    parent = "" if base == root else str(base.parent)
    return jsonify({"root": str(root), "path": str(base), "parent": parent,
                    "dirs": dirs, "files": files})


@bp.route("/api/dub/media/deliver", methods=["POST"])
def media_deliver():
    """A kész szinkron visszamásolása az eredeti médiafájl helyére."""
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    from dubstudio import deliver, media
    out = project.out / media.safe_name(tag, project)
    try:
        info = deliver.copy_back(project, tag, out)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": f"A visszamásolás nem sikerült: {exc}"}), 500
    return jsonify({"ok": True, **info})


@bp.route("/api/dub/pick", methods=["POST"])
def pick_path():
    """Helyi fájl- vagy mappaválasztó a szerver gépén (a böngésző nem ad
    valódi útvonalat). A választó külön folyamatban fut, így a Tk nem kerül
    a Flask szálába."""
    body = request.get_json(silent=True) or {}
    kind = "dir" if str(body.get("kind") or "file") == "dir" else "file"
    initial = str(body.get("initial") or "").strip()
    title = str(body.get("title") or ("Mappa kiválasztása" if kind == "dir"
                                       else "Videófájl kiválasztása"))
    if kind == "dir":
        call = (f"filedialog.askdirectory(title={title!r}, initialdir={initial!r}, "
                f"mustexist=True)")
    else:
        call = (f"filedialog.askopenfilename(title={title!r}, initialdir={initial!r}, "
                f"filetypes=[('Videó', '*.mkv *.mp4 *.avi *.mov *.webm *.m2ts *.ts *.m4v'), "
                f"('Minden fájl', '*.*')])")
    script = (
        "import json, tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        f"picked = {call}\n"
        "root.destroy()\n"
        "print(json.dumps(picked or ''))\n"
    )
    if not _PICK_MUTEX.acquire(blocking=False):
        return jsonify({"error": "Már nyitva van egy fájlválasztó."}), 409
    try:
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=900,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return jsonify({"error": "A fájlválasztó nem zárt be (időtúllépés)."}), 504
    finally:
        _PICK_MUTEX.release()
    if proc.returncode != 0:
        return jsonify({"error": "A fájlválasztó nem indult el: "
                                 f"{(proc.stderr or '').strip()[-300:]}"}), 500
    try:
        picked = json.loads((proc.stdout or "").strip().splitlines()[-1] or '""')
    except (ValueError, IndexError):
        picked = ""
    return jsonify({"ok": True, "path": picked or ""})


def _movie_project_doc(template, name: str, slug: str, source_lang: str) -> dict:
    """Egy új filmprojekt JSON-ja a sablon beállításaiból.

    A modellek abszolút útvonalat kapnak, mert az új film gyökere más lesz,
    mint a sabloné. A hangfelismerés beállításai öröklődnek, a fordítás
    viszont mindig a helyi LLM-re áll (a repó nem küld szöveget külső
    szolgáltatásnak), a vokál/M&E pedig az intake alaphelyére kerül
    (``work/<EP>/``).

    **Sablon nélkül** (üres ``projects/``, azaz friss klón) a dubstudio
    ``newproject`` gyára adja az alapértékeket — így az első film is
    működőképes beállításokkal jön létre.
    """
    if template is None:
        return _newproject().project_doc(name, "movie", 1, source_lang)
    models = {}
    for key, value in (template.models or {}).items():
        try:
            models[key] = str(template.model(key))
        except Exception:                       # noqa: BLE001 — hibás útvonal
            models[key] = str(value)
    intake_doc = dict(template.intake or {})
    sep = dict(intake_doc.get("separate") or {})
    sep["audio_stream"] = 0
    if not sep.get("model_dir"):
        sep["model_dir"] = str(template.root / "models")
    intake_doc["separate"] = sep
    # A sablonból csak a hangolható számok jönnek át; a végpont és a modell
    # nem, mert egy régi projekt külső szolgáltatásra is állhat — az új film
    # fordítása mindig helyi modellé.
    old_tr = dict(intake_doc.get("translate") or {})
    tr = dict(_newproject().TRANSLATE_DEFAULTS)
    for key in ("enabled", "batch", "timeout", "max_tokens"):
        if old_tr.get(key) is not None:
            tr[key] = old_tr[key]
    tr["source_lang"] = source_lang
    intake_doc["translate"] = tr
    # A sablon diarizációs beállításait örökítjük, de a sorozatra kalibrált
    # beszélőszámot (num_speakers / num_speakers_by_ep) nem: egy film
    # szereplőszáma más, és a rossz fix szám kényszerítve szétvágja a
    # szereplőket. Hiányuk esetén a diarizáció automatikus (küszöb-alapú);
    # a film saját számát a kaszt ismeretében lehet beállítani.
    diar_doc = {k: v for k, v in (template.diarization or {}).items()
                if k not in ("num_speakers", "num_speakers_by_ep")}
    return {
        "name": name,
        "season": 1,
        "kind": "movie",
        "root": f"../movies/{slug}",
        # külön munkakönyvtár, hogy két film E01-je ne írja felül egymást
        "cache_dir": f"../work/{slug}",
        "data_dir": "data",
        "subs_dir": "subs",
        "vocals_globs": ["{tag}/vocals.wav"],
        "me_globs": ["{tag}/me.wav"],
        "source_lang": source_lang,
        # filmnél az első hangsáv a forrás; a sablon epizódjaknál ez lehet 1
        "original_audio": 0,
        "tts_sr": template.tts_sr,
        "models": models,
        "tuning": dict(template.tuning or {}),
        "gate": dict(template.gate or {}),
        "llm": dict(template.llm or {}),
        "asr": dict(template.asr or {}),
        "diarization": diar_doc,
        "intake": intake_doc,
        "series": {"auto_process": False, "poll_s": 30},
    }


def _slugify(name: str) -> str:
    """Projekt-ID: ékezet nélküli, kisbetűs, kötőjeles (max 48 karakter)."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:48] or "film"


def _youtube_url(value: str) -> str:
    """Csak nyilvános YouTube URL mehet a letöltőnek (SSRF-védelem)."""
    url = str(value or "").strip()
    return url if _youtube().is_youtube_url(url) else ""


def _youtube_video_id(url: str) -> str:
    """A videó azonosítója a projekt alapértelmezett nevéhez."""
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/").split("/")[0][:24]
    value = (parse_qs(parsed.query).get("v") or [""])[0]
    return value[:24]


@bp.route("/api/dub/movies", methods=["POST"])
def create_movie_project():
    """Új filmprojekt egyetlen fájlból, majd intake + első szinkron indítása.

    A varázsló a kiválasztott projekt beállításait (modellek, kapuk, LLM)
    örökli, de saját média- és work-könyvtárat kap. A fájl hardlinkkel vagy
    másolással kerül a projekt ``data/`` mappájába; utána az intake fordítja
    a feliratot és választja szét a hangot, végül a szinkronlánc elindul.
    """
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Add meg a film nevét."}), 400
    if len(name) > 80:
        return jsonify({"error": "A film neve legfeljebb 80 karakter lehet."}), 400
    raw_path = str(body.get("path") or "").strip().strip('"')
    if not raw_path:
        return jsonify({"error": "Add meg a filmfájl útvonalát."}), 400
    src = Path(raw_path).expanduser()
    if not src.is_absolute():
        src = DUB_ROOT / src
    if not src.is_file():
        return jsonify({"error": f"A filmfájl nem található: {src}"}), 400
    if src.suffix.lower() not in _MOVIE_SUFFIXES:
        return jsonify({"error": "Támogatott videó kell: mkv, mp4, avi, mov, webm, m2ts, ts, m4v."}), 400
    slug = _slugify(name)
    dest = DUB_ROOT / "projects" / f"{slug}.json"
    if dest.exists():
        return jsonify({"error": f"Már van ilyen projekt: {slug}."}), 409
    # üres projects/ (friss klón): nincs sablon, a dubstudio gyára adja az alapot
    template = _project(body.get("template") or None) if _project_names() else None
    default_lang = template.source_lang if template is not None else "eng"
    lang = str(body.get("source_lang") or default_lang or "heb").strip().lower()
    lang = "heb" if lang in ("he", "heb", "héber") else "eng"
    doc = _movie_project_doc(template, name, slug, lang)
    try:
        _write_project_json(dest, doc)
    except OSError as exc:
        return jsonify({"error": f"A projektfájl nem írható: {exc}"}), 500
    config, *_ = _core()
    project = config.load_project(dest)
    try:
        project.data_dir.mkdir(parents=True, exist_ok=True)
        project.subs_dir.mkdir(parents=True, exist_ok=True)
        target, mode = _place_source(project, src, "E01")
    except (ValueError, FileExistsError, OSError) as exc:
        dest.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    warning = ""
    job = None
    try:
        job = _launch_job("intake", slug, ["E01"], ["--then-run", "--allow-fail"])
    except _RunBusy as exc:
        warning = f"A projekt elkészült, de most más fut: {exc}"
    except _RunFailed as exc:
        warning = f"A projekt elkészült, de az intake nem indult: {exc}"
    return jsonify({"ok": True, "project": {"id": slug, "name": name,
                                             "kind": "movie", "is_movie": True},
                    "movie": str(target), "mode": mode, "job": job,
                    "warning": warning})


@bp.route("/api/dub/subtitle", methods=["POST"])
def upload_subtitle():
    """Külső SRT/VTT felirat feltöltése a kiválasztott epizódhoz.

    A magyar felirat a kész szinkronszöveg; az angol/héber a fordítás forrása.
    ``then_run`` esetén a feltöltés után a szokásos intake + szinkron indul,
    így a felhasználónak nem kell külön lépést indítania.
    """
    body = request.form
    project = _project(body.get("project"))
    key = _project_key(body.get("project"))
    tag = str(body.get("episode") or "").strip().upper()
    if not _TAG_RE.match(tag):
        return jsonify({"error": "Érvénytelen epizód-azonosító (pl. E04)."}), 400
    lang = str(body.get("lang") or "eng").strip().lower()
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Válassz feliratfájlt (.srt vagy .vtt)."}), 400
    replace = str(body.get("replace") or "").lower() in ("1", "true", "on", "yes")
    try:
        result = subtitle_upload.save_subtitle(
            project, tag, lang, upload.stream, upload.filename, replace=replace)
    except FileExistsError as exc:
        return jsonify({"error": str(exc)}), 409
    except subtitle_upload.SubtitleUploadError as exc:
        return jsonify({"error": str(exc)}), 400
    warning = ""
    job = None
    if str(body.get("then_run") or "").lower() in ("1", "true", "on", "yes"):
        try:
            job = _launch_job("intake", key, [tag], ["--then-run", "--allow-fail"])
        except _RunBusy as exc:
            warning = f"A felirat elmentve, de most más fut: {exc}"
        except _RunFailed as exc:
            warning = f"A felirat elmentve, de az intake nem indult: {exc}"
    return jsonify({"ok": True, **result, "project": key, "job": job,
                    "warning": warning})


@bp.route("/api/dub/youtube", methods=["POST"])
def create_youtube_project():
    """YouTube URL-ből filmprojekt: letöltés, felirat, majd automatikus szinkron.

    A projektet a kiválasztott sablonból hozza létre (mint a filmvarázsló;
    sablon nélkül a dubstudio gyára adja az alapot), a letöltést és az
    intake+szinkron láncot pedig külön háttérfolyamat végzi, hogy a hosszú
    videó ne blokkolja a webes kérést.
    """
    body = request.get_json(silent=True) or {}
    url = _youtube_url(body.get("url"))
    if not url:
        return jsonify({"error": "Csak nyilvános YouTube URL adható meg."}), 400
    name = str(body.get("name") or "").strip()
    if not name:
        video_id = _youtube_video_id(url)
        name = f"YouTube {video_id}".strip() if video_id else "YouTube videó"
    if len(name) > 80:
        return jsonify({"error": "A cím legfeljebb 80 karakter lehet."}), 400
    slug = _slugify(name)
    dest = DUB_ROOT / "projects" / f"{slug}.json"
    if dest.exists():
        return jsonify({"error": f"Már van ilyen projekt: {slug}."}), 409
    # üres projects/ esetén sablon nélkül, a dubstudio gyárából épül a projekt
    template = _project(body.get("template") or None) if _project_names() else None
    lang = str(body.get("source_lang") or "eng").strip().lower()
    lang = "heb" if lang in ("he", "heb", "héber") else "eng"
    doc = _movie_project_doc(template, name, slug, lang)
    try:
        _write_project_json(dest, doc)
    except OSError as exc:
        return jsonify({"error": f"A projektfájl nem írható: {exc}"}), 500
    config, *_ = _core()
    project = config.load_project(dest)
    project.data_dir.mkdir(parents=True, exist_ok=True)
    project.subs_dir.mkdir(parents=True, exist_ok=True)
    warning = ""
    job = None
    try:
        job = _launch_job("youtube", slug, ["E01"],
                          ["--url", url, "--source-lang", lang,
                           "--then-run", "--allow-fail"])
    except _RunBusy as exc:
        warning = f"A projekt elkészült, de most más fut: {exc}"
    except _RunFailed as exc:
        warning = f"A projekt elkészült, de a letöltés nem indult: {exc}"
    return jsonify({"ok": True, "project": {"id": slug, "name": name,
                                             "kind": "movie", "is_movie": True},
                    "url": url, "job": job, "warning": warning})


@bp.route("/api/dub/series/process", methods=["POST"])
def series_process():
    """A kiválasztott (vagy minden kész) epizód szinkronjának indítása."""
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    key = _project_key(body.get("project"))
    lib = _library().library(project)
    wanted = [str(e).strip().upper() for e in (body.get("episodes") or []) if str(e).strip()]
    if wanted:
        episodes = [e for e in lib["episodes"] if e["tag"] in wanted]
    else:
        # a meglévő kimenet lezárja a részt; QA-bukásnál sem indul újra
        episodes = [e for e in lib["episodes"] if e["ready"] and not e["output"]]
    if not episodes:
        return jsonify({"error": "Nincs feldolgozható epizód (hiányzó felirat/vokál/M&E)."}), 400
    missing = {e["tag"]: e["missing"] for e in episodes if not e["ready"]}
    if missing:
        return jsonify({"error": "Hiányzó előkészített anyag.", "missing": missing}), 400
    flags, err = _run_flags(body)
    if err:
        return jsonify({"error": err}), 400
    if project.kind != "movie":
        flags = [*flags, "--season"]
    try:
        job = _launch_job("run", key, [e["tag"] for e in episodes], flags)
    except _RunBusy as exc:
        return jsonify({"error": str(exc)}), 409
    except _RunFailed as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, **job})


@bp.route("/api/dub/series/watch", methods=["GET", "POST"])
def series_watch():
    """Automatikus új epizód figyelés be/ki + állapot."""
    if request.method == "GET":
        project = _project(request.args.get("project"))
        key = _project_key(request.args.get("project"))
        if key:
            _ensure_watcher(key)
        return jsonify({"ok": True,
                        "watch": _library().library(project)["watch"],
                        "watcher": _WATCH_STATUS.get(key) or {},
                        "job": _run_status_payload()})
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    key = _project_key(body.get("project"))
    if not key:
        return jsonify({"error": "Nincs dubstudio projekt."}), 404
    path = _project_path(key)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"A projektfájl nem olvasható: {exc}"}), 500
    series = dict(doc.get("series") or {})
    series["auto_process"] = bool(body.get("enabled"))
    try:
        poll_s = int(body.get("poll_s") or series.get("poll_s") or 30)
    except (TypeError, ValueError):
        poll_s = 30
    series["poll_s"] = max(10, min(600, poll_s))
    doc["series"] = series
    try:
        _write_project_json(path, doc)
    except OSError as exc:
        return jsonify({"error": f"A beállítás nem menthető: {exc}"}), 500
    _ensure_watcher(key)
    project = _project(key)
    return jsonify({"ok": True, "watch": _library().library(project)["watch"]})


@bp.route("/api/dub/intake", methods=["POST"])
def intake_run():
    """Nyers epizód előkészítése: felirat fordítás + vokál/M&E szétválasztás.

    A folyamat a közös futtatógépezeten indul, ezért a 4. szakasz naplója
    mutatja a haladást. ``then_run`` esetén az intake után ugyanabban a
    folyamatban elindul a szinkronlánc is (a filmvarázsló ezt használja).
    """
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    key = _project_key(body.get("project"))
    if not key:
        return jsonify({"error": "Nincs dubstudio projekt."}), 404
    wanted = [str(e).strip().upper() for e in (body.get("episodes") or [])
              if str(e).strip()]
    tags = []
    if wanted:
        tags = [_tag(tag) for tag in wanted]
    else:
        # a média már megvan, de a felirat/vokál/M&E hiányzik: ezeket vesszük
        for episode in _library().library(project)["episodes"]:
            if episode.get("media") and not episode.get("ready") and not episode.get("output"):
                tags.append(episode["tag"])
        if not tags:
            tags = [_library().library(project)["next_tag"]]
    flags: list[str] = []
    if body.get("translate") is False:
        flags.append("--no-translate")
    if body.get("separate") is False:
        flags.append("--no-separate")
    if body.get("force"):
        flags.append("--force")
    if body.get("then_run"):
        flags += ["--then-run", "--allow-fail"]
    try:
        job = _launch_job("intake", key, tags, flags)
    except _RunBusy as exc:
        return jsonify({"error": str(exc)}), 409
    except _RunFailed as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "episodes": tags, **job})


def _english_cues(project, tag: str):
    """Az angol (dubtitle) felirat sávjai — a jelenet szövege a kártyákhoz.

    A héber beszéd a filmben felirat marad; a szinkronhang az angol sávhoz
    igazodik, ezért a felülvizsgálónak az angol sor kell a döntéshez.
    """
    _config, _identity, _review, speaker_audit, _vd = _core()
    out = project.ep_dir(tag) / "source_audit" / "source_english.srt"
    try:
        path = speaker_audit.extract_subtitles(project, tag, out, lang="en",
                                               prefer_title="Dubtitle")
    except Exception:  # noqa: BLE001 — angol sáv nélkül is mûködik a felület
        return None
    from dubstudio import subs
    try:
        return subs.parse_srt(path)
    except Exception:  # noqa: BLE001
        return None


def _frame_time(spans, start: float) -> float | None:
    """A beszélő leghosszabb fordulójának közepe (abszolút filmidő)."""
    best, best_len = None, 0.0
    for span in spans or []:
        try:
            a, b = float(span[0]), float(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b - a > best_len:
            best, best_len = (a + b) / 2.0, b - a
    return None if best is None else round(start + best, 2)


def _speaker_lines(cues, spans, start: float, limit: int = 4) -> list[dict]:
    """A beszélő fordulóival átfedő angol sorok (ki mit mond, mikor)."""
    if not cues or not spans:
        return []
    words = []
    for span in spans or []:
        try:
            a, b = float(span[0]) + start, float(span[1]) + start
        except (TypeError, ValueError, IndexError):
            continue
        for cue in cues:
            overlap = min(b, cue.end) - max(a, cue.start)
            if overlap >= 0.15:
                words.append((round(max(a, cue.start) - start, 2),
                              round(min(b, cue.end) - start, 2), cue.text))
    words.sort(key=lambda x: x[0])
    seen, out = set(), []
    for a, b, text in words:
        if text in seen:
            continue
        seen.add(text)
        out.append({"start": a, "end": b, "text": text})
        if len(out) >= limit:
            break
    return out


def _has_media(project, tag: str) -> bool:
    try:
        return project.episode_media(tag).is_file()
    except (FileNotFoundError, OSError):
        return False


@bp.route("/api/dub/state")
def state():
    config, identity, review_mod, speaker_audit, voice_decision = _core()
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode") or
               next((p.parent.name for p in project.work.glob("*/utterances.json")), ""))
    # a legfrissebb próba legyen az alapértelmezett (a névsorrend félrevezet:
    # "audit_32_240.json" < "audit_32_30.json")
    audits = sorted((project.ep_dir(tag) / "source_audit").glob("audit_*.json"),
                    key=lambda p: (p.stat().st_mtime, p.name))
    wanted = request.args.get("audit")
    audit_path = None
    if wanted:
        candidate = project.ep_dir(tag) / "source_audit" / Path(wanted).name
        audit_path = candidate if candidate.is_file() else None
    if audit_path is None and audits:
        audit_path = audits[-1]

    total_all = float(sum(float(r.get("duration") or 0.0) for r in _cast_rows(project).values())) \
        or 1.0
    payload: dict = {
        "project": {"name": project.name, "season": project.season,
                    "kind": project.kind, "is_movie": project.kind == "movie",
                    "root": str(project.root), "media": _has_media(project, tag)},
        "episode": tag,
        "audits": [_audit_summary(p) for p in audits],
        "characters": [],
        "review": review_mod.summary(review_mod.load(project, tag)),
        "audit": None,
        "speakers": [],
        "words": [],
        "timbres": _timbre_status(project),
    }
    records = identity.char_records(project.cast_file, project.work / "identity-gallery.json")
    cast_rows = _cast_rows(project)
    payload["characters"] = sorted(
        ({"char_id": cid, "name": row.get("name"), "gender": row.get("gender"),
          "voice": row.get("voice"),
          "episodes": row.get("episodes") or [],
          "origin": row.get("origin"),
          "duration": round(float(row.get("duration") or 0.0), 1),
          "share": round(float(row.get("duration") or 0.0) / total_all, 4),
          "role": (cast_rows.get(cid) or {}).get("role"),
          "role_override": (cast_rows.get(cid) or {}).get("role_override"),
          "rank": (cast_rows.get(cid) or {}).get("rank") or 0,
          "pitch": (cast_rows.get(cid) or {}).get("pitch"),
          "pitch_override": (cast_rows.get(cid) or {}).get("pitch_override"),
          "pitch_auto": (cast_rows.get(cid) or {}).get("pitch_auto"),
          "exclusive": bool((cast_rows.get(cid) or {}).get("exclusive", True))}
         for cid, row in records.items()),
        key=lambda c: (-(c["rank"] or 0) if c.get("rank") else 1e9, c["char_id"]))
    from dubstudio import voices as voices_mod
    payload["voice_pool"] = [
        {"vid": v.vid, "label": v.label, "engine": v.engine, "gender": v.gender,
         "model_present": _model_ok(project, v.engine),
         "users": [{"char_id": str(cid),
                    "role": (cast_rows.get(str(cid)) or {}).get("role"),
                    "pitch": (cast_rows.get(str(cid)) or {}).get("pitch")}
                   for cid, row in records.items() if row.get("voice") == v.vid]}
        for v in voices_mod.CATALOG]
    payload["stages"] = _stage_states(project, tag, review_mod, records)

    if audit_path is None:
        return jsonify(payload)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ident = audit.get("speaker_identity") or {}
    merged = review_mod.apply(ident, review_mod.load(project, tag))
    payload["audit"] = {
        "file": audit_path.name, "start": audit.get("start"),
        "duration": audit.get("duration"),
        "audio_stream": audit.get("audio_stream"),
        "method": audit.get("method") or {},
        "limitations": audit.get("limitations") or [],
        "subtitle_segments": len(audit.get("subtitle_segments") or []),
        "turns": len(audit.get("speaker_turns") or []),
        "candidate_cast_links": audit.get("candidate_cast_links") or [],
        "counts": merged.get("counts") or {},
        "pending_review": merged.get("pending_review", 0),
    }
    payload["words"] = audit.get("words") or []
    payload["source_language"] = audit.get("language")
    cues = _english_cues(project, tag)

    timbres = voice_decision.load_timbres(project.work / "voice-timbres.json")
    usage, cooccur = _cast_context(project, tag, audit)
    main_share = float((project.tuning or {}).get("main_share_min", 0.04))
    total_share = {cid: float(row.get("duration") or 0.0) for cid, row in records.items()}

    speakers = []
    for speaker, entry in sorted((merged.get("speakers") or {}).items()):
        char = records.get(str(entry.get("char_id"))) or {}
        vector = entry.get("vector")
        cosines = voice_decision.cosine_to_prototype(vector, timbres) if vector else {}
        char_view = {
            "char_id": entry.get("char_id") or entry.get("suggested_new_id"),
            "gender": entry.get("gender") or char.get("gender"),
            "voice": entry.get("voice") or char.get("voice"),
            "main": bool(total_share.get(str(entry.get("char_id")), 0.0) and
                         total_share[str(entry.get("char_id"))] /
                         max(sum(total_share.values()), 1e-6) >= main_share),
            "evidence_seconds": entry.get("evidence_seconds") or 0.0,
        }
        proposal = None
        if entry.get("char_id"):
            proposal = voice_decision.decide_voice(
                char_view, timbres=cosines, usage=usage,
                cooccurring=cooccur.get(str(entry.get("char_id"))))
        spans = entry.get("spans") or []
        speakers.append({
            "speaker": speaker,
            "name": entry.get("name") or char.get("name"),
            "frame_t": _frame_time(spans, start=float(audit.get("start") or 0.0)),
            "lines": _speaker_lines(cues, spans, float(audit.get("start") or 0.0)),
            "status": entry.get("status"),
            "reason": entry.get("reason"),
            "char_id": entry.get("char_id"),
            "suggested_new_id": entry.get("suggested_new_id"),
            "gender": entry.get("gender"),
            "voice": entry.get("voice"),
            "decision": entry.get("decision"),
            "manual_note": entry.get("manual_note") or "",
            "review_required": entry.get("review_required", True),
            "evidence_seconds": entry.get("evidence_seconds") or 0.0,
            "evidence_turns": entry.get("evidence_turns") or 0,
            "spans": entry.get("spans") or [],
            "candidates": (entry.get("candidates") or [])[:3],
            "proposal": proposal,
        })
    payload["speakers"] = speakers
    payload["cast_context"] = {"voices_in_use": {k: v for k, v in usage.items() if v},
                               "cooccurrence_known": len(cooccur)}
    payload["casting"] = _casting_view(speakers, records, _cast_rows(project),
                                       total_share, main_share)
    payload["char_frames"] = _char_frames(records, merged, tag,
                                           float(audit.get("start") or 0.0))
    payload["sound"] = _sound_view(project)
    # a lépés-állapottáblát a valódi pending-számmal frissítjük (a korai
    # betöltésnél nincs próba, ott None volt)
    payload["stages"] = _stage_states(
        project, tag, review_mod, records,
        pending=int(merged.get("pending_review") or 0),
        casting=payload["casting"])
    return jsonify(payload)


#: noul-küszöb (emberi döntés valószínűsége) és válasz-valószínűség fölött az
#: automatikus kötegelt elfogadás felelős — alatta a döntés a felületre kerül
CAST_NOUL_OK = 0.40
CAST_CONF_OK = 0.50


def _char_frames(records: dict, merged: dict, tag: str, start: float) -> dict[str, float]:
    """Karakter → képkocka-idő: a galéria beszélő-indexeiből az audit fordulóiig.

    A galéria ``members`` mezője epizódonként a beszélő-sorszámokat tartja
    (``{"E05": [1]}`` = ``SPEAKER_01``). Az audit beszélő-bejegyzéseinek
    fordulóiból vesszük a leghosszabb középpontját, így a szereplő-kártyán
    akkor is látszik arc, ha még senki nincs kézzel azonosítva.
    """
    entries = (merged.get("speakers") or {})
    out: dict[str, float] = {}
    for cid, row in (records or {}).items():
        idxs = (row.get("members") or {}).get(tag) or []
        for idx in idxs:
            entry = entries.get(f"SPEAKER_{int(idx):02d}")
            if not entry:
                continue
            t = _frame_time(entry.get("spans"), start=start)
            if t is not None:
                out[str(cid)] = t
                break
    return out


def _casting_view(speaker_view: list[dict], records: dict, cast_rows: dict,
                  total_share: dict[str, float], main_share: float) -> dict:
    """A hangkiosztás **döntési** nézete: nyitott / biztos / kiosztott.

    A cél, hogy az ember csak ott nyúljon bele, ahol az automatizmus nem
    biztos: minden más automatikusan rendezi a beszédidő-súly (szereplő-
    gyakoriság) alapján. Egy beszélő:
      * „open“   — döntést kér (nincs hangja ÉS bizonytalan a javaslat,
        VAGY a javaslat eltér a már meglévő hangjától);
      * „ready“  — a javaslat biztos (noul alacsony, válasz-valószínűség
        magas) és nincs még hangja → kötegelt elfogadás alighol módó;
      * „assigned“ — van hangja és a javaslat is ugyanaz (automatizmus
        tartás) → nem kér döntést;
    Az elutasított beszélők nem szerepelnek (nincs sem hang, sem döntés).
    """
    total = float(sum(total_share.values())) or 1.0
    open_rows, ready_rows, assigned_rows = [], [], []
    for sp in speaker_view:
        if sp.get("status") == "rejected_by_human":
            continue
        cid = str(sp.get("char_id") or "")
        char = records.get(cid) or {}
        dur = float(total_share.get(cid) or 0.0)
        ovr = (cast_rows.get(cid) or {}).get("role_override")
        weight = {
            "role": ovr if ovr in ("fő", "mellék") else cast_role(dur, total, main_share),
            "share": round(dur / total, 4),
            "duration": round(float(char.get("duration") or dur), 1),
        }
        proto = {
            "speaker": sp.get("speaker"), "char_id": sp.get("char_id"),
            "name": sp.get("name"), "gender": sp.get("gender"),
            "frame_t": sp.get("frame_t"),
            "evidence_seconds": round(float(sp.get("evidence_seconds") or 0.0), 1),
            **weight,
        }
        proposal = sp.get("proposal")
        if not proposal:
            if sp.get("voice"):
                assigned_rows.append({**proto, "voice": sp["voice"]})
            continue
        choice = proposal.get("choice")
        voice_now = sp.get("voice")
        top = [(k, round(float(v) * 100, 1))
               for k, v in list((proposal.get("probabilities") or {}).items())[:6]]
        row = {**proto, "voice": voice_now,
               "choice": choice, "choice_label": proposal.get("label"),
               "score": proposal.get("score"),
               "answer_confidence": proposal.get("answer_confidence"),
               "confidence": proposal.get("confidence"),
               "noul": proposal.get("noul") or {},
               "options": top,
               "hint": proposal.get("hint")}
        if voice_now and choice == voice_now:
            assigned_rows.append(row)
        elif not voice_now and \
                float(row["noul"].get("human_review", 1.0)) < CAST_NOUL_OK and \
                float(row["answer_confidence"] or 0.0) >= CAST_CONF_OK:
            ready_rows.append(row)
        else:
            open_rows.append(row)
    for rows in (open_rows, ready_rows, assigned_rows):
        rows.sort(key=lambda r: (-(r["share"] or 0.0), -(r["evidence_seconds"] or 0.0)))
    return {"open": open_rows, "ready": ready_rows, "assigned": assigned_rows,
            "summary": {"n_open": len(open_rows), "n_ready": len(ready_rows),
                        "n_assigned": len(assigned_rows),
                        **cast_counts(ready_rows, open_rows, assigned_rows)}}


def cast_role(duration: float, total: float, main_share: float) -> str:
    """A beszédidő-arányból: „fő“ vagy „mellék“ (a hangzár szabálya)."""
    return "fő" if (duration and total > 0 and duration / total >= main_share) else "mellék"


def cast_counts(ready_rows, open_rows, assigned_rows) -> dict:
    """N főszereplő / M mellékszereplő a döntési nézetben (súly-összesítés)."""
    rows = list(ready_rows) + list(open_rows) + list(assigned_rows)
    return {"n_main": sum(1 for r in rows if r.get("role") == "fő"),
            "n_side": sum(1 for r in rows if r.get("role") != "fő")}


def _stage_states(project, tag: str, review_mod, records: dict,
                  pending: int | None = None, casting: dict | None = None) -> list[dict]:
    """A filmszinkron workflow lépéseinek állapota — a vizuális követéshez.

    A lépések a valós pipeline-t tükrözik (betöltés → előkészítés →
    beszélő-ellenőrzés → évad-hangzár → hangkiosztás → szinkron → QA → mux);
    az állapot **a fájlrendszerből és a review-állapotból** számolva, nem
    becslésből: ami nincs a lemezen, az nincs kész. ``pending`` a
    felülvizsgálandó beszélők száma a legfrissebb próba szerint (None = nincs
    próba); ``casting`` a hangkiosztás döntési nézete (nyitott/biztos/
    kiosztott), amiből a hangkiosztás lépés info-ja készül.
    """
    try:
        ep = int("".join(c for c in tag if c.isdigit()) or 0)
    except (TypeError, ValueError):
        ep = 0
    ep_dir = project.ep_dir(tag)
    try:
        media = project.episode_media(tag).is_file()
    except (FileNotFoundError, OSError):
        media = False
    audits = sorted((ep_dir / "source_audit").glob("audit_*.json"))
    prepared = (ep_dir / "utterances.json").is_file()
    summary = review_mod.summary(review_mod.load(project, tag))
    reviewed = int(summary.get("total") or 0)
    total_chars = len(records)
    chars_with_voice = sum(1 for r in records.values() if r.get("voice"))
    dub_wav = (ep_dir / "dub.wav").is_file()
    report = (ep_dir / "report.json").is_file()
    out_dir = project.out
    out_mkv = bool(out_dir.is_dir() and any(
        p.name and (project.kind == "movie" or f"E{ep:02d}" in p.name)
        for p in out_dir.glob("*.mkv")))

    def row(step_id, label, state, info="") -> dict:
        return {"id": step_id, "label": label, "state": state, "info": info}

    if pending == 0:
        review_state, review_info = "done", (f"{reviewed} döntés rögzítve"
                                             if reviewed else "nincs felülvizsgálandó beszélő")
    elif pending is None:
        review_state, review_info = "pending", "még nincs próba"
    else:
        review_state, review_info = "active", f"{pending} döntés hátra"
    cast_summary = ((casting or {}).get("summary") or {})
    n_open = int(cast_summary.get("n_open") or 0) + int(cast_summary.get("n_ready") or 0)
    voices_info = f"{chars_with_voice}/{total_chars}" if total_chars else ""
    if n_open:
        voices_info += f" · {n_open} nyitott döntés"

    return [
        row("load", "Epizód hozzáadása" if project.kind != "movie" else "Film betöltése",
            "done" if media else "pending",
            "média a helyén" if media else "nincs média a projektban"),
        row("prepare", "Előkészítés + próba",
            "done" if audits else "pending",
            f"{len(audits)} próba" if audits else "jóváhagyó-próba futtandó"),
        row("review", "Beszélők ellenőrzése", review_state, review_info),
        row("cluster", "Szereplők gyűjtése",
            "done" if total_chars else "pending",
            f"{total_chars} karakter" if total_chars else "klaszterezés futtandó"),
        row("voices", "Hangok kiosztása",
            "done" if total_chars and chars_with_voice == total_chars
            else ("active" if total_chars else "pending"),
            voices_info),
        row("dub", "Szinkronizálás", "done" if dub_wav else "pending"),
        row("qa", "Ellenőrzés", "done" if report else "pending"),
        row("mux", "Kész film" if project.kind == "movie" else "Kész epizód",
            "done" if out_mkv else "pending",
            "out/ mappa" if out_mkv else ""),
    ]


def _timbre_status(project) -> dict:
    p = project.work / "voice-timbres.json"
    if not p.is_file():
        return {"ready": False, "voices": 0, "path": str(p)}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ready": False, "voices": 0, "path": str(p)}
    return {"ready": True, "voices": len(data.get("voices") or {}), "path": str(p)}


def _cast_context(project, tag: str, audit: dict) -> tuple[dict, dict]:
    """Hang-használat és együtt-szólalás a szereplő-rekordokból (ha van)."""
    config, identity, *_ = _core()
    records = identity.char_records(project.cast_file, project.work / "identity-gallery.json")
    usage: dict[str, list[str]] = {}
    for cid, row in records.items():
        voice = row.get("voice")
        if voice:
            usage.setdefault(str(voice), []).append(str(cid))
    cooccur: dict[str, set[str]] = {}
    utt_path = project.ep_dir(tag) / "utterances.json"
    if utt_path.is_file():
        try:
            utts = json.loads(utt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            utts = []
        cluster_char: dict[str, str] = {}
        for cid, row in records.items():
            for ep, clusters in (row.get("members") or {}).items():
                if str(ep).upper() == tag.upper():
                    for cl in clusters:
                        cluster_char[str(cl)] = str(cid)
        by_cluster: dict[str, list[float]] = {}
        for utt in utts:
            cl = utt.get("cluster")
            if cl is None:
                continue
            by_cluster.setdefault(str(cl), []).append(float(utt.get("start") or 0.0))
        window = 20.0
        keys = list(by_cluster)
        for i, a in enumerate(keys):
            char_a = cluster_char.get(a)
            if not char_a:
                continue
            for b in keys[i + 1:]:
                char_b = cluster_char.get(b)
                if not char_b or char_b == char_a:
                    continue
                if any(abs(x - y) <= window for x in by_cluster[a][:40] for y in by_cluster[b][:40]):
                    cooccur.setdefault(char_a, set()).add(char_b)
                    cooccur.setdefault(char_b, set()).add(char_a)
    if not cooccur:
        # nincs évad-szintű klaszter-leképezés: a kivágás beszélői közvetlenül
        cooccur = _cooccurrence_from_turns(audit.get("speaker_turns") or [])
    return usage, cooccur


def _cooccurrence_from_turns(turns: list[dict], window_s: float = 8.0) -> dict[str, set[str]]:
    """Ugyanabban a kivágásban egymás után szólaló beszélők = közös jelenet."""
    out: dict[str, set[str]] = {}
    for i, turn in enumerate(turns):
        a = str(turn.get("speaker"))
        for other in turns[max(0, i - 6): i + 7]:
            b = str(other.get("speaker"))
            if a == b:
                continue
            if abs(float(other.get("start", 0)) - float(turn.get("end", 0))) <= window_s or \
                    abs(float(turn.get("start", 0)) - float(other.get("end", 0))) <= window_s:
                out.setdefault(a, set()).add(b)
    return out


@bp.route("/api/dub/audit", methods=["POST"])
def run_audit():
    _config, _identity, _review, speaker_audit, _vd = _core()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    try:
        start = float(body.get("start") or 30.0)
        duration = float(body.get("duration") or 30.0)
    except (TypeError, ValueError):
        return jsonify({"error": "Érvénytelen időtartomány."}), 400
    if duration <= 0 or duration > 300:
        return jsonify({"error": "A próba 1–300 másodperc közötti lehet."}), 400
    if start < 0:
        return jsonify({"error": "A kezdőpont nem lehet negatív."}), 400
    language = str(body.get("language") or "en").lower()
    if language not in ("en", "he"):
        return jsonify({"error": "A nyelv en vagy he lehet."}), 400
    try:
        path = speaker_audit.run(project, tag, start, duration, language=language)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:                          # pragma: no cover - külső eszköz
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, "audit": path.name, "start": start,
                    "duration": duration, "language": language})


@bp.route("/api/dub/timbres", methods=["POST"])
def measure_timbres():
    _config, _identity, _review, _sa, voice_decision = _core()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    try:
        payload = voice_decision.measure_voice_timbres(project, verbose=False)
    except Exception as exc:                          # pragma: no cover - TTS modell
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, "voices": len(payload.get("voices") or {})})


def _prototype_for(project, tag: str, speaker: str) -> tuple[list | None, list | None, float]:
    """A beszélő mért lenyomata a legfrissebb audit-riportból.

    A felület nem küld 512 dimenziós vektorokat: a jóváhagyott karakter
    galéria-bejegyzéséhez a **példány-minták** kellenek, különben a következő
    epizódban csak a felfújt centroid maradna, és az azonosság nem lenne stabil.
    """
    paths = sorted((project.ep_dir(tag) / "source_audit").glob("audit_*.json"),
                   key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    for path in paths:
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row = ((audit.get("speaker_identity") or {}).get("speakers") or {}).get(speaker)
        if not row or not row.get("vector"):
            continue
        return (row.get("vector"), row.get("samples"),
                float(row.get("evidence_seconds") or 0.0))
    return None, None, 0.0


@bp.route("/api/dub/speaker", methods=["POST"])
def set_speaker():
    _config, _identity, review_mod, _sa, _vd = _core()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    speaker = str(body.get("speaker") or "").strip()
    if not speaker:
        return jsonify({"error": "Hiányzó beszélő."}), 400
    decision = str(body.get("decision") or "").strip().lower()
    char_id = (body.get("char_id") or "").strip() or None
    if decision == "accepted" and not char_id:
        return jsonify({"error": "Az elfogadáshoz karakter-azonosító kell."}), 400
    if decision == "corrected" and not char_id:
        return jsonify({"error": "A javításhoz válassz karaktert."}), 400
    prototype = body.get("vector")
    samples = body.get("samples")
    seconds = float(body.get("evidence_seconds") or 0.0)
    if not prototype:
        prototype, samples, seen = _prototype_for(project, tag, speaker)
        seconds = seconds or seen
    try:
        review_mod.set_speaker(
            project, tag, speaker, decision, char_id=char_id,
            gender=(body.get("gender") or None), voice=(body.get("voice") or None),
            note=str(body.get("note") or ""), prototype=prototype,
            samples=samples, seconds=seconds, name=(body.get("name") or None))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "review": review_mod.summary(review_mod.load(project, tag))})


@bp.route("/api/dub/voice", methods=["POST"])
def set_voice():
    _config, _identity, review_mod, _sa, _vd = _core()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    speaker = str(body.get("speaker") or "").strip()
    voice = str(body.get("voice") or "").strip()
    if not speaker or not voice:
        return jsonify({"error": "Beszélő és hang is kell."}), 400
    review_mod.set_voice(project, tag, speaker, voice,
                         accepted=bool(body.get("accepted", True)),
                         note=str(body.get("note") or ""))
    # Az azonosítás rögzítése is szükséges lehet, ha még nem történt meg.
    if body.get("char_id"):
        vector = body.get("vector")
        samples = body.get("samples")
        seconds = float(body.get("evidence_seconds") or 0.0)
        if not vector:
            vector, samples, seen = _prototype_for(project, tag, speaker)
            seconds = seconds or seen
        review_mod.set_speaker(project, tag, speaker, "accepted",
                               char_id=str(body["char_id"]),
                               gender=(body.get("gender") or None), voice=voice,
                               prototype=vector, samples=samples, seconds=seconds)
    return jsonify({"ok": True, "review": review_mod.summary(review_mod.load(project, tag))})


@bp.route("/api/dub/voice/bulk", methods=["POST"])
def bulk_voice():
    """A **biztos** hangjavaslatok kötegelt elfogadása egy epizódra.

    Az automatizmus maga a cél: minden olyan beszélő, akinek a döntési
    modell javaslata a noul-küszöb alatt van (nem kér emberi döntést) és
    kellően határozott, megkapja a javasolt hangot. Csak **nincs-még-hang**
    esetet ír felül: ahol már van hang, azt az emberi döntés vagy a
    stabilitás tartja. Visszaadja, kik kaptak hangot és kik maradtak nyitva.
    """
    _config, identity, review_mod, _sa, voice_decision = _core()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    try:
        conf_min = float(body.get("confidence_min", CAST_CONF_OK))
        noul_max = float(body.get("noul_max", CAST_NOUL_OK))
    except (TypeError, ValueError):
        return jsonify({"error": "A küszöbök számok."}), 400
    conf_min = max(0.0, min(1.0, conf_min))
    noul_max = max(0.0, min(1.0, noul_max))

    audits = sorted((project.ep_dir(tag) / "source_audit").glob("audit_*.json"),
                    key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not audits:
        return jsonify({"error": "Nincs próba ennek az epizódnak — futtass egyet előbb."}), 400
    try:
        audit = json.loads(audits[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return jsonify({"error": "A legfrissebb próba-riport nem olvasható."}), 500
    ident = audit.get("speaker_identity") or {}
    review = review_mod.load(project, tag)
    merged = review_mod.apply(ident, review)
    records = identity.char_records(project.cast_file, project.work / "identity-gallery.json")
    timbres = voice_decision.load_timbres(project.work / "voice-timbres.json")
    usage, cooccur = _cast_context(project, tag, audit)
    main_share = float((project.tuning or {}).get("main_share_min", 0.04))
    total_share = {cid: float(row.get("duration") or 0.0) for cid, row in records.items()}

    accepted, skipped = [], {"needs_identity": 0, "already": 0, "needs_human": 0,
                            "uncertain": 0}
    for speaker, entry in sorted((merged.get("speakers") or {}).items()):
        char_id = entry.get("char_id")
        if not char_id or entry.get("status") == "rejected_by_human":
            skipped["needs_identity"] += 1
            continue
        char = records.get(str(char_id)) or {}
        char_view = {
            "char_id": str(char_id),
            "gender": entry.get("gender") or char.get("gender"),
            "voice": entry.get("voice") or char.get("voice"),
            "main": cast_role(float(total_share.get(str(char_id)) or 0.0),
                              max(sum(total_share.values()), 1e-6), main_share) == "fő",
            "evidence_seconds": entry.get("evidence_seconds") or 0.0,
        }
        cosines = voice_decision.cosine_to_prototype(entry.get("vector"), timbres) \
            if entry.get("vector") else {}
        proposal = voice_decision.decide_voice(
            char_view, timbres=cosines, usage=usage,
            cooccurring=cooccur.get(str(char_id)))
        choice = proposal.get("choice")
        voice_now = char_view["voice"]
        if not choice:
            skipped["needs_human"] += 1
            continue
        if voice_now:
            skipped["already" if choice == voice_now else "needs_human"] += 1
            continue
        needs = float((proposal.get("noul") or {}).get("human_review", 1.0))
        conf = float(proposal.get("answer_confidence") or 0.0)
        if needs >= noul_max or conf < conf_min:
            skipped["uncertain"] += 1
            continue
        review_mod.set_voice(project, tag, speaker, choice, accepted=True,
                             note="WebUI kötegelt automatikus elfogadás")
        accepted.append({"speaker": speaker, "char_id": str(char_id), "voice": choice,
                         "role": cast_role(total_share.get(str(char_id), 0.0),
                                           max(sum(total_share.values()), 1e-6),
                                           main_share)})
    return jsonify({"ok": True, "episode": tag, "accepted": accepted,
                    "skipped": skipped})


@bp.route("/api/dub/char", methods=["POST"])
def set_char_role():
    """Főszereplők **kijelölése** és visszaállása (szerep-felülírás).

    A hangzár automatikusa a beszédidő-arányból dönt (``main_share_min``).
    Ez a végpont emberi felülírást teszi lehetővé: ``role="fő"`` vagy
    ``role="mellék"`` megmarad két futás között (``role_override`` a
    szereplő-rekordban); üres érték visszatér az automatikus szabályra.
    A főszereplők hangja kizárólagos — a hangzár ennek megfelelően tartalékol.
    """
    registry = _registry()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    char_id = str(body.get("char_id") or "").strip()
    role = str(body.get("role") or "").strip().lower()
    if role not in ("", "fő", "mellék"):
        return jsonify({"error": "A szerep csak fő vagy mellék lehet."}), 400
    if not char_id:
        return jsonify({"error": "Hiányzó szereplő-azonosító (pl. C001)."}), 400
    reg = registry.Registry(project.cast_file, registry.share_threshold(project.tuning))
    row = next((c for c in reg.characters if c.char_id == char_id), None)
    if row is None:
        return jsonify({"error": f"Nincs ilyen szereplő: {char_id} "
                                 f"(klaszterezés futtandó)."}), 404
    row.role_override = role or None
    try:
        reg.save()
    except OSError as exc:
        return jsonify({"error": f"A szereplő-rekord nem menthető: {exc}"}), 500
    return jsonify({"ok": True, "character": row.to_dict()})


# ── Egyszerű finomhangolás (hangszín + keverés) ─────────────────────────────

#: a keverés/szintézis emberi állításai: kulcs -> (min, max), mindkettő elhagyható
_SOUND_KEYS = {
    "duck_db": (0.0, 12.0),            # zene halkítása a beszéd alatt
    "dialogue_over_bed_db": (0.0, 12.0),  # magyar hang a zene fölött
    "synth_threads": (1, 16),           # CPU-szálak
    "synth_steps": (4, 32),             # Supertonic lépésszám (minőség)
}
_SOUND_DEFAULTS = {"duck_db": 4.0, "dialogue_over_bed_db": 4.0,
                   "synth_threads": 4, "synth_steps": 8}
#: a felület három fokozata -> Supertonic lépésszám (None = automatikus)
_QUALITY_STEPS = {"auto": None, "fast": 6, "balanced": 8, "best": 12}


def _sound_view(project) -> dict:
    """A keverés/szintézis **érvényes** értékei + a gyári alapértékek."""
    tn = project.tuning or {}

    def num(key: str, default: float) -> float:
        value = tn.get(key)
        if value in (None, ""):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    steps = tn.get("synth_steps")
    try:
        steps_n = int(steps) if steps not in (None, "") else None
    except (TypeError, ValueError):
        steps_n = None
    quality = {None: "auto", 6: "fast", 8: "balanced", 12: "best"}.get(steps_n, "custom")
    return {"duck_db": num("duck_db", _SOUND_DEFAULTS["duck_db"]),
            "dialogue_over_bed_db": num("dialogue_over_bed_db",
                                        _SOUND_DEFAULTS["dialogue_over_bed_db"]),
            "synth_threads": int(num("synth_threads", _SOUND_DEFAULTS["synth_threads"])),
            "synth_steps": steps_n, "quality": quality,
            "defaults": dict(_SOUND_DEFAULTS)}


def _write_project_json(path: Path, doc: dict) -> None:
    """Atomikus projekt-JSON írás (a fél-beírt fájl ellen)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


@bp.route("/api/dub/pitch", methods=["POST"])
def set_char_pitch():
    """Egy szereplő hangszín-eltolása (félhang) — kézi finomhangolás.

    Alapból az automatizmus állítja be, hogy az azonos hangot osztozó
    szereplők megkülönböztethetők legyenek. Ha a kész szinkronban mégis
    egyformán szól két szereplő, itt eltolható. Üres érték = vissza az
    automatikus létra első szabad fokára. A következő futásnál érvényes
    (a szintézis-gyorsítótár a hangszínt is nézi).
    """
    registry = _registry()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    char_id = str(body.get("char_id") or "").strip()
    if not char_id:
        return jsonify({"error": "Hiányzó szereplő-azonosító (pl. C001)."}), 400
    raw = body.get("pitch")
    value: float | None
    if raw in (None, ""):
        value = None
    else:
        try:
            value = round(float(raw), 2)
        except (TypeError, ValueError):
            return jsonify({"error": "A hangszín csak szám lehet (félhang)."}), 400
        if not -6.0 <= value <= 6.0:
            return jsonify({"error": "A hangszín -6 és +6 félhang között lehet."}), 400
    reg = registry.Registry(project.cast_file,
                            registry.share_threshold(project.tuning))
    row = next((c for c in reg.characters if c.char_id == char_id), None)
    if row is None:
        return jsonify({"error": f"Nincs ilyen szereplő: {char_id} "
                                 f"(klaszterezés futtandó)."}), 404
    if value is None:
        # vissza az automatikusra: ha ismert, az override előtti értékre
        used = {round(c.pitch, 2) for c in reg.characters
                if c.char_id != char_id and c.voice == row.voice
                and c.pitch_override is None}
        row.pitch = (float(row.pitch_auto) if row.pitch_auto is not None
                     else next((p for p in registry.PITCH_LADDER
                                if round(p, 2) not in used), 0.0))
        row.pitch_auto = None
    else:
        if row.pitch_override is None:
            row.pitch_auto = float(row.pitch or 0.0)
        row.pitch = value
    row.pitch_override = value
    try:
        reg.save()
    except OSError as exc:
        return jsonify({"error": f"A szereplő-rekord nem menthető: {exc}"}), 500
    return jsonify({"ok": True, "character": row.to_dict()})


@bp.route("/api/dub/sound", methods=["POST"])
def set_sound():
    """A hang és a keverés egyszerű állításai (projekt ``tuning``).

    Csak azt a négy kulcsot írja, amit a felület kínál; a többi beállítás
    érintetlen. Mindegyik elhagyható (``null``) — akkor visszaáll a gyári
    érték, tehát a teljesen automatikus futás továbbra is működik.
    """
    name, doc, path = _read_project_raw((request.get_json(silent=True) or {}).get("project"))
    body = request.get_json(silent=True) or {}
    tuning = dict(doc.get("tuning") or {})
    quality = body.get("quality")
    if quality not in (None, "", "custom"):
        if quality not in _QUALITY_STEPS:
            return jsonify({"error": f"Ismeretlen minőség: {quality}."}), 400
        steps = _QUALITY_STEPS[quality]
        if steps is None:
            tuning.pop("synth_steps", None)
        else:
            tuning["synth_steps"] = steps
    for key, (lo, hi) in _SOUND_KEYS.items():
        if key not in body:
            continue
        raw = body.get(key)
        if raw in (None, ""):
            tuning.pop(key, None)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return jsonify({"error": f"{key}: csak szám engedélyezett."}), 400
        if not lo <= value <= hi:
            return jsonify({"error": f"{key}: {lo:g} és {hi:g} közötti érték kell."}), 400
        tuning[key] = int(value) if key == "synth_threads" else round(value, 2)
    doc["tuning"] = tuning
    backup_dir = DUB_ROOT / "projects" / "_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{Path(name).stem}.{time.strftime('%Y%m%d-%H%M%S')}.json"
    try:
        shutil.copy2(path, backup)
        _write_project_json(path, doc)
    except OSError as exc:
        return jsonify({"error": f"A mentés nem sikerült: {exc}"}), 500
    from dubstudio import config as config_mod
    return jsonify({"ok": True, "backup": backup.name,
                    "sound": _sound_view(config_mod.load_project(path))})


# ── Hangminta-előhallgatás ──────────────────────────────────────────────────

#: Előhallgatási mintamondatok: hangonként EGYETLEN, természetes magyar
# mondat, és minden hang mást kap (katalógus-sorrend szerint), hogy ne
# ugyanazt kelljen hallgatni. Ha a szöveg vagy a sorrend változik, emeld a
# _PREVIEW_VERSION-t: a régi hangfájlok kihullnak.
_PREVIEW_VERSION = 3
_PREVIEW_TEXTS = (
    "Sziasztok! Ez itt egy szinkronhang-minta, örömmel szolgálok vele.",
    "Hallod, milyen természetesen szól ezen a hangon a magyar szöveg?",
    "A hétfői forgatáson három jelenetünk is volt, mind jól sikerült.",
    "Képzeld el, ahogy ez a hang mondja a főszereplő mondatait!",
    "Este hét órakor találkozunk a moziban, ne késs le róla!",
    "Ez a mondat azért van itt, hogy összehasonlítsd a hangokat.",
    "Nyugodj meg, mindent elintéztem, már úton vagyok haza.",
    "Soha nem gondoltam volna, hogy éppen te jelensz meg itt.",
    "A kapitány azt mondta, hajnalban kifut a hajó a kikötőből.",
    "Kérlek, mondd el újra, mert most nem figyeltem eléggé.",
    "Tizenöt perccel a kezdés előtt már a helyünkön ültünk.",
    "Az öreg ház tetejéről látni az egész várost a folyóig.",
    "Köszönöm, hogy ilyen gyorsan elintézted ezt a dolgot.",
)


def _preview_text(vid: str, order: list[str] | None = None) -> str:
    """A hanghoz tartozó mintamondat: katalógus-sorrend szerint egyedi.

    Ha a ``order`` (a katalógus id-sorrendje) megadott és benne van a hang,
    abból az indexből veszünk — így 13 hang = 13 különböző mondat. Ha nem,
    a stabil hashes visszafelé kompatibilis útvonal.
    """
    if order and vid in order:
        idx = order.index(vid)
    else:
        idx = sum(ord(ch) for ch in vid)
    return _PREVIEW_TEXTS[idx % len(_PREVIEW_TEXTS)]


_PREVIEW_MUTEX = threading.Lock()
_PREVIEW_ENGINES: dict[str, tuple[object, str]] = {}


def _model_ok(project, key: str) -> bool:
    """A modell-útvonal be van állítva ÉS létező könyvtár.

    A ``model()`` üres értékre a projekt-gyökeret adja (relatív ``''``),
    ami mindig könyvtár — ezért az üres útvonalat kifejezetten hiánynak
    vesszük, kifejezetten a katalógus/preview jelzéseihez.
    """
    return bool(str(project.models.get(key) or "").strip()) \
        and project.model(key).is_dir()


def _voice_preview_engine(project, kind: str):
    """A minta-motor betöltése: a dubstudio voices modul lustán, zárolva."""
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import voices as voices_mod
    return _preview_engine(project, kind, voices_mod)


def _preview_engine(project, kind: str, voices_mod):
    """A motor-objektum csak egyszer töltődik be (lustyán, zárolva)."""
    st = str(project.model("supertonic"))
    pi = str(project.model("piper"))
    es = str(project.model("espeak"))
    key = f"{kind}|{st}|{pi}|{es}"
    with _PREVIEW_MUTEX:
        cached = _PREVIEW_ENGINES.get(key)
        if cached:
            return cached[0]
        if kind == "supertonic":
            obj = voices_mod.SupertonicEngine(project.model("supertonic"))
        else:
            obj = voices_mod.PiperEngine(project.model("piper"), project.model("espeak"))
        _PREVIEW_ENGINES[key] = (obj, key)
        return obj


def _write_wav(path: Path, samples, sr: int) -> None:
    """Mono, 16 bites PCM WAV (stdlib wave + numpy) — hiba esetén nem toródik."""
    import wave

    import numpy as np
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("üres TTS-kimenet")
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    pcm = (np.clip(y, -1.0, 1.0) * 32000.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


@bp.route("/api/dub/voicepreview")
def voice_preview():
    """Egy magyar hang rövid mintája — a hangkiosztás előhallgatásához.

    Az első kérés betölti a motort (Supertonic ~1 GB), azután a minta cache-elt
    (``work/_voicepreview/<vid>.wav``) és azonnali.
    """
    project = _project(request.args.get("project"))
    vid = str(request.args.get("voice") or "").strip()
    if not vid:
        abort(400, description="Hiányzó hang-azonosító (pl. SUP_F1).")
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import voices as voices_mod
    # A hang-azonosító érzékeny: "PIP_anna" kisbetűs tagot tartalmaz, ezért
    # NEM nagybetűsítjük — csak pontos találat esetén próbálunk nagybetűset.
    voice = voices_mod.BY_ID.get(vid) or voices_mod.BY_ID.get(vid.upper())
    if voice is None:
        return jsonify({"error": f"Ismeretlen hang: {vid} (a katalógus: "
                                 f"{', '.join(voices_mod.BY_ID)})."}), 400
    engine_ok = (_model_ok(project, "supertonic") if voice.engine == "supertonic"
                 else _model_ok(project, "piper"))
    if not engine_ok:
        kind = "supertonic" if voice.engine == "supertonic" else "piper"
        return jsonify({"error": f"A {kind} modell-mappa hiányzik vagy nincs beállítva "
                                 f"(Beállítások → Modellek)."}), 400
    cache = project.work / f"_voicepreview_{_PREVIEW_VERSION}"
    out = cache / f"{voice.vid}.wav"
    if not out.is_file():
        try:
            engine = _voice_preview_engine(project, voice.engine)
        except FileNotFoundError as exc:
            return jsonify({"error": f"A modell nem tölthető be: {exc}"}), 400
        try:
            with _PREVIEW_MUTEX:
                # katalógus-sorrend: minden hang kap saját, egyedi mondatot
                order = list(voices_mod.BY_ID)
                samples, sr = engine.synth(_preview_text(voice.vid, order), voice)
        except Exception as exc:                     # pragma: no cover — motorfüggő
            return jsonify({"error": f"A mintaszintézis nem sikerült: "
                                     f"{type(exc).__name__}: {exc}"}), 500
        cache.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        try:
            _write_wav(tmp, samples, sr)
        except (ValueError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            return jsonify({"error": f"A mintafájl nem írható: {exc}"}), 500
        tmp.replace(out)
    return send_file(str(out), mimetype="audio/wav", conditional=True)


# ── Film betöltése ──────────────────────────────────────────────────────────

_MOVIE_SUFFIXES = (".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".m4v")
_TAG_PATTERNS = (
    re.compile(r"(?i)[Ss](\d{1,2})[Ee][\. _-]?[Ee]?(\d{1,3})"),
    re.compile(r"(?i)[Ee](\d{1,3})(?!\d)"),
    re.compile(r"(?i)epiz[óo]d[ \-.:_]*(\d{1,3})"),
)


def _suggest_tag_from_name(name: str) -> str | None:
    """Epizódszám-kiolvasás a fájlnévből (S01E05 / E05 / epizód 5)."""
    for pattern in _TAG_PATTERNS:
        m = pattern.search(name or "")
        if m:
            num = int(m.group(m.lastindex))
            return _tag(f"E{num:02d}") if 0 < num < 100 else None
    return None


def _probe_movie(path: Path) -> tuple[dict | None, str]:
    """ffprobe-féle ellenőrzés: hossz, videó-/hang-/felirat-sávok."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_entries", "format=duration:stream=index,codec_type,codec_name",
             "-show_entries", "stream_tags=language,title", str(path)],
            capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return None, "Az ffprobe nem érhető el (telepítsd és tedd a PATH-ra)."
    except subprocess.TimeoutExpired:
        return None, "Az ffprobe idötúlt kifutáson — a fájl rossz vagy túl lassú."
    if proc.returncode != 0:
        return None, f"Az ffprobe nem tudta olvasni: {proc.stderr.strip()[-300:]}"
    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError:
        return None, "Az ffprobe válasz nem értelmezhető."
    streams = data.get("streams") or []
    if not any(s.get("codec_type") == "video" for s in streams):
        return None, "A fájlban nincs videósáv — ez nem filmfájl tűnik."
    fmt = data.get("format") or {}
    try:
        duration = round(float(fmt.get("duration") or 0.0), 1)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "duration": duration,
        "video": bool(any(s.get("codec_type") == "video" for s in streams)),
        "audio": [s.get("codec_name", "?") for s in streams
                  if s.get("codec_type") == "audio"],
        "subtitles": [str((s.get("tags") or {}).get("title") or
                          (s.get("tags") or {}).get("language") or "")
                      for s in streams if s.get("codec_type") == "subtitle"],
        "size": path.stat().st_size,
    }, ""


@bp.route("/api/dub/loadmovie", methods=["POST"])
def load_movie():
    """Filmfájl betöltése a projektba: ffprobe-ellenőrzés + elhelyezés.

    A fájl a projekt ``data_dir``-jébe kerül ``S<se>ee<ep>`` névvel, így a
    dubstudio ``episode_media()``-jének glob-mintája megtalálja. Ha a fájla
    ugyanazon a köteten van, **hardlinket** használ (nincs GB-okba kerülő másolás);
    máskülönben másol.
    """
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = "E01" if project.kind == "movie" else _tag(body.get("episode"))
    raw_path = str(body.get("path") or "").strip().strip('"')
    if not raw_path:
        return jsonify({"error": "Add meg a filmfájl útvonalát."}), 400
    src = Path(raw_path).expanduser()
    if not src.is_absolute():
        src = DUB_ROOT / src
    if not src.is_file():
        return jsonify({"error": f"A fájl nem található: {src}"}), 400
    if src.suffix.lower() not in _MOVIE_SUFFIXES:
        return jsonify({"error": "Nem támogatott fájltípus (mkv/mp4/mov/avi/webm stb.)."}), 400
    probe, probe_err = _probe_movie(src)
    if probe is None:
        return jsonify({"error": probe_err}), 400
    try:
        target, mode = _place_source(project, src, tag)
    except FileExistsError as exc:
        return jsonify({"error": str(exc)}), 400
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "episode": tag, "movie": str(target), "mode": mode,
                    "probe": probe})


@bp.route("/api/dub/frame")
def frame():
    """Egy képkocka a filmből — a felülvizsgáló lássa, ki beszél."""
    _config, _identity, _review, _sa, _vd = _core()
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    try:
        t = float(request.args.get("t") or 0.0)
        width = max(120, min(640, int(request.args.get("w") or 320)))
    except (TypeError, ValueError):
        return jsonify({"error": "Hibás időpont vagy szélesség."}), 400
    src = project.episode_media(tag)
    if not Path(src).is_file():
        return jsonify({"error": f"Nincs forrásvideó: {src}"}), 404
    cache = project.ep_dir(tag) / "frames"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"f_{t:08.2f}_{width}.jpg"
    if not out.is_file():
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)],
            capture_output=True)
        if proc.returncode != 0 or not out.is_file():
            return jsonify({"error": "Nem sikerült képkockát kinyerni."}), 500
    return send_file(out, mimetype="image/jpeg")


@bp.route("/api/dub/clip")
def clip():
    _config, _identity, _review, speaker_audit, _vd = _core()
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    try:
        start = float(request.args.get("start") or 0.0)
        duration = float(request.args.get("duration") or 3.0)
        stream = request.args.get("stream")
        stream = int(stream) if stream not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "Érvénytelen időtartomány."}), 400
    duration = max(0.5, min(duration, 20.0))
    clips = project.work / "_clips"
    clips.mkdir(parents=True, exist_ok=True)
    out = clips / f"{tag}_{start:.2f}_{duration:.2f}_{stream or 'a0'}.wav"
    try:
        path = speaker_audit.extract_source_audio(project, tag, out, start, duration,
                                                  stream=stream)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:                          # pragma: no cover - ffmpeg
        return jsonify({"error": f"Hangkiemelés hiba: {exc}"}), 500
    return send_file(str(path), mimetype="audio/wav", conditional=True)


# ── Böngészős felülvizsgáló lejátszó ──────────────────────────────────────────


def _player_urls(project, tag: str, info: dict) -> dict:
    """A manifest URL-jei (a projekt nevét a kérés nevével adjuk tovább)."""
    key = _project_key(request.args.get("project")) or project.name
    base = {"project": key, "episode": tag}
    for track in info.get("audio") or []:
        track["url"] = "/api/dub/player/media?" + urlencode({**base, "audio": track["id"]})
    for track in info.get("subtitles") or []:
        track["url"] = "/api/dub/player/subtitle?" + urlencode({**base, "id": track["id"]})
    return info


@bp.route("/api/dub/player")
def player_manifest():
    """A forrás lejátszható sávjai (hang + felirat), proxy URL-ekkel."""
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    try:
        info = _player().manifest(project, tag)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(_player_urls(project, tag, info))


@bp.route("/api/dub/player/status")
def player_proxy_status():
    """Egy hangsáv proxyjának állapota (előkészítés közben is válaszol)."""
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    audio_id = str(request.args.get("audio") or "").strip()
    try:
        payload = _player().proxy_status(project, tag, audio_id)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": f"A proxy állapota nem olvasható: {exc}"}), 500
    return jsonify(payload)


@bp.route("/api/dub/player/prepare", methods=["POST"])
def player_prepare():
    """A proxy aszinkron elkészítése; azonnal állapotot ad vissza."""
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    audio_id = str(body.get("audio") or "").strip()
    try:
        payload = _player().prepare(project, tag, audio_id)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": f"A proxy nem készíthető el: {exc}"}), 500
    return jsonify(payload)


@bp.route("/api/dub/player/media")
def player_media():
    """A választott hangsávval rendelkező MP4 proxy (Range-kérésekkel)."""
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    audio_id = str(request.args.get("audio") or "").strip()
    try:
        path = _player().media_for(project, tag, audio_id)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": f"A lejátszó-proxy nem készíthető el: {exc}"}), 500
    return send_file(str(path), mimetype="video/mp4", conditional=True)


@bp.route("/api/dub/player/sample", methods=["POST"])
def player_sample():
    """Rövid magyar ellenőrző klip készítése a kész dub.wav-ból."""
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    try:
        result = _player().sample_for(project, tag)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": f"A minta nem készíthető el: {exc}"}), 500
    key = _project_key(body.get("project")) or project.name
    result["url"] = "/api/dub/player/sample/media?" + urlencode(
        {"project": key, "episode": tag})
    return jsonify(result)


@bp.route("/api/dub/player/sample/media")
def player_sample_media():
    """A rövid magyar ellenőrző klip (Range-kérésekkel)."""
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    try:
        result = _player().sample_for(project, tag)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    return send_file(str(result["path"]), mimetype="video/mp4", conditional=True)


@bp.route("/api/dub/player/subtitle")
def player_subtitle():
    """A választott felirat WebVTT-változata."""
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    sub_id = str(request.args.get("id") or "").strip()
    try:
        path = _player().subtitle_for(project, tag, sub_id)
    except _player().PlayerError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": f"A felirat nem készíthető el: {exc}"}), 500
    return send_file(str(path), mimetype="text/vtt", conditional=True)


# ── Beállítások ──────────────────────────────────────────────────────────────


def _project_path(name: str) -> Path:
    return DUB_ROOT / "projects" / f"{Path(name).stem}.json"


def _read_project_raw(name: str | None) -> tuple[str, dict, Path]:
    """A választott projekt **nyers** JSON (a Project-objektum útvonalak nélkül)."""
    names = _project_names()
    if not names:
        abort(404, description="Nincs dubstudio projekt (projects/*.json).")
    chosen = name if name in names else names[0]
    path = _project_path(chosen)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        abort(500, description=f"A projektfájl nem olvasható: {exc}")
    if not isinstance(doc, dict):
        abort(500, description="A projektfájl nem szakasz-gyűjtemény.")
    return chosen, doc, path


def _num_valid(value, where: str) -> tuple[bool, str]:
    """A tuning/gate szakaszok levél-eleme szám (vagy számokból álló dict)."""
    if isinstance(value, bool):
        return True, ""                                       # true/false mint 0/1 azonos szándék
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return False, f"{where}: nem véges szám."
        return True, ""
    if isinstance(value, dict) and value:
        for key, sub in value.items():
            ok, err = _num_valid(sub, f"{where}.{key}")
            if not ok:
                return False, err
        return True, ""
    return False, f"{where}: csak szám (vagy számokból álló alcsoport) engedélyezett, kapott: {type(value).__name__}"


def _flat_valid(value, where: str) -> tuple[bool, str]:
    """Az llm/asr szakaszok levél-eleme szám, szöveg, igazságérték vagy None."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True, ""
    if isinstance(value, dict) and value:
        for key, sub in value.items():
            ok, err = _flat_valid(sub, f"{where}.{key}")
            if not ok:
                return False, err
        return True, ""
    return False, f"{where}: csak szám, szöveg vagy logikai érték engedélyezett"


@bp.route("/api/dub/config")
def get_config():
    """A projekt beállításai (szerkeszthető szakaszok) + hangkatalógus.

    A hangkatalógus **nem** tölt be modellt: csak a mintaútvonalak létinyét
    nézi, mert a teljes engedélyezés (Engine-példány) ~1 GB betöltés lenne.
    """
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import config, voices as voices_mod
    chosen, doc, path = _read_project_raw(request.args.get("project"))
    project = config.load_project(path)
    catalog = [v.to_dict() for v in voices_mod.CATALOG]
    voices = []
    for v in catalog:
        engine_ok = (project.model("supertonic").is_dir() if v["engine"] == "supertonic"
                     else project.model("piper").is_dir())
        voices.append({**v, "model_present": bool(engine_ok)})
    return jsonify({
        "project": chosen,
        "file": str(path),
        "settings": {key: doc.get(key) for key in _EDITABLE_SECTIONS},
        "meta": {"name": doc.get("name", chosen), "season": doc.get("season", 1),
                 "tts_sr": doc.get("tts_sr", 44100),
                 "source_lang": doc.get("source_lang", "heb")},
        "voices": voices,
    })


@bp.route("/api/dub/config", methods=["POST"])
def set_config():
    """Beállítások mentése **biztonsági mentéssel** (projects/_backup/).

    A küldött szakaszok teljes csereként kerülnek be (a UI mindig a teljes
    szakaszt küldi el), így nincs részleges folt a JSON-ban.
    """
    body = request.get_json(silent=True) or {}
    name, doc, path = _read_project_raw(body.get("project"))
    changes = body.get("config") or body
    if not isinstance(changes, dict):
        return jsonify({"error": "A küldött beállítás nem szakasz-gyűjtemény."}), 400

    for section in ("tuning", "gate"):
        value = changes.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            return jsonify({"error": f"{section}: csak objektum (dict) küldhető."}), 400
        for key, sub in value.items():
            ok, err = _num_valid(sub, f"{section}.{key}")
            if not ok:
                return jsonify({"error": err}), 400
    for section in ("llm", "asr"):
        value = changes.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            return jsonify({"error": f"{section}: csak objektum (dict) küldhető."}), 400
        for key, sub in value.items():
            ok, err = _flat_valid(sub, f"{section}.{key}")
            if not ok:
                return jsonify({"error": err}), 400
    models = changes.get("models")
    if models is not None:
        if not isinstance(models, dict) or not all(isinstance(v, str) for v in models.values()):
            return jsonify({"error": "models: csak szöveg-útvonalakat fogad el."}), 400
    if not all(isinstance(v, str) for v in (changes.get("models") or {}).values()):
        return jsonify({"error": "models: minden mező szöveg kell legyen."}), 400
    for key in ("season", "tts_sr"):
        if key in changes:
            try:
                changes[key] = int(changes[key])
            except (TypeError, ValueError):
                return jsonify({"error": f"{key}: egész szám kell."}), 400
    if "source_lang" in changes:
        changes["source_lang"] = str(changes["source_lang"]).strip().lower()
    if "name" in changes and str(changes["name"]).strip() == "":
        return jsonify({"error": "A projekt neve nem lehet üres."}), 400

    backup_dir = DUB_ROOT / "projects" / "_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = backup_dir / f"{Path(name).stem}.{stamp}.json"
    shutil.copy2(path, backup)

    for key in ("name", "season", "tts_sr", "source_lang"):
        if key in changes:
            doc[key] = changes[key]
    for section in _EDITABLE_SECTIONS:
        if section in changes and changes[section] is not None:
            doc[section] = changes[section]
    try:
        # a fél-beírt fájl ellen is véd: először a .tmp-be írunk, csak aztán cserélünk
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        return jsonify({"error": f"A mentés nem sikerült: {exc}"}), 500
    return jsonify({"ok": True, "backup": backup.name,
                    "settings": {k: doc.get(k) for k in _EDITABLE_SECTIONS}})


# ── Futtatás (dubstudio CLI a WebUI-ból) ─────────────────────────────────────

_RUN: dict = {"state": "idle"}
_RUN_MUTEX = threading.Lock()
_RUN_TAIL_CHARS = 12000


class _RunBusy(RuntimeError):
    """Már fut egy pipeline-feladat."""


class _RunFailed(RuntimeError):
    """A háttérfolyamat nem indítható."""


def _launch_job(step: str, chosen: str, episodes: list[str],
                flags: list[str]) -> dict:
    """Egy CLI-lépés indítása; a WebUI és a sorozatfigyelő közös útja."""
    if step not in _INTERNAL_STEPS:
        raise _RunFailed(f"A lépés ({step}) nem indítható innen.")
    with _RUN_MUTEX:
        if _RUN["state"] == "running":
            raise _RunBusy("Már fut egy feladat: "
                           f"{_RUN.get('step')} ({_RUN.get('started_at')}).")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = DUB_ROOT / "work" / f"webui_{stamp}_{step}_{'_'.join(episodes)}.log"
        cmd = [sys.executable, "-m", "dubstudio", "--project", str(_project_path(chosen)),
               step, *episodes, *flags]
        log_handle = None
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                cmd, cwd=str(DUB_ROOT), stdout=log_handle, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except (OSError, ValueError) as exc:
            if log_handle is not None:
                log_handle.close()
            raise _RunFailed(f"A folyamat nem indítható: {exc}") from exc
        _RUN.clear()
        _RUN.update({
            "state": "running", "step": step, "episodes": episodes,
            "project": chosen, "command": cmd, "pid": proc.pid,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": str(log_path), "log_handle": log_handle, "proc": proc,
        })
    return {"step": step, "episodes": episodes, "pid": proc.pid,
            "log": str(log_path)}


def _run_status_payload() -> dict:
    job = dict(_RUN)
    if job.get("state") == "running" and job.get("proc") is not None:
        rc = job["proc"].poll()
        if rc is not None:
            with _RUN_MUTEX:
                job = _finish_run(_RUN, rc)
    payload = {key: job.get(key) for key in
               ("state", "step", "episodes", "project", "started_at", "finished_at",
                "returncode", "log", "pid")}
    payload["command"] = " ".join(job.get("command") or [])
    if job.get("log"):
        payload["log_tail"] = _read_log_tail(Path(job["log"]))
    return payload


def _run_flags(body: dict) -> tuple[list[str], str]:
    """Érvényesített CLI-flagok; második érték a hibaüzenet (None = rendben)."""
    flags = []
    try:
        limit = int(body.get("limit") or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit < 0:
        return [], "A limit csak nem negatív lehet."
    if limit:
        flags += ["--limit", str(limit)]
    if body.get("no_mux"):
        flags.append("--no-mux")
    if body.get("allow_fail"):
        flags.append("--allow-fail")
    if body.get("season"):
        flags.append("--season")
    out = str(body.get("out") or "").strip()
    if out:
        if not re.match(r"^[\w .-]+\.(mkv|mp4)$", out, re.IGNORECASE):
            return [], "A kimeneti fájlnév nem érvényes (.mkv/.mp4)."
        flags += ["--out", out]
    return flags, ""


def _read_log_tail(path: Path, max_chars: int = _RUN_TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_chars * 3))
            data = fh.read(max_chars * 3).decode("utf-8", errors="replace")
    except OSError:
        return []
    # az onnxruntime verbose figyelmeztetései elnyomják a hasznos naplót
    lines = [ln for ln in data.splitlines()
             if ln.strip() and "[V:onnxruntime" not in ln]
    return lines[-120:]


def _finish_run(job: dict, rc: int | None) -> dict:
    job["state"] = "finished" if rc == 0 else "failed"
    job["returncode"] = rc
    job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        job["log_handle"].close()
    except (OSError, ValueError):
        pass
    return job


@bp.route("/api/dub/run", methods=["POST"])
def start_run():
    """Egy dubstudio CLI-lépés indítása háttérfolyamatként.

    Egyszerre egy futás lehet: a TTS és az ffmpeg nem párhuzamosítható
    értelmesen ezen a gépen, és a kapu-riport is egézenként olvasható.
    """
    body = request.get_json(silent=True) or {}
    step = str(body.get("step") or "").strip().lower()
    if step not in _RUNNABLE_STEPS:
        return jsonify({"error": f"A lépés ({step}) nem elindítható innen: "
                                 f"{', '.join(_RUNNABLE_STEPS)}."}), 400
    names = _project_names()
    chosen = body.get("project") if body.get("project") in names else (names[0] if names else None)
    if not chosen:
        return jsonify({"error": "Nincs dubstudio projekt."}), 404
    episodes = [str(e).strip().upper() for e in (body.get("episodes") or []) if str(e).strip()]
    if not episodes:
        return jsonify({"error": "Jelölj ki legalább egy epizódot."}), 400
    episodes = [_tag(e) for e in episodes]
    flags, err = _run_flags(body)
    if err:
        return jsonify({"error": err}), 400
    try:
        job = _launch_job(step, chosen, episodes, flags)
    except _RunBusy as exc:
        return jsonify({"error": str(exc)}), 409
    except _RunFailed as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, **job})


@bp.route("/api/dub/run")
def run_status():
    return jsonify(_run_status_payload())


@bp.route("/api/dub/run/stop", methods=["POST"])
def stop_run():
    with _RUN_MUTEX:
        if _RUN.get("state") != "running" or _RUN.get("proc") is None:
            return jsonify({"state": _RUN.get("state"), "stopped": False})
        proc = _RUN["proc"]
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        _finish_run(_RUN, proc.returncode)
    return jsonify({"ok": True, "stopped": True, "state": _RUN["state"]})


# ── Sorozatfigyelő: új epizódok automatikus indítása ─────────────────────────

_WATCHERS: dict[str, threading.Thread] = {}
_WATCH_STOP: dict[str, threading.Event] = {}
_WATCH_STATUS: dict[str, dict] = {}
_WATCH_MUTEX = threading.Lock()


def _watch_loop(key: str, stop: threading.Event) -> None:
    """Periodikusan megnézi, van-e kész, még nem szinkronizált epizód.

    Ha a média megvan, de a felirat/vokál/M&E hiányzik, előbb az intake fut
    (fordítás + hangsáv-szétválasztás); csak utána indul a szinkronlánc.
    A kész kimenet megléte a lezárás jele, hogy egy QA-bukás után ne induljon
    végtelen újrapróbálkozás.
    """
    while not stop.is_set():
        poll_s = 30
        try:
            project = _project(key)
            if not (project.series or {}).get("auto_process"):
                break
            try:
                poll_s = max(10, min(600, int((project.series or {}).get("poll_s", 30))))
            except (TypeError, ValueError):
                poll_s = 30
            lib = _library().library(project)
            auto_intake = bool((project.series or {}).get("auto_intake", True))
            intake_pending = [e for e in lib["episodes"]
                              if e.get("media") and not e.get("ready")
                              and not e.get("output")]
            pending = [e for e in lib["episodes"]
                       if e["ready"] and not e["output"]]
            if auto_intake and intake_pending and _RUN.get("state") != "running":
                first = next((e for e in intake_pending
                              if (e.get("intake") or {}).get("retry_allowed", True)), None)
                if first is not None:
                    try:
                        _launch_job("intake", key, [first["tag"]],
                                    ["--then-run", "--allow-fail"])
                        _WATCH_STATUS[key] = {
                            "state": "intake", "episode": first["tag"],
                            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    except _RunBusy:
                        pass
                    except _RunFailed as exc:
                        _WATCH_STATUS[key] = {
                            "state": "error", "episode": first["tag"],
                            "error": str(exc),
                            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
            elif pending and _RUN.get("state") != "running":
                first = pending[0]
                flags = ["--allow-fail"]
                if project.kind != "movie":
                    flags.append("--season")
                try:
                    _launch_job("run", key, [first["tag"]], flags)
                    _WATCH_STATUS[key] = {
                        "state": "started", "episode": first["tag"],
                        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                except _RunBusy:
                    pass
                except _RunFailed as exc:
                    _WATCH_STATUS[key] = {
                        "state": "error", "episode": first["tag"],
                        "error": str(exc),
                        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
            else:
                _WATCH_STATUS[key] = {
                    "state": "watching", "pending": [e["tag"] for e in pending],
                    "intake": [e["tag"] for e in intake_pending],
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
        except Exception as exc:                       # noqa: BLE001 — háttérszál
            _WATCH_STATUS[key] = {
                "state": "error", "error": f"{type(exc).__name__}: {exc}",
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        if stop.wait(poll_s):
            break


def _ensure_watcher(key: str) -> None:
    """A projekt beállítása szerint indítja/leállítja a figyelő szálat."""
    if not key:
        return
    try:
        project = _project(key)
    except Exception:                                  # noqa: BLE001
        return
    enabled = bool((project.series or {}).get("auto_process"))
    with _WATCH_MUTEX:
        current = _WATCHERS.get(key)
        if not enabled:
            stop = _WATCH_STOP.pop(key, None)
            if stop is not None:
                stop.set()
            _WATCHERS.pop(key, None)
            return
        if current is not None and current.is_alive():
            return
        stop = threading.Event()
        thread = threading.Thread(target=_watch_loop, args=(key, stop),
                                  name=f"dub-watch-{key}", daemon=True)
        _WATCHERS[key] = thread
        _WATCH_STOP[key] = stop
        thread.start()
