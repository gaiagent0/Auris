"""Emberi ellenőrző felület: film + felirat együtt, soronkénti címkézés.

A cél az, hogy egy ember a filmet nézve, a feliratok mellett **gyorsan**
megmondja soronként, ki beszél (és hogy képen kívül van-e), illetve hogy
javítsa a sor elcsúszását. Ezért a felület nem kódokat mutat, hanem magyar
keresztneveket (Anna, Éva, Ottó, Laci, Józsi, Panna…), a döntéseket pedig a
``dubstudio.linereview`` állapotába írja (``work/<EP>/review/lines.json``),
amit a szinkronlánc a következő futásnál felhasznál.

Ez a modul nem szinkronizál: csak olvassa a kész előkészítést
(``utterances.json``, cast, ASD-bizonyíték) és rögzíti az emberi döntést.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file

from core.dub_api import (DUB_ROOT, _PREVIEW_MUTEX, _cast_rows, _model_ok,
                          _project, _tag, _project_key, _voice_ok,
                          _voice_preview_engine, _registry, _write_wav)

bp = Blueprint("lines", __name__)

#: a sor állapotai (a felület színezéséhez)
_KIND_LABEL = {"char": "szereplő", "offscreen": "képen kívül",
               "unknown": "nem tudom"}


def _dubstudio():
    """A dubstudio modulok lustán (numpy/sherpa nélkül is betölthető legyen)."""
    from core.dub_api import DUB_ROOT
    import sys

    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import config, identity, linereview
    return config, identity, linereview


def _rows(project, tag: str) -> list[dict]:
    """A kész megnyilatkozások (``prepare`` kimenete)."""
    path = project.ep_dir(tag) / "utterances.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _asd_rows(project, tag: str) -> dict[str, dict]:
    """A képi beszélő-ellenőrzés sorai (bizonyíték, nem kapu)."""
    path = project.ep_dir(tag) / "report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    block = data.get("asd") or (data.get("metrics") or {}).get("asd") or {}
    out: dict[str, dict] = {}
    for row in (block.get("rows") or []):
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        faces = row.get("faces")
        speaking = row.get("speaking")
        if speaking is not None:
            kind, text = "face", f"képen: arc #{int(speaking) + 1}"
        elif faces == 0:
            kind, text = "offscreen", "képen kívül"
        else:
            kind, text = "unknown", "nem eldönthető"
        out[uid] = {"kind": kind, "text": text,
                    "prob": row.get("best_prob")}
    return out


def _cast(project, tag: str) -> tuple[list[dict], dict[str, str]]:
    """A szereplők magyar nevekkel (a hiányzó neveket itt pótoljuk, stabilan)."""
    config, identity, linereview = _dubstudio()
    records = identity.char_records(project.cast_file,
                                   project.work / "identity-gallery.json")
    cast_rows = _cast_rows(project)
    rows = []
    for cid, row in records.items():
        extra = cast_rows.get(str(cid)) or {}
        rows.append({"char_id": str(cid),
                     "name": row.get("name"),
                     "gender": row.get("gender"),
                     "voice": row.get("voice"),
                     "pitch": row.get("pitch"),
                     "role": extra.get("role"),
                     "rank": int(extra.get("rank") or 0),
                     "duration": round(float(row.get("duration") or 0.0), 1)})
    data = linereview.load(project, tag)
    before = dict(data.get("names") or {})
    names = linereview.ensure_names(data, rows)
    if data.get("updated_at") is None or names != before:
        # A nevek első megnyitáskor keletkeznek, de újraklaszterezés után a
        # névkészlet is változik: ilyenkor is elmentjük, különben a fájlban a
        # régi (44 neves, „Anna 2 … Anna 17”) állapot marad, és a felület csak
        # a memóriában mutatná a helyes neveket.
        linereview.save(project, tag, data)
    for row in rows:
        row["name"] = names.get(row["char_id"]) or row.get("name")
    rows.sort(key=lambda r: (r["rank"] or 9999, r["char_id"]))
    return rows, names


def _people(data: dict, cast: list[dict]) -> list[dict]:
    """A felületen létrehozott szereplők (``N###``) is választhatók legyenek."""
    known = {c["char_id"] for c in cast}
    out = []
    for cid, row in (data.get("people") or {}).items():
        if cid in known:
            continue
        out.append({"char_id": cid, "name": (row or {}).get("name"),
                    "gender": (row or {}).get("gender"),
                    "voice": (row or {}).get("voice"),
                    "pitch": (row or {}).get("pitch"),
                    "role": "mellék", "rank": 0, "duration": 0.0})
    return out


def _voice_catalog():
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import voices as voices_mod
    return voices_mod


def _ensure_person_voices(project, tag: str, data: dict,
                          cast: list[dict]) -> bool:
    """A felületen létrehozott szereplők kapjanak szabad magyar hangot.

    Enélkül az új szereplő (``N001``) alá nincs hang rendelve, és a kijelölt
    feliratrész sem hallgatható meg. A választás nemenként a szabad
    hangkészletből történik, és a felületen bármikor átállítható.
    """
    people = data.get("people") or {}
    if not people:
        return False
    used = {str(c.get("voice")) for c in cast if c.get("voice")}
    used |= {str((p or {}).get("voice")) for p in people.values()
             if (p or {}).get("voice")}
    changed = False
    for cid, person in people.items():
        if not person or person.get("voice"):
            continue
        gender = "F" if str(person.get("gender") or "").upper().startswith("F") else "M"
        pick = next((v for v in _voice_catalog().CATALOG
                     if v.gender == gender and v.vid not in used), None)
        if pick is None:
            continue
        person["voice"] = pick.vid
        used.add(pick.vid)
        changed = True
    if changed:
        _, _, linereview = _dubstudio()
        linereview.save(project, tag, data)
    return changed


def _line_view(row: dict, cue: dict, data: dict, names: dict,
               asd: dict[str, dict], valid_uids: set[str] | None = None) -> dict:
    lit = linereview_line(data, row.get("uid"), cue.get("index"), valid_uids)
    char = row.get("char")
    return {
        "key": lit["key"], "uid": row.get("uid"), "cue": cue.get("index"),
        "start": round(float(cue.get("start") or 0.0), 2),
        "end": round(float(cue.get("end") or 0.0), 2),
        "text": cue.get("text") or "",
        "char": char,
        "name": names.get(str(char)) if char else None,
        "gender": data.get("gender_of", {}).get(str(char)) if char else None,
        "voice": row.get("voice"),
        "p_female": row.get("p_female"),
        "multi_diar": int(row.get("multi_diar") or 0),
        "asd": (asd.get(str(row.get("uid"))) or None),
        "offset": lit["offset"],
        "marks": lit["marks"],
    }


def linereview_line(data: dict, uid, cue, valid_uids=None) -> dict:
    """Egy sor mentett állapota (kulcs, elcsúszás, megjelölések).

    Ha a megjelölés egy korábbi futás uid-jére mutat (a csoportosítás azóta
    megváltozott), a feliratsor sorszáma alapján mégis megjelenik — a kulcs a
    mostani, tehát a mentés már a helyére ír.
    """
    _, _, linereview = _dubstudio()
    key, entry = linereview.resolve(data, str(uid), cue, valid_uids)
    marks = []
    for mark in (entry.get("marks") or []):
        mark = dict(mark)
        mark["name"] = linereview.display_name(data, mark.get("char"))
        mark["label"] = _KIND_LABEL.get(str(mark.get("kind")), str(mark.get("kind")))
        marks.append(mark)
    return {"key": key, "offset": round(float(entry.get("offset") or 0.0), 2),
            "marks": marks}


def _payload(project, tag: str, key: str | None = None) -> dict:
    config, identity, linereview = _dubstudio()
    rows = _rows(project, tag)
    data = linereview.load(project, tag)
    cast, names = _cast(project, tag)

    gender_of = {c["char_id"]: c.get("gender") for c in cast if c.get("gender")}
    for cid, person in (data.get("people") or {}).items():
        if (person or {}).get("gender"):
            gender_of[cid] = person["gender"]
    data["gender_of"] = gender_of

    _ensure_person_voices(project, tag, data, cast)
    people = cast + _people(data, cast)
    for i, person in enumerate(people):
        person["hue"] = ((i * 47 + 18) if person["char_id"].startswith("C")
                         else (i * 89 + 140)) % 360

    asd = _asd_rows(project, tag)
    valid_uids = {str(row.get("uid")) for row in rows if row.get("uid")}
    lines = []
    for row in rows:
        for cue in (row.get("cues") or []):
            # a zene/dalszöveg/háttérhang sor nem dialógus: nincs kit címkézni
            if cue.get("music"):
                continue
            lines.append(_line_view(row, cue, data, names, asd, valid_uids))
    lines.sort(key=lambda l: (l["start"], l["key"]))
    for i, line in enumerate(lines):
        line["n"] = i + 1

    summary = linereview.summary(project, tag)
    summary["total_lines"] = len(lines)
    summary["labelled_lines"] = len({l["key"] for l in lines if l["marks"]})
    summary["duration"] = round(max((l["end"] for l in lines), default=0.0), 1)
    return {
        "project": key or _project_key(request.args.get("project")) or project.name,
        "project_name": project.name,
        "episode": tag,
        "is_movie": project.kind == "movie",
        "has_utterances": bool(rows),
        "cast": people,
        "voices": [{"vid": v.vid, "label": v.label, "engine": v.engine,
                    "gender": v.gender, "model_present": _voice_ok(project, v)}
                   for v in _voice_catalog().CATALOG],
        "lines": lines,
        "stats": summary,
        "asd_ready": bool(asd),
    }


@bp.route("/felulvizsgalat")
def review_page():
    return render_template("felulvizsgalat.html")


@bp.route("/videoteszt")
def video_test_page():
    """Önálló lejátszó-tesztoldal (csak videó + fájlválasztó + diagnosztika)."""
    return render_template("videoteszt.html")


@bp.route("/api/dub/lines")
def lines_view():
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    return jsonify(_payload(project, tag))


def _line_body():
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    uid = str(body.get("uid") or "").strip()
    if not uid:
        raise ValueError("hiányzó megnyitás-azonosító (uid)")
    try:
        cue = int(body.get("cue"))
    except (TypeError, ValueError) as exc:
        raise ValueError("hiányzó feliratsor-sorszám") from exc
    return body, project, tag, uid, cue


@bp.route("/api/dub/lines/label", methods=["POST"])
def set_line_label():
    _, _, linereview = _dubstudio()
    try:
        body, project, tag, uid, cue = _line_body()
        data, key = linereview.set_label(
            project, tag, uid, cue,
            kind=str(body.get("kind") or "char"),
            char_id=(str(body.get("char_id") or "").strip() or None),
            gender=(str(body.get("gender") or "").strip() or None),
            name=(str(body.get("name") or "").strip() or None),
            words=body.get("words"))
        _ensure_person_voices(project, tag, data, _cast(project, tag)[0])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    payload = _payload(project, tag, _project_key(body.get("project")))
    line = next((l for l in payload["lines"] if l["key"] == key), None)
    return jsonify({"ok": True, "key": key, "line": line,
                    "cast": payload["cast"], "voices": payload["voices"],
                    "stats": linereview.summary(project, tag)})


@bp.route("/api/dub/lines/offset", methods=["POST"])
def set_line_offset():
    _, _, linereview = _dubstudio()
    try:
        body, project, tag, uid, cue = _line_body()
        data, key = linereview.set_offset(project, tag, uid, cue,
                                          float(body.get("offset") or 0.0))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    line = next((l for l in _payload(project, tag, _project_key(body.get("project")))["lines"]
                 if l["key"] == key), None)
    return jsonify({"ok": True, "key": key, "line": line,
                    "stats": linereview.summary(project, tag)})


# ── a kijelölt feliratrész meghallgatása egy hangon ────────────────
#: a minta-gyorsítótár verziója (ha a szintézis vagy a vágás változik, emelni)
_LINE_PREVIEW_VERSION = 1


def _words_arg(value) -> list[int] | None:
    """``words=2-5`` -> ``[2, 5]`` (a kijelölt szótartomány a soron belül)."""
    if not value:
        return None
    text = str(value).strip().replace(",", "-")
    try:
        first, last = (int(part) for part in text.split("-", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("a szótartomány „első-utolsó” alakú legyen (pl. 2-5)") from exc
    if first < 0 or last < first:
        raise ValueError("érvénytelen szótartomány")
    return [first, last]


def _cue_text(project, tag: str, uid: str, cue_index: int,
              words: list[int] | None) -> str:
    """A felirat (vagy a kijelölt szótartomány) szövege az előkészítésből."""
    for row in _rows(project, tag):
        if str(row.get("uid") or "") != str(uid):
            continue
        for cue in (row.get("cues") or []):
            if int(cue.get("index") or -1) != int(cue_index):
                continue
            text = " ".join(str(cue.get("text") or "").split())
            if words is None:
                return text
            parts = text.split(" ")
            first, last = words
            if last >= len(parts):
                raise ValueError("a kijelölt szótartomány nincs a sorban")
            return " ".join(parts[first:last + 1])
    return ""


@bp.route("/api/dub/lines/preview")
def preview_line():
    """A kijelölt feliratrész meghallgatása egy magyar hangon.

    A felülvizsgáló így a szinkronfutás **előtt** hallhatja, hogy a kiválasztott
    hang (és a szereplő fix hangszíne) hogyan szól ugyanazon a szövegen. A
    szöveg a kész előkészítésből jön (uid + feliratsor index + szótartomány),
    a WAV pedig gyorsítótárból (``work/_linepreview_v1/``); a TTS-motor csak az
    első kérésnél töltődik be.
    """
    project = _project(request.args.get("project"))
    tag = _tag(request.args.get("episode"))
    uid = str(request.args.get("uid") or "").strip()
    if not uid:
        return jsonify({"error": "Hiányzó megnyitás-azonosító (uid)."}), 400
    try:
        cue = int(request.args.get("cue"))
        words = _words_arg(request.args.get("words"))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Hiányzó vagy hibás feliratsor: {exc}"}), 400
    vid = str(request.args.get("voice") or "").strip()
    root = str(DUB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dubstudio import audio as audio_mod
    from dubstudio import voices as voices_mod

    voice = voices_mod.BY_ID.get(vid) or voices_mod.BY_ID.get(vid.upper())
    if voice is None:
        return jsonify({"error": f"Ismeretlen hang: {vid}."}), 400
    try:
        pitch = round(float(request.args.get("pitch") or 0.0), 2)
    except (TypeError, ValueError):
        pitch = 0.0
    pitch = max(-6.0, min(6.0, pitch))
    try:
        text = _cue_text(project, tag, uid, cue, words)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not text:
        return jsonify({"error": "Ehhez a sorhoz nincs szöveg (az előkészítés futtatandó)."}), 404
    if not _voice_ok(project, voice):
        if voice.engine == "xtts":
            return jsonify({"error": "A klónhanghoz XTTS modellkönyvtár "
                                     "(models.xtts) és referenciafelvétel "
                                     "(models.xtts_ref) kell."}), 400
        return jsonify({"error": f"A {voice.engine} modell-mappa hiányzik vagy "
                                 f"nincs beállítva (Beállítások → Modellek)."}), 400

    digest = hashlib.sha1(
        f"{_LINE_PREVIEW_VERSION}|{voice.vid}|{pitch:.2f}|{text}".encode("utf-8")
    ).hexdigest()[:20]
    cache = project.work / f"_linepreview_{_LINE_PREVIEW_VERSION}"
    out = cache / f"{digest}.wav"
    if not out.is_file():
        try:
            engine = _voice_preview_engine(project, voice.engine)
        except FileNotFoundError as exc:
            return jsonify({"error": f"A modell nem tölthető be: {exc}"}), 400
        tmp = out.with_suffix(".part")
        try:
            with _PREVIEW_MUTEX:
                samples, sr = engine.synth(text, voice)
                if pitch:
                    samples = audio_mod.pitch_shift(samples, sr, pitch)
                cache.mkdir(parents=True, exist_ok=True)
                _write_wav(tmp, samples, sr)
            tmp.replace(out)
        except Exception as exc:                     # pragma: no cover — motorfüggő
            tmp.unlink(missing_ok=True)
            return jsonify({"error": f"A minta nem készíthető el: "
                                     f"{type(exc).__name__}: {exc}"}), 500
    return send_file(str(out), mimetype="audio/wav", conditional=True)


@bp.route("/api/dub/lines/name", methods=["POST"])
def rename_character():
    _, _, linereview = _dubstudio()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    try:
        linereview.set_name(project, tag, str(body.get("char_id") or ""),
                            str(body.get("name") or ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    payload = _payload(project, tag, _project_key(body.get("project")))
    return jsonify({"ok": True, "cast": payload["cast"], "lines": payload["lines"],
                    "stats": payload["stats"]})


@bp.route("/api/dub/lines/person", methods=["POST"])
def edit_person():
    """Egy szereplő nevének és magyar hangjának szerkesztése.

    A ``C###`` szereplők hangja a hangzárba kerül (a következő szinkronfutás
    ezt használja), a felületen létrehozott ``N###`` szereplőké a
    felülvizsgálati állományba (``review/lines.json``). Így a meghallgatás
    mindkét fajtánál a kiválasztott hangon szól.
    """
    _, _, linereview = _dubstudio()
    body = request.get_json(silent=True) or {}
    project = _project(body.get("project"))
    tag = _tag(body.get("episode"))
    cid = str(body.get("char_id") or "").strip()
    if not cid:
        return jsonify({"error": "Hiányzó szereplő-azonosító."}), 400
    name = str(body.get("name") or "").strip() or None
    has_voice = "voice" in body
    voice = str(body.get("voice") or "").strip()
    if has_voice and voice and voice not in {v.vid for v in _voice_catalog().CATALOG}:
        return jsonify({"error": f"Ismeretlen hang: {voice}."}), 400

    records = _cast_rows(project)
    people = linereview.load(project, tag).get("people") or {}
    if cid in records:
        if name:
            linereview.set_name(project, tag, cid, name)
        if has_voice:
            registry = _registry()
            reg = registry.Registry(project.cast_file,
                                    registry.share_threshold(project.tuning))
            row = next((c for c in reg.characters if c.char_id == cid), None)
            if row is None:
                return jsonify({"error": f"Nincs ilyen szereplő: {cid}."}), 404
            row.voice = voice or None
            try:
                reg.save()
            except OSError as exc:
                return jsonify({"error": f"A szereplő-rekord nem menthető: {exc}"}), 500
    elif cid in people:
        try:
            linereview.set_person(project, tag, cid, name=name,
                                  voice=voice if has_voice else None)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        return jsonify({"error": f"Nincs ilyen szereplő ebben az epizódban: {cid}."}), 404

    payload = _payload(project, tag, _project_key(body.get("project")))
    return jsonify({"ok": True, "cast": payload["cast"], "lines": payload["lines"],
                    "voices": payload["voices"], "stats": payload["stats"]})
