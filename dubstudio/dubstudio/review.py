"""Emberi felülvizsgálat (review) állapota a beszélő- és hangkiosztáshoz.

A felület és a parancssor ugyanezt a fájlt írja: ``work/<EP>/source_audit/
review.json``. A döntések **emberiek**, ezért elsőbbségük van minden
automatikus találattal szemben, és túlélik az újrafuttatást.

Háromféle döntés van, szándékosan külön:

* ``accepted`` — a diarizált beszélő azonosítása jó (esetleg másik ``C###``);
* ``rejected`` — ez a beszélő nem kap karaktert (zörej, kórus, bemondó);
* ``voice``    — a javasolt hangot elfogadta vagy másikat választott.

Az azonosítás megerősítése egyben a galériát is frissíti
(``identity.confirm``), tehát a **következő epizódban** ugyanaz a hang már
stabilan ugyanazt a ``C###`` kapja.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PIPELINE = "dubstudio.review v1"
DECISIONS = ("accepted", "corrected", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def path_for(project, tag: str) -> Path:
    return project.ep_dir(tag.upper()) / "source_audit" / "review.json"


def load(project, tag: str) -> dict:
    p = path_for(project, tag)
    if not p.is_file():
        return {"pipeline": PIPELINE, "episode": tag.upper(), "speakers": {},
                "updated_at": None}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"pipeline": PIPELINE, "episode": tag.upper(), "speakers": {},
                "updated_at": None}
    data.setdefault("pipeline", PIPELINE)
    data.setdefault("episode", tag.upper())
    data.setdefault("speakers", {})
    return data


def save(project, tag: str, review: dict) -> Path:
    p = path_for(project, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    review["pipeline"] = PIPELINE
    review["episode"] = tag.upper()
    review["updated_at"] = _now()
    p.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def set_speaker(project, tag: str, speaker: str, decision: str,
                char_id: str | None = None, gender: str | None = None,
                voice: str | None = None, note: str = "",
                gallery_path: str | Path | None = None,
                prototype=None, samples=None, seconds: float = 0.0,
                name: str | None = None) -> dict:
    """Egy beszélő emberi döntésének rögzítése (+ galéria-frissítés).

    ``samples`` nélkül a galéria csak centroidot őriz, és centroidból nem lehet
    stabilan újraazonosítani (a különböző beszélők centroidjai 0,8–0,9
    koszinuszt adnak egymással). Ezért a hívónak a beszélő példány-lenyomatait
    is át kell adnia, ha vannak.
    """
    from . import identity

    decision = str(decision or "").strip().lower()
    if decision not in DECISIONS:
        raise ValueError(f"ismeretlen döntés: {decision!r} (lehetséges: {DECISIONS})")
    review = load(project, tag)
    entry = {
        "decision": decision,
        "char_id": char_id if decision != "rejected" else None,
        "gender": gender,
        "voice": voice,
        "note": str(note or "")[:400],
        "at": _now(),
    }
    if name:
        entry["name"] = str(name).strip()[:80]
    review["speakers"][str(speaker)] = entry
    save(project, tag, review)
    gallery_file = gallery_path or (project.work / "identity-gallery.json")
    gallery = identity.load_gallery(gallery_file)
    identity.confirm(gallery, tag, speaker, entry["char_id"], vec=prototype,
                     samples=samples, gender=gender, voice=voice,
                     seconds=seconds, note=note, name=entry.get("name"))
    identity.save_gallery(gallery_file, gallery)
    return review


def set_voice(project, tag: str, speaker: str, voice: str, accepted: bool = True,
              note: str = "") -> dict:
    """A hangkiosztás jóváhagyása vagy felülírása (az azonosítást nem érinti)."""
    review = load(project, tag)
    entry = review["speakers"].setdefault(str(speaker), {"decision": "accepted"})
    entry["voice"] = str(voice)
    entry["voice_accepted"] = bool(accepted)
    entry["voice_at"] = _now()
    if note:
        entry["note"] = str(note)[:400]
    save(project, tag, review)
    return review


def apply(identity_result: dict, review: dict) -> dict:
    """Az emberi döntéseket ráolvasztja az automatikus találatra (nézet)."""
    speakers = {k: dict(v) for k, v in (identity_result.get("speakers") or {}).items()}
    for speaker, entry in (review.get("speakers") or {}).items():
        row = speakers.setdefault(str(speaker), {"speaker": str(speaker),
                                                 "candidates": [], "status": "no_evidence",
                                                 "reason": "", "review_required": True})
        row["manual"] = True
        row["decision"] = entry.get("decision")
        row["manual_note"] = entry.get("note") or ""
        if entry.get("decision") == "rejected":
            row["char_id"] = None
            row["status"] = "rejected_by_human"
            row["review_required"] = False
        if entry.get("name"):
            row["name"] = entry["name"]
        if entry.get("char_id"):
            row["char_id"] = entry["char_id"]
            row["status"] = "confirmed" if entry.get("decision") == "accepted" else "corrected"
            row["review_required"] = False
            if entry.get("gender"):
                row["gender"] = entry["gender"]
        if entry.get("voice"):
            row["voice"] = entry["voice"]
            row["voice_accepted"] = bool(entry.get("voice_accepted"))
    identity_result = dict(identity_result)
    identity_result["speakers"] = speakers
    counts: dict[str, int] = {}
    for row in speakers.values():
        counts[row.get("status") or "unknown"] = counts.get(row.get("status") or "unknown", 0) + 1
    identity_result["counts"] = counts
    identity_result["pending_review"] = sum(
        1 for row in speakers.values() if row.get("review_required"))
    return identity_result


def summary(review: dict) -> dict:
    rows = list((review.get("speakers") or {}).values())
    return {
        "total": len(list(rows)),
        "accepted": sum(1 for r in rows if r.get("decision") == "accepted"),
        "corrected": sum(1 for r in rows if r.get("decision") == "corrected"),
        "rejected": sum(1 for r in rows if r.get("decision") == "rejected"),
        "voices": sum(1 for r in rows if r.get("voice")),
        "updated_at": review.get("updated_at"),
    }
