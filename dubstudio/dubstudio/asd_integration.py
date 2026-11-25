"""Képi ASD (aktív beszélő detektálás) integráció a szinkronláncba.

A kép **sosem dönt egyedül** a hang helyett: a besorolás a hangból jön, a
kép csak azt mutatja, hogy a beszélő arca látszik-e és szája jár-e a
hallható beszéd alatt. Ezért az ASD eredmények **bizonyíték** (evidence) a
riportban, nem kapu — per ``docs/asd.md``.

Az integráció három feladatot lát el:

1. ``load_asd_results(ep_dir)`` — a ``work/<EP>/visual/asd.json`` betöltése.
2. ``apply_asd_evidence(utts, asd_rows)`` — minden megnyilatkozáshoz
   hozzárendeli az ASD-eredményt (arc látszik-e, beszél-e, prob, verdict)
   az ``utterance.asd`` mezőbe (dict), ha van egyező uid/idő.
3. ``asd_metrics(utts)`` — QA-szintű összefoglaló: hány soron volt látható
   beszélő, hányon képen kívüli, hány nem eldönthető; ezt a riport ``asd``
   szakasza használja.

A modul **nem** módosít klasztert, char-t vagy voice-t: csak evidence-t
tárol, hogy a felülvizsgáló és a riport lássa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utterances import Utterance


def load_asd_results(ep_dir: Path) -> list[dict]:
    """Az epizód visual/asd.json betöltése (ha létezik)."""
    p = Path(ep_dir) / "visual" / "asd.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "rows" in data:
        return list(data["rows"])
    return []


def _match_row(utt: Utterance, rows: list[dict]) -> dict | None:
    """Megnyilatkozáshoz illő ASD-sor (uid egyezés, vagy időpont)."""
    # 1) uid egyezés
    for r in rows:
        if r.get("uid") is not None and str(r.get("uid")) == str(utt.uid):
            return r
    # 2) időpont közelség
    best, best_dt = None, 1e9
    for r in rows:
        t = r.get("t")
        if t is None:
            continue
        dt = abs(float(t) - float(utt.start))
        if dt < best_dt and dt <= 0.5:
            best, best_dt = r, dt
    return best


def apply_asd_evidence(utts: list[Utterance],
                       asd_rows: list[dict] | None = None,
                       ep_dir: Path | None = None) -> int:
    """ASD-eredmények hozzárendelése a megnyilatkozásokhoz.

    Visszaadja, hány megnyilatkozás kapott evidence-t.
    Az utterance.asd mezőbe (dict) kerül: {verdict, speaking_face, prob,
    faces, present_s, speech_s, weight}.
    """
    if asd_rows is None:
        if ep_dir is None:
            return 0
        asd_rows = load_asd_results(ep_dir)
    if not asd_rows:
        return 0
    n = 0
    for u in utts:
        row = _match_row(u, asd_rows)
        if row is None:
            continue
        light = row.get("light") or {}
        verdict = light.get("verdict") or row.get("verdict")
        speaking = light.get("speaking")
        cands = light.get("cands") or []
        best_prob = None
        best_face = None
        top = None
        if cands:
            # a legnagyobb prob-ú arc
            top = max(cands, key=lambda c: float(c.get("prob", 0.0)))
            best_prob = float(top.get("prob", 0.0))
            best_face = top.get("face")
        # A present_s / speech_s ugyanahhoz az archoz tartozik, mint a prob
        # (korábban a cands[0]-ból jött, ami másik arc is lehetett).
        u.asd = {
            "verdict": verdict,
            "speaking_face": speaking,
            "prob": best_prob,
            "face": best_face,
            "faces": row.get("faces", len(cands)),
            "present_s": (top.get("present_s") if top else None),
            "speech_s": (top.get("speech_s") if top else None),
            "weight": row.get("weight"),
        }
        n += 1
    return n


def asd_metrics(utts: list[Utterance]) -> dict:
    """QA-szintű összefoglaló az ASD evidence alapján."""
    total = 0
    visible_speaker = 0
    offscreen = 0
    undecided = 0
    no_face = 0
    with_evidence = 0
    examples = []
    for u in utts:
        a = getattr(u, "asd", None)
        if not a:
            continue
        with_evidence += 1
        total += 1
        verdict = str(a.get("verdict") or "")
        prob = a.get("prob")
        # A kategóriát a ``decide_light`` tényleges kimenetei szerint kell
        # bontani (a korábbi „beszél” részstring a „nincs beszélő arc”-ot is
        # elkaphatta, a „nincs beszélő arc a képen” pedig sehova nem esett).
        low = verdict.lower()
        if a.get("speaking_face") is not None or "arc #" in low:
            visible_speaker += 1
        elif low.startswith("nincs beszélő arc") or "képen kívüli" in low:
            offscreen += 1
        elif low.startswith("nincs arc"):
            no_face += 1
        else:
            undecided += 1
        if len(examples) < 12:
            examples.append({
                "uid": u.uid, "start": round(float(u.start), 2),
                "verdict": verdict, "prob": prob, "face": a.get("face"),
                "voice": u.voice, "char": u.char,
            })
    return {
        "n": with_evidence,
        "visible_speaker": visible_speaker,
        "offscreen_speaker": offscreen,
        "undecided": undecided,
        "no_face": no_face,
        "examples": examples,
    }


# Utterance dataclass bővítése: asd mező (ha még nincs)
def ensure_asd_field(utt: Utterance) -> None:
    if not hasattr(utt, "asd"):
        utt.asd = None
