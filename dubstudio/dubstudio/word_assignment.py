"""Szó-szintű hozzárendelés (word-level speaker assignment).

A ``plan.build`` sor-szintű: egy megnyilatkozás egy hangon szól. De ha a
diarizáció egy feliratsorban két beszélőt lát, és a szó-idők is megvannak,
akkor **szóhatáron** is szétvághatjuk a sort: a szavak egy része az egyik
beszélőé, a többi a másiké. Ez a „szó-szintű hozzárendelés” (a soron belüli
hangváltás pontos helyének meghatározása).

A modul két függvényt ad:

* ``assign(words, turns, ...)`` — minden szóhoz beszélő-azonosítót (klaszter
  vagy diarizációs címke) rendel az időbeli átfedés alapján, bizonytalanságot
  megtartva (ambiguous_overlap, no_turn_overlap, no_word_timing).
* ``split_utterance_by_words(utt, word_speakers)`` — ha a megnyilatkozás
  szavainak beszélője vált, szétvágja a megnyilatkozást szóhatáron, és
  külön Utterance-öket ad vissza (az eredeti uid-t megtartva indexszel).

A modul **nem** módosítja a hangot vagy a klasztert: csak evidence-t és
szétvágási javaslatot ad. A tényleges vágást a hívó (plan/linereview) végzi.

A ``plan.build`` bekötése az ``auto_marks`` útvonalon megy: **a vágás
előkészítése a ``prepare`` lépésben történik** (``segments_by_label`` +
``save_marks``), mert ott a diarizációs fordulók már a memóriában vannak — a
``dub`` nem futtat diarizációt csak a markokért. A ``prepare`` a
**fordulócímkéket** menti (nem karaktert, mert a karakter csak a klaszterezés
után ismert), a karakterre képezést a ``marks_for_project`` végzi a
megnyilatkozásokból. A ``plan.build`` a kézi mark hiányában ezekkel vágja a
sort: a vágás a meglévő, tesztelt útvonalon történik, és a kézi döntés mindig
elsőbbséget élvez.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .utterances import Utterance

# A speaker_audit.turn_segment_ids már megcsinálja az átfedés-alapú
# hozzárendelést; itt csak egy gazdagabb változatot adunk, amely a
# diarizációs címkéket és a klasztereket is kezeli.


@dataclass
class WordSpeaker:
    word: str
    start: float
    end: float
    speaker: str | None          # diarizációs címke (str) vagy None
    confidence: str              # turn_overlap_only / ambiguous_overlap / no_turn_overlap / no_word_timing
    char: str | None = None      # hozzárendelt karakter (ha van mapping)
    voice: str | None = None     # hozzárendelt hang (ha van mapping)


def assign(words: list[dict], turns: list[dict],
           turn_to_char: dict[str, str] | None = None,
           turn_to_voice: dict[str, str] | None = None,
           margin_s: float = 0.08) -> list[WordSpeaker]:
    """Minden szóhoz beszélő (és opcionálisan char/voice) hozzárendelés.

    ``turns``: lista of {start, end, speaker, ...} (diarizációs fordulók).
    ``turn_to_char`` / ``turn_to_voice``: diarizációs címke → karakter/hang.
    """
    turn_to_char = turn_to_char or {}
    turn_to_voice = turn_to_voice or {}
    out: list[WordSpeaker] = []
    for w in words or []:
        a, b = w.get("start"), w.get("end")
        if a is None or b is None:
            out.append(WordSpeaker(
                word=str(w.get("word", "")), start=0.0, end=0.0,
                speaker=None, confidence="no_word_timing"))
            continue
        a, b = float(a), float(b)
        scores: dict[str, float] = defaultdict(float)
        for t in turns or []:
            lo = max(a, float(t["start"]) - margin_s)
            hi = min(b, float(t["end"]) + margin_s)
            if hi > lo:
                scores[str(t["speaker"])] += hi - lo
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            ws = WordSpeaker(word=str(w.get("word", "")), start=a, end=b,
                             speaker=None, confidence="no_turn_overlap")
        elif len(ranked) > 1 and ranked[1][1] >= 0.2 * max(ranked[0][1], 1e-9):
            ws = WordSpeaker(word=str(w.get("word", "")), start=a, end=b,
                             speaker=None, confidence="ambiguous_overlap")
        else:
            sid = ranked[0][0]
            ws = WordSpeaker(word=str(w.get("word", "")), start=a, end=b,
                             speaker=sid, confidence="turn_overlap_only")
            ws.char = turn_to_char.get(sid)
            ws.voice = turn_to_voice.get(sid)
        out.append(ws)
    return out


def segments_from_word_speakers(ws: list[WordSpeaker]) -> list[dict]:
    """Szó-szintű beszélő-váltásokból folytonos szegmensek.

    Visszaad: lista of {start, end, speaker, char, voice, words: [str]}.
    Egy szegmens addig tart, amíg a beszélő nem vált.
    """
    segs: list[dict] = []
    cur: dict | None = None
    for w in ws:
        if cur is None or cur["speaker"] != w.speaker:
            if cur is not None:
                segs.append(cur)
            cur = {"start": w.start, "end": w.end, "speaker": w.speaker,
                   "char": w.char, "voice": w.voice, "words": [w.word]}
        else:
            cur["end"] = w.end
            cur["words"].append(w.word)
    if cur is not None:
        segs.append(cur)
    return segs


def split_points(ws: list[WordSpeaker]) -> list[float]:
    """Azok a szóhatárok (másodperc), ahol a beszélő vált.

    Ezek a pontok használhatók a megnyilatkozás szétvágására.
    """
    pts: list[float] = []
    prev_spk = None
    for w in ws:
        if prev_spk is not None and w.speaker != prev_spk and w.speaker is not None:
            pts.append(w.start)
        if w.speaker is not None:
            prev_spk = w.speaker
    return pts


def split_utterance_by_words(utt: Utterance, ws: list[WordSpeaker],
                             min_words: int = 2) -> list[Utterance]:
    """Megnyilatkozás szétvágása szóhatáron, ha a beszélő vált.

    Visszaad: lista of Utterance (ha nem vált, egyelemű lista az eredetivel).
    A szétvágás csak akkor történik, ha legalább ``min_words`` szó van
    mindkét oldalon. Az új uid-k az eredeti uid + index.
    """
    if not ws or len(ws) < 2 * min_words:
        return [utt]
    segs = segments_from_word_speakers(ws)
    # Csak akkor vágunk, ha legalább 2 különböző beszélő van
    speakers = {s["speaker"] for s in segs if s["speaker"] is not None}
    if len(speakers) < 2:
        return [utt]
    # A rövid szegmenst az előzőhöz csatoljuk: a szövege **nem** veszhet el
    # (a korábbi `continue` a középső töredék szövegét eldobta).
    merged: list[dict] = []
    for seg in segs:
        if len(seg["words"]) < min_words and merged:
            prev = merged[-1]
            prev["end"] = seg["end"]
            prev["words"] = prev["words"] + seg["words"]
            continue
        merged.append(dict(seg))
    out: list[Utterance] = []
    for i, seg in enumerate(merged):
        text = " ".join(seg["words"])
        u = Utterance(uid=f"{utt.uid}w{i}", ep=utt.ep,
                      start=round(seg["start"], 3), end=round(seg["end"], 3),
                      cues=list(utt.cues), text=text, en=utt.en)
        u.cluster = utt.cluster
        u.char = seg.get("char") or utt.char
        u.voice = seg.get("voice") or utt.voice
        u.p_female = utt.p_female
        u.emb = utt.emb
        u.diar = utt.diar
        u.cue_embs = list(utt.cue_embs)
        u.asd = utt.asd
        # a vágott rész egy beszélőé: nincs benne két beszélő
        u.multi_diar = 0
        out.append(u)
    if not out:
        return [utt]
    return out


def coverage_report(ws: list[WordSpeaker]) -> dict:
    """Összefoglaló: hány szó kapott beszélőt, hány bizonytalan."""
    total = len(ws)
    assigned = sum(1 for w in ws if w.speaker is not None)
    ambiguous = sum(1 for w in ws if w.confidence == "ambiguous_overlap")
    no_overlap = sum(1 for w in ws if w.confidence == "no_turn_overlap")
    no_timing = sum(1 for w in ws if w.confidence == "no_word_timing")
    return {
        "total": total,
        "assigned": assigned,
        "ambiguous": ambiguous,
        "no_overlap": no_overlap,
        "no_timing": no_timing,
        "coverage": round(assigned / max(total, 1), 4),
    }


# ── bekötés a plan.build-be ─────────────────────────────────────────────

def cue_word_times(cue) -> list[dict]:
    """A feliratsor szavainak időbecslése (a sor hosszán arányosan).

    Ha van valódi ASR szóidő, azt kell használni; ez a függvény arra az
    esetre van, amikor csak a feliratsor ideje ismert. A szóidő a szó
    hosszával arányos, mert a magyar szövegben a hosszabb szó tovább tart.
    """
    words = str(getattr(cue, "text", "") or "").split()
    if not words:
        return []
    weights = [max(1, len(w)) for w in words]
    total = float(sum(weights))
    dur = max(float(cue.end) - float(cue.start), 1e-3)
    out: list[dict] = []
    acc = 0.0
    pos = float(cue.start)
    for word, w in zip(words, weights):
        acc += w
        end = float(cue.start) + dur * acc / total
        out.append({"word": word, "start": round(pos, 3), "end": round(end, 3)})
        pos = end
    return out


def _label_to_char(utts: list[Utterance]) -> dict[str, str]:
    """Diarizációs címke -> karakter (a leghosszabb beszédidő szerint).

    A nyers címke nem döntés: a klaszterezés több címkét is összevonhat egy
    szereplővé. Ezért a szó-szintű vágás is **karakterre** csoportosít, hogy
    ne bontsa vissza a már helyesen összevont címkéket.
    """
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for u in utts:
        if u.diar is None or not u.char:
            continue
        votes[str(u.diar)][str(u.char)] += max(u.dur, 0.01)
    return {lab: max(by.items(), key=lambda kv: kv[1])[0]
            for lab, by in votes.items()}


def segments_by_label(utts: list[Utterance], turns: list[dict],
                      min_words: int = 2, min_seg_s: float = 0.20
                      ) -> dict[tuple[str, int], list[dict]]:
    """Szótartományos beszélő-váltások **nyers fordulócímkékkel**.

    Ez a ``prepare`` lépés kimenete: csak a diarizáció fordulóit és a
    feliratsor szó-időit használja, karaktert (szereplőt) **nem** — az akkor
    még nem ismert. A karakterre képezést a ``resolve_marks`` végzi.

    Visszaadás: ``{(uid, feliratsor index): [{"label", "words": [a, b]}]}``.
    """
    out: dict[tuple[str, int], list[dict]] = {}
    for u in utts:
        for cue in u.cues:
            if getattr(cue, "music", False):
                continue
            words = cue_word_times(cue)
            if len(words) < 2 * min_words:
                continue
            local = [t for t in turns
                     if min(float(cue.end), float(t["end"]))
                     - max(float(cue.start), float(t["start"])) > 0.0]
            if not local:
                continue
            segs = segments_from_word_speakers(assign(words, local))
            if len(segs) < 2:
                continue
            if any(s["speaker"] is None for s in segs):
                continue
            if any((s["end"] - s["start"]) < min_seg_s for s in segs):
                continue
            marks: list[dict] = []
            i = 0
            for s in segs:
                marks.append({"label": str(s["speaker"]),
                              "words": [i, i + len(s["words"]) - 1]})
                i += len(s["words"])
            out[(u.uid, int(cue.index))] = marks
    return out


def resolve_marks(labeled: dict[tuple[str, int], list[dict]],
                  label_to_char: dict[str, str] | None = None
                  ) -> dict[tuple[str, int], list[dict]]:
    """Fordulócímkék → karakter (szereplő), a ``plan.build`` mark-formátumában.

    Csak akkor ad markot egy sorra, ha a címkék **két külön karakterre**
    váltanak: a nyers címkeváltás önmagában nem elég, mert a klaszterezés
    több címkét is egy szereplővé vonhatott össze. Az azonos karakterre eső
    szomszédos szegmenseket összevonjuk (különben a sor feleslegesen vágódna
    ketté ugyanarra a hangra).
    """
    l2c = {str(k): str(v) for k, v in (label_to_char or {}).items()}
    out: dict[tuple[str, int], list[dict]] = {}
    for key, segs in (labeled or {}).items():
        merged: list[dict] = []
        for seg in segs or []:
            cid = l2c.get(str(seg.get("label")))
            if cid is None:
                merged = []          # ismeretlen címke: nincs biztos szereplő
                break
            try:
                a, b = int(seg["words"][0]), int(seg["words"][1])
            except (TypeError, ValueError, IndexError, KeyError):
                merged = []
                break
            if merged and merged[-1]["char"] == cid \
                    and merged[-1]["words"][1] + 1 == a:
                merged[-1]["words"][1] = b
                continue
            merged.append({"char": cid, "words": [a, b],
                           "kind": "word_auto"})
        if len(merged) < 2:
            continue
        if len({m["char"] for m in merged}) < 2:
            continue
        out[key] = merged
    return out


def auto_marks(utts: list[Utterance], turns: list[dict],
               label_to_char: dict[str, str] | None = None,
               min_words: int = 2, min_seg_s: float = 0.20
               ) -> dict[tuple[str, int], list[dict]]:
    """Automatikus szótartományos markok egy lépésben (kényelmi függvény).

    Ugyanaz, mint ``resolve_marks(segments_by_label(...), label_to_char)``;
    a láncban a két lépés külön fut (a címkék a ``prepare``-ben, a karakterre
    képezés a ``dub``-ban).
    """
    return resolve_marks(segments_by_label(utts, turns, min_words=min_words,
                                          min_seg_s=min_seg_s),
                        label_to_char)


# ── tartósítás (prepare -> dub) ─────────────────────────────────────────

MARKS_FILE = "word_marks.json"


def marks_path(project, tag: str) -> Path:
    """Az epizód szó-szintű vágásjelöltjei (a ``prepare`` írja)."""
    return project.ep_dir(str(tag).upper()) / MARKS_FILE


def save_marks(project, tag: str,
              labeled: dict[tuple[str, int], list[dict]]) -> Path:
    """Szótartományos címke-váltások mentése (``prepare`` lépés)."""
    path = marks_path(project, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    cues: dict[str, list[list]] = {}
    for (uid, idx), segs in sorted(labeled.items()):
        rows = []
        for s in segs or []:
            try:
                rows.append([str(s["label"]), int(s["words"][0]),
                             int(s["words"][1])])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        if len(rows) >= 2:
            cues[f"{uid}#{int(idx)}"] = rows
    data = {"pipeline": "dubstudio.word_marks v1",
            "episode": str(tag).upper(), "cues": cues}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load_segments(project, tag: str) -> dict[tuple[str, int], list[dict]]:
    """A ``prepare`` által mentett címke-váltások betöltése."""
    path = marks_path(project, tag)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[tuple[str, int], list[dict]] = {}
    for k, segs in (data.get("cues") or {}).items():
        uid, _, idx = str(k).rpartition("#")
        if not uid:
            continue
        try:
            idx_i = int(idx)
        except ValueError:
            continue
        rows = []
        for s in segs or []:
            try:
                rows.append({"label": str(s[0]), "words": [int(s[1]), int(s[2])]})
            except (TypeError, ValueError, IndexError):
                continue
        if len(rows) >= 2:
            out[(uid, idx_i)] = rows
    return out


def marks_for_project(project, tag: str,
                      utts: list[Utterance]) -> dict[tuple[str, int], list[dict]]:
    """A ``plan.build`` mark-jai: mentett címkék + a mostani karakter-kiosztás.

    Ha nincs mentett fájl (régi, ``prepare`` előtti munkakönyvtár), üres dict
    jön vissza — a terv ilyenkor a sor-szintű viselkedést követi.
    """
    return resolve_marks(load_segments(project, tag), _label_to_char(utts))
