"""Időzítési terv: feliratsorok -> szintetizálandó egységek.

**A hangot itt már nem választjuk** — a szereplő (és így a hang) a
megnyilatkozás szintjén rögzült (``utterances`` + ``registry``). Ez a modul
csak az időzítést és a szöveget készíti elő.

Szerkezeti garancia: egy megnyilatkozás minden sora **ugyanazt** a
``voice``/``char`` értéket kapja, tehát egy mondaton belül nem szólalhat meg
két hang. A ``qa.py`` ezt mérésként is ellenőrzi.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import audio, subs
from .utterances import Utterance

PAD_HEAD = 0.04
PAD_TAIL = 0.03
MIN_WINDOW = 0.30
GAP = 0.02            # ennyi csend marad két sor között (nincs átfedés)
SCREEN_TEXT_DB = -58.0


@dataclass
class Entry:
    """Egy szintetizálandó sor."""

    i: int
    uid: str
    ep: str
    hu: str
    cue_index: int = 0
    en: str = ""
    cue_start: float = 0.0
    cue_end: float = 0.0
    start: float = 0.0
    end: float = 0.0
    char: str | None = None
    voice: str | None = None
    cluster: int | None = None
    pitch: float = 0.0
    skip: bool = False
    reason: str = ""
    speech_db: float | None = None
    #: emberi döntés (felülvizsgálati mark) áll a sor hangja mögött
    manual: bool = False

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"i": self.i, "uid": self.uid, "ep": self.ep, "hu": self.hu,
                "en": self.en, "cue_start": round(self.cue_start, 3),
                "cue_end": round(self.cue_end, 3), "cue_index": self.cue_index,
                "start": round(self.start, 3),
                "end": round(self.end, 3), "dur": round(self.dur, 3),
                "char": self.char, "voice": self.voice, "cluster": self.cluster,
                "pitch": self.pitch, "skip": self.skip, "reason": self.reason,
                "manual": self.manual, "speech_db": self.speech_db}


def speech_db(x, sr: int, a: float, b: float) -> float | None:
    i0, i1 = int(max(0.0, a) * sr), int(max(0.0, b) * sr)
    seg = x[i0:i1]
    if len(seg) < int(0.05 * sr):
        return None
    return round(audio.rms_db(seg), 1)


def _segments(text: str, marks: list[dict] | None,
              default_char: str | None) -> list[tuple[str | None, str]]:
    """A feliratsor szövegét szereplő-szakaszokra vágja (szó szerint).

    A szótartományos megjelölés a sort a szavak arányában ketté (többé)
    választja: a kijelölt rész más hangon szól, a többi az automatikus
    szereplőé. A szótartomány nélküli mark az egész sorra vonatkozik.
    """
    words = str(text or "").split()
    if not words:
        return []
    owner: list[str | None] = [default_char] * len(words)
    for mark in marks or []:
        cid = str(mark.get("char") or "") or default_char
        span = mark.get("words")
        if not span:
            owner = [cid] * len(words)
            continue
        try:
            a, b = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        for i in range(max(0, a), min(len(words) - 1, b) + 1):
            owner[i] = cid
    out: list[tuple[str | None, str]] = []
    start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or owner[i] != owner[start]:
            out.append((owner[start], " ".join(words[start:i])))
            start = i
    return out


def build(utts: list[Utterance], x=None, sr: int | None = None,
          verbose: bool = True,
          pitch_of: dict[str, float] | None = None,
          offsets: dict[tuple[str, int], float] | None = None,
          labels: dict[tuple[str, int], list[dict]] | None = None,
          voices: dict[str, str] | None = None,
          auto_marks: dict[tuple[str, int], list[dict]] | None = None
          ) -> list[Entry]:
    """Megnyilatkozások -> időzített terv (egy sor = egy feliratsor).

    ``pitch_of``: szereplő -> fix hangszín-eltolás (félhang) a hangzárból.
    Enélkül két szereplő, aki ugyanazt a hangot kapta, **azonosan szólna** —
    ezt a lépcsőt eddig a terv nem vitte át a szintézisbe.

    ``offsets``: ``(uid, feliratsor index) -> másodperc`` — az emberi
    felülvizsgáló kézi elcsúszás-javítása (``work/<EP>/review/lines.json``).

    ``labels``: ``(uid, feliratsor index) -> markok`` — a felülvizsgáló kézi
    szereplő-megjelölései (``linereview.labels``). A teljes sorra szóló mark
    felülírja a sor hangját, a szótartományos mark kettévágja a sort. Ez a
    döntés **elsőbbséget élvez** az automatával szemben (``Entry.manual``).
    ``voices``: szereplő -> magyar hang (a hangzárból és a felületi ``N###``
    szereplőkből) — a kézi döntés hangjának feloldásához.

    ``auto_marks``: ``(uid, feliratsor index) -> markok`` — a szó-szintű
    hozzárendelésből (``word_assignment.marks_for_project``, a ``prepare``
    lépésben mentett fordulócímkékből) származó vágási javaslat. Csak akkor érvényesül, ha a sorra **nincs kézi mark**; a kézi
    döntés mindig elsõbbséget élvez. Az így vágott sor nem lesz ``manual``.
    """
    entries: list[Entry] = []
    pitch_map = {str(k): float(v) for k, v in (pitch_of or {}).items()}
    shifts = {k: float(v) for k, v in (offsets or {}).items()}
    voice_map = {str(k): str(v) for k, v in (voices or {}).items()}
    manual = dict(labels or {})
    n = 0
    music = 0
    hand = 0
    for u in utts:
        for cue in u.cues:
            if cue.music:
                # zene/dalszöveg/háttérhang: nem dialógus, nincs TTS-sor
                music += 1
                continue
            key = (u.uid, int(cue.index))
            marks = manual.get(key) or []
            auto = (auto_marks or {}).get(key) or []
            shift = shifts.get(key, 0.0)
            start = cue.start + shift
            end = cue.end + shift
            if start < 0 or end <= start:
                start, end = cue.start, cue.end
            if marks:
                hand += 1
            segs = _segments(cue.text, marks or auto, u.char)
            if not segs:
                segs = [(u.char, "")]
            counts = [max(1, len(part.split())) for _, part in segs]
            total = float(sum(counts))
            span_start = start
            spans: list[tuple[str | None, str, float, float]] = []
            for k, (cid, part) in enumerate(segs):
                take = (end - start) * counts[k] / total
                seg_end = end if k == len(segs) - 1 else round(span_start + take, 3)
                spans.append((cid, part, span_start, seg_end))
                span_start = seg_end
            for cid, part, a, b in spans:
                text = subs.tts_text(part)
                voice = voice_map.get(str(cid)) if cid else None
                if voice is None and cid == u.char:
                    voice = u.voice
                n += 1
                e = Entry(i=n, uid=u.uid, ep=u.ep, hu=text, en=u.en,
                          cue_index=int(cue.index),
                          cue_start=round(a, 3), cue_end=round(b, 3),
                          start=round(max(0.0, a - PAD_HEAD), 3),
                          end=round(b + PAD_TAIL, 3),
                          char=cid, voice=voice, cluster=u.cluster,
                          pitch=pitch_map.get(str(cid), 0.0),
                          manual=bool(marks))
                if x is not None and sr:
                    e.speech_db = speech_db(x, sr, a, b)
                if not text.strip():
                    e.skip, e.reason = True, "üres szöveg"
                elif subs.is_screen_text(cue.text) and (
                        e.speech_db is None or e.speech_db < SCREEN_TEXT_DB):
                    e.skip, e.reason = True, "képernyőszöveg, nincs beszéd"
                elif not e.voice:
                    # a klaszterezés képernyőszövegként kiszűrte: nincs
                    # szereplő és hang, ezért nem mehet a TTS-be
                    e.skip, e.reason = True, "nincs szereplő/hang"
                entries.append(e)

    # rövid sorok minimális ablaka — az átfedés kezelése ELŐTT, hogy a
    # későbbi kiterjesztés ne hozzon vissza átfedést
    live = [e for e in entries if not e.skip]
    live.sort(key=lambda e: e.start)
    for e in live:
        if e.dur < MIN_WINDOW:
            e.end = round(e.start + MIN_WINDOW, 3)

    # ablakok: **sosem lóghatnak egymásba**. A régi MAX_OVERLAP=0,12 a gyors
    # párbeszédeknél a régi sor farkát az új sor első szótagja alá tette
    # („beragad a régi hang”). Előbb az előző sor végét húzzuk vissza, és csak
    # ha az már nem mehet (0,30 s alá), akkor toljuk az új sor kezdetét.
    for a, b in zip(live, live[1:]):
        ov = (a.end + GAP) - b.start
        if ov <= 0:
            continue
        room_a = max(0.0, a.dur - MIN_WINDOW)
        take = min(ov, room_a)
        if take > 0:
            a.end = round(a.end - take, 3)
            ov -= take
        if ov > 0:
            b.start = round(b.start + ov, 3)

    if verbose:
        kept = [e for e in entries if not e.skip]
        print(f"[terv] {len(entries)} sor, {len(kept)} szinkronizálandó, "
              f"{sum(1 for e in entries if e.skip)} kihagyva"
              + (f", {music} zenei sor (nem dialógus)" if music else "")
              + (f", {hand} sor kézi szereplő-döntéssel" if hand else ""))
    return entries


def voice_consistency(entries: list[Entry]) -> list[dict]:
    """Egy megnyilatkozáson belül eltérő hang keresése (a garancia ellenőrzése)."""
    by_uid: dict[str, set] = {}
    for e in entries:
        # a kézi szereplő-döntés szándékos: nem „egy hangon két ember” hiba
        if e.skip or e.manual:
            continue
        by_uid.setdefault(e.uid, set()).add((e.voice, e.char))
    return [{"uid": uid, "voices": sorted(str(v) for v in voices)}
            for uid, voices in by_uid.items() if len(voices) > 1]
