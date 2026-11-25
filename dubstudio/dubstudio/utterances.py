"""Feliratsorok -> **akusztikusan összefüggő megnyilatkozások**.

Ez az a lépés, ami a „színész egy mondaton belül hangot vált” hibát
szerkezetileg megszünteti: a szinkronhang a *megnyilatkozás* szintjén
rögzül, a megnyilatkozás pedig akkor szakad meg, amikor a hang maga
megváltozik.

A csoportosítás bizonyítékai, erősségi sorrendben:
  1. **a hang** — a sorok saját CAM++ embeddingjének hasonlósága
     (``cos_same`` felett egy beszélő, ``cos_diff`` alatt más);
  2. **a szöveg folytonossága** — ha az előző sor nem zár mondatot, és a
     következő kisbetűvel kezdődik, ugyanaz a mondat (tehát ugyanaz a hang):
     ez felülírja a hang-bizonytalanságot, mert egy mondat nem válthat hangot;
  3. **az időzítés** — a feliratsorok közti szünet.

Az átfedő beszédet nem próbáljuk szétvágni: ott a hangok valóban egyszerre
szólnak, és a magyar szinkron is egyszerre szólal meg (mint az eredetiben).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import subs
from .embeddings import centroid, cosine, norm

#: hang-alapú döntési küszöbök (CAM++ koszinusz; a projekt felülírhatja)
DEFAULTS = {
    "gap_merge": 0.90,     # ennél nagyobb szünetnél csak szöveg-folytonosságra olvadunk
    "gap_cont": 2.20,      # e fölött akkor sem olvadunk, ha folytatódik a mondat
                            # (mért hiba: 1,36 mp szünet szakította ketté a mondatot)
    "cos_same": 0.55,      # ennyi felett: ugyanaz a beszélő
    "cos_diff": 0.22,      # ennyi alatt: más beszélő (ha van szünet)
    "diff_min_gap": 0.10,  # rövidebb szünettel nem váltunk (egy levegővétel)
    "split_cos": 0.40,     # egy feliratsoron belüli kétbeszélős vágás küszöbe
    "split_min_s": 0.75,   # ennél rövidebb sort nem vágunk ketté (a 0,84 s-os
                            # „- Igen? - Igen.” is két beszélő)
}


@dataclass
class Utterance:
    """Egy beszélő egy folyamatos megszólalása (egy vagy több feliratsor)."""

    uid: str
    ep: str
    start: float
    end: float
    cues: list[subs.Cue] = field(default_factory=list)
    text: str = ""
    en: str = ""
    emb: np.ndarray | None = None
    p_female: float | None = None
    cue_embs: list = field(default_factory=list)
    #: nyers diarizációs címke (a klaszterezés előtti beszélőforduló)
    diar: int | None = None
    cluster: int | None = None
    char: str | None = None
    voice: str | None = None
    #: hány külön beszélő szól a megnyilatkozás idejében a diarizáció szerint
    #: (2 vagy több = egy hangon megy ki két ember: ez a kapu alapja)
    multi_diar: int = 0
    parts: list[dict] = field(default_factory=list)
    #: képi ASD evidence (dict vagy None): verdict, speaking_face, prob, ...
    asd: dict | None = None

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def music(self) -> bool:
        """Csak zene/dalszöveg/háttérhang sorokból áll (nincs benne beszéd)?"""
        return bool(self.cues) and all(c.music for c in self.cues)

    @property
    def first(self) -> subs.Cue:
        return self.cues[0]

    @property
    def last(self) -> subs.Cue:
        return self.cues[-1]

    def to_dict(self) -> dict:
        return {"uid": self.uid, "ep": self.ep, "start": round(self.start, 3),
                "end": round(self.end, 3), "text": self.text, "en": self.en,
                "diar": self.diar, "cluster": self.cluster,
                "char": self.char, "voice": self.voice,
                "dur": round(self.dur, 3),
                "cues": [{"index": c.index, "start": round(c.start, 3),
                          "end": round(c.end, 3), "text": c.text,
                          "music": bool(c.music)} for c in self.cues],
                "multi_diar": int(self.multi_diar),
                "p_female": None if self.p_female is None else round(self.p_female, 3),
                "emb": None if self.emb is None else [round(float(v), 5) for v in self.emb],
                "asd": self.asd}

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        u = cls(uid=d["uid"], ep=d["ep"], start=float(d["start"]),
                end=float(d["end"]), text=d.get("text", ""), en=d.get("en", ""))
        u.cues = [subs.Cue(index=c.get("index", i + 1), start=float(c["start"]),
                           end=float(c["end"]), text=c.get("text", ""),
                           music=bool(c.get("music")))
                  for i, c in enumerate(d.get("cues", []))]
        u.diar = d.get("diar")
        u.cluster = d.get("cluster")
        u.char = d.get("char")
        u.voice = d.get("voice")
        u.p_female = d.get("p_female")
        u.multi_diar = int(d.get("multi_diar") or 0)
        u.emb = (None if not d.get("emb")
                 else np.asarray(d["emb"], dtype=np.float32))
        u.asd = d.get("asd")
        return u


def _continuation(prev: subs.Cue, cur: subs.Cue) -> bool:
    # Az elhagyás-jel („…”) nem zár mondatot: a levegővétel után ugyanaz a
    # beszélő fejezi be a mondatot. Máskülönben egy mondat két hangra szakad
    # (mért hiba: „A zsidók meg…” + „visszamehetnek oda…” két hangon szólt).
    t = subs.one_line(prev.text).rstrip()
    unfinished = t.endswith(("…", "...")) or not subs.ends_sentence(t)
    return unfinished and subs.starts_lower(cur.text)


def group(cues: list[subs.Cue], embs: list[np.ndarray | None],
          tuning: dict | None = None) -> list[list[int]]:
    """A sorok indexeiből álló csoportok (megnyilatkozások)."""
    t = {**DEFAULTS, **(tuning or {})}
    groups: list[list[int]] = []
    for i, cue in enumerate(cues):
        if not groups:
            groups.append([i])
            continue
        prev_idx = groups[-1][-1]
        prev, e_prev, e_cur = cues[prev_idx], embs[prev_idx], embs[i]
        gap = cue.start - prev.end
        sim = cosine(e_prev, e_cur) if (e_prev is not None and e_cur is not None) else None
        cont = _continuation(prev, cue)

        # A sor eleji „- ”/„[” új beszélő fordulója: két forduló nem kerülhet
        # egy megnyilatkozásba (különben egy hangon szólal meg két színész).
        # Kisbetűs folytatásnál (cont) maradhat egy mondat.
        # A sor végén lógó fordulójel („- A? -” / „B.”) is beszélőváltás:
        # a jel a *következő* sor fordulóját vezeti be. Enélkül a két színész
        # szövege egy megnyilatkozásba (és egy hangra) kerül.
        if subs.starts_new_turn(cue.text) and not cont:
            merge = False
        elif subs.ends_with_turn_dash(prev.text) and not cont:
            merge = False
        elif gap > t["gap_cont"]:
            merge = False
        elif gap > t["gap_merge"]:
            merge = cont
        elif sim is None:
            merge = True
        elif sim >= t["cos_same"]:
            merge = True
        elif sim <= t["cos_diff"] and gap > t["diff_min_gap"] and not cont:
            merge = False
        else:
            merge = cont

        if merge:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


#: kétbeszélős sor diarizáció-alapú vágása: ennél rövidebb fordulót nem
#: tekintünk valódi beszélőnek (a szomszéd beszéd átszivárgása ennél rövid)
SPLIT_RUN_MIN_S = 0.6
#: két *egymás utáni* forduló közt ennyi szünetnek kell lennie. A diarizációs
#: szegmenshatár önmagában nem bizonyíték fordulóváltásra: a határnál a
#: szegmensek összeérnek (mért eset: E04 250,1 s és 1409,3 s sora — a mérés
#: szerint mindkettő egy beszélő, P(nő)=0,001).
SEQUENTIAL_GAP_S = 0.15
#: ennél rövidebb forduló a szegmenshatár zajára utal, nem valódi beszélőre.
#: Mért eset (The Twin Who Lived, 4432 s): a szomszédos címkék 0,5–0,9 s-os
#: szigetei ugyanazt a színészt vágták el, és a kétbeszélős kaput hamisan
#: billentették — a szinkronhang helyesen egy volt.
MULTI_RUN_MIN_S = 1.2

#: mondathatárok (a szöveg vágásához, idő helyett)
_SENTENCE = re.compile(r"(?<=[.!?…\"])\s+")


def split_by_speaker(cue: subs.Cue, runs: list[tuple[int, float, float]],
                     tuning: dict | None = None,
                     min_run: float = SPLIT_RUN_MIN_S
                     ) -> list[subs.Cue] | None:
    """Kétbeszélős sor szétvágása **diarizáció + mondathatár** alapján.

    Akkor kell, amikor a felirat a fordulót nem jelölte (nincs „- ” a sor
    elején), de a hang szerint **két beszélő** szól benne — mért hiba: a
    341,12 s-os sor (férfi az autóban) egy női hangon ment ki, mert a sort a
    nő dominálta.

    A vágás **ideje** a diarizációs váltás, a **szöveg** vágása a hozzá
    legközelebbi mondathatár; ha nincs mondathatár, a szavak arányában
    osztunk. Így a két forduló külön megnyilatkozás — és külön hang — lesz.
    """
    t = {**DEFAULTS, **(tuning or {})}
    if cue.dur < t["split_min_s"] or not runs:
        return None
    clips = [(lab, max(cue.start, a), min(cue.end, b))
             for lab, a, b in runs]
    clips = sorted((c for c in clips if c[2] - c[1] >= min_run),
                   key=lambda c: c[1])
    if len({lab for lab, _, _ in clips}) < 2:
        return None
    cut: float | None = None
    for (l1, _a1, b1), (l2, a2, _b2) in zip(clips, clips[1:]):
        if l1 != l2:
            cut = (b1 + a2) / 2.0
            break
    if cut is None or not (cue.start + 0.15 < cut < cue.end - 0.15):
        return None

    text = subs.one_line(cue.text)
    words = text.split()
    if len(words) < 3:
        return None
    # mondathatárok karakter-pozíciói
    bounds = [m.start() for m in _SENTENCE.finditer(text)]
    ratio = (cut - cue.start) / max(cue.dur, 1e-6)
    target = ratio * len(text)
    if bounds:
        best = min(bounds, key=lambda b: abs(b - target))
    else:                                  # nincs mondathatár: szóarány
        best = len(" ".join(words[:max(1, round(len(words) * ratio))]))
    best = max(1, min(len(text) - 1, best))
    first, second = text[:best].strip(), text[best:].strip()
    if len(first.split()) < 2 or len(second.split()) < 1:
        return None
    return [subs.Cue(index=cue.index, start=round(cue.start, 3),
                     end=round(cut, 3), text=first),
            subs.Cue(index=cue.index, start=round(cut, 3),
                     end=round(cue.end, 3), text=second)]


def split_turns(cue: subs.Cue, tuning: dict | None = None
                ) -> list[subs.Cue] | None:
    """Egy feliratsor **összes** beszélő-fordulójának kivágása.

    A feliratkészítő szöveges jelölése („- A? - B.”, „[A] [B.]”) mondja meg,
    hol vált a beszélő; ez erősebb bizonyíték, mint egy másfél másodperces,
    zenével kevert ablak akusztikája. A fordulók időtartamát a szavak
    arányában osztjuk el a feliratsor hosszán.

    Visszaadás: legalább két külön ``Cue`` (azonos indexszel), különben None.
    """
    t = {**DEFAULTS, **(tuning or {})}
    if cue.dur < t["split_min_s"]:
        return None
    parts = subs.split_dialogue_parts(cue.text)
    if not parts:
        return None
    weights = [max(1, len(p.split())) for p in parts]
    total = float(sum(weights))
    out: list[subs.Cue] = []
    acc = 0.0
    start = cue.start
    for i, (text, w) in enumerate(zip(parts, weights)):
        acc += w
        end = cue.end if i == len(parts) - 1 else cue.start + (acc / total) * cue.dur
        out.append(subs.Cue(index=cue.index, start=round(start, 3),
                            end=round(end, 3), text=text))
        start = end
    return out if len(out) >= 2 else None


def build(ep: str, cues: list[subs.Cue], embs: list[np.ndarray | None],
          pgen: list[float | None] | None = None,
          embed_span=None, tuning: dict | None = None,
          en_cues: list[subs.Cue] | None = None,
          gender_span=None) -> list[Utterance]:
    """Feliratsorokból megnyilatkozások (fordulónkénti kétbeszélős vágással).

    ``gender_span(a, b)``: a kivágott forduló nem-mérése (P(nő)); a teljes
    feliratsorra mért érték a kétbeszélős soron értelmetlen (két nem
    átlaga), ezért a vágott részeket külön kell mérni.
    """
    t = {**DEFAULTS, **(tuning or {})}
    english = subs.SubtitlePair(eng=list(en_cues or []))
    pgen = pgen or [None] * len(cues)

    by_cue = {id(c): embs[i] for i, c in enumerate(cues)}
    p_by_cue = {id(c): pgen[i] for i, c in enumerate(cues)}

    def make(group_cues: list[subs.Cue]) -> Utterance:
        uid = f"{ep}:{len(items) + 1:04d}"
        u = Utterance(uid=uid, ep=ep,
                      start=round(min(c.start for c in group_cues), 3),
                      end=round(max(c.end for c in group_cues), 3),
                      cues=list(group_cues))
        # A zene/dalszöveg sorok nem dialógusok: sem a szövegbe, sem az
        # embeddingbe, sem a nem-mérésbe nem kerülnek bele. A sor a csoportban
        # megmarad, így a megnyilatkozás-azonosítók (uid) és a kézi
        # felülvizsgálat kulcsai nem csúsznak el.
        spoken = [c for c in group_cues if not c.music]
        u.text = " ".join(subs.one_line(c.text) for c in spoken).strip()
        m = english.match(spoken[0]) if spoken else None
        u.en = subs.one_line(m.text) if m else ""
        vs = []
        for c in spoken:
            v = by_cue.get(id(c))
            if v is None and embed_span is not None:
                v = embed_span(c.start, c.end)   # vágott rész: saját ablakból
            vs.append(v)
        ws = [max(0.2, c.dur) for c in spoken]
        u.emb = centroid([v for v in vs if v is not None],
                         [w for v, w in zip(vs, ws) if v is not None])
        u.cue_embs = list(vs)
        ps = []
        for c in spoken:
            p = p_by_cue.get(id(c))
            if p is None and id(c) not in p_by_cue and gender_span is not None:
                p = gender_span(c.start, c.end)   # vágott forduló: saját mérés
            if p is not None:
                ps.append((p, max(0.2, c.dur)))
        if ps:
            w = sum(x[1] for x in ps)
            u.p_female = float(sum(x[0] * x[1] for x in ps) / max(w, 1e-6))
        return u

    items: list[Utterance] = []

    def emit(members: list[subs.Cue], depth: int = 0) -> None:
        """A csoport kimenete; minden többbeszélős sor fordulónként válik szét."""
        if not members:
            return
        for k, c in enumerate(members):
            parts = split_turns(c, t) if depth < 8 else None
            if not parts:
                continue
            # az előzó sorok az első fordulóhoz, a következők az utolsóhoz
            items.append(make(members[:k] + [parts[0]]))
            for sc in parts[1:-1]:
                items.append(make([sc]))
            emit([parts[-1]] + members[k + 1:], depth + 1)
            return
        items.append(make(members))

    for g in group(cues, embs, t):
        emit([cues[i] for i in g])
    return items


def build_diarized(ep: str, cues: list[subs.Cue], labels: list[int | None],
                   embs: list[np.ndarray | None],
                   pgen: list[float | None] | None = None,
                   embed_span=None, tuning: dict | None = None,
                   en_cues: list[subs.Cue] | None = None,
                   gender_span=None, label_span=None,
                   runs_span=None) -> list[Utterance]:
    """Feliratsorok → megnyilatkozások **diarizációs címkék** alapján.

    A csoportosítás nem a rövid feliratsor embeddingjére, hanem a teljes sávon
    mért beszélőfordulóra támaszkodik. Egy fordulón belül a sorok ugyanazt a
    címkét kapják, ezért nem válthat hangot minden mondatnál. A többbeszélős
    feliratsorokat a szöveges fordulók mentén továbbra is szétvágjuk; a
    vágott részek címkéjét az időablakukból mérjük.
    """
    t = {**DEFAULTS, **(tuning or {})}
    english = subs.SubtitlePair(eng=list(en_cues or []))
    pgen = pgen or [None] * len(cues)
    by_cue = {id(c): embs[i] for i, c in enumerate(cues)}
    p_by_cue = {id(c): pgen[i] for i, c in enumerate(cues)}

    def make(group_cues: list[subs.Cue], label: int | None) -> Utterance:
        uid = f"{ep}:{len(items) + 1:04d}"
        u = Utterance(uid=uid, ep=ep,
                      start=round(min(c.start for c in group_cues), 3),
                      end=round(max(c.end for c in group_cues), 3),
                      cues=list(group_cues))
        # A zene/dalszöveg sorok nem dialógusok: sem a szövegbe, sem az
        # embeddingbe, sem a nem-mérésbe nem kerülnek bele. A sor a csoportban
        # megmarad, így a megnyilatkozás-azonosítók (uid) és a kézi
        # felülvizsgálat kulcsai nem csúsznak el.
        spoken = [c for c in group_cues if not c.music]
        u.text = " ".join(subs.one_line(c.text) for c in spoken).strip()
        m = english.match(spoken[0]) if spoken else None
        u.en = subs.one_line(m.text) if m else ""
        vs = []
        for c in spoken:
            v = by_cue.get(id(c))
            if v is None and embed_span is not None:
                v = embed_span(c.start, c.end)
            vs.append(v)
        ws = [max(0.2, c.dur) for c in spoken]
        u.emb = centroid([v for v in vs if v is not None],
                         [w for v, w in zip(vs, ws) if v is not None])
        u.cue_embs = list(vs)
        ps = []
        for c in spoken:
            p = p_by_cue.get(id(c))
            if p is None and id(c) not in p_by_cue and gender_span is not None:
                p = gender_span(c.start, c.end)
            if p is not None:
                ps.append((p, max(0.2, c.dur)))
        if ps:
            w = sum(x[1] for x in ps)
            u.p_female = float(sum(x[0] * x[1] for x in ps) / max(w, 1e-6))
        # A diarizációs címke lesz a kezdeti klaszter. A `clustering` ezt
        # nem dobja el, hanem nem szerint ellenőrzi és csak indokolt esetben
        # vonja össze más fordulókkal. A nyers címkét külön tároljuk a
        # diagnosztikához és a címkén belüli szétvágáshoz.
        u.diar = label
        # A tiszta zene/dalszöveg megnyilatkozás nem szereplő: nincs klasztere,
        # így nem kap hangot, és nem hígítja a beszédidő-alapú kiosztást.
        u.cluster = None if u.music else label
        if runs_span is not None:
            # Hány külön beszélő szól a megnyilatkozás idejében?
            # Csak az **egymás utáni** (nem átfedő) beszélők számítanak:
            # ha ketten egyszerre beszélnek, a szinkronban is egyszerre
            # szólnak — az nem hiba, hanem átfedő beszéd.
            multi = 0
            for c in group_cues:
                runs = [r for r in runs_span(c.start, c.end)
                        if min(c.end, r[2]) - max(c.start, r[1]) >= SPLIT_RUN_MIN_S]
                runs.sort(key=lambda r: r[1])
                for a, b in zip(runs, runs[1:]):
                    if a[0] == b[0]:
                        continue
                    if b[1] - a[2] < SEQUENTIAL_GAP_S:
                        continue
                    # Mindkét fordulónak érdemi hosszúnak kell lennie: a
                    # rövid sziget a diarizáció jittere, nem második beszélő.
                    if min(a[2] - a[1], b[2] - b[1]) < MULTI_RUN_MIN_S:
                        continue
                    multi = 2
                    break
                if multi:
                    break
            u.multi_diar = multi
        return u

    # (cue, címke) párok; a kétbeszélős sorok szöveges fordulói külön cserét
    # kapnak, és a diarizációból megkapják a saját időablakukhoz tartozó címkét.
    # (cue, címke, kényszerített határ?) — a szétvágott fordulók nem
    # olvadhatnak vissza egy megnyilatkozássá (különben két beszélő kerül
    # egy hangra, mért hiba: a 250 s-os „Nos, vezessen biztonságosan. Örülök…”).
    pieces: list[tuple[subs.Cue, int | None, bool]] = []
    for i, c in enumerate(cues):
        lab = labels[i] if i < len(labels) else None
        if c.music:
            # dalszöveget nem vágunk ketté két beszélőre
            pieces.append((c, lab, False))
            continue
        parts = split_turns(c, t)
        if not parts and runs_span is not None:
            # A felirat nem jelölte a fordulót, de a hang szerint két beszélő
            # szól: diarizáció + mondathatár mentén vágunk (mért hiba: a
            # 341 s-os sor férfi szövege női hangon ment ki).
            parts = split_by_speaker(c, runs_span(c.start, c.end), t)
        if not parts:
            pieces.append((c, lab, False))
            continue
        for k, part in enumerate(parts):
            plab = label_span(part.start, part.end) if label_span is not None else lab
            pieces.append((part, plab if plab is not None else lab, k > 0))

    items: list[Utterance] = []
    group: list[subs.Cue] = []
    group_label: int | None = None
    for c, lab, forced in pieces:
        if not group:
            group, group_label = [c], lab
            continue
        prev = group[-1]
        gap = c.start - prev.end
        cont = _continuation(prev, c)
        same = lab == group_label
        if (forced or not same or gap > t["gap_cont"]
                or (gap > t["gap_merge"] and not cont)
                or (subs.starts_new_turn(c.text) and not cont)
                or (subs.ends_with_turn_dash(prev.text) and not cont)):
            items.append(make(group, group_label))
            group, group_label = [c], lab
        else:
            group.append(c)
    if group:
        items.append(make(group, group_label))
    return items


def normalize_parts(utt: Utterance) -> None:
    """A megnyilatkozás részeit (feliratsorait) időrendbe teszi (kimenethez)."""
    utt.parts.sort(key=lambda p: p["start"])
