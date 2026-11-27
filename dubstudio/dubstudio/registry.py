"""Szereplő-regiszter és **hangzár** (``cast.json``).

A hangkeveredés elleni védelem nem a klaszterezésben, hanem itt van: a
klasztert egy **szereplő-rekordhoz** kötjük, a szereplő hangja pedig zárolt.
A szintézis nem választhat hangot — csak azt kapja meg, ami a rekordban áll.

Szabályok:
  1. egy szereplőhöz **pontosan egy** hang és egy fix hangszín-eltolás tartozik;
  2. két szereplő **ugyanabban a jelenetben** (együtt szólalnak fel) nem kaphatja
     ugyanazt a hangot, amíg van szabad hang — ezt a színezés kényszeríti ki;
  3. ha mégis osztozniuk kell (elfogytak a hangok), **külön fix pitch**
     különbözteti meg őket;
  4. az újrafuttatás **megtartja** a korábbi kiosztást (stabil), csak az új
     szereplőket osztja be — így egy epizód újragenerálása nem cserél hangot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import subs
from .utterances import Utterance
from .voices import CATALOG, Voice, VoiceBank, is_clone

#: fix hangszín-létra az azonos hangot osztozó szereplők szétválasztására
PITCH_LADDER = [0.0, 1.0, -1.0, 1.8, -1.8, 0.5, -0.5, 1.4, -1.4]

#: beszédidő-arány (az összes beszédhez képest), amitől valaki **főszereplő**;
#: a főszereplő hangja kizárólagos, a mellékszereplők osztozhatnak rajta
MAIN_SHARE_MIN = 0.04
#: visszatérő mellékszereplő: legalább 2 epizód és legalább 1% évadidő
SUPPORT_SHARE_MIN = 0.01
#: ennél rövidebb, egyébként epizód-/statisztaszerep
EPISODIC_MIN_S = 10.0
#: rétegsorrend a kiosztási prioritáshoz (0 = legfontosabb)
TIER_ORDER = {"fő": 0, "mellék": 1, "epizód": 2, "statiszta": 3}


def recall_threshold(tuning: dict | None) -> float:
    """A projekt `tuning.recall_cos` értéke (vagy a kalibrált alapértelmezés)."""
    try:
        return float((tuning or {}).get("recall_cos", RECALL_COS))
    except (TypeError, ValueError):
        return RECALL_COS


def share_threshold(tuning: dict | None) -> float:
    """A projekt `tuning.main_share_min` értéke (vagy az alapértelmezés)."""
    try:
        return float((tuning or {}).get("main_share_min", MAIN_SHARE_MIN))
    except (TypeError, ValueError):
        return MAIN_SHARE_MIN


def speaker_count(project) -> int:
    """A projekt ténylegesen beszélő szereplőinek száma a hangzár szerint.

    A kézi „egy hang az egész projektre” választó csak **egyszemélyes**
    videónál érvényesülhet: egy sorozatnál ugyanis minden szereplő hangját
    felülírná. Beszélőnek számít minden olyan szereplő, akinek van mért
    beszédideje — a töredék-klaszterek is, mert inkább figyelmeztessünk,
    mint hogy észrevétlenül elhangoljunk egy egész sorozatot.
    """
    try:
        reg = Registry(project.cast_file, share_threshold(project.tuning))
    except (OSError, ValueError):
        return 0
    return sum(1 for c in reg.characters if float(c.duration or 0.0) > 0.0)


CO_OCCUR_S = 20.0     # ennyin belül megszólaló szereplők = egy jelenet
CO_OCCUR_CAP = 3      # egy pár egy epizódban ennyit számítson (ne domináljon)
#: ennyi centroid-hasonlóság fölött ugyanaz a szereplő. Kalibrált érték az
#: eres2net centroidokon (E04, `work/recall_calibrate.py`): az egy szereplőhöz
#: tartozó töredék-párok 0,37–0,57, a **más** szereplők p90-e 0,54 — ezért a
#: 0,60 gyakorlatilag elérhetetlen volt (semmit nem ismert fel újra, minden
#: futás új szereplő-azonosítókat gyártott), a 0,50 a mért határ közepén van.
#: A projekt `tuning.recall_cos`-szal felülírható.
RECALL_COS = 0.50


@dataclass
class Character:
    char_id: str
    gender: str
    voice: str
    pitch: float = 0.0
    duration: float = 0.0
    episodes: list[str] = field(default_factory=list)
    members: dict[str, list[int]] = field(default_factory=dict)
    #: epizódonkénti beszédidő — az évadranghoz; futásonként nem íródik felül
    ep_durations: dict[str, float] = field(default_factory=dict)
    #: a szereplő hanglenyomata — a stabil újrafelismeréshez (nem a klaszter-számhoz)
    centroid: list[float] = field(default_factory=list)
    #: gyakorisági rangsor (1 = a legtöbbet beszél)
    rank: int = 0
    #: beszédidő-arány az összes beszédidőhöz képest
    share: float = 0.0
    #: "fő" = kizárólagos hang; "mellék" = osztozhat a hangon
    role: str = "mellék"
    #: részletes réteg: "fő" | "mellék" | "epizód" | "statiszta"
    tier: str = "mellék"
    #: emberi felülírás a szerep automatikájára ("fő"/"mellék", None = automat)
    #: a WebUI-ból a főszereplők kijelölésére; futásról futásra megmarad
    role_override: str | None = None
    #: emberi felülírás a hangszínre (félhang, None = az automatikus létra)
    #: akkor kell, ha két szereplő mégis egyformán szól a kész szinkronban
    pitch_override: float | None = None
    #: a felülírás előtti automatikus hangszín (a visszaállításhoz)
    pitch_auto: float | None = None
    #: a hangját egyetlen másik szereplő sem használja
    exclusive: bool = True

    def to_dict(self) -> dict:
        return {"char_id": self.char_id, "gender": self.gender, "voice": self.voice,
                "pitch": round(self.pitch, 2), "duration": round(self.duration, 1),
                "episodes": sorted(set(self.episodes)),
                "rank": self.rank, "share": round(self.share, 4),
                "role": self.role, "tier": self.tier,
                "exclusive": bool(self.exclusive),
                "role_override": self.role_override,
                "pitch_override": self.pitch_override,
                "pitch_auto": self.pitch_auto,
                "ep_durations": {ep: round(float(v), 1)
                                 for ep, v in sorted(self.ep_durations.items())},
                "centroid": [round(float(v), 5) for v in self.centroid],
                "members": {ep: sorted(set(c)) for ep, c in sorted(self.members.items())}}


def co_occurrence(utts: list[Utterance]) -> dict[tuple[str, int, int], int]:
    """Szereplő-párok együtt-szólalása epizódonként (jeleneten belül)."""
    by_ep: dict[str, list[Utterance]] = {}
    for u in utts:
        by_ep.setdefault(u.ep, []).append(u)
    out: dict[tuple[str, int, int], int] = {}
    for ep, seq in by_ep.items():
        seq.sort(key=lambda u: u.start)
        seen: dict[tuple[int, int], int] = {}
        for i, a in enumerate(seq):
            if a.cluster is None:
                continue
            for b in seq[i + 1:]:
                if b.start - a.end > CO_OCCUR_S:
                    break
                if b.cluster is None or b.cluster == a.cluster:
                    continue
                key = (min(a.cluster, b.cluster), max(a.cluster, b.cluster))
                seen[key] = min(CO_OCCUR_CAP, seen.get(key, 0) + 1)
        for (x, y), k in seen.items():
            out[(ep, x, y)] = k
    return out


class Registry:
    """A projekt szereplő-rekordjai (perzisztens, stabil kiosztással)."""

    def __init__(self, path: str | Path, main_share_min: float = MAIN_SHARE_MIN,
                 recall_cos: float = RECALL_COS, embed_model: str = ""):
        self.path = Path(path)
        self.main_share_min = float(main_share_min)
        self.recall_cos = float(recall_cos)
        #: a centroidokat adó hanglenyomat-modell; modellcsere után a régi
        #: vektorok más térben vannak, ezért nem használhatók felismerésre
        self.embed_model = str(embed_model or "")
        self.characters: list[Character] = []
        self.voice_pool: list[str] = []
        self.usage: dict = {}
        if self.path.exists():
            self.load()

    # ── I/O ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        d = json.loads(self.path.read_text(encoding="utf-8"))
        old_model = str(d.get("embed_model") or "")
        if old_model and self.embed_model and old_model != self.embed_model:
            # a régi centroidok más beágyazási térben vannak: a szereplőket és
            # a hangzárjukat megtartjuk, de a felismeréshez újra kell mérni
            for c in d.get("characters", []):
                c["centroid"] = []
        self.voice_pool = list(d.get("voice_pool", []))
        self.usage = dict(d.get("usage") or {})
        self.characters = [
            Character(char_id=c["char_id"], gender=c["gender"], voice=c["voice"],
                      pitch=float(c.get("pitch", 0.0)),
                      duration=float(c.get("duration", 0.0)),
                      episodes=list(c.get("episodes", [])),
                      rank=int(c.get("rank", 0)),
                      share=float(c.get("share", 0.0)),
                      role=str(c.get("role", "mellék")),
                      tier=str(c.get("tier", "mellék")),
                      role_override=(c.get("role_override") or None),
                      pitch_override=(None if c.get("pitch_override") is None
                                      else float(c["pitch_override"])),
                      pitch_auto=(None if c.get("pitch_auto") is None
                                  else float(c["pitch_auto"])),
                      exclusive=bool(c.get("exclusive", True)),
                      ep_durations={ep: float(v) for ep, v in
                                    (c.get("ep_durations") or {}).items()},
                      centroid=[float(x) for x in (c.get("centroid") or [])],
                      members={ep: [int(x) for x in v]
                               for ep, v in (c.get("members") or {}).items()})
            for c in d.get("characters", [])]

    def save(self, pipeline: str = "dubstudio.registry v1") -> None:
        self.analyze()          # rangsor + szerep + kizárólagosság + hanggyakoriság
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"pipeline": pipeline, "main_share_min": self.main_share_min,
             "recall_cos": self.recall_cos, "embed_model": self.embed_model,
             "usage": self.usage,
             "voice_pool": self.voice_pool,
             "characters": [c.to_dict() for c in self.characters]},
            ensure_ascii=False, indent=1), encoding="utf-8")

    def analyze(self) -> dict:
        """**Rögzíti, ki milyen gyakran beszél** és hogy kinek melyik hang.

        A beszédidő a színészi hang gyakorisága (klaszter-idő); ebből lesz:
          * ``rank``   — helyezés beszédidő szerint,
          * ``share``  — arány az összes beszédidőből,
          * ``role``   — ``fő``, ha ``share >= main_share_min`` (kizárólagos hang),
          * ``exclusive`` — a hangját senki más nem használja,
          * ``usage``  — hangonkénti összesítés: melyik hangot hány szereplő
            használja és mennyi idővel (a riport és a `usage` parancs adata).
        """
        total = float(sum(c.duration for c in self.characters))
        holders: dict[str, list[Character]] = {}
        for c in self.characters:
            holders.setdefault(c.voice, []).append(c)
        for i, c in enumerate(sorted(self.characters,
                                     key=lambda c: -c.duration), 1):
            c.rank = i
            c.share = (c.duration / total) if total > 0 else 0.0
            # Részletes réteg: a főszereplőn kívül a visszatérő (legalább
            # 2 epizód, 1% idő), az epizódszerep (legalább 10 s) és a
            # statiszta (ennél rövidebb) különül el. A `role` marad a
            # kompatibilis fő/mellék érték a QA-hez.
            forced = c.role_override if c.role_override in ("fő", "mellék") else None
            if forced == "fő" or (forced is None
                                  and c.share >= self.main_share_min):
                c.tier = "fő"
            elif len(set(c.episodes)) >= 2 and c.share >= SUPPORT_SHARE_MIN:
                c.tier = "mellék"
            elif c.duration >= EPISODIC_MIN_S:
                c.tier = "epizód"
            else:
                c.tier = "statiszta"
            c.role = "fő" if c.tier == "fő" else "mellék"
            c.exclusive = len(holders.get(c.voice, [])) == 1
        pool = [v for v in CATALOG if v.vid in (self.voice_pool or [v.vid for v in CATALOG])]
        n_f = sum(1 for c in self.characters if c.gender == "F")
        n_m = sum(1 for c in self.characters if c.gender == "M")
        n_main_f = sum(1 for c in self.characters if c.role == "fő" and c.gender == "F")
        n_main_m = sum(1 for c in self.characters if c.role == "fő" and c.gender == "M")
        episodes = []
        for ep in sorted({e for c in self.characters for e in c.ep_durations}):
            cs = [c for c in self.characters if c.ep_durations.get(ep, 0) > 0]
            episodes.append({"ep": ep,
                             "seconds": round(sum(c.ep_durations.get(ep, 0) for c in cs), 1),
                             "n_chars": len(cs),
                             "n_f": sum(1 for c in cs if c.gender == "F"),
                             "n_m": sum(1 for c in cs if c.gender == "M"),
                             "n_main": sum(1 for c in cs if c.role == "fő")})
        voices = []
        for voice, lst in sorted(holders.items(),
                                 key=lambda kv: -sum(c.duration for c in kv[1])):
            sec = sum(c.duration for c in lst)
            voices.append({"voice": voice,
                           "seconds": round(sec, 1),
                           "share": round(sec / total, 4) if total else 0.0,
                           "n_chars": len(lst),
                           "exclusive": len(lst) == 1,
                           "chars": [{"char_id": c.char_id,
                                      "duration": round(c.duration, 1),                           "rank": c.rank, "role": c.role, "tier": c.tier,
                           "pitch": c.pitch}
                                     for c in sorted(lst, key=lambda c: c.rank)]})
        self.usage = {
            "total_seconds": round(total, 1),
            "main_share_min": self.main_share_min,
            "n_chars": len(self.characters),
            "n_f": n_f, "n_m": n_m,
            "n_main": sum(1 for c in self.characters if c.role == "fő"),
            "n_main_f": n_main_f, "n_main_m": n_main_m,
            "tiers": {t: sum(1 for c in self.characters if c.tier == t)
                      for t in TIER_ORDER},
            "n_voices": len(holders),
            "n_pool": len(pool),
            "n_pool_f": sum(1 for v in pool if v.gender == "F"),
            "n_pool_m": sum(1 for v in pool if v.gender == "M"),
            "shared_voices": sum(1 for v in voices if not v["exclusive"]),
            "episodes": episodes,
            "voices": voices}
        return self.usage

    def table_usage(self) -> str:
        """A hanggyakoriság szöveges táblája (CLI)."""
        if not self.usage:
            self.analyze()
        u = self.usage
        lines = [f"{'rang':>4} {'szereplő':>8} {'hang':>9} {'pitch':>6} {'nem':>4} "
                 f"{'idő':>7} {'%':>5} {'réteg':>7}  hang"]
        for c in sorted(self.characters, key=lambda c: c.rank):
            lines.append(f"{c.rank:>4} {c.char_id:>8} {c.voice:>9} {c.pitch:>+6.2f} "
                         f"{c.gender:>4} {c.duration:>6.0f}s {100 * c.share:>4.1f}% "
                         f"{c.tier:>7}  {'kizárólagos' if c.exclusive else 'MEGOSZTVA'}")
        lines.append(f"\n{'hang':>9} {'szereplő':>8} {'beszédidő':>10} {'%':>5}  kik")
        for v in u.get("voices", []):
            who = ", ".join(f"{c['char_id']}({c['duration']:.0f}s)"
                            for c in v["chars"])
            lines.append(f"{v['voice']:>9} {v['n_chars']:>8} {v['seconds']:>9.0f}s "
                         f"{100 * v['share']:>4.1f}%  {who}")
        lines.append(f"\nösszesen: {u.get('n_chars', 0)} szereplő, "
                     f"{u.get('n_main', 0)} fő (share ≥ {u.get('main_share_min', 0):.0%}), "
                     f"{u.get('n_voices', 0)} hang, "
                     f"ebből {u.get('shared_voices', 0)} megosztott, "
                     f"{u.get('total_seconds', 0):.0f} s beszéd; rétegek: "
                     f"fő {u.get('tiers', {}).get('fő', 0)}, "
                     f"mellék {u.get('tiers', {}).get('mellék', 0)}, "
                     f"epizód {u.get('tiers', {}).get('epizód', 0)}, "
                     f"statiszta {u.get('tiers', {}).get('statiszta', 0)}")
        return "\n".join(lines)

    # ── besorolás ────────────────────────────────────────────────────────
    def assign(self, clusters: dict, utts: list[Utterance], bank: VoiceBank,
               verbose: bool = True) -> dict[int, Character]:
        """Klaszterek -> szereplők, **hangzárral**.

        A visszatérés klaszter-azonosító -> szereplő. A ``utts`` klaszter- és
        hangmezői is beállításra kerülnek.
        """
        import numpy as np

        # A tisztán képernyőszöveg-klaszter (helyszín, epizód-fejléc) nem
        # szereplő: nincs beszéd, a `plan` kihagyja. Korábban ezek is kaptak
        # karaktert és hangot, felfújták a kasztot és elvették a helyet a
        # valódi szereplők elől.
        by_cluster: dict[int, list[Utterance]] = {}
        for u in utts:
            if u.cluster is not None:
                by_cluster.setdefault(u.cluster, []).append(u)
        clusters = {
            cid: cl for cid, cl in clusters.items()
            if not (by_cluster.get(cid)
                    and all(subs.is_screen_text(u.text) for u in by_cluster[cid]))}
        # a kiszűrt (képernyőszöveg) klaszterek sorairól töröljük a korábbi
        # futás szereplő-/hanghivatkozását, különben a kasztban már nem létező
        # karakterre hivatkoznának, és a lefedettséget is rontanák
        kept = set(clusters)
        for u in utts:
            if u.cluster not in kept:
                u.char = None
                u.voice = None

        co = co_occurrence(utts)
        # a klónhang nem automatikus választék: csak kézi választással szólal meg
        self.voice_pool = [v.vid for v in bank.available.values() if not v.clone]

        # beszédidő szerint csökkenő sorrend: a főszereplők kapnak először hangot
        order = sorted(clusters, key=lambda cid: -clusters[cid].dur)
        used_voice: dict[str, str] = {}          # voice -> char_id
        voice_pitch: dict[str, list[float]] = {}  # voice -> használt pitch-ek
        result: dict[int, Character] = {}
        new_chars: list[Character] = []
        kept = 0

        # korábbi rekordok pitch/voice foglalása
        for c in self.characters:
            used_voice.setdefault(c.voice, c.char_id)
            voice_pitch.setdefault(c.voice, []).append(c.pitch)

        def co_weight(cid: int, other: Character, ep: str) -> int:
            w = 0
            for ep2, cids in other.members.items():
                if ep2 != ep:
                    continue
                for oc in cids:
                    a, b = min(cid, oc), max(cid, oc)
                    w += co.get((ep, a, b), 0)
            return w

        def cos(a, b) -> float:
            a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            if not len(a) or not len(b):
                return 0.0
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

        def recall(cid: int) -> Character | None:
            """A **hanglenyomat** alapján ismeri fel ugyanazt a szereplőt.

            Nem a klaszter-számra támaszkodunk (az minden futásnál változik),
            hanem a centroidra: így a korábban kiosztott hang megmarad.
            """
            vec = clusters[cid].centroid
            if vec is None:
                return None
            best, best_s = None, self.recall_cos
            for c in self.characters:
                if not c.centroid:
                    continue
                if clusters[cid].gender and c.gender in ("F", "M") \
                        and clusters[cid].gender != c.gender:
                    continue
                s = cos(vec, c.centroid)
                if s > best_s:
                    best, best_s = c, s
            return best

        claimed: set[str] = set()
        # az aktuális futás epizódjai: a régi tagokat és a régi beszédidőt
        # újraírjuk (különben futásról futásra összegződik és torzul)
        current_eps = {u.ep for u in utts}
        for c in self.characters:
            for e in list(c.members):
                if e in current_eps:
                    c.members[e] = []
        for cid in order:
            cl = clusters[cid]
            ep = cl.episodes[0] if cl.episodes else "?"
            rec = recall(cid)
            if rec is not None and rec.char_id not in claimed:
                claimed.add(rec.char_id)
                result[cid] = rec
                rec.members.setdefault(ep, [])
                if cid not in rec.members[ep]:
                    rec.members[ep].append(cid)
                if ep not in rec.episodes:
                    rec.episodes.append(ep)
                if cl.centroid is not None:
                    n = max(1, len(rec.episodes))
                    rec.centroid = ((np.asarray(rec.centroid, dtype=np.float32) * (n - 1)
                                     + np.asarray(cl.centroid, dtype=np.float32)) / n).tolist()
                kept += 1
                continue
            new_chars.append(Character(char_id="", gender=cl.gender or "?",
                                       voice="", duration=cl.dur,
                                       episodes=[ep], members={ep: [cid]},
                                       centroid=([] if cl.centroid is None else
                                                 [float(v) for v in cl.centroid])))

        # Nem felismert régi rekordok: ha az aktuális epizódokban már nem
        # jelennek meg, töröljük — különben futásról futásra gyűlnek, foglalják
        # a hangokat, és a gyakorisági rangsor is torzul. (A más epizódokból
        # ismert szereplők megmaradnak.)
        self.characters = [c for c in self.characters
                           if c.char_id in claimed
                           or not (set(c.members) <= current_eps)]

        # új szereplők azonosítói (a beszédidő újraszámítása előtt)
        existing = [c for c in self.characters]
        next_n = 1
        for ch in new_chars:
            used_ids = {c.char_id for c in existing} | {c.char_id for c in new_chars}
            while f"C{next_n:03d}" in used_ids:
                next_n += 1
            ch.char_id = f"C{next_n:03d}"

        # **Beszédidő = epizódonkénti összeg**: az aktuális epizódok idejét
        # cseréljük, a többi epizódét megtartjuk. Korábban az egész `duration`
        # felülíródott az utolsó futás idejével, ezért az évadrang csak az E07-et
        # látta, és a főszereplők is megosztott hangot kaptak.
        cur_dur: dict[str, dict[str, float]] = {}
        char_of_cluster = {cid: ch for cid, ch in result.items()}
        for u in utts:
            ch = char_of_cluster.get(u.cluster)
            if ch is not None and u.dur:
                cur_dur.setdefault(ch.char_id, {}).setdefault(u.ep, 0.0)
                cur_dur[ch.char_id][u.ep] += u.dur
        for ch in existing:
            if ch.char_id in cur_dur:
                for ep in current_eps:
                    ch.ep_durations.pop(ep, None)
                ch.ep_durations.update(cur_dur[ch.char_id])
                ch.duration = round(sum(ch.ep_durations.values()), 1)
        for ch in new_chars:
            for ep in ch.episodes:
                ch.ep_durations[ep] = round(ch.duration, 1)
            ch.duration = round(sum(ch.ep_durations.values()), 1)

        # új szereplők hangkiosztása: színezés az együtt-szólás gráfján
        # **Gyakoriság = szerep**: a beszédidő-arány dönti el, ki főszereplő.
        # A főszereplők előbb jönnek (idő szerint csökkenő sorrendben), és
        # kizárólagos hangot kapnak; a mellékszereplők a megmaradó hangokon
        # osztoznak — a főszereplő hangját csak akkor vehetik át, ha a
        # nemen belül nincs több szabad hang (ezt a `main_shared` kapu méri).
        total = (float(sum(c.duration for c in existing))
                 + float(sum(c.duration for c in new_chars)))

        def is_main(c: Character) -> bool:
            if c.role_override in ("fő", "mellék"):
                return c.role_override == "fő"
            return total > 0 and c.duration / total >= self.main_share_min

        reserved: set[str] = {c.voice for c in existing if is_main(c)}
        for ch in new_chars:
            cid = ch.members[ch.episodes[0]][0]
            cands = bank.pool(ch.gender) if ch.gender in ("F", "M") else []
            if not cands:
                cands = bank.pool("F") + bank.pool("M")
            if not cands:
                raise RuntimeError("egyetlen TTS-hang sem elérhető")
            # tilalom: az együtt szóló szereplők hangja
            forbidden: dict[str, int] = {}
            for other in existing + new_chars[:-1]:
                if other is ch:
                    continue
                if other.episodes and ch.episodes and not (set(other.episodes) & set(ch.episodes)):
                    continue
                w = co_weight(cid, other, ch.episodes[0])
                if w:
                    forbidden[other.voice] = forbidden.get(other.voice, 0) + w
            free = [v for v in cands if v.vid not in used_voice]
            if free:
                # főszereplő: elsőnek a még senki által nem használt hang jár,
                # így a hangja végig kizárólagos marad
                ch.voice = free[0].vid
            else:
                allowed = [v for v in cands if v.vid not in forbidden]
                # Rangsor a mellékszereplőnek (a fő hangja erősebb, mint a
                # jeleneti színezés — ott a pitch amazt is megkülönbözteti):
                #   1. nem fő hangja ÉS nem jeleneti ütközés,
                #   2. nem fő hangja (jeleneti büntetéssel),
                #   3. nem fő hangja (nincs más), 4. bármi (fő hangja — kapu jelzi)
                non_res = [v for v in cands if v.vid not in reserved]
                if is_main(ch):
                    pick = allowed or non_res or [v for v in cands]
                else:
                    clean = [v for v in allowed if v.vid not in reserved]
                    pick = clean or non_res or allowed or [v for v in cands]
                ch.voice = min(pick, key=lambda v: (forbidden.get(v.vid, 0),
                                                    len(voice_pitch.get(v.vid, [])))) .vid
            # pitch: azonos hangot osztozó szereplők fix eltolással különböznek
            # (az emberi felülírás mindig erősebb az automatánál)
            used_p = set(voice_pitch.get(ch.voice, []))
            if ch.pitch_override is not None:
                ch.pitch = float(ch.pitch_override)
            else:
                ch.pitch = next((p for p in PITCH_LADDER if p not in used_p), 0.0)
                ch.pitch_auto = None      # az automatikus létra újraszámolódott
            voice_pitch.setdefault(ch.voice, []).append(ch.pitch)
            used_voice.setdefault(ch.voice, ch.char_id)
            if is_main(ch):
                reserved.add(ch.voice)      # a fő hangját a kisebbek nem kérik
            existing.append(ch)
            result[cid] = ch

        self.characters = existing
        self.analyze()          # rangsor, szerep (fő/mellék), kizárólagosság
        for cid, ch in result.items():
            for u in utts:
                if u.cluster == cid:
                    u.char = ch.char_id
                    u.voice = ch.voice
        if verbose:
            print(f"[szereplők] {len(new_chars)} új, {kept} megtartott rekord; "
                  f"összesen {len(self.characters)} szereplő")
        return result

    # ── évadszintű újrakiosztás ──────────────────────────────────────────
    def recast(self, utts: list[Utterance] | None = None,
               bank: VoiceBank | None = None,
               verbose: bool = False) -> dict:
        """Az **egész évad** hangkiosztásának újraszámítása.

        A ``cluster`` csak az új szereplőket osztja ki, a meglévő hangokat
        megtartja. Ezért egy évadközi futás után a főszereplő hangján maradhattak
        mellékszereplők. Ez a lépés minden szereplőt újraoszt:

          1. a beszédidőt epizódonként összegzi (a évadranghoz);
          2. a főszereplők (share >= ``main_share_min``) először **egyedi**
             hangot kapnak, a jelenlegi hangot lehetőleg megtartva;
          3. a mellékszereplők a maradék hangokon osztoznak, DSATUR-szerű
             színezéssel: előbb a legtöbb jelenetkapcsolattal rendelkező
             szereplő választ, és a már egy jelenetben használt hangot kerüli;
          4. az azonos hangot osztozó szereplők fix, eltérő pitch-et kapnak.

        Így egy szereplő végig egy hangot használ, a főszereplő hangja
        kizárólagos, és a párbeszédben lehetőleg nincs két azonos hang.
        """
        from collections import Counter

        from .voices import BY_ID

        if not self.characters:
            return self.analyze()

        # 1) epizódonkénti beszédidő (csak a megadott epizódokat cseréljük)
        if utts:
            eps = {u.ep for u in utts}
            fresh: dict[str, dict[str, float]] = {}
            for u in utts:
                if u.char and u.dur:
                    fresh.setdefault(u.char, {}).setdefault(u.ep, 0.0)
                    fresh[u.char][u.ep] += u.dur
            for c in self.characters:
                if c.char_id in fresh:
                    for ep in eps:
                        c.ep_durations.pop(ep, None)
                    c.ep_durations.update(fresh[c.char_id])
                    c.duration = round(sum(c.ep_durations.values()), 1)
                    # A megjelenő epizódok listája az epizódonkénti beszédidőből
                    # származik; a régi rekordból örökölt lista különben hiányos
                    # lehet (pl. E01,E05,E07 egy 11 epizódos főszereplőnél).
                    c.episodes = sorted(c.ep_durations)

        self.analyze()
        pool_ids = ([v.vid for v in bank.available.values() if not v.clone]
                    if bank else (self.voice_pool
                                  or [v.vid for v in CATALOG if not v.clone]))
        by_gender = {g: [v for v in pool_ids
                         if BY_ID.get(v) and BY_ID[v].gender == g]
                     for g in ("F", "M")}
        if not by_gender["F"] and not by_gender["M"]:
            return self.usage

        chars = {c.char_id: c for c in self.characters}
        # A **kézzel klónhangra állított** szereplőt az újrakiosztás nem bántja:
        # a klónhang nem tagja az automatikus választéknak, így a szokásos
        # kiosztás lecserélné — pedig az ember választotta
        clone_held = {c.char_id for c in self.characters if is_clone(c.voice)}
        reserved: set[str] = set()
        owner: dict[str, str] = {}

        # 2) főszereplők: nemenként egyedi hang, a jelenlegi hangot megtartva
        mains = [c for c in self.characters
                 if c.role == "fő" and c.gender in ("F", "M")]
        mains.sort(key=lambda c: (-c.duration, c.char_id))
        for c in mains:
            if c.char_id in clone_held:
                continue
            pool = by_gender[c.gender]
            free = [v for v in pool if v not in reserved]
            if not free:
                if verbose:
                    print(f"[recast] ! {c.char_id}: elfogytak a {c.gender} hangok")
                continue
            pick = c.voice if c.voice in free else free[0]
            c.voice = pick
            c.pitch = (float(c.pitch_override)
                       if c.pitch_override is not None else 0.0)
            reserved.add(pick)
            owner[pick] = c.char_id

        # 3) jelenetgráf a karakterek szintjén (20 s = egy jelenet)
        co: dict[tuple[str, str], int] = {}
        if utts:
            by_ep: dict[str, list[Utterance]] = {}
            for u in utts:
                if u.char:
                    by_ep.setdefault(u.ep, []).append(u)
            for seq in by_ep.values():
                seq.sort(key=lambda u: u.start)
                for i, a in enumerate(seq):
                    for b in seq[i + 1:]:
                        if b.start - a.end > CO_OCCUR_S:
                            break
                        if not b.char or a.char == b.char:
                            continue
                        key = tuple(sorted((a.char, b.char)))
                        co[key] = min(CO_OCCUR_CAP, co.get(key, 0) + 1)
        neigh: dict[str, set[str]] = {}
        for a, b in co:
            neigh.setdefault(a, set()).add(b)
            neigh.setdefault(b, set()).add(a)

        # 4) mellékszereplők: a maradék hangokon, jelenet-ütközés nélkül
        secondaries = [c for c in self.characters
                       if c.role != "fő" and c.gender in ("F", "M")
                       and c.char_id not in clone_held]
        pending = {c.char_id: c for c in secondaries}
        assigned = {c.char_id for c in mains if c.voice in reserved} | clone_held
        used_count = Counter(owner.values())
        while pending:
            def key(c: Character) -> tuple[int, int, int, float]:
                used_colors = {chars[cid].voice for cid in assigned
                               if cid in neigh.get(c.char_id, ())}
                # A visszatérő mellékszereplők előbb választhatnak, mint az
                # epizód-/statisztaszerepek; a színezés a megkötöttséget követi.
                return (TIER_ORDER.get(c.tier, 3), len(used_colors),
                        len(neigh.get(c.char_id, ())), c.duration)

            c = max(pending.values(), key=key)
            pending.pop(c.char_id)
            pool = [v for v in by_gender[c.gender] if v not in reserved]
            if not pool:
                pool = by_gender[c.gender]

            def cost(v: str) -> tuple[int, int, int, int, int]:
                conflict = sum(
                    co.get(tuple(sorted((c.char_id, o.char_id))), 0)
                    for o in self.characters
                    if o.char_id != c.char_id and o.voice == v
                    and o.char_id in assigned)
                # Az epizód-/statisztaszerep ne vegye el a visszatérő
                # mellékszereplő hangját, ha van szabad alternatíva.
                reserve = sum(
                    1 for o in self.characters
                    if o.char_id != c.char_id and o.voice == v
                    and o.char_id in assigned and o.tier == "mellék"
                    and c.tier in ("epizód", "statiszta"))
                return (conflict, reserve, 0 if v == c.voice else 1,
                        used_count[v], pool.index(v))

            pick = min(pool, key=cost)
            c.voice = pick
            used_p = {o.pitch for o in self.characters
                      if o.char_id != c.char_id and o.voice == pick}
            if c.pitch_override is not None:
                c.pitch = float(c.pitch_override)
            else:
                c.pitch = next((p for p in PITCH_LADDER if p not in used_p), 0.0)
            used_count[pick] += 1
            assigned.add(c.char_id)

        # 5) szigorú jelenetütközés javítása: ha egy szereplő áthelyezésével
        # csökken az azonos hangú, egy jelenetben szereplő párok száma, tegyük meg.
        def strict_count() -> int:
            return sum(1 for a, b in co
                       if a != b and chars[a].voice == chars[b].voice)

        for _ in range(8):
            bad = [(a, b) for a, b in co
                   if a != b and chars[a].voice == chars[b].voice
                   and a not in clone_held and b not in clone_held]
            if not bad:
                break
            improved = False
            for a, b in bad:
                for cid in (a, b):
                    c = chars[cid]
                    if c.role == "fő":
                        continue
                    old_voice, old_pitch = c.voice, c.pitch
                    for v in by_gender.get(c.gender, []):
                        if v in reserved or v == old_voice:
                            continue
                        before = strict_count()
                        c.voice = v
                        used_p = {o.pitch for o in self.characters
                                  if o is not c and o.voice == v}
                        c.pitch = next((p for p in PITCH_LADDER
                                        if p not in used_p), 0.0)
                        if strict_count() < before:
                            improved = True
                            break
                        c.voice, c.pitch = old_voice, old_pitch
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        # 5b) Ha a szabad hangok elfogytak, az eddigi javítás csak „szabad”
        # hangot keresett. Ilyenkor két mellékszereplő hangjának **cseréje**
        # is megszüntethet egy azonos hangú jelenetpárt. A főszereplőket nem
        # bántjuk; csak akkor fogadjuk el a cserét, ha a teljes évad szigorú
        # ütközésszáma csökken.
        for _ in range(8):
            bad = [(a, b) for a, b in co
                   if a != b and chars[a].voice == chars[b].voice
                   and a not in clone_held and b not in clone_held]
            if not bad:
                break
            improved = False
            for a, b in bad:
                for x, y in ((a, b), (b, a)):
                    cx, cy = chars[x], chars[y]
                    if cx.role == "fő" or cy.role == "fő":
                        continue
                    if cx.gender != cy.gender or cx.voice == cy.voice:
                        continue
                    old_x = (cx.voice, cx.pitch)
                    old_y = (cy.voice, cy.pitch)
                    before = strict_count()
                    cx.voice, cy.voice = old_y[0], old_x[0]
                    for z in (cx, cy):
                        used_p = {o.pitch for o in self.characters
                                  if o is not z and o.voice == z.voice}
                        z.pitch = next((p for p in PITCH_LADDER
                                        if p not in used_p), 0.0)
                    if strict_count() < before:
                        improved = True
                        break
                    cx.voice, cx.pitch = old_x
                    cy.voice, cy.pitch = old_y
                if improved:
                    break
            if not improved:
                break

        # 6) a meglévő sorok hangjának frissítése + új rangsor
        if utts:
            for u in utts:
                if u.char in chars:
                    u.voice = chars[u.char].voice
        self.analyze()
        if verbose:
            tiers = self.usage.get("tiers", {})
            print(f"[recast] {len(mains)} fő, {len(secondaries)} nem fő "
                  f"(mellék {tiers.get('mellék', 0)}, epizód {tiers.get('epizód', 0)}, "
                  f"statiszta {tiers.get('statiszta', 0)}); "
                  f"{len(reserved)} foglalt hang, {len(self.usage['voices'])} használt")
        return self.usage

    # ── ellenőrzés ───────────────────────────────────────────────────────
    def conflicts(self, utts: list[Utterance], strict: bool = False) -> list[dict]:
        """Egy jelenetben azonos hangot használó szereplők (a hangzár megsértése).

        A hangzár tervezett tartaléka a fix ``pitch``-eltolás: ha elfogytak a
        hangok, ugyanazt a hangot két szereplő is kaphatja, de különböző
        félhangnyi eltolással — így hallhatóan különböznek. A kapu ezért csak
        azt tekinti konfliktusnak, ha **ugyanaz a hang és ugyanaz a pitch**
        áll a két szereplőn (azaz hallhatóan megkülönböztethetetlenek).
        """
        co = co_occurrence(utts)
        # A klaszter-azonosítók epizódonként újrahasználódnak (E05:0 és E07:0
        # más szereplő lehet), ezért a kulcs (epizód, klaszter) — különben
        # hamis jelenetütközést látnánk két különböző epizód szereplői között.
        char_of: dict[tuple[str, int], str] = {}
        voice_of: dict[tuple[str, int], str] = {}
        pitch_of: dict[tuple[str, int], float] = {}
        char_pitch = {c.char_id: c.pitch for c in self.characters}
        for u in utts:
            if u.cluster is not None:
                key = (u.ep, u.cluster)
                char_of.setdefault(key, u.char or "?")
                voice_of.setdefault(key, u.voice or "?")
        for key, ch in char_of.items():
            pitch_of[key] = float(char_pitch.get(ch, 0.0))
        out: list[dict] = []
        for (ep, a, b), w in sorted(co.items()):
            ka, kb = (ep, a), (ep, b)
            if char_of.get(ka) == char_of.get(kb):
                continue
            if voice_of.get(ka) == voice_of.get(kb) \
                    and (strict or abs(pitch_of.get(ka, 0.0)
                                       - pitch_of.get(kb, 0.0)) < 0.01):
                ca, cb = char_of.get(ka), char_of.get(kb)
                out.append({"ep": ep, "char_a": ca, "char_b": cb,
                            "voice": voice_of.get(ka), "co_occurrence": w})
        return out

    def voices_in_use(self) -> dict[str, list[str]]:
        d: dict[str, list[str]] = {}
        for c in self.characters:
            d.setdefault(c.voice, []).append(c.char_id)
        return d

    def table(self) -> str:
        lines = [f"{'szereplő':>8} {'hang':>9} {'pitch':>6} {'nem':>4} {'idő':>9}  epizódok"]
        for c in sorted(self.characters, key=lambda c: -c.duration):
            lines.append(f"{c.char_id:>8} {c.voice:>9} {c.pitch:>+6.2f} {c.gender:>4} "
                         f"{c.duration:>8.0f}s  {','.join(sorted(set(c.episodes)))}")
        return "\n".join(lines)
