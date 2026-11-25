"""Szereplő-klaszterezés: **globális** csoportosítás + szekvenciális finomítás.

A kutatás és a mérés egyaránt ezt a felépítést támasztja alá:
  1. **epizódon belül** átlag-linkage agglomeratív klaszterezés a
     megnyilatkozás-embeddingeken (kezdő csoportok);
  2. **epizódok között** centroid-összevonás — így ugyanaz a színész a
     következő epizódban is ugyanabba a klaszterbe kerül (ezt a régi lánc
     epizódonként újraosztotta);
  3. **szekvenciális finomítás** (VB-HMM/VBx elv): epizódonként a
     megnyilatkozások sorrendjén Viterbi, erős váltási büntetéssel és
     klaszter-oldali normalizált hasonlósággal (AS-Norm) — ez szünteti meg
     az izolált téves címkéket;
  4. **nem-konzisztencia javítása**: ha egy megnyilatkozás nemét erősen
     mérjük, és az ellentétes a klaszter nemével, akkor a helyes nemű
     klaszterhez kerül (vagy új klaszter lesz) — ez a „női soron férfi hang”
     hiba közvetlen kezelése;
  5. **töredék-felszívás**: a néhány másodperces, bizonytalan klasztereket
     a legközelebbi azonos nemű klaszterbe olvasztjuk.

A klaszterek száma **nincs** a TTS-hangkészlethez kötve: előbb a valódi
szereplőket keressük meg, és csak utána osztunk hangot (``registry.py``).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import subs
from .embeddings import centroid, cosine, norm
from .plda import PLDAModel
from .utterances import Utterance

DEFAULTS = {
    "sim_same": 0.55,        # régi, nyers AHC-küszöb (visszafelé kompatibilitás)
    "sim_gender": 0.30,      # nemenkénti AHC küszöb **középpontos** vektorokon
    "u_attach_cos": 0.10,    # bizonytalan nemű sor csatolási küszöbe (középpontos)
    "link_cos": 0.62,        # epizódok közötti összevonás (szigorúbb)
    "viterbi_pen": 0.30,     # váltási büntetés alapsúlya
    "viterbi_pen_gap": 1.20, # ennyi másodperc szünet után a büntetés 0
    "z_gain": 1.00,          # AS-Norm hasonlóság súlya
    "strong_f": 0.85,        # erős női jel
    "strong_m": 0.15,        # erős férfi jel
    "absorb_min_s": 4.0,     # ennél rövidebb klaszter összeolvasztható
    "absorb_cos": 0.40,
    "min_utt_s": 0.45,       # rövidebb megnyilatkozásból nincs döntés
    # rövid töredék ne kapjon saját hangot: ennyi alatt a klasztert a
    # leghosszabb, azonos nemű klaszterbe olvasztjuk (a beszélő-embedding
    # rövid, zenés ablakon nem elég megbízható önálló szereplőhöz)
    "frag_merge_max_s": 3.0,
    "frag_host_min_s": 6.0,         # a befogadó klaszter legalább ennyi legyen
    "frag_screen_max": 0.30,        # ennyi képernyőszöveg fölött nem olvasztunk
    # hangbizonyíték nélküli sor: ennél nagyobb szünetnél már nem csatoljuk
    # a szomszéd beszélőhöz (különben egy gyerek kiáltása a narrátor hangját kapja).
    # 60 s: egy jeleneten belül a rövid, halk sor is a szomszéd folytatása; a
    # 8 s-os korlát túl sok 1 s-os „szereplőt” hagyott saját hanggal.
    "attach_max_gap": 60.0,
    "gender_move_cos": 0.30,        # nem-javításhoz ennyi középpontos hasonlóság kell
    "gender_move_margin": 0.10,     # és ennyivel jobb, mint a saját klaszteré
    "gap_cont": 2.20,        # ennél kisebb szünetnél a sor a szomszéd folytatása
    # Egy szomszédos, külön hangú pár csak akkor hangváltás, ha a két
    # klaszter mintái sem támasztják alá, hogy két különböző beszélő. A
    # rövid felkiáltások („Békesség!”, „Igen.”) önmagukban zajosak; a
    # klaszter-szintű saját/idegen különbség a beszélő-azonosítás bizonyítéka.
    "switch_cluster_margin": 0.05,
    # kalibrált beszélő-verifikátor (AS-Norm + döntési sáv)
    "verifier_same": 0.65,
    "verifier_diff": 0.35,
    "verifier_temp": 2.0,
    # PLDA: a döntés null-normalizált (z) skálán történik, lásd plda.py.
    "plda_temp": 2.0,
    "plda_weight": 0.5,      # PLDA z súlya az AS-Norm z-hez képest
    "verifier_cohort_min": 3,
}


@dataclass
class Cluster:
    cid: int
    gender: str | None = None
    centroid: np.ndarray | None = None
    dur: float = 0.0
    uids: list[str] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    gender_vote: float = 0.0
    n_long: int = 0

    def to_dict(self) -> dict:
        return {"cid": self.cid, "gender": self.gender, "dur": round(self.dur, 1),
                "episodes": sorted(set(self.episodes)), "n_utt": len(self.uids),
                "gender_vote": round(self.gender_vote, 3),
                "centroid": None if self.centroid is None else
                [round(float(v), 5) for v in self.centroid]}


def _ahc(vecs: np.ndarray, thr: float) -> list[int]:
    """Átlag-linkage agglomeratív klaszterezés koszinusz-távolsággal."""
    from scipy.cluster.hierarchy import fcluster, linkage

    if len(vecs) == 1:
        return [0]
    Z = linkage(vecs, method="average", metric="cosine")
    return [int(v) - 1 for v in fcluster(Z, t=1.0 - thr, criterion="distance")]


class _UF:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, a: int) -> int:
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _centered(utts: list[Utterance]) -> dict[int, np.ndarray]:
    """Epizódonkénti átlagvektor-kivont, újranormalizált embeddingek.

    A nyers CAM++ koszinusz nem kalibrált: két különböző beszélő (férfi és
    nő) centroidja is 0,8–0,9-et ad, mert minden vektorban van egy nagy közös
    komponens. Az átlag kivonása után ugyanaz a beszélő ~0,5–0,8, két
    különböző ~0,0 körül van — ez teszi mérhetővé az összevonást.

    Az átlag **epizódonként** számolódik: a különböző részek más mikrofon- és
    keverési láncot használnak, a közös epizódátlag nem távolítaná el a
    csatorna-hatást. A visszaadott vektorok ezért csak azonos epizódon belül
    hasonlíthatók össze.

    Nagyon rövid epizódnál (néhány minta) az átlag kivonása instabil: két
    vektor középpontja éppen a két beszélő közé esik, és a két azonos beszélő
    különböző epizódjai ellentétes irányba fordulhatnak. Ilyenkor a nyers,
    normalizált vektort használjuk; a valódi epizódokban (több száz minta)
    a középpontosítás a helyes.
    """
    by_ep: dict[str, list[Utterance]] = {}
    for u in utts:
        if u.emb is not None:
            by_ep.setdefault(u.ep, []).append(u)
    out: dict[int, np.ndarray] = {}
    for rows in by_ep.values():
        if len(rows) < 8:
            for u in rows:
                v = np.asarray(u.emb, dtype=np.float32)
                n = float(np.linalg.norm(v))
                if n > 0:
                    out[id(u)] = v / n
            continue
        mean = np.stack([u.emb for u in rows]).mean(axis=0)
        for u in rows:
            c = np.asarray(u.emb, dtype=np.float32) - mean
            n = float(np.linalg.norm(c))
            if n > 0:
                out[id(u)] = c / n
    return out


def cluster(utts: list[Utterance], tuning: dict | None = None,
            verbose: bool = True) -> dict[int, Cluster]:
    """Megnyilatkozások -> globális szereplő-klaszterek."""
    t = {**DEFAULTS, **(tuning or {})}
    by_ep: dict[str, list[Utterance]] = {}
    for u in utts:
        # tiszta lap: a korábbi futás klaszter-azonosítói ne szivárogjanak át
        # (különben az embedding nélküli sorok régi címkéje ütközhet az újakkal,
        #  és az újrafuttatás nem ugyanazt az eredményt adja)
        u.cluster = None
        by_ep.setdefault(u.ep, []).append(u)
    for v in by_ep.values():
        v.sort(key=lambda u: u.start)

    next_cid = 0

    def gender_of(u: Utterance) -> str | None:
        if u.p_female is None:
            return None
        if u.p_female >= t["strong_f"]:
            return "F"
        if u.p_female <= t["strong_m"]:
            return "M"
        return None

    # ── 1. epizódon belüli csoportok, nemenként ──────────────────────────
    # A nyers, nem-agnosztikus AHC a mérés szerint összemosta a két
    # főszereplőt (egy klaszterben 75 s női és 75 s férfi bizonyíték), a régi
    # „nem-javítás” pedig éppen ezt leplezte azzal, hogy egyes sorokat
    # átdobált — a hallható eredmény a jeleneten belüli ellenkező nemű
    # hangváltás volt. Ezért a nem **kemény feltétel**: ismert nemű sorokat
    # csak azonos nemű sorokkal csoportosítunk, a bizonytalan neműeket pedig
    # a legjobban hasonló klaszterhez csatoljuk (nem indítanak új hangot).
    for ep, group_utts in by_ep.items():
        have = [u for u in group_utts if u.emb is not None]
        if not have:
            for u in group_utts:
                u.cluster = next_cid
                next_cid += 1
            continue
        vec = _centered(have)
        cents: dict[int, np.ndarray] = {}
        local: set[int] = set()          # id(u), amely már klasztert kapott

        def make(members: list[Utterance]) -> None:
            nonlocal next_cid
            cid = next_cid
            next_cid += 1
            for u in members:
                u.cluster = cid
                local.add(id(u))
            v = np.mean([vec[id(u)] for u in members], axis=0)
            n = float(np.linalg.norm(v))
            if n > 0:
                cents[cid] = v / n

        for g in ("F", "M"):
            rows = [u for u in have if gender_of(u) == g]
            if not rows:
                continue
            if len(rows) == 1:
                labels = [0]
            else:
                labels = _ahc(np.stack([vec[id(u)] for u in rows]), t["sim_gender"])
            for lab in sorted(set(labels)):
                make([u for u, l in zip(rows, labels) if l == lab])

        if not cents:
            # nincs erős nem-mérés az epizódban: marad a középpontos AHC
            labels = _ahc(np.stack([vec[id(u)] for u in have]), t["sim_gender"])
            for lab in sorted(set(labels)):
                make([u for u, l in zip(have, labels) if l == lab])

        # bizonytalan nemű sor: a legjobban hasonló klaszterhoz (azonos
        # epizódon belül), hogy ne kapjon önálló hangot egyetlen zajos mérés
        for u in have:
            if id(u) in local:
                continue
            best, best_s = None, -2.0
            for cid, cv in cents.items():
                s = float(vec[id(u)] @ cv)
                if s > best_s:
                    best, best_s = cid, s
            if best is not None and best_s >= t["u_attach_cos"]:
                u.cluster = best
            else:
                make([u])

        # embedding nélküli sorok külön klaszterbe; a 7. lépés csatolja őket
        for u in group_utts:
            if u.cluster is None:
                u.cluster = next_cid
                next_cid += 1

    # ── 2. epizódok közötti összevonás (centroid-hasonlóság) ─────────────
    # A cross-episode összevonáshoz **középpontosított** centroid kell: a
    # nyers CAM++ vektorokban nagy közös komponens van, ezért két különböző
    # beszélő nyers koszinusza is 0,8–0,9 lehet. A teljes évadnál ez a hiba
    # láncreakcióként összevonta az összes azonos nemű szereplőt (17 klaszter,
    # egy férfi és egy női óriásklaszter). A `_centered` epizódonként kivonja
    # az epizódátlagot, így a hasonlóság a tényleges hangszínt méri.
    vec_all = _centered(utts)

    def centered_centroid(members: list[Utterance]) -> np.ndarray | None:
        vs = [vec_all[id(u)] for u in members if id(u) in vec_all]
        ws = [u.dur for u in members if id(u) in vec_all]
        if not vs:
            return None
        v = np.average(np.stack(vs), axis=0, weights=np.asarray(ws, dtype=np.float32))
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def snapshot() -> dict[int, list[Utterance]]:
        d: dict[int, list[Utterance]] = {}
        for u in utts:
            if u.cluster is not None:
                d.setdefault(u.cluster, []).append(u)
        return d

    def info(members: list[Utterance]) -> tuple[np.ndarray | None, float, str | None]:
        vs = [u.emb for u in members if u.emb is not None]
        ws = [u.dur for u in members if u.emb is not None]
        c = centroid(vs, ws)
        dur = float(sum(u.dur for u in members))
        # A nemet az **erős** (kétségtelen) mérések döntik el: a QA kapu is
        # csak ezeket nézi, így a klaszter neme és az egyes sorok neme így
        # sem ütheti egymást. Ha nincs erős mérés, a súlyozott szavazat dönt.
        sf = sum(u.dur for u in members if u.p_female is not None
                 and u.p_female >= t["strong_f"])
        sm = sum(u.dur for u in members if u.p_female is not None
                 and u.p_female <= t["strong_m"])
        if sf or sm:
            g = "F" if sf >= sm else "M"
        else:
            vote = float(sum((u.p_female - 0.5) * u.dur for u in members
                             if u.p_female is not None))
            g = "F" if vote > 0 else ("M" if vote < 0 else None)
        return c, dur, g

    for _round in range(6):
        groups = snapshot()
        ids = sorted(groups)
        ccents, durs, gens, eps = {}, {}, {}, {}
        for cid in ids:
            _, durs[cid], gens[cid] = info(groups[cid])
            ccents[cid] = centered_centroid(groups[cid])
            eps[cid] = {u.ep for u in groups[cid]}
        uf = _UF()
        for i, a in enumerate(ids):
            if ccents[a] is None:
                continue
            for b in ids[i + 1:]:
                if ccents[b] is None:
                    continue
                if eps[a] & eps[b]:
                    continue                      # azonos epizód: tiltott
                if gens[a] and gens[b] and gens[a] != gens[b]:
                    continue                      # ellentétes nem: tiltott
                if cosine(ccents[a], ccents[b]) >= t["link_cos"]:
                    uf.union(a, b)
        merged = 0
        for u in utts:
            if u.cluster is None:
                continue
            r = uf.find(u.cluster)
            if r != u.cluster:
                u.cluster = r
                merged += 1
        if merged == 0:
            break
        if verbose:
            print(f"  → cross-episode összevonás: {merged} megnyilatkozás áthelyezve")

    # ── 3. szekvenciális finomítás (epizódonkénti Viterbi) ───────────────
    changed = 0
    for ep, group_utts in by_ep.items():
        groups = snapshot()
        present = sorted({u.cluster for u in group_utts if u.cluster is not None})
        # A finomítás is **epizódon belüli, középpontosított** vektorokon
        # történik. Nyers embeddinggel a teljes évadnál minden férfi sort a
        # legnagyobb férfi klaszterbe húzott a Viterbi (és ugyanezt tette a
        # női oldalon), mert a nyers koszinusz minden azonos nemű beszélőnél
        # 0,8 fölött van. A jelöltek is csak az epizódban jelen lévő klaszterek.
        epvec = _centered(group_utts)
        cents: dict[int, np.ndarray] = {}
        for c in present:
            vs = [epvec[id(u)] for u in groups[c] if u.ep == ep and id(u) in epvec]
            ws = [u.dur for u in groups[c] if u.ep == ep and id(u) in epvec]
            if not vs:
                continue
            v = np.average(np.stack(vs), axis=0, weights=np.asarray(ws, dtype=np.float32))
            n = float(np.linalg.norm(v))
            if n > 0:
                cents[c] = v / n
        gens = {c: info(m)[2] for c, m in groups.items() if m}
        sims: dict[tuple[int, int], float] = {}
        raw: dict[int, dict[int, float]] = {}
        for i, u in enumerate(group_utts):
            if u.emb is None or id(u) not in epvec:
                continue
            raw[i] = {}
            for c, cvec in cents.items():
                raw[i][c] = cosine(epvec[id(u)], cvec)
        if not raw:
            continue
        # AS-Norm: klaszterenként a saját átlag/szórás levonása
        all_vals: dict[int, list[float]] = {}
        for i, d in raw.items():
            for c, v in d.items():
                all_vals.setdefault(c, []).append(v)
        mu = {c: float(np.mean(v)) for c, v in all_vals.items()}
        sd = {c: float(np.std(v)) + 1e-3 for c, v in all_vals.items()}
        cand = sorted({c for d in raw.values() for c in d
                       if d[c] >= 0.25 or c in present})
        if not cand:
            continue
        for i, d in raw.items():
            for c in cand:
                if c in d:
                    sims[(i, c)] = float(np.tanh((d[c] - mu[c]) / (2.0 * sd[c])))
                else:
                    sims[(i, c)] = -1.0

        idx = {c: k for k, c in enumerate(cand)}
        n = len(cand)
        order = [i for i in range(len(group_utts)) if i in raw]
        score: list[float] = []
        back: list[list[int]] = []
        for pos, i in enumerate(order):
            em = [t["z_gain"] * sims.get((i, c), -1.0) for c in cand]
            # kemény nem-korlát: ismert nemű sort nem engedünk ellentétes
            # nemű klaszterbe (ez volt a jeleneten belüli hangváltás okozója)
            ug = gender_of(group_utts[i])
            if ug:
                for k, c in enumerate(cand):
                    if gens.get(c) not in (None, ug):
                        em[k] = -1e9
            if pos == 0:
                score, back = em, [[-1] * n]
                continue
            prev_i = order[pos - 1]
            gap = group_utts[i].start - group_utts[prev_i].end
            pen = t["viterbi_pen"] * max(0.0, 1.0 - max(0.0, gap) / t["viterbi_pen_gap"])
            dp = [0.0] * n
            bp = [0] * n
            for k, c in enumerate(cand):
                best, bk = -1e18, k
                for k2, c2 in enumerate(cand):
                    v = score[k2] - (pen if c2 != c else 0.0)
                    if v > best:
                        best, bk = v, k2
                dp[k] = em[k] + best
                bp[k] = bk
            score, _ = dp, back.append(bp)
        path = [0] * len(order)
        k = max(range(n), key=lambda j: score[j])
        for pos in range(len(order) - 1, -1, -1):
            path[pos] = cand[k]
            if back[pos][k] >= 0:
                k = back[pos][k]
        for pos, i in enumerate(order):
            if group_utts[i].cluster != path[pos]:
                group_utts[i].cluster = path[pos]
                changed += 1
    if verbose and changed:
        print(f"  → Viterbi-finomítás: {changed} megnyilatkozás klasztere javítva")

    # ── 4. nem-konzisztencia javítása (csak akusztikai bizonyítékkal) ────
    # A régi lánc **kizárólag** a P(nő) mért értékére mozgatott sort, és így
    # egyetlen zajos mérés kiszakíthatott egy színészt a klaszteréből: a
    # következő sor már ellenkező nemű hangon szólt. Most a mozgatáshoz a
    # hangnak is a célklaszterhez kell hasonlítania (középpontosított
    # koszinusz), méghozzá érdemben jobban, mint a saját klaszteréhez.
    vec = _centered(utts)
    groups = snapshot()
    gens = {c: info(members)[2] for c, members in groups.items()}
    ccents: dict[int, np.ndarray] = {}
    for c, members in groups.items():
        vs = [vec[id(u)] for u in members if id(u) in vec]
        if vs:
            v = np.mean(vs, axis=0)
            n = float(np.linalg.norm(v))
            if n > 0:
                ccents[c] = v / n
    fixed = 0
    for u in utts:
        if u.p_female is None or u.cluster is None or id(u) not in vec:
            continue
        want = ("F" if u.p_female >= t["strong_f"]
                else ("M" if u.p_female <= t["strong_m"] else None))
        if want is None or gens.get(u.cluster) in (None, want):
            continue
        own = ccents.get(u.cluster)
        scored = [(float(vec[id(u)] @ v), c) for c, v in ccents.items()
                  if gens.get(c) == want and c != u.cluster]
        if not scored:
            continue
        best_s, best = max(scored)
        own_s = float(vec[id(u)] @ own) if own is not None else -1.0
        if best_s < t["gender_move_cos"] or best_s - own_s < t["gender_move_margin"]:
            continue
        u.cluster = best
        fixed += 1
    if verbose and fixed:
        print(f"  → nem-javítás: {fixed} megnyilatkozás került a mért nemének "
              f"megfelelő klaszterbe (akusztikai egyezéssel)")

    # ── 5. töredék-klaszterek felszívása ─────────────────────────────────
    groups = snapshot()
    cents = {c: info(members)[0] for c, members in groups.items()}
    durs = {c: info(members)[1] for c, members in groups.items()}
    gens = {c: info(members)[2] for c, members in groups.items()}
    absorbed = 0
    for c in sorted(durs, key=lambda x: durs[x]):
        if durs[c] >= t["absorb_min_s"]:
            continue
        others = [o for o in durs if o != c and cents.get(o) is not None
                  and durs[o] >= t["absorb_min_s"]
                  and (gens[c] is None or gens[o] in (None, gens[c]))]
        if not others or cents.get(c) is None:
            continue
        best = max(others, key=lambda o: cosine(cents[c], cents[o]))
        if cosine(cents[c], cents[best]) >= t["absorb_cos"]:
            for u in groups[c]:
                u.cluster = best
            absorbed += 1
    if verbose and absorbed:
        print(f"  → {absorbed} töredék-klaszter beolvasztva a legközelebbi szereplőbe")

    # ── 5b. rövid töredékek: ne kapjanak saját hangot ────────────────
    # A kétbeszélős sorok fordulónkénti szétvágása után sok 1–3 s-os
    # megnyilatkozás keletkezik. Ezek hanglenyomata rövid, zenés ablakon
    # megbízhatatlan, ezért önálló szereplőt és hangot nem kaphatnak: a
    # leghosszabb, azonos nemű (vagy ismeretlen nemű) klaszterbe kerülnek.
    # Ez a „minden fél sor új szinkronhangot kap” hiba szerkezeti zárja.
    vec = _centered(utts)
    groups = snapshot()
    durs = {c: float(sum(u.dur for u in m)) for c, m in groups.items()}
    gens = {c: info(m)[2] for c, m in groups.items()}
    ccents: dict[int, np.ndarray] = {}
    for c, members in groups.items():
        vs = [vec[id(u)] for u in members if id(u) in vec]
        if vs:
            v = np.mean(vs, axis=0)
            n = float(np.linalg.norm(v))
            if n > 0:
                ccents[c] = v / n

    def screenish(members: list[Utterance]) -> bool:
        n = sum(1 for u in members if subs.is_screen_text(u.text))
        return n >= 2 and n / max(len(members), 1) >= t["frag_screen_max"]

    frag_uf = _UF()
    frag_pairs = 0
    for c in sorted(durs, key=lambda x: durs[x]):
        if durs[c] >= t["frag_merge_max_s"] or c not in ccents:
            continue
        if screenish(groups[c]):
            continue
        best, best_s = None, -1.0
        for o, co in ccents.items():
            if o == c or durs[o] < max(t["frag_host_min_s"], 2.0 * durs[c]):
                continue
            if gens[c] and gens[o] and gens[c] != gens[o]:
                continue
            if screenish(groups[o]):
                continue
            sims = [float(vec[id(u)] @ co) for u in groups[c] if id(u) in vec]
            if not sims:
                continue
            s = float(np.median(sims))
            if s > best_s:
                best, best_s = o, s
        if best is not None:
            frag_uf.union(best, c)
            frag_pairs += 1
    merged_frag = 0
    for u in utts:
        if u.cluster is None:
            continue
        r = frag_uf.find(u.cluster)
        if r != u.cluster:
            u.cluster = r
            merged_frag += 1
    if verbose and frag_pairs:
        print(f"  → {frag_pairs} rövid töredék (3 s alatt) olvadt be hosszabb, "
              f"azonos nemű szereplőbe ({merged_frag} megnyilatkozás)")

    # ── 6. kapu-vezérelt összevonás ───────────────────────────────────
    # A mérés kimutatta: az egymás után következő, koszinusz
    # >= cos_same hasonlóságú megszólalások gyakran két klaszterben végződnek
    # (átlag-linkage + Viterbi szétválaszt), és így két különböző magyar hangon
    # szólnak — ez a „színész több hangon beszél" hallható hiba. Ezeket a
    # klaszterpárokat egybe vonjuk, amíg a hangváltás meg nem szűnik.
    # Feltétel: a két klaszter erős nem-mérése nem ütközhet (nő/férfi tiltott).
    merged_pairs = 0
    for _round in range(8):
        ev = gender_evidence(utts, tuning)
        snap = snapshot()
        gens6 = {c: info(m)[2] for c, m in snap.items() if m}

        def strong(ca: int, cb: int) -> bool:
            # ismert, ellentétes nemű klaszterpár soha nem olvad össze
            ga, gb = gens6.get(ca), gens6.get(cb)
            if ga and gb and ga != gb:
                return True
            a, b = ev.get(ca, {}), ev.get(cb, {})
            return (a.get("F", 0) >= 5.0 and b.get("M", 0) >= 5.0) \
                or (a.get("M", 0) >= 5.0 and b.get("F", 0) >= 5.0)

        uf = _UF()
        hit = 0
        for ep, seq in by_ep.items():
            epvec = _centered(seq)
            for a, b in zip(seq, seq[1:]):
                if a.emb is None or b.emb is None or a.cluster is None \
                        or b.cluster is None or a.cluster == b.cluster:
                    continue
                if id(a) not in epvec or id(b) not in epvec:
                    continue
                if cosine(epvec[id(a)], epvec[id(b)]) < t.get("cos_same", 0.55):
                    continue
                if strong(a.cluster, b.cluster):
                    continue          # erősen ellentétes nem: nem vonjuk össze
                uf.union(a.cluster, b.cluster)
                hit += 1
        if hit == 0:
            break
        moved = 0
        for u in utts:
            if u.cluster is None:
                continue
            r = uf.find(u.cluster)
            if r != u.cluster:
                u.cluster = r
                moved += 1
        merged_pairs += hit
        if moved == 0:
            break
        if verbose:
            print(f"  → hang-azonos szomszédok összevonása: {moved} megnyilatkozás "
                  f"({hit} pár)")
    # Az összevonás után **nem** mozgatunk többé sort kizárólag a mért nem
    # alapján: a régi lánc itt 148 megnyilatkozást dobált át, és ezzel
    # szabdálta szét a színészeket. A nem a klaszter tulajdonsága marad; a
    # vegyes klasztert a 4. lépés akusztikai bizonyítéka és a QA jelzi.

    # ── 7. hangbizonyíték nélküli sorok ───────────────────────────────────
    # A rövid/nem hallható sorokból (nincs embedding) nem lehet szereplőt
    # csinálni: az önálló klaszterjük szereplőt és hangot pazarolna, miközben
    # róluk semmilyen mérés nincs. A szomszédos megszólalás szereplőjéhez
    # kötjük őket (a mondatot ott is ugyanaz a beszélő folytatja).
    unattached = 0
    detached = 0
    for _ep, seq in by_ep.items():
        for i, u in enumerate(seq):
            if u.emb is not None or u.cluster is None:
                continue
            # képernyőszöveg (helyszín/fejléc) nem beszélő: nem kap szereplőt
            if subs.is_screen_text(u.text):
                continue
            prev = nxt = None
            for j in range(i - 1, -1, -1):          # legközelebbi előző hang
                v = seq[j]
                if v.emb is not None and v.cluster is not None:
                    prev = v
                    break
            for j in range(i + 1, len(seq)):         # legközelebbi következő hang
                v = seq[j]
                if v.emb is not None and v.cluster is not None:
                    nxt = v
                    break
            if prev is None and nxt is None:
                unattached += 1                      # a epizódban nincs más hang
                continue
            gp = (u.start - prev.end) if prev else None
            gn = (nxt.start - u.end) if nxt else None
            if prev is not None and gp <= t["gap_cont"]:
                host = prev                          # rövid szünet az előző után
            elif nxt is not None and gn <= t["gap_cont"]:
                host = nxt                           # rövid szünet a következő előtt
            else:
                # nincs közeli szomszéd: csak akkor csatoljuk a legközelebbi
                # megszólaláshoz, ha az még ugyanabban a jelenetben van.
                # Különben egy gyerek kiáltása távoli, más nemű szereplő
                # hangját kapná (mért hiba: „Anya!” a narrátor hangján).
                cand = [c for c in (prev, nxt) if c is not None]
                if not cand:
                    unattached += 1
                    continue
                host = min(cand, key=lambda c: (c.start - u.end)
                           if c.start >= u.end else (u.start - c.end))
                gap = (host.start - u.end) if host.start >= u.end else (u.start - host.end)
                if gap > t["attach_max_gap"]:
                    detached += 1
                    continue
            if host.cluster != u.cluster:
                u.cluster = host.cluster
    if verbose and (unattached or detached):
        print(f"  → {detached} hangbizonyíték nélküli sor maradt a saját "
              f"klaszterében (nincs közeli szomszéd), {unattached} pedig "
              f"egyáltalán nem csatolható")

    # ── 8. fordulónkénti cannot-link ─────────────────────────────────────
    # Egy „- A? - B.” feliratsor fordulói **két különböző beszélő**. A szöveg
    # szétvágása önmagában nem elég: a rövid második forduló hanglenyomata
    # gyakran az első beszélőhöz hasonlít, és a klaszterezés összevonja őket
    # (mért példa: 47 szétvágott sorból 20-ban ugyanaz a szereplő maradt).
    # Itt a leghosszabb forduló marad a klaszterében, a többit átmozgatjuk
    # egy másik, azonos nemű klaszterbe; ha nincs alkalmas, új klasztert kap.
    cue_parts: dict[tuple[str, object], list[Utterance]] = {}
    for u in utts:
        if u.cues:
            cue_parts.setdefault((u.ep, u.cues[0].index), []).append(u)
    # A zárójeles fordulók néha **külön feliratsorban** vannak (rádió/telefon:
    # „[Szavamat adom.]” + „[Gyertek ki!]”), ezért a cue-index nem fogja össze
    # őket. Az egymást követő, rövid szünettel induló zárójeles sorok külön
    # beszélők; ezeket is cannot-link párba tesszük.
    for ep, seq in by_ep.items():
        for a, b in zip(seq, seq[1:]):
            ta, tb = a.text.strip(), b.text.strip()
            if (ta.startswith("[") and ta.endswith("]")
                    and tb.startswith("[") and tb.endswith("]")
                    and b.start - a.end <= 0.8):
                cue_parts.setdefault((ep, f"br{a.cues[0].index}"), []).extend([a, b])
    snap8 = snapshot()
    gens8 = {c: info(m)[2] for c, m in snap8.items() if m}
    cent8 = _centered(utts)
    ccents8: dict[tuple[int, str], np.ndarray] = {}
    for c, members in snap8.items():
        for ep in {u.ep for u in members}:
            vs = [cent8[id(u)] for u in members if u.ep == ep and id(u) in cent8]
            if not vs:
                continue
            v = np.mean(vs, axis=0)
            n = float(np.linalg.norm(v))
            if n > 0:
                ccents8[(c, ep)] = v / n
    cannot = 0
    for _key, parts in cue_parts.items():
        if len(parts) < 2:
            continue
        parts = list({id(u): u for u in parts}.values())
        if len(parts) < 2:
            continue
        parts = sorted(parts, key=lambda u: -u.dur)
        used = {parts[0].cluster}
        for u in parts[1:]:
            if u.cluster not in used:
                used.add(u.cluster)
                continue
            ug = gender_of(u)
            ep = u.ep
            best, best_s = None, -2.0
            if u.emb is not None and id(u) in cent8:
                for (c, cep), v in ccents8.items():
                    if cep != ep or c in used:
                        continue
                    if ug and gens8.get(c) not in (None, ug):
                        continue
                    s = float(cent8[id(u)] @ v)
                    if s > best_s:
                        best, best_s = c, s
            if best is None:
                # nincs embedding: a szomszédos sor beszélője (a második
                # forduló folytatása) a legjobb tipp, ha más a klasztere
                seq = by_ep.get(ep, [])
                i = next((k for k, v in enumerate(seq) if v is u), -1)
                order = list(range(i + 1, len(seq))) + list(range(i - 1, -1, -1))
                for j in order:
                    v = seq[j]
                    if v.cluster is None or v.cluster in used:
                        continue
                    if ug and gens8.get(v.cluster) not in (None, ug):
                        continue
                    best = v.cluster
                    break
            if best is None:
                best = next_cid
                next_cid += 1
            u.cluster = best
            used.add(best)
            cannot += 1
    if verbose and cannot:
        print(f"  → fordulónkénti cannot-link: {cannot} forduló került másik szereplőhöz")

    # ── 9. kétoldalú töredék-összevonás ──────────────────────────────────
    # Ha egy szomszédos pár mindkét tagja közelebb van a **másik** klaszter
    # mintáihoz, mint a sajátjához, akkor a diarizáció ugyanazt a beszélőt
    # vágta ketté (mért példa: E04 C053/C001). Ilyenkor a rövidebb klasztert
    # a hosszabbhoz csatoljuk. Ez nem a cannot-link ellen dolgozik: ott a két
    # forduló szövegesen bizonyítottan két beszélő; itt viszont mindkét oldal
    # akusztikai bizonyítéka a másik klaszter felé mutat.
    raw = {id(u): norm(np.asarray(u.emb, dtype=np.float32))
           for u in utts if u.emb is not None}

    def pair_margins(a: Utterance, b: Utterance, groups9: dict[int, list[Utterance]]
                     ) -> tuple[float, float] | None:
        """(a saját-idegen, b saját-idegen) klaszterátlag-különbség."""
        if a.cluster is None or b.cluster is None or id(a) not in raw or id(b) not in raw:
            return None
        ga, gb = groups9.get(a.cluster, []), groups9.get(b.cluster, [])
        if a.cluster == b.cluster or len(ga) < 3 or len(gb) < 3:
            return None

        def mean_to(u: Utterance, members: list[Utterance]) -> float | None:
            vals = [float(raw[id(u)] @ raw[id(v)]) for v in members
                    if v is not u and id(v) in raw]
            return float(np.mean(vals)) if len(vals) >= 2 else None

        a_own, a_other = mean_to(a, ga), mean_to(a, gb)
        b_own, b_other = mean_to(b, gb), mean_to(b, ga)
        if None in (a_own, a_other, b_own, b_other):
            return None
        return float(a_own - a_other), float(b_own - b_other)

    merged_split = 0
    for _round in range(4):
        groups9 = snapshot()
        gens9 = {c: info(m)[2] for c, m in groups9.items() if m}
        durs9 = {c: float(sum(u.dur for u in m)) for c, m in groups9.items() if m}
        ev9 = gender_evidence(utts, tuning)
        uf9 = _UF()
        hit9 = 0
        for _ep, seq in by_ep.items():
            epvec = _centered(seq)
            for a, b in zip(seq, seq[1:]):
                if a.cluster is None or b.cluster is None or a.cluster == b.cluster:
                    continue
                if id(a) not in epvec or id(b) not in epvec:
                    continue
                if cosine(epvec[id(a)], epvec[id(b)]) < t.get("cos_same", 0.55):
                    continue
                ca, cb = a.cluster, b.cluster
                ga, gb = gens9.get(ca), gens9.get(cb)
                if ga and gb and ga != gb:
                    continue
                ea, eb = ev9.get(ca, {}), ev9.get(cb, {})
                if ((ea.get("F", 0) >= 5.0 and eb.get("M", 0) >= 5.0)
                        or (ea.get("M", 0) >= 5.0 and eb.get("F", 0) >= 5.0)):
                    continue
                if min(durs9.get(ca, 0.0), durs9.get(cb, 0.0)) > 90.0:
                    continue
                margins = pair_margins(a, b, groups9)
                if margins is None or margins[0] >= 0.0 or margins[1] >= 0.0:
                    continue
                short, host = (ca, cb) if durs9[ca] <= durs9[cb] else (cb, ca)
                uf9.union(host, short)
                hit9 += 1
        if not hit9:
            break
        moved9 = 0
        for u in utts:
            if u.cluster is None:
                continue
            r = uf9.find(u.cluster)
            if r != u.cluster:
                u.cluster = r
                moved9 += 1
        merged_split += hit9
        if not moved9:
            break
        if verbose:
            print(f"  → kétoldalú töredék-összevonás: {hit9} pár, "
                  f"{moved9} megnyilatkozás")

    # ── végleges klaszterek ──────────────────────────────────────────────
    final: dict[int, Cluster] = {}
    for cid, members in snapshot().items():
        # a tisztán képernyőszöveg-klaszter nem szereplő: a `plan` úgyis
        # kihagyja a szinkronból, itt csak a kasztot és a statisztikát hízlalná
        if all(subs.is_screen_text(u.text) for u in members):
            continue
        c = Cluster(cid=cid, uids=[u.uid for u in members])
        c.centroid, c.dur, c.gender = info(members)
        c.episodes = [u.ep for u in members]
        c.gender_vote = float(sum((u.p_female - 0.5) * u.dur for u in members
                                  if u.p_female is not None))
        c.n_long = sum(1 for u in members if u.dur >= 2.0)
        for u in members:
            u.cluster = cid
        final[cid] = c
    # újraszámozás beszédidő szerint (a leghosszabb a legkisebb azonosító)
    order = sorted(final, key=lambda c: -final[c].dur)
    remap = {old: new for new, old in enumerate(order)}
    out: dict[int, Cluster] = {}
    # Egyetlen bejárás: az azonosítót egyszer cseréljük le. (Különben egy már
    # átnevezett érték újra beleeshet egy későbbi kulcsba, és másodszor is
    # átneveződik — ez klaszter-összefolyáshoz és a rekordok szétszórásához
    # vezetett.)
    for u in utts:
        if u.cluster in remap:
            u.cluster = remap[u.cluster]
    for old in order:
        cl = final[old]
        cl.cid = remap[old]
        out[remap[old]] = cl
    if verbose:
        print(f"  → {len(out)} szereplő-klaszter "
              f"({sum(1 for c in out.values() if c.gender == 'F')} nő, "
              f"{sum(1 for c in out.values() if c.gender == 'M')} férfi, "
              f"{sum(1 for c in out.values() if c.gender is None)} bizonytalan)")
    return out


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _turn_gender(members: list[Utterance]) -> tuple[str | None, float]:
    """Forduló/klaszter neme súlyozott log-odds összeggel.

    A rövid, kiabálós sorok P(nő) értéke zajos; a log-odds összegzés a
    bizonyíték erejét használja, a súly a beszédidő (2 s felett nem ad több
    független bizonyítékot). Ez ugyanaz az elv, mint a profi diarizáció
    nem-döntésénél: nem szavazunk, hanem valószínűséget összegzünk.
    """
    num = wsum = 0.0
    for u in members:
        if u.p_female is None or u.dur <= 0:
            continue
        w = min(float(u.dur), 2.0)
        num += w * _logit(u.p_female)
        wsum += w
    if wsum <= 0:
        return None, 0.5
    p = _sigmoid(num / wsum)
    if p >= 0.85:
        return "F", p
    if p <= 0.15:
        return "M", p
    return None, p


#: jelenethatár: ennél hosszabb szünet választja el a jeleneteket. A rövid
#: válaszok diarizációs címkéje csak a jeleneten belül értelmezhető.
SCENE_GAP_S = 18.0
#: a jelenet azonos nemű beszédének ekkora részét kell egy szereplőnek vinnie
#: ahhoz, hogy a többi azonos nemű, rövid betét az ő töredéke legyen.
SCENE_LEAD_MIN_S = 6.0
SCENE_LEAD_SHARE = 0.60
#: e fölötti jelenet-beszédidő már önálló szereplő, nem töredék
SCENE_MINOR_S = 4.0
#: ennél hosszabb összbeszédű klaszter soha nem töredék (lehet csendes mellékszereplő)
SCENE_FRAGMENT_TOTAL_S = 60.0
#: a QA-ban ennél rövidebb összbeszédű szereplő számít töredéknek
SCENE_FRAGMENT_MAX_S = 20.0
#: a töredék és a jelenet fő beszélője közti hanghasonlóság alsó határa
SCENE_MERGE_COS = 0.40
#: e fölötti összbeszédű klaszter „nagy” szereplő. Ha egy jelenetben adott
#: nemből **csak egy** nagy szereplő van, a jelenet kisebb azonos nemű
#: klaszterei az ő töredékei — akkor is, ha a jelenetben többször szólnak
#: (a főszereplő elszórt sorai: mért eset a 4021–4037 s, „Nem tudod a teljes
#: történetet, Mark.”, ami külön hangon ment ki).
SCENE_MAJOR_MIN_S = 60.0
SCENE_MAJOR_COS = 0.15
SCENE_MAJOR_MAX_S = 12.0


def scenes(utts: list[Utterance], gap_s: float = SCENE_GAP_S) -> list[list[Utterance]]:
    """Jelenetek: rövid szünetekkel összefüggő beszédblokkok."""
    seq = sorted((u for u in utts if u.cluster is not None), key=lambda u: u.start)
    out: list[list[Utterance]] = []
    for u in seq:
        if out and u.start - out[-1][-1].end <= gap_s:
            out[-1].append(u)
        else:
            out.append([u])
    return out


#: megszólítás-környezet: ennyi másodpercen belüli szomszéd számít
#: (a jelenethatárral azonos, tehát a jelenet szomszédos sorai)
VOCATIVE_NEIGHBOR_S = SCENE_GAP_S


def _addressed_nearby(scene: list[Utterance], i: int,
                      known: set[str] | None = None,
                      gap_s: float = VOCATIVE_NEIGHBOR_S) -> bool:
    """Megszólít-e a sor vagy a jelenet szomszédos sora?"""
    if subs.vocative(scene[i].text, known):
        return True
    if i > 0:
        prev = scene[i - 1]
        if scene[i].start - prev.end <= gap_s \
                and subs.vocative(prev.text, known):
            return True
    if i + 1 < len(scene):
        nxt = scene[i + 1]
        if nxt.start - scene[i].end <= gap_s \
                and subs.vocative(nxt.text, known):
            return True
    return False


def vocative_fragments(scene: list[Utterance], genders: dict[int, str | None],
                       total_dur: dict[int, float],
                       known: set[str] | None = None) -> list[tuple[Utterance, int]]:
    """Rövid jelenet-töredékek, amelyeket a **szöveg megszólítása** köt be.

    A QA ``scene_minor`` mérőszáma azt fogja meg, amikor egy jelenet azonos
    nemű beszédének nagy részét egy szereplő viszi, mégis egy másik, nagyon
    rövid szereplő szólal meg benne — a diarizáció ugyanazt a színészt vágta
    több címkére. A címkék rövidsége miatt a hang önmagában nem dönthető el
    (1–3 s zenés ablakon a beszélő-embedding zajos), a **szöveg viszont
    igen**: ha a töredék sorában vagy a közvetlen szomszédjában megszólítás
    áll („Hé, Claire, ne mondd ezt!”, „Claire, lányom.”, „Nem tudod a teljes
    történetet, Mark.”, „Kicsim.”), akkor a sor egy folyó dialógus része, nem
    új szereplő. Ilyenkor a jelenet azonos nemű, nagy beszélőjéhez kerül.

    Feltételek (a téves összevonás ellen):
      * a jelenetben legyen legalább két azonos nemű szereplő;
      * a befogadó a jelenet azonos nemű, legtöbbet beszélő szereplője, és
        összességében is nagy (``SCENE_MAJOR_MIN_S``);
      * a töredék a jelenetben és összességében is rövid
        (``SCENE_MINOR_S`` / ``SCENE_FRAGMENT_MAX_S``);
      * legyen megszólítás a szövegkörnyezetben (``known``: az epizód
        bizonyított névkészlete, lásd ``subs.vocative_names``).
    """
    out: list[tuple[Utterance, int]] = []
    if len(scene) < 2:
        return out
    for gender in ("F", "M"):
        per_dur: dict[int, float] = {}
        for u in scene:
            if u.cluster is None or genders.get(u.cluster) != gender:
                continue
            per_dur[u.cluster] = per_dur.get(u.cluster, 0.0) + u.dur
        if len(per_dur) < 2:
            continue
        lead = max(per_dur, key=per_dur.get)
        if total_dur.get(lead, 0.0) < SCENE_MAJOR_MIN_S:
            continue
        for i, u in enumerate(scene):
            cid = u.cluster
            if cid is None or cid == lead or genders.get(cid) != gender:
                continue
            if per_dur.get(cid, 0.0) > SCENE_MINOR_S:
                continue
            if total_dur.get(cid, 0.0) > SCENE_FRAGMENT_MAX_S:
                continue
            if not _addressed_nearby(scene, i, known):
                continue
            out.append((u, lead))
    return out


def _labeled_vectors(utts: list[Utterance]) -> list[tuple[int, np.ndarray]]:
    """(klaszter, centered embedding) párok — a PLDA tanítóadata."""
    vecs = _centered(utts)
    return [(int(u.cluster), vecs[id(u)]) for u in utts
            if id(u) in vecs and u.cluster is not None]


def fit_plda(utts: list[Utterance]) -> PLDAModel | None:
    """PLDA-modell tanítása az utterance-ek centered embeddingjeiből.

    A klaszter-azonosítók beszélő-címkékként szolgálnak. Ha nincs elég
    címke/minta, None-t ad vissza (visszaesés AS-Norm-ra).
    """
    labeled = _labeled_vectors(utts)
    if len(labeled) < 10:
        return None
    model = PLDAModel.fit([v for _c, v in labeled], [c for c, _v in labeled])
    return model if model.fitted else None


def verify_pair_plda(a: np.ndarray, b: np.ndarray,
                     cohort: list[np.ndarray] | None = None,
                     plda_model: PLDAModel | None = None,
                     tuning: dict | None = None) -> dict:
    """PLDA-alapú verifikáció, AS-Norm-mal kalibrálva.

    Ha a PLDA-modell nincs tanítva (vagy a keresztvalidációból nem maradt rá
    modell), visszaesik a verify_pair-ra.

    A küszöbök a **null-normalizált z skálán** értendők (``plda_z_same`` /
    ``plda_z_diff``); ha ezek nincsenek beállítva, a modell mért operációs
    pontja, azután a ``plda.Z_SAME``/``Z_DIFF`` érvényes. A ``verifier_same`` /
    ``verifier_diff`` csak az AS-Norm ágon (``verify_pair``) számít.
    """
    if plda_model is None or not plda_model.fitted:
        return verify_pair(a, b, cohort=cohort, tuning=tuning)
    t = tuning or {}
    return plda_model.decide(a, b, cohort=cohort,
                             temp=float(t.get("plda_temp", 2.0)),
                             plda_weight=float(t.get("plda_weight", 0.5)),
                             z_same=t.get("plda_z_same"),
                             z_diff=t.get("plda_z_diff"))


@dataclass
class PLDACrossFit:
    """Keresztvalidált PLDA a klaszter-összevonáshoz (körkörösség ellen).

    Ha a modellt ugyanazokon a klaszter-címkéken tanítjuk, amelyeket épp
    összevonni próbálunk, a modell a saját korábbi döntéseit igazolja vissza.
    Ezért a klasztereket foldokba osztjuk, és minden vizsgált pár olyan modellt
    kap, amely **egyik** klasztert sem látta tanításkor:

      * azonos foldban lévő pár → az adott fold kihagyásával tanított modell;
      * különböző foldban lévő pár → a két fold kihagyásával tanított modell.

    Ha ilyen modell nincs (kevés klaszter), a hívó AS-Norm-ra esik vissza — az
    cohort-alapú z nem használ klaszter-címkéket, tehát címkefüggetlen.
    """

    folds: dict[int, int]
    models: dict[frozenset[int], PLDAModel]
    n_folds: int = 3

    def model_for(self, a: int, b: int) -> PLDAModel | None:
        fa, fb = self.folds.get(a), self.folds.get(b)
        if fa is None or fb is None:
            return None
        m = self.models.get(frozenset((fa, fb)))
        return m if (m is not None and m.fitted) else None

    def describe(self) -> str:
        fitted = sum(1 for m in self.models.values() if m.fitted)
        return (f"{fitted}/{len(self.models)} keresztvalidált modell "
                f"({self.n_folds} fold, {len(self.folds)} klaszter)")


def plda_crossfit(utts: list[Utterance], n_folds: int = 3
                  ) -> PLDACrossFit | None:
    """Foldonként kihagyó PLDA-modellek tanítása az epizód klasztereiből.

    A tanító címkék a klaszter-azonosítók, de a modellek mindig **kihagynak**
    egy (vagy két) foldot, így a merge-ciklusban egyetlen vizsgált pár sem
    kap olyan modellt, amely a saját tagjait látta volna.
    """
    if n_folds < 2:
        return None
    members: dict[int, list[np.ndarray]] = {}
    for cid, v in _labeled_vectors(utts):
        members.setdefault(cid, []).append(v)
    ids = sorted(members)
    if len(ids) < 2 * n_folds:
        return None
    folds = {cid: i % n_folds for i, cid in enumerate(ids)}
    # Csak a ``model_for`` által kérdezhető kulcsok: egy fold kihagyása (azonos
    # foldban lévő pár) és két fold kihagyása (különböző foldban lévő pár).
    keys = [frozenset((f,)) for f in range(n_folds)]
    keys += [frozenset(c) for c in itertools.combinations(range(n_folds), 2)]
    models: dict[frozenset[int], PLDAModel] = {}
    for key in keys:
        emb: list[np.ndarray] = []
        lab: list[int] = []
        for cid in ids:
            if folds[cid] in key:
                continue
            emb.extend(members[cid])
            lab.extend([cid] * len(members[cid]))
        models[key] = (PLDAModel.fit(emb, lab) if len(emb) >= 10
                       else PLDAModel())
    if not any(m.fitted for m in models.values()):
        return None
    return PLDACrossFit(folds=folds, models=models, n_folds=n_folds)


def from_diarization(utts: list[Utterance], tuning: dict | None = None,
                     verbose: bool = True,
                     plda_model: PLDAModel | "PLDACrossFit" | None = None
                     ) -> dict[int, Cluster]:
    """Diarizációs fordulók → végleges szereplő-klaszterek.

    A fordulócímkék a teljes sávon stabilak, de a nyers klaszterezés
    túlszegmentálhat (ugyanaz a színész több címkén) vagy összevonhat
    (két, hasonló hangú szereplő). Ezért itt már **nem** feliratsoronként
    klaszterezünk, hanem fordulókon:

      1. a kevert nemű fordulót szétvágjuk a P(nő) jel mentén;
      2. az azonos hosszú önbemutatást („A nevem …”) tartalmazó címkéket
         összekötjük (ugyanaz a felvétel/szereplő);
      3. csak azonos nemű fordulókat vonunk össze; két forduló akkor vonható
         össze, ha a hang hasonló, és vagy nagyon erős a hasonlóság (ugyanaz
         a színész töredéke), vagy a két forduló nem ugyanabban a jelenetben
         szólal meg (nem co-occurál);
      4. a rövid töredékeket a legközelebbi, nem co-occuráló azonos nemű
         szereplőhöz csatoljuk.

    Ez a „egy színész = egy hang” garancia, miközben a párbeszédben
    egymással szemben álló két szereplő nem olvad össze.
    """
    t = {**DEFAULTS, **(tuning or {})}
    groups: dict[int, list[Utterance]] = {}
    for u in utts:
        # A `diar` a nyers fordulócímke; a `cluster` lehet egy korábbi futás
        # eredménye. A szétvágás mindig a nyers címkéből indul.
        key = u.cluster
        if key is not None:
            groups.setdefault(int(key), []).append(u)
    if not groups:
        return {}
    vec = _centered(utts)

    def centroid_of(members: list[Utterance]) -> np.ndarray | None:
        vs = [vec[id(u)] for u in members if id(u) in vec]
        ws = [u.dur for u in members if id(u) in vec]
        if not vs:
            return None
        v = np.average(np.stack(vs), axis=0,
                       weights=np.asarray(ws, dtype=np.float32))
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else None

    def duration(members: list[Utterance]) -> float:
        return float(sum(u.dur for u in members))

    def co_occurs(a: list[Utterance], b: list[Utterance],
                  gap_s: float = 8.0) -> bool:
        """Ugyanabban a jelenetben (szoros időbeli szomszédságban) vannak-e?"""
        for x in a:
            for y in b:
                if min(x.end, y.end) - max(x.start, y.start) > 0.05:
                    return True
                if y.start >= x.end and y.start - x.end <= gap_s:
                    return True
                if x.start >= y.end and x.start - y.end <= gap_s:
                    return True
        return False

    # ── 1. kevert nemű fordulók szétvágása ───────────────────────────────
    next_id = max(groups) + 1
    split = 0
    for cid in sorted(groups):
        members = groups[cid]
        sf = sum(u.dur for u in members
                 if u.p_female is not None and u.p_female >= 0.85)
        sm = sum(u.dur for u in members
                 if u.p_female is not None and u.p_female <= 0.15)
        total = duration(members)
        if min(sf, sm) < 3.0 or min(sf, sm) < 0.15 * max(total, 1e-6):
            continue
        female = [u for u in members if u.p_female is not None and u.p_female >= 0.55]
        male = [u for u in members if u.p_female is not None and u.p_female < 0.55]
        unknown = [u for u in members if u.p_female is None]
        if len(female) < 2 or len(male) < 2:
            continue
        for u in unknown:
            (female if len(female) >= len(male) else male).append(u)
        groups[cid] = female
        groups[next_id] = male
        for u in male:
            u.cluster = next_id
        next_id += 1
        split += 1
    if verbose and split:
        print(f"  → diarizáció: {split} kevert nemű forduló szétvágva")

    # ── 1b. önbemutatás = ugyanaz a szereplő ────────────────────────────
    # A „A nevem …” típusú, hosszú önbemutatás egyedi: ha ugyanaz a név
    # két címkében szerepel, a két címke ugyanahhoz a beszélőhöz tartozik.
    # Ez oldja a visszajátszás/ismételt felvétel miatti kettészakadást.
    def norm_text(s: str) -> str:
        return " ".join("".join(ch.lower() if ch.isalnum() else " "
                                for ch in s).split())

    intros = [(norm_text(u.text), int(u.cluster)) for u in utts
              if u.cluster is not None and norm_text(u.text).startswith("a nevem ")]
    intro_links = 0
    for _ in range(3):
        merged_intro = False
        for i, (ta, ca) in enumerate(intros):
            if ca not in groups:
                continue
            for tb, cb in intros[i + 1:]:
                if cb not in groups or ca == cb:
                    continue
                if ta not in tb and tb not in ta:
                    continue
                a, b = sorted((ca, cb))
                ga, _pa = _turn_gender(groups[a])
                gb, _pb = _turn_gender(groups[b])
                if ga and gb and ga != gb:
                    continue
                groups[a] = groups[a] + groups.pop(b)
                for u in groups[a]:
                    u.cluster = a
                intro_links += 1
                merged_intro = True
                break
            if merged_intro:
                break
        if not merged_intro:
            break
    if verbose and intro_links:
        print(f"  → diarizáció: {intro_links} önbemutatás alapján összekötve")

    # ── 2. összevonás ──────────────────────────────────────────────────
    def info(cid: int) -> tuple[np.ndarray | None, float, str | None, float]:
        members = groups[cid]
        g, p = _turn_gender(members)
        return centroid_of(members), duration(members), g, p

    cache = {cid: info(cid) for cid in groups}
    merged = 0
    # pár → kapott-e keresztvalidált PLDA-modellt (a záró jelentéshez)
    scored: dict[tuple[int, int], bool] = {}
    for _ in range(80):
        best: tuple[float, int, int] | None = None
        ids = sorted(groups)
        centroids = [(c, cache[c][0]) for c in ids if cache[c][0] is not None]
        for ai, a in enumerate(ids):
            ca, da, ga, _pa = cache[a]
            if ca is None:
                continue
            for b in ids[ai + 1:]:
                cb, db, gb, _pb = cache[b]
                if cb is None:
                    continue
                if ga and gb and ga != gb:
                    continue
                s = cosine(ca, cb)
                # nagyon erős hangegyezés: ugyanannak a színésznek a
                # töredéke akkor is, ha ugyanabban a jelenetben van
                strong = s >= 0.80
                # gyengébb egyezés csak különböző jelenetek között, és csak
                # szigorúbb küszöbbel: a különböző, hasonló hangú szereplők
                # ne olvadjanak össze pusztán azért, mert más jelenetben vannak
                cross = s >= 0.55 and not co_occurs(groups[a], groups[b])
                # Kalibrált verifikátor (AS-Norm): ha biztosan különböző
                # beszélő, a nyers koszinusz nem elég bizonyíték — veto.
                cohort = [v for c, v in centroids if c not in (a, b)]
                # Keresztvalidált PLDA: a pár egyik klaszterét sem látta
                # modell dönt; ha nincs ilyen, az AS-Norm z (címkefüggetlen)
                # adja a vétót. Így a döntés nem igazolja vissza önmagát.
                pm = (plda_model.model_for(a, b)
                      if isinstance(plda_model, PLDACrossFit) else plda_model)
                scored[(a, b)] = pm is not None
                vcal = verify_pair_plda(ca, cb,
                                        cohort=cohort if len(cohort) >= 3 else None,
                                        plda_model=pm, tuning=t)
                if vcal["decision"] == "different":
                    strong = False
                    cross = False
                # FIGYELEM: co-occuráló (egymással párbeszédben álló) fordulókat
                # szándékosan **nem** vonunk össze a legerősebb egyedi sorpár
                # alapján. Mért ellenpélda (E04, 893 s): két különböző
                # szereplő („Dr. Ottes vagyok…” / „Még egy gyakornok is
                # látja…”) koszinusza 0,93, de a diarizáció és a szöveg is
                # két beszélőt bizonyít — az összevonás két figurát egy
                # hangra tett volna. A rövid, egymást követő sorok
                # hasonlósága nem elég bizonyíték.
                if (strong or cross) and (best is None or s > best[0]):
                    best = (s, a, b)
        if best is None:
            break
        _s, a, b = best
        groups[a] = groups[a] + groups.pop(b)
        cache[a] = info(a)
        cache.pop(b, None)
        merged += 1
    if verbose and merged:
        print(f"  → diarizáció: {merged} forduló összevonva "
              f"(azonos hang, nem co-occuráló jelenet)")
    if verbose and scored:
        n_plda = sum(scored.values())
        print(f"  → verifikátor: {n_plda} pár keresztvalidált PLDA-val, "
              f"{len(scored) - n_plda} pár AS-Norm-mal (címkefüggetlen)")

    # ── 2b. jelenet-szintű töredék-összevonás ───────────────────────────
    # Egy jelenet azonos nemű beszédének túlnyomó részét egy szereplő viszi.
    # A jelenet rövid (≤ 4 s) azonos nemű klaszterei nem új figurák: a rövid
    # válaszok diarizációs címkéje megbízhatatlan. Mért hiba: a 911-jelenetben
    # ugyanaz a nő (Claire) három címkén, három hangon szólt (C001/C009/C012);
    # a „főszereplő váltakozva más-más hangon” panasz innen jön. Csak akkor
    # vonunk össze, ha a hang is hasonló, és a töredék sehol nem domináns.
    scene_moved = 0
    total_dur = {cid: duration(m) for cid, m in groups.items()}
    for scene in scenes(utts):
        if len(scene) < 2:
            continue
        for gender in ("F", "M"):
            per: dict[int, list[Utterance]] = {}
            for u in scene:
                if u.cluster not in groups:
                    continue
                if cache[u.cluster][2] == gender:
                    per.setdefault(u.cluster, []).append(u)
            if len(per) < 2:
                continue
            dur = {cid: sum(u.dur for u in m) for cid, m in per.items()}
            lead = max(dur, key=dur.get)
            majors = [c for c in dur
                      if total_dur.get(c, 0.0) >= SCENE_MAJOR_MIN_S]
            if len(majors) == 1:
                lead = majors[0]
                min_cos, max_secs = SCENE_MAJOR_COS, SCENE_MAJOR_MAX_S
            elif dur[lead] >= SCENE_LEAD_MIN_S \
                    and dur[lead] >= SCENE_LEAD_SHARE * sum(dur.values()):
                min_cos, max_secs = SCENE_MERGE_COS, SCENE_MINOR_S
            else:
                continue
            for cid, secs in sorted(dur.items(), key=lambda kv: -kv[1]):
                if cid == lead or secs > max_secs:
                    continue
                if total_dur.get(cid, 0.0) > SCENE_FRAGMENT_TOTAL_S:
                    continue
                cvec, lvec = cache[cid][0], cache[lead][0]
                if cvec is None or lvec is None or cosine(cvec, lvec) < min_cos:
                    continue
                groups[lead] += groups.pop(cid)
                for u in groups[lead]:
                    u.cluster = lead
                scene_moved += len(per[cid])
                total_dur[lead] += total_dur.pop(cid, 0.0)
                cache[lead] = info(lead)
                cache.pop(cid, None)
    if verbose and scene_moved:
        print(f"  → diarizáció: {scene_moved} rövid jelenet-sor a jelenet "
              f"azonos nemű fő beszélőjéhez csatolva")

    # ── 2c. szöveg-alapú megszólítás ────────────────────────────────────
    # A 2b. lépés a hanghasonlóságra támaszkodik, de egy 1–3 s-os, zenés
    # töredék embeddingje nem elég megbízható: a rövid sorok így is saját
    # címkén maradhatnak (mért eset: E01 „Hová mész?”, „Kicsim.”, „Nem tudod
    # a teljes történetet, Mark.”). A szöveg viszont egyértelmű: ahol a sor
    # vagy a szomszédja megszólít valakit, ott folyó dialógus van, a töredék
    # nem új szereplő — a jelenet azonos nemű fő beszélőjéhez kerül.
    voc_moved = 0
    touched: set[int] = set()
    known: set[str] = set()
    for u in utts:
        known |= subs.vocative_names(u.text)
    for scene in scenes(utts):
        gens = {c: cache[c][2] for c in cache}
        for u, lead in vocative_fragments(scene, gens, total_dur, known):
            old = u.cluster
            if old is None or old == lead or old not in groups:
                continue
            groups[old] = [x for x in groups[old] if x is not u]
            groups[lead].append(u)
            u.cluster = lead
            touched.update((old, lead))
            voc_moved += 1
    for cid in sorted(touched):
        if not groups.get(cid):
            groups.pop(cid, None)
            cache.pop(cid, None)
            total_dur.pop(cid, None)
        else:
            cache[cid] = info(cid)
            total_dur[cid] = duration(groups[cid])
    if verbose and voc_moved:
        print(f"  → diarizáció: {voc_moved} rövid sor a szöveg megszólítása "
              f"alapján a jelenet azonos nemű fő beszélőjéhez került")

    # ── 2d. félbemaradt mondat folytatása ──────────────────────────────
    # Ha az előző sor végén nincs mondatvégi írásjel, és a következő sor
    # kisbetűvel folytatja („…, amikor még mindketten” / „nagyon fiatalok
    # voltatok.”), akkor a diarizáció mondat közben váltott címkét: a sor az
    # előző beszélőé. Csak rövid töredéket csatolunk, hogy egy valódi
    # kérdés-válasz („Hol vagy?”) ne olvadjon össze.
    cont_moved = 0
    for scene in scenes(utts):
        for i in range(1, len(scene)):
            u, prev = scene[i], scene[i - 1]
            if u.cluster is None or prev.cluster is None \
                    or u.cluster == prev.cluster:
                continue
            if u.cluster not in groups or prev.cluster not in groups:
                continue
            if duration(groups[u.cluster]) > SCENE_MINOR_S:
                continue
            if total_dur.get(u.cluster, 0.0) > SCENE_FRAGMENT_MAX_S:
                continue
            gu = cache.get(u.cluster, (None, 0.0, None, 0.5))[2]
            gp = cache.get(prev.cluster, (None, 0.0, None, 0.5))[2]
            if gu and gp and gu != gp:
                continue
            if not subs.continues(prev.text, u.text):
                continue
            if u.start - prev.end > 1.0:
                continue
            groups[u.cluster].remove(u)
            groups[prev.cluster].append(u)
            u.cluster = prev.cluster
            cont_moved += 1
    if cont_moved:
        for cid in list(groups):
            if not groups[cid]:
                groups.pop(cid)
                cache.pop(cid, None)
                total_dur.pop(cid, None)
        for cid in groups:
            cache[cid] = info(cid)
            total_dur[cid] = duration(groups[cid])
        if verbose:
            print(f"  → diarizáció: {cont_moved} félbemaradt mondat folytatása "
                  f"az előző beszélőhöz került")

    # ── 3. rövid töredékek csatolása ────────────────────────────────────
    for cid in sorted(list(groups), key=lambda c: duration(groups[c])):
        if cid not in groups or duration(groups[cid]) >= 2.5:
            continue
        cvec, _d, cg, _p = info(cid)
        if cvec is None:
            continue
        best: tuple[float, int] | None = None
        for other in groups:
            if other == cid:
                continue
            ovec, od, og, _op = cache.get(other, (None, 0.0, None, 0.5))
            if ovec is None or od < 3.0:
                continue
            if cg and og and cg != og:
                continue
            if co_occurs(groups[cid], groups[other]):
                continue
            s = cosine(cvec, ovec)
            if s >= 0.45 and (best is None or s > best[0]):
                best = (s, other)
        if best is not None:
            groups[best[1]] += groups.pop(cid)
            cache[best[1]] = info(best[1])
            cache.pop(cid, None)

    # ── 4. erős nem-ellentmondás javítása ───────────────────────────────
    moved = 0
    for cid in sorted(groups):
        _vec, _dur, cg, _p = info(cid)
        if cg is None:
            continue
        for u in list(groups[cid]):
            if u.p_female is None:
                continue
            want = ("F" if u.p_female >= 0.90
                    else "M" if u.p_female <= 0.10 else None)
            if want is None or want == cg:
                continue
            candidates = [(cosine(vec[id(u)], cache[o][0]), o)
                          for o in groups if o != cid and cache.get(o, (None,))[0] is not None
                          and cache[o][2] == want]
            if not candidates:
                continue
            s, target = max(candidates)
            if s < 0.30:
                continue
            groups[cid].remove(u)
            groups[target].append(u)
            u.cluster = target
            moved += 1
    if verbose and moved:
        print(f"  → diarizáció: {moved} erős nem-ellentmondás áthelyezve")

    # üres csoportok törlése, újraszámozás beszédidő szerint
    final = {cid: m for cid, m in groups.items() if m}
    order = sorted(final, key=lambda c: -duration(final[c]))
    remap = {old: new for new, old in enumerate(order)}
    # Az összevonásnál a tagok `cluster` mezője még a régi azonosítón maradhat;
    # ezért a végleges csoporttagság alapján írjuk vissza, nem a régi kulcsból.
    for old in order:
        for u in final[old]:
            u.cluster = remap[old]
    out: dict[int, Cluster] = {}
    for old in order:
        members = final[old]
        cl = Cluster(cid=remap[old], uids=[u.uid for u in members])
        cl.centroid = centroid_of(members)
        cl.dur = duration(members)
        cl.gender, _p = _turn_gender(members)
        cl.episodes = [u.ep for u in members]
        cl.gender_vote = float(sum((u.p_female - 0.5) * u.dur for u in members
                                   if u.p_female is not None))
        cl.n_long = sum(1 for u in members if u.dur >= 2.0)
        out[cl.cid] = cl
    if verbose:
        print(f"  → {len(out)} szereplő-klaszter a diarizációból "
              f"({sum(1 for c in out.values() if c.gender == 'F')} nő, "
              f"{sum(1 for c in out.values() if c.gender == 'M')} férfi, "
              f"{sum(1 for c in out.values() if c.gender is None)} bizonytalan)")
    return out


def gender_evidence(utts: list[Utterance], tuning: dict | None = None
                    ) -> dict[int, dict[str, float]]:
    """Klaszterenkénti **erős** nem-bizonyíték (beszédidő másodpercben).

    A megbízható egység nem az egy sorhoz tartozó pár pillanatnyi mérése
    (a rövid, kiabálós/zajos ablakoknál az osztályozó bizonytalan), hanem a
    teljes klaszter fölött összegzett beszédidő. Ezt használja az
    összevonási tiltás és a hangváltás-mérés is.
    """
    t = {**DEFAULTS, **(tuning or {})}
    out: dict[int, dict[str, float]] = {}
    for u in utts:
        if u.cluster is None or u.p_female is None:
            continue
        d = out.setdefault(u.cluster, {"F": 0.0, "M": 0.0})
        if u.p_female >= t["strong_f"]:
            d["F"] += u.dur
        elif u.p_female <= t["strong_m"]:
            d["M"] += u.dur
    return out


def violations(utts: list[Utterance], tuning: dict | None = None,
               cos_same: float = 0.55,
               min_evidence_s: float = 5.0,
               context: list[Utterance] | None = None) -> list[dict]:
    """Hangváltások, amelyeket a hang **nem** indokol.

    Ha két egymást követő megnyilatkozás hangja bizonyítottan ugyanaz
    (koszinusz >= küszöb), akkor a magyar hangnak is ugyanannak kell lennie.
    Ez a hallható hiba közvetlen mutatója; a QA-kapu ezt méri.

    Kizárás: ha a két klaszter **erősen ellentétes nemű** (mindkét oldalon
    legalább ``min_evidence_s`` erős mérés), akkor nem lehetnek ugyanaz a
    beszélő — ott a különböző magyar hang a helyes. Ez nem a kapu lazítása,
    hanem a mérés téves pozitívjainak kiszűrése (mért példa: női
    interjúalany vs. férfi kihallgató, koszinusz 0,74).
    """
    t = {**DEFAULTS, **(tuning or {})}
    thr = t.get("cos_same", cos_same)
    ev = gender_evidence(utts, tuning)

    def opposed(ca: int | None, cb: int | None) -> bool:
        if ca is None or cb is None:
            return False
        a, b = ev.get(ca, {}), ev.get(cb, {})
        return (a.get("F", 0) >= min_evidence_s and b.get("M", 0) >= min_evidence_s) \
            or (a.get("M", 0) >= min_evidence_s and b.get("F", 0) >= min_evidence_s)

    from .voices import BY_ID

    def siblings(a: Utterance, b: Utterance) -> bool:
        """Két forduló **szövegesen** külön beszélő (egy feliratsor „- A? - B.”
        vagy szomszédos zárójeles sorok). Ilyenkor a hangváltás helyes, akkor
        is, ha a rövid felkiáltások embeddingje véletlenül hasonló.
        """
        if a.ep != b.ep:
            return False
        if a.cues and b.cues and a.cues[0].index == b.cues[0].index:
            return True
        ta, tb = a.text.strip(), b.text.strip()
        return (ta.startswith("[") and ta.endswith("]")
                and tb.startswith("[") and tb.endswith("]")
                and b.start - a.end <= 0.8)

    by_ep: dict[str, list[Utterance]] = {}
    for u in utts:
        by_ep.setdefault(u.ep, []).append(u)
    # A klaszter-szintű mintaösszehasonlítás a hangváltás valódi
    # bizonyítéka: az egymáshoz hasonló rövid felkiáltások önmagukban nem
    # bizonyítják, hogy ugyanaz a színész beszél. Ha a sor a saját
    # klaszterének többi mintájához érdemben jobban hasonlít, mint a másik
    # klaszterhez, a két karakter külön beszélő — a magyar hangváltás helyes.
    # A szereplő azonossága évadszintű: a többi epizód ugyanazon
    # klaszterének mintái is bizonyítékot adnak. Az epizódon belüli szomszédos
    # párt változatlanul vizsgáljuk; csak a minta-összehasonlítás gazdagabb.
    evidence: list[Utterance] = list(utts)
    if context:
        seen = {id(u) for u in utts}
        evidence += [u for u in context if id(u) not in seen]
    raw: dict[int, np.ndarray] = {
        id(u): norm(np.asarray(u.emb, dtype=np.float32))
        for u in evidence if u.emb is not None}
    by_cluster: dict[int, list[Utterance]] = {}
    for u in evidence:
        if u.cluster is not None and id(u) in raw:
            by_cluster.setdefault(u.cluster, []).append(u)

    def exemplar_mean(u: Utterance, cid: int) -> float | None:
        vals = [float(raw[id(u)] @ raw[id(v)])
                for v in by_cluster.get(cid, [])
                if v is not u and id(v) in raw]
        return float(np.mean(vals)) if len(vals) >= 2 else None

    def cluster_centroid(cid: int, vecs: dict[int, np.ndarray]) -> np.ndarray | None:
        members = [v for v in by_cluster.get(cid, []) if id(v) in vecs]
        if len(members) < 2:
            return None
        vals = np.stack([vecs[id(v)] for v in members])
        ws = np.asarray([max(v.dur, 0.2) for v in members], dtype=np.float32)
        m = np.average(vals, axis=0, weights=ws)
        n = float(np.linalg.norm(m))
        return (m / n).astype(np.float32) if n > 0 else None

    def distinct_clusters(a: Utterance, b: Utterance, margin: float,
                          vecs: dict[int, np.ndarray],
                          centroid_max: float = 0.25) -> bool:
        if a.cluster is None or b.cluster is None or a.cluster == b.cluster:
            return False
        # Erős, klaszter-szintű bizonyíték: a két klaszter középpontja a
        # középpontosított térben távol van. Ez a rövid felkiáltások zajos
        # egyedi embeddingjénél megbízhatóbb (mért példa: két női szereplő
        # ugyanabban a jelenetben, „Végül.” vs. „Ez az ember.”).
        ca = cluster_centroid(a.cluster, vecs)
        cb = cluster_centroid(b.cluster, vecs)
        if ca is not None and cb is not None and cosine(ca, cb) <= centroid_max:
            return True
        a_own = exemplar_mean(a, a.cluster)
        a_other = exemplar_mean(a, b.cluster)
        b_own = exemplar_mean(b, b.cluster)
        b_other = exemplar_mean(b, a.cluster)
        if None in (a_own, a_other, b_own, b_other):
            return False
        return max(a_own - a_other, b_own - b_other) >= margin

    def distinct_centered(a: Utterance, b: Utterance, margin: float,
                          vecs: dict[int, np.ndarray]) -> bool:
        """A pár két beszélőjét a **klaszter-hozzáférés** bizonyítja-e?

        A pár koszinusza önmagában kevés: két hasonló hangú, különböző
        színész is 0,88-ra hasonlíthat (mért eset, E04:0124/0125). A
        megbízhatóbb teszt ugyanabban a (középpontosított) térben: az adott
        sor a saját klaszterének tagjaihoz jobban hasonlít-e, mint az összes
        többiekhez. Ha mindkét félre igaz, két külön beszélő — a hangváltás
        helyes, nem hiba. Külön a régi, nyers terekből számoló teszthez képest
        itt **     egyetlen** saját minta is elfogadható, mert nem egy minta értékét,
        hanem két átlag különbségét mérjük.
        """
        if a.cluster is None or b.cluster is None or a.cluster == b.cluster:
            return False
        if id(a) not in vecs or id(b) not in vecs:
            return False

        def own_minus_other(u: Utterance) -> float | None:
            own = [cosine(vecs[id(u)], vecs[id(v)])
                   for v in by_cluster.get(u.cluster, [])
                   if v is not u and id(v) in vecs]
            other = [cosine(vecs[id(u)], vecs[id(v)])
                     for v in evidence
                     if v.cluster is not None and v.cluster != u.cluster
                     and id(v) in vecs]
            if not own or not other:
                return None
            return float(np.mean(own)) - float(np.mean(other))

        da, db = own_minus_other(a), own_minus_other(b)
        if da is None or db is None:
            return False
        return min(da, db) >= margin

    out: list[dict] = []
    for ep, seq in by_ep.items():
        seq.sort(key=lambda u: u.start)
        # A hangváltás-mérés is középpontosított: a nyers CAM++ koszinusz
        # azonos nemű, de különböző beszélőknél is 0,55 fölött lehet, ezért
        # hamis „ugyanaz a hang” párokat jelentene a teljes évadban.
        epvec = _centered(seq)
        for a, b in zip(seq, seq[1:]):
            if a.emb is None or b.emb is None:
                continue
            if id(a) not in epvec or id(b) not in epvec:
                continue
            if siblings(a, b):
                continue          # a szöveg két beszélőt jelöl: helyes váltás
            cos_ab = cosine(epvec[id(a)], epvec[id(b)])
            # Szoros időbeli szomszédság (ugyanaz a párbeszéd) esetén a
            # rövid sorok embeddingje zajos, és két különböző szereplő is
            # 0,55–0,78 közé eshet — ilyenkor a „más hang” helyes. Az azonos
            # beszélőt csak a nagyon erős egyezés bizonyítja; a casting is
            # ezt a küszöböt használja a co-occuráló fordulók összevonására.
            gap_ab = b.start - a.end
            thr_here = (float(t.get("cos_same_strong", 0.80))
                        if gap_ab <= float(t.get("co_occur_gap", 8.0)) else thr)
            if cos_ab < thr_here:
                continue
            if a.voice is not None and a.voice == b.voice:
                continue          # ugyanaz a magyar hang: rendben
            va, vb = BY_ID.get(a.voice or ""), BY_ID.get(b.voice or "")
            if va is not None and vb is not None and va.gender != vb.gender:
                continue          # férfi↔nő: két különböző szereplő, helyes váltás
            if opposed(a.cluster, b.cluster):
                continue          # bizonyítottan két különböző beszélő
            if distinct_clusters(a, b, float(t.get("switch_cluster_margin", 0.05)),
                                 epvec,
                                 float(t.get("switch_centroid_max", 0.25))):
                continue          # a klaszter-minták külön beszélőt bizonyítanak
            if distinct_centered(a, b,
                                 float(t.get("switch_own_margin", 0.05)),
                                 epvec):
                continue          # mindkettő a saját klaszteréhez illik: két beszélő
            out.append({"ep": ep, "a": a.uid, "b": b.uid,
                        "cos": round(cos_ab, 3),
                        "a_voice": a.voice, "b_voice": b.voice,
                        "a_cluster": a.cluster, "b_cluster": b.cluster,
                        "gap": round(b.start - a.end, 3),
                        "a_text": a.text[:60], "b_text": b.text[:60]})
    return out


# ── Kalibrált beszélő-verifikáció (AS-Norm + döntési sáv) ─────────────────
#
# A nyers koszinusz egyetlen küszöbe nem működik: ugyanaz a színész 0,82-nél
# is szétválik, más színészek 0,60-nál összemosódnak (saját mérésen).
# A verifikátor cohort-alapú AS-Norm-mal kalibrál, és same/different/uncertain
# döntést ad. A döntési modell (Jev/Laya/Kev) csak az uncertain sávban
# (0,35–0,65) szól bele; a nyers hangot ez a modell dolgozza fel.
#
# A cohort az összes többi klaszter centroidja (imposztorok). Ha nincs elég
# cohort, nyers koszinuszra esünk vissza (visszafelé kompatibilis).


def _as_norm_score(a: np.ndarray, b: np.ndarray,
                   cohort: list[np.ndarray] | None) -> float:
    """Adaptive Score Normalization: z-score of cos(a,b) against cohort."""
    raw = float(cosine(a, b))
    if not cohort or len(cohort) < 3:
        return raw
    sa = np.asarray([float(cosine(a, c)) for c in cohort], dtype=np.float64)
    sb = np.asarray([float(cosine(b, c)) for c in cohort], dtype=np.float64)
    mu_a, sd_a = float(sa.mean()), float(sa.std()) + 1e-6
    mu_b, sd_b = float(sb.mean()), float(sb.std()) + 1e-6
    z_a = (raw - mu_a) / sd_a
    z_b = (raw - mu_b) / sd_b
    return float((z_a + z_b) / 2.0)


def verify_pair(a: np.ndarray, b: np.ndarray,
                cohort: list[np.ndarray] | None = None,
                tuning: dict | None = None) -> dict:
    """Kalibrált beszélő-verifikáció két embeddingre (centroidra).

    Visszaad: {'raw': nyers koszinusz, 'as_norm': normalizált z-score,
               'prob': kalibrált valószínűség (0..1),
               'decision': 'same'/'different'/'uncertain'}
    """
    t = {**DEFAULTS, **(tuning or {})}
    thr_same = float(t.get("verifier_same", 0.65))
    thr_diff = float(t.get("verifier_diff", 0.35))
    temp = float(t.get("verifier_temp", 2.0))
    cohort_min = int(t.get("verifier_cohort_min", 3))
    raw = float(cosine(a, b))
    if not cohort or len(cohort) < cohort_min:
        # nincs cohort: nem kalibrálható → uncertain (a veto nem lép életbe,
        # az eredeti küszöbök maradnak érvényben; visszafelé kompatibilis)
        return {"raw": raw, "as_norm": raw, "prob": 0.5,
                "decision": "uncertain"}
    else:
        z = _as_norm_score(a, b, cohort)
        prob = float(1.0 / (1.0 + np.exp(-z / max(temp, 1e-6))))
    if prob >= thr_same:
        decision = "same"
    elif prob <= thr_diff:
        decision = "different"
    else:
        decision = "uncertain"
    return {"raw": raw, "as_norm": z, "prob": prob, "decision": decision}


def verify_clusters(a_members: list, b_members: list,
                    vecs: dict, cohort: list[np.ndarray] | None = None,
                    tuning: dict | None = None) -> dict:
    """Klaszter-szintű verifikáció: az adott klaszterek mintáinak
    átlag-embeddingjére futtatja a verify_pair-t.

    a_members, b_members: Utterance-listák (a klaszter tagjai).
    vecs: id(u) -> centered embedding (mint _centered() ad).
    """
    def _centroid(members):
        vs = [vecs[id(u)] for u in members if id(u) in vecs]
        ws = [u.dur for u in members if id(u) in vecs]
        if not vs:
            return None
        v = np.average(np.stack(vs), axis=0,
                       weights=np.asarray(ws, dtype=np.float32))
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else None

    ca = _centroid(a_members)
    cb = _centroid(b_members)
    if ca is None or cb is None:
        return {"raw": 0.0, "as_norm": 0.0, "prob": 0.5,
                "decision": "uncertain"}
    return verify_pair(ca, cb, cohort=cohort, tuning=tuning)


def cluster_verdict_summary(utts, tuning=None) -> dict:
    """Összefoglaló a kalibrált verifikátor alkalmazhatóságáról:
    hány pár 'same'/'different'/'uncertain' a szomszédos megnyilatkozásokon.
    Diagnosztikai segéd; nem módosít adatot.
    """
    t = {**DEFAULTS, **(tuning or {})}
    vecs = _centered(utts)
    by_ep = {}
    for u in utts:
        by_ep.setdefault(u.ep, []).append(u)
    counts = {"same": 0, "different": 0, "uncertain": 0}
    cohort = None
    # cohort: az összes utterance centered embeddingje
    all_vecs = [vecs[id(u)] for u in utts if id(u) in vecs]
    for ep, seq in by_ep.items():
        seq.sort(key=lambda u: u.start)
        epvec = _centered(seq)
        for a, b in zip(seq, seq[1:]):
            if a.emb is None or b.emb is None:
                continue
            if id(a) not in epvec or id(b) not in epvec:
                continue
            ca = epvec[id(a)]
            cb = epvec[id(b)]
            # cohort: imposztorok — más epizódok utterance-jei; ha nincs
            # (film, egyetlen epizód), akkor a jelenettől távoli sorok
            imp = [vecs[id(u)] for u in utts
                   if u.ep != ep and id(u) in vecs]
            if len(imp) < 3:
                imp = [vecs[id(u)] for u in utts
                       if id(u) in vecs and u is not a and u is not b
                       and abs(u.start - a.start) > 30.0]
            v = verify_pair(ca, cb, cohort=imp if len(imp) >= 3 else None,
                            tuning=t)
            counts[v["decision"]] += 1
    return counts
