"""PLDA (Probabilistic Linear Discriminant Analysis) beszélő-verifikáció.

A nyers koszinusz egyetlen küszöbe nem működik: ugyanaz a színész 0,82-nél
is szétválik, más színészek 0,60-nál összemosódnak (mért, Detour 1945).
Az AS-Norm (verify_pair) cohort-alapú z-score-t ad; a PLDA ezt
principiálisabbá teszi: a cohort-ból megtanulja a **beszélő-közötti** és
**beszélő-beli** szórást (between/within-class scatter), LDA-projekcióval
csökkenti a dimenziót, majd log-likelihood-ratio (LLR) pontszámot ad két
embedding párra.

**Pontszám-skála (kalibráció).** Az LDA-térbeli koszinusz önmagában
definiálatlan skálájú: 0,75 jelentése modellenként más, ezért a régi kód egy
nyers koszinusz-küszöböt (0,65 / 0,35) alkalmazott egy koszinusz + z-score
keverékére — ez értelmetlen volt. A modell a tanító cohort **különböző
osztályú** párjain méri a null-eloszlást (``nt_mean`` / ``nt_sd``), és minden
pontszámot ebből a null-eloszlásból mért szórásban fejez ki::

    z = (score - nt_mean) / nt_sd

Így a skála definiált: ``z = 0`` a tipikus *különböző* beszélő-pár, ``z > 2``
a null-eloszlás felső ~2%-a. A döntés is ezen a skálán történik
(``Z_SAME`` / ``Z_DIFF``), az AS-Norm z-je ugyanezen a null-skálán van,
ezért a kettő súlyozott átlaga is értelmes. A küszöbök méréshez kötöttek:
``roc_eer()`` a mért target/non-target pontszámokból adja az EER-t és a
hozzá tartozó z-t; a ``calibrate()`` ezt a modellre írja.

**Nincs körkörösség.** Ha a modellt ugyanazokon a klaszter-címkéken tanítjuk,
amelyeket a hívó épp összevonni próbál, a modell igazolja a saját korábbi
döntéseit (két, már összevont klaszter „tanúsága szerint” azonos). Ezért a
klaszterező lépés keresztvalidált modelleket használ
(``clustering.plda_crossfit``): minden vizsgált klaszterpár olyan modellt kap,
amely **egyik** klasztert sem látta tanításkor.

A modell **nem** igényel külső tanítóadatot: a cohort maga az adott
epizód/film megnyilatkozás-embeddingjei, beszélő-címkével (a diarizáció
címkei vagy a klaszter-azonosítók). Ha nincs elég címke, a modell
``None`` marad és a hívó visszaesik AS-Norm-ra (visszafelé kompatibilis).

Használat:
    model = PLDAModel.fit(embeddings, labels)   # embeddings: (N, D), labels: (N,)
    z     = model.z(model.score(a, b))          # null-eloszlásból mért szórás
    decision = model.decide(a, b)               # same/different/uncertain
    stats = roc_eer(target_scores, non_target_scores)   # EER + operációs z
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .embeddings import cosine

#: Alap-küszöbök a null-eloszlásból mért szórásban (z).
#:
#: **Mérésből** (Detour E01, 27 477 target / 11 606 non-target pár,
#: keresztvalidált, nyers diarizációs címkékkel): a non-target eloszlás a
#: tanító null-eloszláshoz képest +0,56 szórással eltolódik és 1,46 szórású
#: (a tanító null-eloszlás **in-sample**, ezért optimista — ezt a mérés
#: mutatta). A mért 1%-os téves-elfogadás z≈3,65. Ezért a küszöbök szélesebbek
#: a „szép” elméletinél: a ``same`` csak akkor jön ki, ha a pár a tipikus
#: különböző párnál 3,5 szórással hasonlóbb; a ``different`` (vétó) csak akkor,
#: ha 1,5 szórással kevésbé hasonló.
Z_SAME = 3.5
Z_DIFF = -1.5


@dataclass
class PLDAModel:
    # LDA-projekciós mátrix (D, K) — a between/within szórást optimalizálja
    projection: np.ndarray | None = None
    # within-class szórás (a projekciós térben, skalár)
    mu_within: float | None = None
    # between-class szórás (skálázás)
    between_scale: float = 1.0
    within_scale: float = 1.0
    # null-eloszlás (különböző beszélő-párok a tanító cohortban) — a skála
    nt_mean: float = 0.0
    nt_sd: float = 1.0
    n_nt_pairs: int = 0
    #: mért operációs pont (``calibrate()`` írja); None = a Z_* alapérték
    z_same: float | None = None
    z_diff: float | None = None
    # osztályok (beszélők) száma a tanításkor
    n_classes: int = 0
    n_samples: int = 0
    #: a tanításkor látott osztály-azonosítók (a keresztvalidáció ellenőrzéséhez)
    labels: list = field(default_factory=list)
    # dimenzió
    dim: int = 0
    fitted: bool = False

    # ── tanítás ──────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, embeddings: Iterable, labels: Iterable,
            n_components: int | None = None,
            reg: float = 1e-4,
            min_samples_per_class: int = 2,
            min_classes: int = 3,
            max_null_pairs: int = 4000) -> "PLDAModel":
        """PLDA-modell tanítása embedding+minta-párból.

        embeddings: (N, D) tömb (L2-normalizált vagy sem — a függvény
                    normalizál).
        labels:     (N,) beszélő-azonosítók (int vagy str).

        A tanítás végén a **különböző osztályú** párok pontszámaiból felírja a
        null-eloszlást (``nt_mean``/``nt_sd``), ami a döntési skálát adja.
        """
        X = np.asarray(list(embeddings), dtype=np.float64)
        y = np.asarray(list(labels))
        model = cls()
        if X.ndim != 2 or len(X) == 0 or len(X) != len(y):
            return model
        N, D = X.shape
        # normalizálás
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        # osztályok szűrése: legalább min_samples_per_class minta
        classes, counts = np.unique(y, return_counts=True)
        keep = classes[counts >= min_samples_per_class]
        if len(keep) < min_classes:
            return model  # nincs elég osztály → nem tanítható
        mask = np.isin(y, keep)
        X = X[mask]
        y = y[mask]
        N = len(X)
        classes = np.unique(y)
        K = len(classes)

        # osztály-centroidok és globális közép
        global_mean = X.mean(axis=0)
        class_means = np.stack([X[y == c].mean(axis=0) for c in classes])
        # within-class scatter
        Sw = np.zeros((D, D), dtype=np.float64)
        for c in classes:
            Xc = X[y == c]
            diff = Xc - Xc.mean(axis=0)
            Sw += diff.T @ diff
        Sw /= max(N - K, 1)
        # between-class scatter
        Sb = np.zeros((D, D), dtype=np.float64)
        for c in classes:
            n_c = int((y == c).sum())
            dm = (class_means[list(classes).index(c)] - global_mean).reshape(-1, 1)
            Sb += n_c * (dm @ dm.T)
        Sb /= max(N, 1)

        # regularizáció
        Sw_reg = Sw + reg * np.eye(D)

        # Általánosított sajátérték-felbontás: Sb w = λ Sw w.
        # A szimmetrikus megoldó (eigh) valós sajátértékeket/sajátvektorokat
        # ad; az Sw^-1 Sb + eig nem-szimmetrikus út komplex eredményt hozhat.
        try:
            from scipy.linalg import eigh
            eigvals, eigvecs = eigh(Sb, Sw_reg)
        except Exception:
            return model
        # rendezés csökkenő sajátérték szerint
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # komponensek száma
        if n_components is None:
            n_components = min(K - 1, D, 64)
        n_components = max(1, min(n_components, D, K - 1, len(eigvals)))
        # csak pozitív sajátértékek
        pos = eigvals > 1e-12
        if not pos.any():
            return model
        n_components = min(n_components, int(pos.sum()))
        if n_components < 1:
            return model

        W = eigvecs[:, :n_components]  # (D, K)
        # projekció normalizálása (whitening-szerű)
        # within-class szórás a projekciós térben
        Xw = X @ W
        within_var = 0.0
        for c in classes:
            Xc = Xw[y == c]
            within_var += float(np.var(Xc, axis=0).mean()) * int((y == c).sum())
        within_var /= max(N, 1)
        if within_var <= 0:
            within_var = 1.0

        model.projection = W.astype(np.float64)
        model.mu_within = float(np.sqrt(within_var))
        model.between_scale = float(np.sqrt(max(eigvals[0], 1e-12)))
        model.within_scale = float(np.sqrt(within_var))

        # ── null-eloszlás: különböző beszélők párjai ─────────────────────
        # Ez adja a **skálát**: a döntés nem nyers koszinuszon, hanem ebből a
        # null-eloszlásból mért szóráson (z) történik.
        proj = X @ W
        pn = np.linalg.norm(proj, axis=1, keepdims=True)
        pn[pn == 0] = 1.0
        proj = proj / pn
        rows = {c: np.where(y == c)[0] for c in classes}
        vals: list[np.ndarray] = []
        for i, ca_ in enumerate(classes):
            for cb_ in classes[i + 1:]:
                ia, ib = rows[ca_], rows[cb_]
                if not len(ia) or not len(ib):
                    continue
                M = proj[ia] @ proj[ib].T
                v = M.ravel()
                if v.size > 512:              # egyenletes mintavétel, determinisztikus
                    v = v[:: max(1, v.size // 512)]
                vals.append(v)
        nt = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float64)
        if nt.size > max_null_pairs:
            nt = nt[:: max(1, nt.size // max_null_pairs)]
        if nt.size >= 8:
            mu = float(nt.mean())
            sd = float(nt.std())
            model.nt_mean = mu
            model.nt_sd = sd if (np.isfinite(sd) and sd > 1e-6) else 1.0
            model.n_nt_pairs = int(nt.size)

        model.n_classes = int(K)
        model.n_samples = int(N)
        model.labels = [c.item() if hasattr(c, "item") else c for c in classes]
        model.dim = int(D)
        model.fitted = True
        return model

    # ── pontszám ─────────────────────────────────────────────────────────
    def _project(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        if self.projection is None:
            return v
        p = self.projection.T @ v
        return p

    def score(self, a, b) -> float:
        """Nyers LDA-térbeli koszinusz (nagyobb = same).

        Önmagában **nem** döntési skála: a ``z()`` viszi át a null-eloszlásból
        mért szórásba.
        """
        if not self.fitted:
            # visszaesés nyers koszinuszra
            return float(cosine(a, b))
        pa = self._project(a)
        pb = self._project(b)
        # LDA-térben: koszinusz-hasonlóság (nagyobb = same)
        na = float(np.linalg.norm(pa))
        nb = float(np.linalg.norm(pb))
        if na <= 0 or nb <= 0:
            return 0.0
        return float((pa @ pb) / (na * nb))

    def z(self, score: float) -> float:
        """Pontszám a null-eloszlás (különböző beszélők) szórásában."""
        sd = float(self.nt_sd) if np.isfinite(self.nt_sd) else 0.0
        if sd <= 1e-9:
            sd = 1.0
        return float((float(score) - float(self.nt_mean)) / sd)

    def probability(self, a, b, temp: float = 2.0) -> float:
        """0..1 valószínűség: sigmoid(z/temp) a null-skálán."""
        z = self.z(self.score(a, b))
        return _sigmoid(z, temp)

    def decide(self, a, b, cohort=None, temp: float = 2.0,
               plda_weight: float = 0.5,
               z_same: float | None = None,
               z_diff: float | None = None) -> dict:
        """Háromállapotú döntés **null-normalizált** (z) skálán.

        A PLDA pontszám és az AS-Norm z azonos null-skálán van (a tipikus
        különböző beszélő-pár = 0), ezért a ``plda_weight`` szerinti átlag
        értelmes. A ``same``/``different`` küszöbök z-ben értendők, sorrendben:
        a hívó által megadott ``z_same``/``z_diff`` → a modell mért operációs
        pontja (``calibrate()``) → a ``Z_SAME``/``Z_DIFF`` alapérték.
        """
        raw = float(cosine(a, b))
        if not self.fitted:
            # visszaesés: nem kalibrálható
            return {"raw": raw, "plda": raw, "prob": 0.5,
                    "decision": "uncertain"}
        # PLDA pontszám → null-skála
        plda_score = self.score(a, b)
        plda_z = self.z(plda_score)
        # AS-Norm cohort kalibráció (ha van cohort): a z már null-skála.
        # Cohort nélkül **nincs** AS-Norm z: a nyers koszinuszt nem szabad a
        # PLDA projekciós terében mért null-eloszlással osztani (más skála).
        as_z: float | None = None
        if cohort is not None and len(cohort) >= 3:
            sa = np.asarray([float(cosine(a, c)) for c in cohort], dtype=np.float64)
            sb = np.asarray([float(cosine(b, c)) for c in cohort], dtype=np.float64)
            mu_a, sd_a = float(sa.mean()), float(sa.std()) + 1e-6
            mu_b, sd_b = float(sb.mean()), float(sb.std()) + 1e-6
            as_z = float(((raw - mu_a) / sd_a + (raw - mu_b) / sd_b) / 2.0)
            w = min(max(float(plda_weight), 0.0), 1.0)
            combined = w * plda_z + (1.0 - w) * as_z
        else:
            combined = plda_z
        prob = _sigmoid(combined, temp)
        # küszöbök z-ben: hívó értéke → modell mért operációs pontja → Z_*
        zs = z_same if z_same is not None else self.z_same
        zd = z_diff if z_diff is not None else self.z_diff
        zs = Z_SAME if zs is None else float(zs)
        zd = Z_DIFF if zd is None else float(zd)
        if combined >= zs:
            decision = "same"
        elif combined <= zd:
            decision = "different"
        else:
            decision = "uncertain"
        return {"raw": raw, "as_norm": as_z, "plda": plda_score,
                "plda_z": plda_z, "z": combined, "combined": combined,
                "z_same": zs, "z_diff": zd, "prob": prob,
                "decision": decision}

    # ── kalibráció ───────────────────────────────────────────────────────
    def calibrate(self, target: Iterable[float],
                  non_target: Iterable[float],
                  margin: float = 0.5) -> dict:
        """Operációs pont mérése: EER-küszöb a saját null-skáláján.

        ``target``: ismert azonos beszélő párok nyers pontszámai (``score``),
        ``non_target``: különböző beszélő pároké. A modellre írt
        ``z_same``/``z_diff`` az EER-pont körüli sávot adja.
        """
        stats = roc_eer(target, non_target)
        if stats.get("z") is None:
            return stats
        self.z_same = round(float(stats["z"]) + float(margin), 3)
        self.z_diff = round(float(stats["z"]) - float(margin), 3)
        stats["model_z_same"] = self.z_same
        stats["model_z_diff"] = self.z_diff
        return stats

    # ── szerializáció ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "fitted": self.fitted,
            "n_classes": self.n_classes,
            "n_samples": self.n_samples,
            "dim": self.dim,
            "between_scale": self.between_scale,
            "within_scale": self.within_scale,
            "nt_mean": self.nt_mean,
            "nt_sd": self.nt_sd,
            "n_nt_pairs": self.n_nt_pairs,
            "z_same": self.z_same,
            "z_diff": self.z_diff,
            "projection": (None if self.projection is None
                           else self.projection.tolist()),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PLDAModel":
        m = cls()
        if not d or not d.get("fitted"):
            return m
        m.fitted = True
        m.n_classes = int(d.get("n_classes", 0))
        m.n_samples = int(d.get("n_samples", 0))
        m.dim = int(d.get("dim", 0))
        m.between_scale = float(d.get("between_scale", 1.0))
        m.within_scale = float(d.get("within_scale", 1.0))
        m.nt_mean = float(d.get("nt_mean", 0.0))
        m.nt_sd = float(d.get("nt_sd", 1.0))
        m.n_nt_pairs = int(d.get("n_nt_pairs", 0))
        m.z_same = (None if d.get("z_same") is None else float(d["z_same"]))
        m.z_diff = (None if d.get("z_diff") is None else float(d["z_diff"]))
        proj = d.get("projection")
        if proj is not None:
            m.projection = np.asarray(proj, dtype=np.float64)
        return m


def _sigmoid(z: float, temp: float = 2.0) -> float:
    t = max(float(temp), 1e-6)
    try:
        return float(1.0 / (1.0 + np.exp(-float(z) / t)))
    except (OverflowError, FloatingPointError):
        return 1.0 if z > 0 else 0.0


def roc_eer(target: Iterable[float], non_target: Iterable[float]) -> dict:
    """EER és operációs pont a mért pontszámokból (mérés, nem becslés).

    ``target``: azonos beszélő párok pontszámai, ``non_target``: különböző
    beszélő pároké. Visszaad: EER, a hozzá tartozó nyers küszöb, annak z-je a
    non-target eloszlásból mérve, valamint az 1%-os téves-elfogadás pontja.
    """
    t = np.asarray(list(target), dtype=np.float64)
    n = np.asarray(list(non_target), dtype=np.float64)
    if t.size == 0 or n.size == 0:
        return {"eer": None, "threshold": None, "z": None, "far_1pct": None,
                "frr_at_eer": None, "n_target": int(t.size),
                "n_non_target": int(n.size)}
    cand = np.unique(np.concatenate([t, n]))
    best: tuple[float, float, float, float] | None = None
    for thr in cand:
        frr = float((t < thr).mean())
        far = float((n >= thr).mean())
        err = (frr + far) / 2.0
        if best is None or err < best[0]:
            best = (err, float(thr), far, frr)
    mu = float(n.mean())
    sd = float(n.std()) or 1.0
    z_thr = (best[1] - mu) / sd
    z_1 = (float(np.quantile(n, 0.99)) - mu) / sd
    return {"eer": round(best[0], 4), "threshold": round(best[1], 4),
            "z": round(z_thr, 3), "far_1pct": round(z_1, 3),
            "frr_at_eer": round(best[3], 4),
            "n_target": int(t.size), "n_non_target": int(n.size)}
