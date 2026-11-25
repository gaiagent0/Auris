"""Stabil, epizódokon átívelő karakterazonosság a diarizált fordulókhoz.

A diarizálás ``SPEAKER_00…`` címkéi **csak az adott futásra** érvényesek: a
klaszterező minden epizódban nulláról számoz. A szinkronhoz viszont arra van
szükség, hogy ugyanaz a színész **minden epizódban ugyanazt a ``C###``
rekordot** kapja, függetlenül attól, hogy éppen hányadik beszélőként szólalt
meg.

**Miért példány-szintű az összevetés?** A beszélő-embeddingek átlaga (centroid)
a gömb középpontja felé húz, ezért két különböző színész centroidja is 0,8–0,9
koszinuszt ad — mértük: a cast 46 karaktere közt a centroid-koszinusz átlaga
0,27, de az azonos klaszterbe tartozó beszélők centroidjai egymással 0,85–0,91
között vannak. A megnyilatkozás-szintű összevetés viszont jól szétválik
(azonos beszélő ≈ 0,5–0,7, idegen ≈ 0,28). Ezért egy beszélő lenyomatát a
karakter **egyedi mintáihoz** hasonlítjuk, és a találat mellé odaírjuk a
mintaszámot — kevés mintából nem lesz magabiztos döntés.

A galéria (``work/identity-gallery.json``) a tekintély: a megerősített
karakterek lenyomatait és mintáit őrzi, így egy újraklaszterezés nem írja felül
a korábbi döntést, és a következő epizód ugyanazt a ``C###``-t kapja.

Döntés soha nem automatikus átcímkézés: a találat ``linked`` / ``weak`` /
``ambiguous`` / ``new`` állapotot és indoklást kap, emberi jóváhagyás nélkül
``review_required`` marad.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audio
from .embeddings import cosine, norm

PIPELINE = "dubstudio.identity v1"

#: példány-szintű koszinusz-küszöbök (mért eloszlásra kalibrálva)
DEFAULTS = {
    "link_cos": 0.55,     # ennyi fölött vállalható azonos karakter
    "weak_cos": 0.42,     # ez alatt új karakternek számít
    "margin": 0.05,       # a legjobb és a második *megalapozott* jelölt között
    "min_exemplars": 3,   # ennyi minta alatt nincs automatikus azonosítás
    "min_linked_s": 2.0,  # ennyi beszédidő alatt nincs automatikus azonosítás
    "min_turn_s": 0.7,    # rövidebb fordulóból nincs lenyomat
    "max_turn_s": 4.0,    # a hosszú fordulóból ennyit használunk
    "max_samples": 6,     # ennyi minta kerül a riportba / galériába
}

#: a galériában karakterenként ennyi minta marad meg
GALLERY_SAMPLES_MAX = 24
#: összehasonlításkor ennyi mintát használunk karakterenként
COMPARE_SAMPLES_MAX = 40

DEFAULT_CENTROID_DIM = 512


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def thresholds(tuning: dict | None = None) -> dict:
    t = {**DEFAULTS, **((tuning or {}).get("identity") or {})}
    for key in ("link_cos", "weak_cos", "margin", "min_turn_s", "max_turn_s",
                "min_linked_s"):
        try:
            t[key] = float(t[key])
        except (TypeError, ValueError):
            t[key] = float(DEFAULTS[key])
    for key in ("min_exemplars", "max_samples"):
        try:
            t[key] = int(t[key])
        except (TypeError, ValueError):
            t[key] = int(DEFAULTS[key])
    return t


# ── galéria ──────────────────────────────────────────────────────────────
def load_gallery(path: str | Path) -> dict:
    p = Path(path)
    empty = {"pipeline": PIPELINE, "characters": {}, "aliases": {}}
    if not p.is_file():
        return empty
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty
    data.setdefault("pipeline", PIPELINE)
    data.setdefault("characters", {})
    data.setdefault("aliases", {})
    return data


def save_gallery(path: str | Path, gallery: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps({"pipeline": PIPELINE, **gallery},
                                    default=lambda o: list(o)))
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _as_matrix(rows) -> np.ndarray | None:
    if rows is None:
        return None
    try:
        m = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if m.ndim == 1:
        m = m[None, :]
    if m.ndim != 2 or m.shape[1] < 8:
        return None
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def char_records(cast_path: str | Path, gallery_path: str | Path) -> dict[str, dict]:
    """A cast és a galéria egyesített képe (a galéria nyer, ha van adata)."""
    records: dict[str, dict] = {}
    cp = Path(cast_path)
    if cp.is_file():
        try:
            cast = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cast = {}
        if not isinstance(cast, dict):
            cast = {}
        raw_chars = cast.get("characters") or {}
        # a registry listát ír (char_id mezővel), a galéria szótárt használ
        pairs = ([(row.get("char_id"), row) for row in raw_chars]
                 if isinstance(raw_chars, list) else list(raw_chars.items()))
        for cid, row in pairs:
            if not cid or not isinstance(row, dict):
                continue
            records[str(cid)] = {
                "char_id": str(cid),
                "gender": row.get("gender"),
                "voice": row.get("voice"),
                "pitch": row.get("pitch"),
                "duration": float(row.get("duration") or 0.0),
                "episodes": list(row.get("episodes") or []),
                "members": dict(row.get("members") or {}),
                "centroid": list(row.get("centroid") or []),
                "samples": [],
                "origin": "cast",
            }
    for cid, row in (load_gallery(gallery_path).get("characters") or {}).items():
        base = records.get(str(cid), {})
        records[str(cid)] = {
            **base,
            "char_id": str(cid),
            "name": row.get("name", base.get("name")),
            "gender": row.get("gender", base.get("gender")),
            "voice": row.get("voice", base.get("voice")),
            "pitch": base.get("pitch"),
            "duration": float(row.get("seconds") or base.get("duration") or 0.0),
            "episodes": sorted(set(base.get("episodes") or [])
                               | set(row.get("episodes") or [])),
            "members": {**base.get("members", {}), **(row.get("members") or {})},
            "centroid": list(row.get("centroid") or base.get("centroid") or []),
            "samples": list(row.get("samples") or []),
            "origin": "gallery" if not base else "gallery+cast",
        }
    return records


def next_char_id(records: dict) -> str:
    """A következő szabad ``C###`` azonosító (nem lyukas újraszámozás)."""
    used = set()
    for cid in records:
        digits = "".join(ch for ch in str(cid) if ch.isdigit())
        if digits:
            used.add(int(digits))
    return f"C{(max(used) + 1 if used else 1):03d}"


# ── lenyomatok fordulókból ───────────────────────────────────────────────
def prototypes_from_turns(turns: list[dict], wav_path: str | Path,
                          embedding_model: str | Path,
                          min_turn_s: float = DEFAULTS["min_turn_s"],
                          max_turn_s: float = DEFAULTS["max_turn_s"],
                          max_samples: int = DEFAULTS["max_samples"],
                          embedder=None) -> dict[str, dict]:
    """Beszélőnként egy L2-normalizált centroid + egyedi minták + bizonyíték.

    Csak a tényleges forduló-audióból mérünk (nem a teljes sávból), mert a
    beszélőváltás körüli zene/effekt elrontaná a lenyomatot.
    """
    from .embeddings import SpeakerEmbedder

    source, sr = audio.load_mono(wav_path, sr=audio.SR16)
    model = embedder or SpeakerEmbedder(embedding_model)
    per_speaker: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for turn in turns:
        a, b = float(turn["start"]), float(turn["end"])
        if b - a < min_turn_s:
            continue
        core = audio.speech_core(source, sr, a, b, min_s=min_turn_s, max_s=max_turn_s)
        if core is None:
            continue
        vec = model.embed(core)
        if vec is None:
            continue
        per_speaker[str(turn["speaker"])].append((vec, b - a))
        spans[str(turn["speaker"])].append((round(a, 3), round(b, 3)))
    out: dict[str, dict] = {}
    for speaker, items in per_speaker.items():
        weights = np.asarray([w for _, w in items], dtype=np.float32)
        stack = np.stack([v for v, _ in items])
        vec = norm((stack * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-6))
        # a leghosszabb fordulók mintái a legtisztábbak
        order = sorted(range(len(items)), key=lambda i: items[i][1], reverse=True)
        samples = [stack[i] for i in order[:max(1, int(max_samples))]]
        out[speaker] = {"vector": vec, "samples": samples,
                        "seconds": round(float(weights.sum()), 2),
                        "turns": len(items), "spans": spans[speaker]}
    return out


def prototypes_from_audit(audit: dict) -> dict[str, dict]:
    """A korábbi riportban eltárolt lenyomatok (újramérés nélkül)."""
    out: dict[str, dict] = {}
    for speaker, row in ((audit.get("speaker_identity") or {}).get("speakers") or {}).items():
        vec = row.get("vector")
        if not vec:
            continue
        samples = _as_matrix(row.get("samples"))
        out[str(speaker)] = {
            "vector": norm(np.asarray(vec, dtype=np.float32)),
            "samples": [] if samples is None else list(samples),
            "seconds": float(row.get("evidence_seconds") or 0.0),
            "turns": int(row.get("evidence_turns") or 0),
            "spans": row.get("spans") or [],
        }
    return out


# ── minták karakterenként ────────────────────────────────────────────────
def character_exemplars(project, records: dict) -> dict[str, np.ndarray]:
    """Karakterenként a beszélő-embeddingek, epizód-klaszter tagokból + galériából."""
    per_char: dict[str, list] = defaultdict(list)
    cache: dict[str, list] = {}
    for cid, row in records.items():
        samples = _as_matrix(row.get("samples"))
        if samples is not None:
            per_char[cid].extend(list(samples[:COMPARE_SAMPLES_MAX]))
        for ep, clusters in (row.get("members") or {}).items():
            ep = str(ep).upper()
            if not clusters:
                continue
            if ep not in cache:
                path = project.work / ep / "utterances.json"
                try:
                    cache[ep] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
                except (OSError, ValueError):
                    cache[ep] = []
            wanted = {str(c) for c in clusters}
            for utt in cache[ep]:
                if utt.get("emb") and str(utt.get("cluster")) in wanted:
                    per_char[cid].append(utt["emb"])
    out: dict[str, np.ndarray] = {}
    for cid, rows in per_char.items():
        m = _as_matrix(rows[:COMPARE_SAMPLES_MAX])
        if m is not None:
            out[cid] = m
    return out


def rank_candidates(vec, exemplars: dict[str, np.ndarray],
                    records: dict, top_k: int = 3) -> list[dict]:
    """Beszélő lenyomat vs karakter-minták: átlagos koszinusz + mintaszám."""
    v = np.asarray(vec, dtype=np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    scored = []
    for cid, matrix in exemplars.items():
        sims = matrix @ v
        scored.append({
            "char_id": str(cid),
            "cosine": round(float(sims.mean()), 4),
            "max_cosine": round(float(sims.max()), 4),
            "exemplars": int(sims.size),
            "gender": (records.get(str(cid)) or {}).get("gender"),
            "voice": (records.get(str(cid)) or {}).get("voice"),
            "episodes": (records.get(str(cid)) or {}).get("episodes") or [],
            "origin": (records.get(str(cid)) or {}).get("origin"),
        })
    scored.sort(key=lambda c: c["cosine"], reverse=True)
    return scored[:max(1, int(top_k))]


def decide(candidates: list[dict], t: dict | None = None,
           gender: str | None = None) -> tuple[str, str, str | None]:
    """``(status, reason, char_id)`` — soha nem dönt a felhasználó helyett.

    * ``linked``    — megalapozott, küszöb fölötti találat;
    * ``ambiguous`` — két megalapozott jelölt túl közel van egymáshoz;
    * ``weak``      — van hasonló, de kevés minta vagy a küszöb alatt;
    * ``new``       — nincs értékelhető hasonlóság (új szereplő lesz).
    """
    t = thresholds({"identity": t} if t else None)
    if not candidates:
        return "new", "nincs összehasonlítható karakterlenyomat", None
    min_n = t["min_exemplars"]
    qualified = [c for c in candidates if int(c.get("exemplars") or 0) >= min_n]
    best = candidates[0]
    if not qualified:
        n = int(best.get("exemplars") or 0)
        return ("weak", (f"egyik jelöltnek sincs elég mintája "
                         f"(max {n} < {min_n}); a legjobb {best['char_id']} "
                         f"cos={best['cosine']:.2f}"), None)
    top = qualified[0]
    cos = float(top["cosine"])
    if gender and top.get("gender") and top["gender"] != gender:
        return ("weak", (f"a legjobb megalapozott találat ({top['char_id']}) neme "
                         f"{top['gender']}, a mért {gender}"), None)
    if cos < t["weak_cos"]:
        return ("new", (f"a legjobb megalapozott hasonlóság {cos:.2f} < "
                         f"{t['weak_cos']:.2f}"), None)
    if cos < t["link_cos"]:
        return ("weak", (f"közepes hasonlóság {cos:.2f} ({top['char_id']}, "
                         f"{top['exemplars']} minta)"), None)
    second = qualified[1] if len(qualified) > 1 else None
    if second is not None and cos - float(second["cosine"]) < t["margin"]:
        return ("ambiguous", (f"{top['char_id']} és {second['char_id']} közt csak "
                              f"{cos - float(second['cosine']):.3f} a különbség"), None)
    return ("linked", (f"{top['char_id']} cos={cos:.2f} "
                       f"({top['exemplars']} minta)"), str(top["char_id"]))


# ── emberi megerősítés ───────────────────────────────────────────────────
def confirm(gallery: dict, episode: str, speaker: str, char_id: str | None,
            vec=None, samples=None, gender: str | None = None,
            voice: str | None = None, seconds: float = 0.0, note: str = "",
            name: str | None = None) -> dict:
    """Emberi döntés rögzítése: alias + (ha van minta) galéria-frissítés.

    ``char_id=None`` azt jelenti, hogy a beszélő **nem** kap karaktert (zörej,
    kórus, bemondó) — ezt is eltároljuk, hogy ne kérdezze újra.
    """
    episode = str(episode).upper()
    gallery.setdefault("characters", {})
    gallery.setdefault("aliases", {})
    gallery["aliases"].setdefault(episode, {})[str(speaker)] = char_id
    if char_id:
        row = gallery["characters"].setdefault(str(char_id), {
            "gender": gender, "voice": voice, "seconds": 0.0,
            "episodes": [], "centroid": [], "samples": [], "sources": []})
        row["episodes"] = sorted(set(row.get("episodes") or []) | {episode})
        if name:
            row["name"] = str(name)[:80]
        if gender:
            row["gender"] = gender
        if voice:
            row["voice"] = voice
        row["seconds"] = round(float(row.get("seconds") or 0.0)
                               + max(0.0, float(seconds)), 2)
        matrix = _as_matrix(samples)
        if matrix is not None:
            old = _as_matrix(row.get("samples"))
            if old is None or old.shape[1] != matrix.shape[1]:
                merged = matrix
            else:
                merged = np.vstack([old, matrix])
            # a legfrissebb minták maradnak meg, és a centroid is belőlük jön
            merged = merged[-GALLERY_SAMPLES_MAX:]
            row["samples"] = [[round(float(x), 5) for x in v] for v in merged]
            row["centroid"] = [round(float(x), 5) for x in norm(merged.mean(axis=0))]
        elif vec is not None:
            old = np.asarray(row.get("centroid") or [], dtype=np.float32)
            new = np.asarray(vec, dtype=np.float32)
            if old.size == new.size and float(old.any()):
                w_old = max(0.0, row["seconds"] - float(seconds))
                mix = (old * w_old + new * float(seconds)) / max(w_old + float(seconds), 1e-6)
                row["centroid"] = [round(float(v), 5) for v in norm(mix)]
            else:
                row["centroid"] = [round(float(v), 5) for v in norm(new)]
        row.setdefault("sources", []).append(
            {"episode": episode, "speaker": str(speaker), "confirmed": True,
             "seconds": round(float(seconds), 2), "at": _now(), "note": note[:200]})
    return gallery


# ── teljes lánc ──────────────────────────────────────────────────────────
def link_speakers(project, tag: str, turns: list[dict], wav_path: str | Path,
                  embedding_model: str | Path, start: float, duration: float,
                  embedder=None, write: bool = True,
                  prototypes: dict | None = None) -> dict:
    """Diarizált fordulók -> stabil ``C###`` azonosság (galéria + cast).

    ``prototypes`` átadható, ha a hívó már kiszámolta (egy beágyazási kör);
    ilyenkor a függvény nem nyúl a hanghoz.
    """
    tag = str(tag).upper()
    t = thresholds(project.tuning)
    gallery_path = project.work / "identity-gallery.json"
    gallery = load_gallery(gallery_path)
    records = char_records(project.cast_file, gallery_path)
    protos = prototypes if prototypes is not None else prototypes_from_turns(
        turns, wav_path, embedding_model, min_turn_s=t["min_turn_s"],
        max_turn_s=t["max_turn_s"], max_samples=t["max_samples"], embedder=embedder)
    aliases = (gallery.get("aliases") or {}).get(tag) or {}
    exemplars = character_exemplars(project, records)

    speakers = {}
    for speaker in sorted({str(x["speaker"]) for x in turns} | set(aliases)):
        proto = protos.get(speaker)
        manual = speaker in aliases
        alias = aliases.get(speaker)
        entry = {
            "speaker": speaker,
            "char_id": alias if manual else None,
            "status": "manual" if manual else "no_evidence",
            "reason": ("ember által rögzített azonosság" if manual and alias
                       else "ember szerint nincs karaktere" if manual
                       else "nincs elég hosszú forduló a lenyomathoz"),
            "evidence_seconds": 0.0,
            "evidence_turns": 0,
            "candidates": [],
            "review_required": not manual,
        }
        if proto is not None:
            entry["evidence_seconds"] = proto["seconds"]
            entry["evidence_turns"] = proto["turns"]
            entry["spans"] = proto["spans"]
            entry["vector"] = [round(float(v), 5) for v in proto["vector"]]
            samples = _as_matrix(proto.get("samples"))
            if samples is not None:
                entry["samples"] = [[round(float(x), 5) for x in v]
                                    for v in samples[:t["max_samples"]]]
            cands = rank_candidates(proto["vector"], exemplars, records)
            entry["candidates"] = cands
            if not manual:
                status, reason, match = decide(cands, t)
                if status == "linked" and float(proto["seconds"]) < t["min_linked_s"]:
                    status, match = "weak", None
                    reason = (f"csak {proto['seconds']:.1f} s beszédidő a biztos "
                              f"azonosításhoz ({t['min_linked_s']:.0f} s kell); ezért "
                              f"a jelölt {cands[0]['char_id']} csak javaslat")
                entry["status"], entry["reason"] = status, reason
                entry["char_id"] = match
        if entry["char_id"] and entry["char_id"] in records:
            entry["gender"] = records[entry["char_id"]].get("gender")
            entry["voice"] = records[entry["char_id"]].get("voice")
        else:
            entry.setdefault("gender", None)
            entry.setdefault("voice", None)
        speakers[speaker] = entry

    # Két külön diarizált beszélő ugyanarra a karakterre: ez vagy túlszabdalt
    # diarizálás, vagy téves találat — egyik sem mehet át automatikusan.
    conflicts: dict[str, list[str]] = defaultdict(list)
    for speaker, entry in speakers.items():
        if entry.get("char_id"):
            conflicts[str(entry["char_id"])].append(speaker)
    for char_id, who in conflicts.items():
        if len(who) < 2:
            continue
        for speaker in who:
            entry = speakers[speaker]
            if entry["status"] == "manual":
                continue
            entry["status"] = "ambiguous"
            entry["reason"] = (f"ugyanarra a karakterre ({char_id}) két beszélő is "
                               f"illeszkedik: {', '.join(sorted(who))}")
            entry["char_id"] = None

    free = next_char_id(records)
    nxt = int("".join(ch for ch in free if ch.isdigit()) or 1)
    for entry in speakers.values():
        if not entry.get("char_id"):
            entry["suggested_new_id"] = f"C{nxt:03d}"
            if entry["status"] == "manual":
                continue
            nxt += 1

    result = {
        "pipeline": PIPELINE,
        "episode": tag,
        "start": start,
        "duration": duration,
        "model": str(embedding_model),
        "thresholds": {k: t[k] for k in
                       ("link_cos", "weak_cos", "margin", "min_exemplars")},
        "characters_known": len(records),
        "characters_with_exemplars": len(exemplars),
        "speakers": speakers,
        "counts": {},
        "generated_at": _now(),
        "notes": [
            "Példány-szintű CAM++ koszinusz: a karakter mintáihoz mért átlag, "
            "mert a puszta centroid 0,8–0,9-re fújja a különböző beszélőket.",
            "Automatikus azonosítás csak legalább "
            f"{t['min_exemplars']} mintás jelöltnél lehetséges.",
            "Az emberi megerősítés a work/identity-gallery.json aliasába és "
            "mintái közé kerül, így a következő epizódokban már stabil.",
        ],
    }
    counts: dict[str, int] = defaultdict(int)
    for entry in speakers.values():
        counts[entry["status"]] += 1
    result["counts"] = dict(counts)

    if write:
        out_dir = project.ep_dir(tag) / "source_audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"identity_{start:.0f}_{duration:.0f}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   default=lambda o: list(o)), encoding="utf-8")
        result_path = path
    else:
        result_path = None
    result["path"] = str(result_path) if result_path else None
    return result


def relink_from_audit(project, tag: str, audit: dict) -> dict:
    """Újraszámolja az azonosságot egy meglévő riport lenyomataiból."""
    protos = prototypes_from_audit(audit)
    turns = audit.get("speaker_turns") or []
    return link_speakers(project, tag, turns, Path(""), project.model("embedding"),
                         float(audit.get("start") or 0.0),
                         float(audit.get("duration") or 0.0),
                         prototypes=protos)


def gallery_key(wav_path: str | Path, turns: list[dict], model: str | Path) -> str:
    """Cache-kulcs: ugyanaz a kivágás + fordulók + modell = ugyanaz a lenyomat."""
    model_path = Path(model)
    h = hashlib.sha256()
    h.update(f"{Path(wav_path).resolve()}|{Path(wav_path).stat().st_size}".encode())
    h.update(f"|{model_path.resolve()}|{model_path.stat().st_size}".encode())
    h.update(json.dumps(turns, sort_keys=True).encode())
    return h.hexdigest()
