"""Tipizált döntési modell (S1M/DM) a szinkronhang kiosztásához.

A System One szerződés három primitívje, változatlan jelentéssel:

* **choice** — melyik hang kapja a szereplőt (opciók: a 13 hang katalógusa),
  kimenet: ``choice`` + ``probabilities`` + ``confidence``;
* **score** — mennyire jó a kiosztás (0 = rossz … 3 = ideális);
* **noul** — igen/nem: kell-e hangszín-eltolás, kell-e emberi döntés.

A modell **nem** nyelvi modell: nem kitalálja, hanem a mérhető bizonyítékokat
értékeli. Ez szándékos — egy szinkronban a hangkiosztásnak megismételhetőnek
és indokolhatónak kell lennie, nem kreatívnak. A bizonyítékok:

1. **nem** — mérve (``gender.py``) vagy a stabil karakterrekordból;
2. **hangszín** — a TTS-hangok és a forrás-színész CAM++ beágyazásának
   koszinusza (a hangokat egyszer, fix mondattal szintetizáljuk és beágyazzuk);
3. **elérhetőség** — a hangot használó szereplők száma, és hogy van-e olyan
   köztük, aki **ugyanabban a jelenetben** szólal meg (ez tiltó feltétel);
4. **stabilitás** — a szereplő korábbi hangja erős előnyt kap, hogy egy
   újrafuttatás ne cserélje ki a megszokott hangot;
5. **főszerep** — a főszereplő kizárólagos, megkülönböztető hangot kap.

Külső neurális S1M (TypeSafe/Jev/Laya/d1-omni ``/v1/systemone``) beköthető:
ugyanezt a kérdéssémát továbbítja, de csak a **score** és a **noul** minőségi
skálájára kérdezünk rá — a hangválasztás bizonyíték-alapú marad.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import audio
from .voices import CATALOG, BY_ID, Voice

#: a choice-softmax hőmérséklete (alacsonyabb = határozottabb)
TEMPERATURE = 0.35
#: a hangszín-koszinuszt efölött tekintjük „egyértelmű egyezésnek”
TIMBRE_GOOD = 0.55
#: a bizonyíték ennyi másodperc alatt számít stabilnak
EVIDENCE_OK_S = 2.0


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def _softmax(scores: dict[str, float], temperature: float = TEMPERATURE) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.asarray([v / max(temperature, 1e-6) for v in scores.values()], dtype=np.float64)
    vals -= vals.max()
    ex = np.exp(vals)
    total = float(ex.sum()) or 1.0
    return {k: float(v) for k, v in zip(scores, ex / total)}


def confidence(probs: dict[str, float]) -> float:
    """Normalizált entrópia: 1 = egyértelmű, 0 = teljesen bizonytalan."""
    p = np.asarray([v for v in probs.values() if v > 0], dtype=np.float64)
    if p.size <= 1:
        return 1.0 if p.size == 1 else 0.0
    h = -float((p * np.log(p)).sum())
    return max(0.0, min(1.0, 1.0 - h / math.log(p.size)))


def timbre_similarity(cos: float | None) -> float:
    """Koszinusz (-0,2…0,8) -> 0…1; ismeretlen mérés = semleges 0,5."""
    if cos is None:
        return 0.5
    return round(_norm01(cos, -0.20, 0.80), 4)


def gender_fit(voice_gender: str | None, char_gender: str | None) -> tuple[float, str]:
    """Kemény feltétel: ismert nemnél csak azonos nemű hang jöhet szóba."""
    if not char_gender:
        return 0.55, "a szereplő neme nem ismert, a hang neme nem szűrhető"
    if not voice_gender:
        return 0.45, "a hang neméhez nincs adat"
    if voice_gender == char_gender:
        return 1.0, f"nem egyezik ({char_gender})"
    return 0.0, f"nem ütközik (hang {voice_gender}, szereplő {char_gender})"


def availability(voice_id: str, usage: dict[str, list[str]] | None,
                 cooccurring: set[str] | None = None) -> tuple[float, str, bool]:
    """``(pontszám, indok, tiltott)`` — a közös jelenet tiltó feltétel."""
    users = list((usage or {}).get(voice_id) or [])
    if not users:
        return 1.0, "szabad hang", False
    clash = sorted(set(users) & set(cooccurring or ()))
    if clash:
        return 0.0, ("ugyanabban a jelenetben már használja: "
                     + ", ".join(clash)), True
    if len(users) == 1:
        return 0.35, f"már használja: {users[0]} (nincs közös jelenet)", False
    return max(0.05, 0.35 - 0.08 * len(users)), (
        f"{len(users)} szereplő használja, közös jelenet nélkül"), False


def stability_bonus(voice_id: str, current_voice: str | None) -> tuple[float, str | None]:
    """A meglévő kiosztás megtartása erős előny (ne cserélődjön újrafuttatáskor)."""
    if current_voice and voice_id == current_voice:
        return 1.0, f"ez a szereplő jelenlegi hangja ({current_voice})"
    if current_voice:
        return 0.0, None
    return 0.5, None


def engine_consistency(voice: Voice, engine: str | None) -> tuple[float, str | None]:
    """Egy szinkronban lehetőleg egy motor család szólaljon meg."""
    if not engine:
        return 0.6, None
    return (1.0, f"a szinkron motorja ({engine})") if voice.engine == engine else \
           (0.25, f"más motor ({voice.engine})")


def main_role_preference(voice: Voice, is_main: bool) -> tuple[float, str | None]:
    """A főszereplő a katalógus élére preferált (megkülönböztető) hangot kapja."""
    if not is_main:
        return 0.6, None
    idx = next((i for i, v in enumerate(CATALOG) if v.vid == voice.vid), len(CATALOG))
    return round(max(0.2, 1.0 - idx / max(len(CATALOG) - 1, 1)), 4), None


WEIGHTS = {
    "gender": 0.34,
    "timbre": 0.20,
    "availability": 0.18,
    "stability": 0.16,
    "engine": 0.07,
    "main_role": 0.05,
}


def relative_timbre(timbres: dict | None, pool: list[Voice]) -> dict[str, float]:
    """A hangszín-koszinusz **a készleten belül** normalizálva 0…1-re.

    Az abszolút koszinusz itt kevéssé megkülönböztető (a TTS-hangok egymáshoz
    is 0,5 körül hasonlítanak), ezért az a kérdés, hogy a forrás-színészhez
    *melyik* hang illik a legjobban. Ha a készlet nem szór (0,05 alatt),
    nincs értékelhető jel, és semleges értéket adunk.
    """
    cos = {}
    for v in pool:
        row = (timbres or {}).get(v.vid) or {}
        value = row.get("cosine")
        if value is not None:
            cos[v.vid] = float(value)
    if len(cos) < 2:
        return {}
    lo, hi = min(cos.values()), max(cos.values())
    if hi - lo < 0.05:
        return {}
    return {k: (v - lo) / (hi - lo) for k, v in cos.items()}


def score_voice(voice: Voice, character: dict, timbres: dict | None,
                usage: dict | None, cooccurring: set[str] | None,
                engine: str | None = None,
                timbre_rel: dict | None = None) -> dict:
    """Egy hang és egy szereplő illesztése — minden részponthoz indoklás."""
    gender = character.get("gender")
    g_score, g_reason = gender_fit(voice.gender, gender)
    timbre = (timbres or {}).get(voice.vid) or {}
    cos = timbre.get("cosine")
    rel = (timbre_rel or {}).get(voice.vid)
    t_score = rel if rel is not None else timbre_similarity(cos)
    a_score, a_reason, blocked = availability(voice.vid, usage, cooccurring)
    s_score, s_reason = stability_bonus(voice.vid, character.get("voice"))
    e_score, e_reason = engine_consistency(voice, engine)
    m_score, m_reason = main_role_preference(voice, bool(character.get("main")))

    parts = {"gender": g_score, "timbre": t_score, "availability": a_score,
             "stability": s_score, "engine": e_score, "main_role": m_score}
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    if blocked:
        total = 0.0
    reasons = [r for r in (g_reason, a_reason, s_reason, e_reason, m_reason) if r]
    if cos is not None:
        rel_txt = f", a készleten belül {rel:.0%}" if rel is not None else ""
        reasons.insert(1, f"hangszín-koszinusz {cos:+.2f}{rel_txt}")
    else:
        reasons.insert(1, "nincs hangszín-mérés (a modell még nem futott)")
    return {
        "voice": voice.vid,
        "label": voice.label,
        "gender": voice.gender,
        "engine": voice.engine,
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "score": round(total, 4),
        "blocked": bool(blocked),
        "reasons": reasons,
        "timbre_cosine": cos,
    }


def usage_without(usage: dict | None, char_id: str | None) -> dict:
    """A használat-térkép a szereplő **saját** hangja nélkül.

    A saját hang nem „foglalt”: ha a szereplő már ezt használja, az a helyes
    és megtartandó állapot, nem ütközés.
    """
    if not usage:
        return {}
    if not char_id:
        return dict(usage)
    out: dict[str, list[str]] = {}
    for vid, users in usage.items():
        rest = [u for u in (users or []) if str(u) != str(char_id)]
        if rest:
            out[str(vid)] = rest
    return out


def decide_voice(character: dict, voices: list[Voice] | None = None,
                 timbres: dict | None = None, usage: dict | None = None,
                 cooccurring: set[str] | None = None,
                 engine: str | None = None,
                 temperature: float = TEMPERATURE) -> dict:
    """**choice**: a szereplő hangja, valószínűségekkel és indoklással."""
    pool = [v for v in (voices or CATALOG)]
    usage = usage_without(usage, character.get("char_id"))
    if not pool:
        return {"choice": None, "probabilities": {}, "confidence": 0.0,
                "answer_confidence": 0.0, "candidates": [],
                "noul": {"human_review": 1.0, "pitch_shift": 0.0},
                "score": 0, "notes": ["nincs elérhető hang"]}
    rel = relative_timbre(timbres, pool)
    scored = [score_voice(v, character, timbres, usage, cooccurring, engine,
                          timbre_rel=rel) for v in pool]
    allowed = [s for s in scored if not s["blocked"] and s["parts"]["gender"] > 0]
    if not allowed:
        allowed = [s for s in scored if not s["blocked"]] or scored
    probs = _softmax({s["voice"]: s["score"] for s in allowed}, temperature)
    for item in scored:
        item["probability"] = round(probs.get(item["voice"], 0.0), 4)
    # A tiltott hangok is benne maradnak a listában 0 valószínűséggel, hogy a
    # felület meg tudja mutatni, miért esett ki (nem tűnik el nyomtalanul).
    ranked = sorted(scored, key=lambda s: s["probability"], reverse=True)
    best = next(s for s in ranked if not s["blocked"] and s["probability"] > 0)
    conf = confidence(probs)
    return {
        "choice": best["voice"],
        "label": best["label"],
        "probabilities": {k: round(v, 4) for k, v in
                          sorted(probs.items(), key=lambda kv: kv[1], reverse=True)},
        "confidence": round(conf, 4),
        "answer_confidence": round(probs.get(best["voice"], 0.0), 4),
        "candidates": ranked,
        "score": score_level(best, character),
        "noul": {
            "human_review": noul_human_review(best, character, conf),
            "pitch_shift": noul_pitch_shift(best, usage),
        },
        "hint": _hint(best, usage),
        "notes": [],
    }


def _hint(best: dict, usage: dict | None) -> str | None:
    """Egy mondat, ha a döntés fenntartása ugyan helyes, de van jobb lehetőség."""
    users = list((usage or {}).get(best["voice"]) or [])
    if best["parts"]["availability"] < 0.35 and users:
        return (f"A jelenlegi hangot {len(users) + 1} szereplő osztja; érdemes lehet "
                f"külön hangot adni neki (a stabilitás miatt most megtartottam).")
    if best["parts"]["availability"] >= 1.0 and best["score"] >= 0.75:
        return "Ez a hang jelenleg szabad, és jól illeszkedik."
    return None


def score_level(best: dict, character: dict) -> int:
    """**score**: 0 rossz, 1 használható, 2 jó, 3 ideális.

    Ismert nemnél a 3-as szint megköveteli a jó hangszín-egyezést és a szabad
    hangot; ismeretlen nemnél (0,55) legfeljebb 2 adható, mert a legfontosabb
    szűrő bizonyíték nélkül futott.
    """
    parts = best["parts"]
    if best["blocked"] or parts["gender"] == 0:
        return 0
    timbre_ok = best.get("timbre_cosine") is None or best["timbre_cosine"] >= TIMBRE_GOOD
    if parts["gender"] >= 1.0 and parts["availability"] >= 1.0 and timbre_ok:
        return 3
    if parts["gender"] >= 0.55 and parts["availability"] >= 0.35:
        return 2
    return 1


def noul_human_review(best: dict, character: dict, conf: float) -> float:
    """**noul**: mekkora a valószínűsége, hogy embernek kell döntenie."""
    p = 0.0
    if not character.get("gender"):
        p += 0.35
    if best.get("timbre_cosine") is None:
        p += 0.15
    if best["parts"]["availability"] < 1.0:
        p += 0.25
    if conf < 0.5:
        p += 0.25
    if not character.get("evidence_seconds") or \
            float(character["evidence_seconds"]) < EVIDENCE_OK_S:
        p += 0.15
    return round(min(0.99, p), 3)


def noul_pitch_shift(best: dict, usage: dict | None) -> float:
    """**noul**: kell-e fix hangszín-eltolás, mert a hangot más is használja."""
    users = list((usage or {}).get(best["voice"]) or [])
    return round(min(0.95, 0.15 + 0.25 * len(users)), 3) if users else 0.02


# ── hangszín-galéria (TTS hangok beágyazása) ─────────────────────────────
TIMBRE_TEXT = ("A szinkronpróba rövid mondata: este a kikötőben hideg szél fúj, "
               "de a csapat nyugodtan vár.")


def measure_voice_timbres(project, out_path: str | Path | None = None,
                          text: str = TIMBRE_TEXT, verbose: bool = True) -> dict:
    """Minden elérhető TTS-hangot beágyazunk, hogy legyen valódi hangszín-adat.

    A CAM++ emberi beszédre tanult, ezért a szintetikus és emberi beágyazás
    összevetése **durva** hasonlóság (főleg a hangszín mély/magas jellege),
    nem azonosítás — ezt a döntés súlya és a felület is így kezeli.
    """
    from .embeddings import SpeakerEmbedder, cosine, norm
    from .voices import VoiceBank

    out = Path(out_path or (project.work / "voice-timbres.json"))
    bank = VoiceBank(project, verbose=False)
    embedder = SpeakerEmbedder(project.model("embedding"))
    data = {}
    for vid in sorted(bank.available):
        voice = BY_ID[vid]
        try:
            x, sr = bank.synth(text, voice)
        except Exception as exc:                      # pragma: no cover - modell
            if verbose:
                print(f"[hangszín] {vid} kihagyva: {exc}")
            continue
        vec = embedder.embed(audio.to_16k(x, sr))
        if vec is None:
            continue
        data[vid] = {"vector": [round(float(v), 6) for v in norm(vec)],
                     "seconds": round(len(x) / max(sr, 1), 2),
                     "text": text}
        if verbose:
            print(f"[hangszín] {vid}: {len(x) / max(sr, 1):.2f} s")
    payload = {"pipeline": "dubstudio.voice-timbres v1",
               "model": str(project.model("embedding")), "voices": data}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def load_timbres(path: str | Path | None) -> dict:
    """Galéria betöltése; hiányzó fájl = üres (a döntés semleges hangszínnel fut)."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return {}
    try:
        return (json.loads(p.read_text(encoding="utf-8")).get("voices") or {})
    except (OSError, ValueError):
        return {}


def cosine_to_prototype(proto, timbres: dict, model_path: str | Path | None = None) -> dict:
    """Besorolás: minden hang koszinusza a forrás-színész lenyomatához."""
    from .embeddings import cosine

    out = {}
    for vid, row in (timbres or {}).items():
        vec = row.get("vector") or []
        if len(vec) < 8:
            continue
        out[vid] = {"cosine": round(float(cosine(proto, np.asarray(vec, dtype=np.float32))), 4),
                    "seconds": row.get("seconds")}
    return out


# ── külső neurális S1M (opcionális) ──────────────────────────────────────
def remote_system_one(config: dict | None, state: str,
                      questions: dict, timeout: float = 60.0) -> dict:
    """TypeSafe/Jev-kompatibilis ``/v1/systemone`` hívás (Laya, d1-omni, Kev).

    Bekapcsolás: ``project.json`` → ``"s1m": {"enabled": true, "base_url":
    "http://127.0.0.1:8000", "api_key": ""}``. Bármilyen hiba esetén üres
    választ ad, és a hívó a helyi tipizált döntést használja tovább — a
    felület soha nem függ a külső modell elérhetőségétől.
    """
    cfg = config or {}
    if not cfg.get("enabled") or not cfg.get("base_url"):
        return {}
    import urllib.error
    import urllib.request

    url = str(cfg["base_url"]).rstrip("/") + "/v1/systemone"
    body = json.dumps({"state": state, "questions": questions},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    if cfg.get("api_key"):
        req.add_header("Authorization", f"Bearer {cfg['api_key']}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}
