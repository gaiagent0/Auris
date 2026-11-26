"""Szövegminőség-ellenőrző pont — a hanggenerálás ELŐTT.

Két egymást kiegészítő réteg:

1. **Szabályok** (gyors, determinisztikus): szóköz- és írásjel-hibák, amiket
   biztonsággal magunk javítunk (dupla szóköz, írásjel előtti szóköz, hiányzó
   szóköz mondat végén, ismételt szó / hiányzó zárójel jelzése).
2. **LLM** (helyi Ollama, alapból ``qwen3.5:9b``): magyar nyelvi minőség —
   elírás, nyelvtan, szórend, értelmezhetőség. Batch-enként JSON-t kérünk, és
   csak a ténylegesen hibás sorokat adja vissza. A javításokat
   hasonlóság- és hossz-kontroll után alkalmazzuk, így egy modell sem írhatja
   át szabadon a szinkronszöveget.

A ``pipeline.dub`` a terv után, a szintézis előtt hívja, tehát a javított
szöveg megy a TTS-be; a szintézis-gyorsítótár szövegre kulcsol, ezért csak a
ténylegesen változott sorok szintetizálódnak újra. Az eredmény a QA
``text_errors`` kapujába és a riportba kerül.

Sor-szintű gyorsítótár (``work/<ep>/textcheck_cache.json``): ami egyszer már
tisztának bizonyult, azt a következő futtatás nem kéri le újra — így a napi
``dub`` nem fizet újra teljes LLM-kört.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import subs
from .config import Project
from .plan import Entry

DEFAULTS = {
    "enabled": True,
    "base": "http://127.0.0.1:11435",
    "fallback_base": "http://127.0.0.1:11434",
    "model": "qwen3.5:9b",
    "batch": 10,
    "timeout": 180,
    "num_predict": 600,     # a hosszú válasz CPU-n ne lépje túl a időt
    "num_thread": 0,        # 0 = az Ollama alapértéke; >0 = ennyi CPU-szál
    "required": True,       # ha az LLM elérhetetlen, bukjon a kapu
    "min_ratio": 0.55,      # a javított szöveg hasonlósága az eredetihez
    "max_len_factor": 1.6,  # legfeljebb ennyiszer hosszabb lehet a javítás
}

# ── 1. réteg: szabályok ──────────────────────────────────────────────────────

_SPACE_BEFORE = re.compile(r"\s+([.,;:!?])(?![.,;:!?…])")
_SPACE_AFTER_SENT = re.compile(r"([.!?…])(?=[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű])")
_SPACE_AFTER_COMMA = re.compile(r"([,;])(?=[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű])")
_REPEAT = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)

#: az erősítő szóismétlés nem hiba („nem nem nem”)
_EMPHASIS = {"nem", "igen", "na", "no", "jó", "hát", "soha", "többé", "mindig",
             "rossz", "szép", "pont", "itt", "ott", "most", "akkor", "újra",
             "minden", "semmi", "teljes", "elég", "már", "még", "itt-ott"}


def rule_pass(text: str) -> tuple[str, list[str]]:
    """Biztonságos automatikus javítások + jelzések (fix nélkül).

    Vissza: (javított szöveg, jelzések listája). A jelzések nem automatikus
    hibák — azokat az LLM vagy a riport nézi meg.
    """
    t = re.sub(r"\s{2,}", " ", text).strip()
    t = _SPACE_BEFORE.sub(r"\1", t)
    t = _SPACE_AFTER_SENT.sub(r"\1 ", t)
    t = _SPACE_AFTER_COMMA.sub(r"\1 ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()

    flags: list[str] = []
    m = _REPEAT.search(t)
    if m and m.group(1).lower() not in _EMPHASIS:
        flags.append(f"ismételt szó: {m.group(0)!r}")
    for a, b in (("(", ")"), ("[", "]")):
        if t.count(a) != t.count(b):
            flags.append(f"hiányzó/záró {b} zárójel")
    if re.search(r"[!?]{3,}", t):
        flags.append("túl sok írásjel")
    return t, flags


# ── 2. réteg: helyi LLM ──────────────────────────────────────────────────────

_SYSTEM = """Te magyar szinkronszöveg-korrektor vagy filmek és sorozatok magyar szinkronjához. A szöveg TTS-ből hangzik majd, feliratsorokból áll.
Feladat: csak a VALÓDI hibás sorokat javítsd (elírás, elgépelt szó, nyelvtani hiba, rossz írásjel-elhelyezés, értelmetlen szórend).
SZABÁLYOK: ne adj hozzá és ne vegyél el záró pontot; ne változtasd a stílust, a hangulatot és a beszédtempót; ne rövidíts és ne bővíts lényegesen; a neveket, számokat, idézeteket és a szereplő hangját (kiabálás, suttogás) őrizd meg; természetes, beszélt magyarsággal maradj.
A hibátlan sorokat NE add vissza. Válaszolj CSAK JSON-ban ebben a formában: {"results":[{"id":<szám>,"fixed":"<javított sor>","errors":"<mi volt benne>"}]} ; ha egy sor sem hibás: {"results":[]}
Az "errors" mező LEGFELJEBB 6 szó, csak a hiba típusa (pl. "elírás: csinált"). Ne indokolj, ne magyarázz, ne ismételd a szöveget, ne tegyél idézőjeles fejtegetést — hosszú magyarázattól a válasz félbeszakad, és a javítás elveszik."""


def cfg_of(project: Project) -> dict:
    cfg = {**DEFAULTS, **(getattr(project, "llm", None) or {})}
    if os.environ.get("DUBSTUDIO_LLM_BASE"):
        cfg["base"] = os.environ["DUBSTUDIO_LLM_BASE"]
    if os.environ.get("DUBSTUDIO_LLM_MODEL"):
        cfg["model"] = os.environ["DUBSTUDIO_LLM_MODEL"]
    cfg["required"] = bool(cfg.get("required", True))
    return cfg


def _reachable(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/api/tags", timeout=2.5) as r:
            return r.status == 200
    except Exception:
        return False


def _pick_base(cfg: dict) -> str | None:
    seen = []
    for b in (cfg.get("base"), cfg.get("fallback_base")):
        if b and b not in seen:
            seen.append(b)
    for b in seen:
        if _reachable(b):
            return b
    return None


def _parse_json_loose(resp: str) -> dict:
    """A modell JSON-jának kiszedése, toleránsan.

    Strukturált kimenetnél a grammar néha "csúszik" (a válasz végén plusz
    záró zárójel marad) — a legelső kiegyensúlyozott, idézet-tudatos
    JSON-objektumot kiolvasjuk, a maradékot eldobjuk.
    """
    resp = (resp or "").strip()
    try:
        return json.loads(resp)
    except Exception:
        pass
    start = resp.find("{")
    if start < 0:
        raise json.JSONDecodeError("nincs JSON a válaszban", resp, 0)
    depth, in_str, esc = 0, False, False
    for i in range(start, len(resp)):
        c = resp[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(resp[start:i + 1])
    raise json.JSONDecodeError("nincs zárt JSON-objektum", resp, start)


def _post_json(base: str, cfg: dict, system: str, prompt: str) -> dict:
    options = {"temperature": 0,
               "num_predict": int(cfg.get("num_predict", 800))}
    # erőforrás-korlát: egygépes futásnál ne vigye el az összes CPU-szálat
    if int(cfg.get("num_thread") or 0) > 0:
        options["num_thread"] = int(cfg["num_thread"])
    payload = json.dumps({
        "model": cfg["model"], "system": system, "prompt": prompt,
        "format": "json", "think": False, "stream": False,
        "options": options,
    }).encode("utf-8")
    # egyszeri azonnali újrapróbálás: a szerver a sok javítást tartalmazó
    # batch-eket néha HTTP 500-zal adja vissza, de a második kérés rendben
    # válaszol (mért: kézi újrapróbálásnál 200)
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(f"{base.rstrip('/')}/api/generate",
                                         data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=float(cfg["timeout"])) as r:
                data = json.loads(r.read().decode("utf-8"))
            return _parse_json_loose(str(data.get("response") or ""))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                OSError, json.JSONDecodeError, ValueError, KeyError) as ex:
            last = ex
            if attempt == 1:
                time.sleep(2.0)
    raise RuntimeError(f"LLM-válasz értelmezhetetlen/időtúllépés: {last}")


def _accepted(orig: str, fixed: str, cfg: dict) -> bool:
    """A modell-javítás csak akkor mehet a TTS-be, ha nem írta át szabadon."""
    if not isinstance(fixed, str) or not fixed.strip():
        return False
    fixed = fixed.strip()
    if len(fixed) > len(orig) * float(cfg["max_len_factor"]) + 12:
        return False
    a, b = orig.lower(), fixed.lower()
    # egyszerű hasonlóság (difflib nélkül, fedezék: karakter 2-grammok)
    def grams(s: str) -> set:
        s = re.sub(r"\s+", " ", s)
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
    ga, gb = grams(a), grams(b)
    jac = len(ga & gb) / max(1, len(ga | gb))
    return jac >= float(cfg["min_ratio"])


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}|{text}".encode("utf-8")).hexdigest()[:20]


# ── a checkpoint ─────────────────────────────────────────────────────────────

def run(entries: list[Entry], project: Project, verbose: bool = True) -> dict:
    """Szabályok + LLM a tervezett sorokon; javítja az ``entry.hu``-t."""
    cfg = cfg_of(project)
    t0 = time.time()
    todo = [e for e in entries if not e.skip]

    rows: list[dict] = []
    flagged: list[dict] = []
    rejected: list[dict] = []
    rule_fixed = llm_fixed = 0

    # 1) szabályok — minden soron, olcsó és idempotens
    for e in todo:
        fixed, flags = rule_pass(e.hu or "")
        if fixed != (e.hu or ""):
            rows.append({"src": "szabály", "i": e.i, "uid": e.uid,
                         "start": e.cue_start, "before": e.hu, "after": fixed,
                         "errors": "szóköz/írásjel"})
            e.hu = fixed
            rule_fixed += 1
        if flags:
            flagged.append({"src": "jelzés", "i": e.i, "uid": e.uid,
                            "start": e.cue_start, "before": e.hu, "after": "",
                            "errors": "; ".join(flags)})

    # gyorsítótár + jóváhagyott felülírások (a LLM állapotától függetlenül)
    tag = todo[0].ep if todo else "E00"
    cache_path = project.ep_dir(tag) / "textcheck_cache.json"
    cache = _load_cache(cache_path)
    model = str(cfg["model"])
    override_fixed = 0
    ov_path = project.ep_dir(tag) / "textcheck_overrides.json"
    try:
        overrides = json.loads(ov_path.read_text(encoding="utf-8"))
    except Exception:
        overrides = {}
    for e in todo:
        o = overrides.get((e.hu or "").strip())
        if not isinstance(o, dict):
            continue
        if o.get("fixed"):
            cleaned, _ = rule_pass(str(o["fixed"]))
            if cleaned != e.hu:
                rows.append({"src": "felülírás", "i": e.i, "uid": e.uid,
                             "start": e.cue_start, "before": e.hu,
                             "after": cleaned,
                             "errors": str(o.get("reason") or
                                            "jóváhagyott javítás")})
                e.hu = cleaned
                override_fixed += 1
                cache[_key(model, cleaned)] = "clean"
        elif o.get("reason"):
            # nyitott, emberi döntésre váró sor — látszik a riportban,
            # és nem záródik le: minden futtatásban újrakérdezzük
            flagged.append({"src": "nyitott", "i": e.i, "uid": e.uid,
                            "start": e.cue_start, "before": e.hu, "after": "",
                            "errors": str(o["reason"])})

    status = "disabled"
    checked = unchecked = 0
    base = None

    if not cfg.get("enabled", True):
        status = "disabled"
    else:
        base = _pick_base(cfg)
        if base is None:
            status = "unreachable"
            if verbose:
                print(f"[szöveg-LLM] ! nincs elérhető LLM-szerver "
                      f"({cfg['base']}, {cfg.get('fallback_base')}) — "
                      f"csak a szabályok futottak", flush=True)
        else:
            status = "ok"
            pending: list[Entry] = []
            for e in todo:
                if cache.get(_key(model, e.hu or "")) == "clean":
                    checked += 1
                    continue
                if cache.get(_key(model, e.hu or "")) == "rejected":
                    rejected.append({"src": "LLM", "i": e.i, "uid": e.uid,
                                     "start": e.cue_start, "before": e.hu,
                                     "after": "", "errors": "a modell-javítás elutasítva"})
                    checked += 1
                    continue
                pending.append(e)

            batches = [pending[i:i + int(cfg["batch"])]
                       for i in range(0, len(pending), int(cfg["batch"]))]
            n_batches_done = 0
            consec_fail = 0
            for bi, batch in enumerate(batches):
                prompt = json.dumps({"batch": [
                    {"id": e.i, "text": e.hu} for e in batch]},
                    ensure_ascii=False)
                # Elsőbbség a beállított szerver; ha egy batch elhasal, a
                # tartalék szerver CSAK akkor válik véglegessé, ha ténylegesen
                # válaszol (a beragadt 11434 így nem viszi el a többi batch-et).
                out = None
                last_err = None
                cands = [base] + [b for b in (cfg.get("fallback_base"),)
                                  if b and b != base]
                for cand in cands:
                    try:
                        out = _post_json(cand, cfg, _SYSTEM, prompt)
                        base = cand
                        break
                    except Exception as ex:    # időtúllépés, JSON-hiba, hálózat
                        last_err = ex
                if out is None:
                    unchecked += len(batch)
                    status = "partial"
                    consec_fail += 1
                    if verbose:
                        print(f"[szöveg-LLM] ! batch {bi + 1} hibás "
                              f"({consec_fail}. egymás utáni): {last_err}",
                              flush=True)
                    if consec_fail >= 2:
                        # a szerver nem válaszol — ne égessük a többi batch-et
                        left = sum(len(b) for b in batches[bi + 1:])
                        unchecked += left
                        if verbose:
                            print(f"[szöveg-LLM] ! LLM-leállítás — további "
                                  f"{left} sor ellenőrzése kimarad "
                                  f"(státusz: partial)", flush=True)
                        break
                    continue
                consec_fail = 0

                results = out.get("results")
                if not isinstance(results, list):
                    raise RuntimeError(f"LLM-válasz nem JSON-lista: {str(out)[:120]}")
                by_id = {int(r["id"]): r for r in results
                         if isinstance(r, dict) and "id" in r
                         and str(r.get("id", "")).isdigit()}
                # Szabály: ha a modell EBBŐL a batch-ből bármit visszaadott,
                # a kimaradt sorokat nem zárjuk le „tisztának” — a modell
                # nondeterminisztikus, a következő futtatás újra kérdezi őket.
                for e in batch:
                    r = by_id.get(e.i)
                    if r is None:
                        if not by_id:
                            cache[_key(model, e.hu or "")] = "clean"
                        checked += 1
                        continue
                    fixed = str(r.get("fixed") or "").strip()
                    if fixed and fixed != e.hu:
                        cleaned, _ = rule_pass(fixed)
                        if _accepted(e.hu, cleaned, cfg):
                            rows.append({"src": "LLM", "i": e.i, "uid": e.uid,
                                         "start": e.cue_start, "before": e.hu,
                                         "after": cleaned,
                                         "errors": str(r.get("errors") or "")[:120]})
                            e.hu = cleaned
                            llm_fixed += 1
                            cache[_key(model, cleaned)] = "clean"
                            checked += 1
                            continue
                    # vagy nem javított, vagy elutasítottuk
                    if fixed and fixed != e.hu:
                        rejected.append({"src": "LLM", "i": e.i, "uid": e.uid,
                                         "start": e.cue_start, "before": e.hu,
                                         "after": fixed,
                                         "errors": "a modell-javítás elutasítva "
                                                   "(túl nagy átírás)"})
                        cache[_key(model, e.hu or "")] = "rejected"
                    else:
                        cache[_key(model, e.hu or "")] = "clean"
                    checked += 1
                n_batches_done += 1
                if verbose and (n_batches_done % 4 == 0 or bi == len(batches) - 1):
                    print(f"  [szöveg-LLM] {n_batches_done}/{len(batches)} batch "
                          f"({llm_fixed} javítás)", flush=True)

    try:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:
        pass
    remaining = len(rejected)
    stats = {
        "llm_status": status,
        "llm_base": base or cfg.get("base"),
        "llm_model": cfg["model"],
        "required": bool(cfg.get("required", True)) and status != "disabled",
        "checked": checked,
        "unchecked": unchecked,
        "rule_fixed": rule_fixed,
        "override_fixed": override_fixed,
        "llm_fixed": llm_fixed,
        "llm_rejected": len(rejected),
        "flagged": len(flagged),
        "remaining": remaining,
        "rows": (rows + rejected + flagged)[:300],
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[szöveg-ellenőrzés] {checked} sor ellenőrizve · "
              f"{rule_fixed} szabály- + {override_fixed} felülírás + "
              f"{llm_fixed} LLM-javítás · "
              f"{remaining} megoldatlan · LLM: {status} "
              f"({stats['llm_base']}, {cfg['model']}) · {stats['seconds']}s",
              flush=True)
        if status in ("unreachable", "partial") and stats["required"]:
            print(f"  ! az LLM nem futott teljesen — a text_errors kapu ezért "
                  "bukni fog (ellenőrizd az Ollama szervert: "
                  f"{stats['llm_base']} / {cfg['model']})", flush=True)
    return stats
