"""Soronkénti emberi felülvizsgálat: ki beszél, képen kívül van-e, elcsúszás.

A felület és a szinkronlánc ugyanezt a fájlt használja:
``work/<EP>/review/lines.json``. A döntések **emberiek**, ezért elsőbbségük
van az automatával szemben, és túlélik az újrafuttatást.

Egy sor (egy feliratsor) bejegyzése:

* ``marks`` — szó- vagy mondatszintű megjelölések. Egy mark a sor egy
  szótartományára mondja meg, ki mondja: egy szereplő (``C###`` vagy a
  felületen létrehozott ``N###``), „képen kívül”, vagy „nem tudom”.
* ``offset`` — a sor elcsúszása másodpercben (a szinkronhang idejére kerül).

A nevek magyar keresztnevek: a kód helyett a felület ezeket mutatja, és a
szinkronfutás is ezt a nevet viszi tovább a riportba.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PIPELINE = "dubstudio.linereview v1"

#: magyar keresztnevek a szereplők megkülönböztetéséhez (nem nemenként)
FEMALE_NAMES = ("Anna", "Éva", "Panna", "Zsófi", "Kata", "Mari", "Réka",
                "Bori", "Nóra", "Lilla", "Juli", "Tímea", "Eszter", "Dóra",
                "Vera", "Klára", "Ildi", "Szilvi", "Ági", "Rita",
                "Kinga", "Melinda", "Orsi", "Bea", "Csilla", "Emese", "Fanni",
                "Gabi", "Hajni", "Judit", "Lena", "Móni", "Niki", "Piri",
                "Rózsa", "Sári", "Teri", "Viki", "Zita", "Adél")
MALE_NAMES = ("Ottó", "Laci", "Józsi", "Gábor", "Peti", "Bence", "Misi",
              "Ákos", "Zoli", "Tivadar", "Sanyi", "Imi", "Feri", "Karcsi",
              "Robi", "Dávid", "Tamás", "Balázs", "Norbi", "Csaba",
              "András", "Béla", "Csongor", "Dani", "Ernő", "Zsolti",
              "Gergő", "Gyuri", "Huba", "István", "Jani", "Kálmán", "Levente",
              "Márk", "Nándor", "Olivér", "Pali", "Ricsi", "Soma", "Viktor")

#: a megjelölés fajtái
KINDS = ("char", "offscreen", "unknown")

#: az elcsúszás határa (másodperc) — ennél nagyobb már nem „finom javítás”
OFFSET_MAX = 10.0


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def path_for(project, tag: str) -> Path:
    return project.ep_dir(tag.upper()) / "review" / "lines.json"


def _empty(tag: str) -> dict:
    return {"pipeline": PIPELINE, "episode": tag.upper(), "names": {},
            "people": {}, "lines": {}, "updated_at": None}


def load(project, tag: str) -> dict:
    p = path_for(project, tag)
    if not p.is_file():
        return _empty(tag)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty(tag)
    if not isinstance(data, dict):
        return _empty(tag)
    base = _empty(tag)
    for key, value in base.items():
        data.setdefault(key, value)
    if not isinstance(data.get("lines"), dict):
        data["lines"] = {}
    if not isinstance(data.get("names"), dict):
        data["names"] = {}
    if not isinstance(data.get("people"), dict):
        data["people"] = {}
    return data


def save(project, tag: str, data: dict) -> Path:
    p = path_for(project, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["pipeline"] = PIPELINE
    data["episode"] = tag.upper()
    data["updated_at"] = _now()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def key(uid: str, cue) -> str:
    """A sor kulcsa: ``<megnyilatkozás>#<feliratsor index>``."""
    return f"{str(uid)}#{int(cue)}"


def split_key(k: str) -> tuple[str, int] | None:
    uid, _, cue = str(k).rpartition("#")
    if not uid:
        return None
    try:
        return uid, int(cue)
    except ValueError:
        return None


def name_pool(gender: str | None) -> tuple[str, ...]:
    return FEMALE_NAMES if str(gender or "").upper().startswith("F") else MALE_NAMES


def suggest_names(rows: list[dict]) -> dict[str, str]:
    """Magyar keresztnevek a szereplőkhöz (rangsor szerint, nemenként).

    ``rows``: ``[{"char_id", "gender", "rank", "name"}]``. A már meglévő nevek
    elsőbbséget kapnak, utána a rangsor szerint osztunk szabad nevet. Így a
    felületen nem kódok, hanem megjegyzett nevek szerepelnek.
    """
    rows = [r for r in rows if str(r.get("char_id") or "")]

    def keepable(name, gender) -> bool:
        """A mentett név megtartható-e az adott nemhez.

        Újraklaszterezés után ugyanaz a ``C###`` más nemű szereplő lehet, ezért
        a korábbi, automatikusan adott keresztnév („Ottó” egy női szereplőn)
        nem örökölhető. A korábbi kimerült névkészlet maradéka („Anna 2”,
        „Anna 17”) sem valódi név: az is újraosztható.
        """
        text = str(name or "")
        if not text:
            return False
        g = str(gender or "").upper()[:1]
        if g == "F" and text in MALE_NAMES:
            return False
        if g == "M" and text in FEMALE_NAMES:
            return False
        base, _, tail = text.rpartition(" ")
        if base and tail.isdigit() and base in (*FEMALE_NAMES, *MALE_NAMES):
            return False
        return True

    used = {str(r.get("name")) for r in rows
            if keepable(r.get("name"), r.get("gender"))}
    out: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: (int(r.get("rank") or 9999),
                                           str(r["char_id"]))):
        cid = str(row["char_id"])
        if keepable(row.get("name"), row.get("gender")) \
                and str(row["name"]) not in out.values():
            out[cid] = str(row["name"])
            continue
        pool = name_pool(row.get("gender"))
        pick = next((n for n in pool if n not in used), "")
        if not pick:
            base = pool[0]
            k = 2
            while f"{base} {k}" in used:
                k += 1
            pick = f"{base} {k}"
        used.add(pick)
        out[cid] = pick
    return out


def ensure_names(data: dict, rows: list[dict]) -> dict[str, str]:
    """A hiányzó nevek pótlása a mentett állapotban (stabil marad).

    A már nem létező szereplők neveit eldobjuk: újraklaszterezés után a régi
    ``C0xx`` azonosítók más szereplőt jelölhetnek, a maradék nevek pedig
    felfújnák a névkészletet (innen jöttek a régi „Anna 2 … Anna 17” nevek).
    """
    valid = {str(r.get("char_id")) for r in rows
             if str(r.get("char_id") or "")}
    # A már nem létező szereplőre mutató mark a felületen **név nélkül**
    # jelenne meg (a régi ``C034`` újraklaszterezés után eltűnt vagy más
    # szereplőt jelöl). A hagyott jelölés hamis beszélőt írna a sorra, ezért
    # eldobjuk; a felületen létrehozott ``N###`` szereplők maradhatnak.
    for k, row in list((data.get("lines") or {}).items()):
        marks = [m for m in (row.get("marks") or [])
                 if m.get("kind") != "char"
                 or str(m.get("char") or "") in valid
                 or str(m.get("char") or "").startswith("N")]
        if marks == (row.get("marks") or []):
            continue
        row["marks"] = marks
        try:
            offset = float(row.get("offset") or 0.0)
        except (TypeError, ValueError):
            offset = 0.0
        if not marks and not offset:
            data["lines"].pop(k, None)
    names = {cid: name for cid, name in (data.get("names") or {}).items()
             if cid in valid}
    merged = []
    for row in rows:
        cid = str(row.get("char_id") or "")
        if not cid:
            continue
        merged.append({**row, "name": names.get(cid) or row.get("name")})
    # A `suggest_names` a megtartható (neműhöz illő, valódi) neveket változatlanul
    # visszaadja, ezért itt felülírhatunk: a nemhez nem illő vagy kimerült
    # névkészletből származó régi név így nem ragad be újraklaszterezés után sem.
    for cid, name in suggest_names(merged).items():
        names[cid] = name
    data["names"] = names
    return names


def _name_fits(name: str, gender: str | None) -> bool:
    """A név illik-e a szereplő neméhez (a keresztnévkészlet alapján)."""
    g = str(gender or "").upper()[:1]
    if g == "F" and name in MALE_NAMES:
        return False
    if g == "M" and name in FEMALE_NAMES:
        return False
    return True


def carry(data: dict, before: dict[str, str], after: dict[str, str],
          genders: dict[str, str] | None = None) -> dict[str, str]:
    """Nevek és markok átvitele újraklaszterezés után.

    A sorok ``uid``-je stabil, a klaszter-azonosító viszont minden futásnál
    újraszámozódik. Ha egy eltűnt szereplő sorai többségükben egy **névtelen**
    új szereplőnél kötnek ki, a neve és a rá mutató markjai átkerülnek — így
    a ``C###`` átszámozása nem dobja el a kézi jelölést. A már elnevezett cél
    saját neve mindig elsőbbséget élvez (egy töredék neve soha nem írja felül
    a főszereplőét).

    ``before``/``after``: ``uid -> char_id`` a klaszterezés előtt/után.
    ``genders``: az új szereplők neme (a név nemének ellenőrzéséhez).
    """
    counts: dict[tuple[str, str], int] = {}
    total: dict[str, int] = {}
    for uid, old in before.items():
        if not old:
            continue
        total[old] = total.get(old, 0) + 1
        new = after.get(uid)
        if new and old != new:
            counts[(old, new)] = counts.get((old, new), 0) + 1
    best_for: dict[str, tuple[int, str]] = {}
    for (old, new), n in counts.items():
        if n > best_for.get(old, (0, ""))[0]:
            best_for[old] = (n, new)
    names = data.get("names") or {}
    taken: dict[str, str] = {}
    remap: dict[str, str] = {}
    for old, (n, new) in sorted(best_for.items(), key=lambda kv: -kv[1][0]):
        if not names.get(old) or new in taken:
            continue
        # A célen már van név: az a szereplő élte a futást, a neve elsőbbséget
        # élvez. Csak névtelen cél kaphat nevet (különben egy eltűnt töredék
        # neve felülírná a főszereplőét — mért hiba).
        if names.get(new):
            continue
        # Gyenge átfedés nem bizonyíték: a sorok többsége és legalább 2 sor.
        if n < 2 or n * 2 < total.get(old, n):
            continue
        if not _name_fits(str(names[old]), (genders or {}).get(new)):
            continue
        taken[new] = old
        remap[old] = new
    if not remap:
        return {}
    data["names"] = {cid: name for cid, name in names.items()
                     if cid not in remap}
    for old, new in remap.items():
        data["names"][new] = names[old]
    for row in (data.get("lines") or {}).values():
        for mark in row.get("marks") or []:
            if mark.get("kind") == "char" and mark.get("char") in remap:
                mark["char"] = remap[mark["char"]]
    return remap


def _same_mark(a: dict, b: dict) -> bool:
    """Két megjelölés ugyanaz-e (a szótár-egyezéshez)."""
    return (a.get("kind"), a.get("char"), a.get("gender"),
            list(a.get("words") or []) or None) == \
           (b.get("kind"), b.get("char"), b.get("gender"),
            list(b.get("words") or []) or None)


def resolve(data: dict, uid, cue, valid_uids=None) -> tuple[str, dict]:
    """Egy sor bejegyzése: a pontos kulcs, különben a feliratsor sorszáma.

    A kulcs ``<megnyilatkozás-uid>#<feliratsor>``: a sorszám stabil, az uid
    viszont az újracsoportosítással megváltozhat. Ha egy jelölés régi uid-re
    mutat, nem tűnhet el a felületről: a sorszám alapján értelmezzük, és a
    visszaadott kulcs mindig a mostani (``uid#sorszám``), így a mentés is oda
    ír. A ``valid_uids`` a mostani megnyilatkozások azonosítói — az ő saját
    bejegyzésüket nem használjuk tartalékként, mert az más soré.
    """
    lines = data.get("lines") or {}
    k = key(uid, cue)
    entry = lines.get(k)
    if entry is not None:
        return k, entry
    try:
        target = int(cue)
    except (TypeError, ValueError):
        return k, {}
    live = {str(u) for u in valid_uids} if valid_uids is not None else set()
    found = []
    for other, row in lines.items():
        pair = split_key(other)
        if pair is None or pair[1] != target or not row:
            continue
        if pair[0] in live:
            continue
        found.append(row)
    if not found:
        return k, {}
    marks: list[dict] = []
    offset = 0.0
    for row in found:
        for mark in (row.get("marks") or []):
            if not any(_same_mark(mark, m) for m in marks):
                marks.append(dict(mark))
        if not offset:
            try:
                offset = float(row.get("offset") or 0.0)
            except (TypeError, ValueError):
                offset = 0.0
    if not marks and not offset:
        return k, {}
    return k, {"marks": marks, "offset": round(offset, 2)}


def display_name(data: dict, char_id: str | None) -> str | None:
    if not char_id:
        return None
    cid = str(char_id)
    return ((data.get("people") or {}).get(cid) or {}).get("name") \
        or (data.get("names") or {}).get(cid)


def _new_person(data: dict, gender: str | None, name: str | None) -> str:
    """Új szereplő a felületről (``N001``, ``N002``…), magyar névvel."""
    people = data.setdefault("people", {})
    g = "F" if str(gender or "").upper().startswith("F") else "M"
    used = {str((p or {}).get("name") or "") for p in people.values()}
    clean = " ".join(str(name or "").split())[:24]
    if not clean:
        clean = next((n for n in name_pool(g) if n not in used), "")
    if not clean:
        base = name_pool(g)[0]
        k = 2
        while f"{base} {k}" in used:
            k += 1
        clean = f"{base} {k}"
    n = 1
    while f"N{n:03d}" in people:
        n += 1
    cid = f"N{n:03d}"
    people[cid] = {"name": clean, "gender": g}
    data.setdefault("names", {})[cid] = clean
    return cid


def _overlap(a: list | None, b: list | None) -> bool:
    """Két szótartomány ütközik-e (``None`` = a teljes sor)."""
    if a is None or b is None:
        return True
    return not (int(b[1]) < int(a[0]) or int(a[1]) < int(b[0]))


def _words(value) -> list[int] | None:
    if value is None:
        return None
    try:
        a, b = int(value[0]), int(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if a < 0 or b < a:
        raise ValueError("érvénytelen szótartomány")
    return [a, b]


def set_label(project, tag: str, uid: str, cue, kind: str = "char",
              char_id: str | None = None, gender: str | None = None,
              name: str | None = None, words=None) -> tuple[dict, str]:
    """Egy sor (vagy szótartomány) megjelölése.

    ``kind``: ``char`` (szereplő), ``offscreen`` (képen kívüli),
    ``unknown`` (nem tudom), ``new`` (új szereplő a felületről), ``clear``
    (a megjelölés törlése az adott tartományon).
    """
    kind = str(kind or "char").strip().lower()
    if kind not in (*KINDS, "new", "clear"):
        raise ValueError(f"ismeretlen megjelölés: {kind!r}")
    data = load(project, tag)
    k = key(uid, cue)
    entry = data["lines"].setdefault(k, {})
    span = _words(words)

    if kind == "new":
        char_id = _new_person(data, gender, name)
        kind = "char"
    if kind == "char" and not char_id:
        raise ValueError("a szereplő-megjelöléshez karakter kell")

    marks = [m for m in (entry.get("marks") or [])
             if not _overlap(span, m.get("words"))]
    if kind != "clear":
        if kind == "char":
            data.setdefault("names", {}).setdefault(str(char_id), name or
                                                    display_name(data, char_id) or
                                                    str(char_id))
        marks.append({
            "words": span,
            "kind": kind,
            "char": str(char_id) if kind == "char" else None,
            "gender": (str(gender).upper()[:1] if gender else
                       ((data.get("people") or {}).get(str(char_id)) or {}).get("gender")),
            "offscreen": kind == "offscreen",
        })
        marks.sort(key=lambda m: (1 if m.get("words") is None else 0,
                                  (m.get("words") or [0])[0]))
    entry["marks"] = marks
    if not marks and not float(entry.get("offset") or 0.0):
        data["lines"].pop(k, None)
    save(project, tag, data)
    return data, k


def set_offset(project, tag: str, uid: str, cue, offset: float) -> tuple[dict, str]:
    """Egy sor elcsúszása (másodperc); 0 = nincs elcsúszás."""
    try:
        value = round(float(offset or 0.0), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("az elcsúszás csak szám lehet") from exc
    if not -OFFSET_MAX <= value <= OFFSET_MAX:
        raise ValueError(f"az elcsúszás {OFFSET_MAX:g} és +{OFFSET_MAX:g} "
                         f"másodperc között lehet")
    data = load(project, tag)
    k = key(uid, cue)
    entry = data["lines"].setdefault(k, {})
    entry["offset"] = value
    if not entry.get("marks") and not value:
        data["lines"].pop(k, None)
    save(project, tag, data)
    return data, k


def set_name(project, tag: str, char_id: str, name: str) -> dict:
    """Egy szereplő megjelenített nevének átírása."""
    cid = str(char_id or "").strip()
    clean = " ".join(str(name or "").split())[:24]
    if not cid:
        raise ValueError("hiányzó szereplő-azonosító")
    if not clean:
        raise ValueError("a név nem lehet üres")
    data = load(project, tag)
    data.setdefault("names", {})[cid] = clean
    person = (data.get("people") or {}).get(cid)
    if person is not None:
        person["name"] = clean
    save(project, tag, data)
    return data


def set_person(project, tag: str, char_id: str, name: str | None = None,
               voice: str | None = None) -> dict:
    """A felületen létrehozott szereplő neve és magyar hangja.

    A ``C###`` szereplők hangja a hangzárban (cast) él, ezt a hívó kezeli; ez a
    függvény a ``people`` szótárban tárolt (``N###``) szereplőket frissíti,
    hogy a felületi meghallgatásnak is legyen hangja.
    """
    cid = str(char_id or "").strip()
    if not cid:
        raise ValueError("hiányzó szereplő-azonosító")
    data = load(project, tag)
    person = (data.get("people") or {}).get(cid)
    if person is None:
        raise ValueError(f"nincs felületi szereplő: {cid}")
    if name is not None:
        clean = " ".join(str(name).split())[:24]
        if not clean:
            raise ValueError("a név nem lehet üres")
        person["name"] = clean
        data.setdefault("names", {})[cid] = clean
    if voice is not None:
        person["voice"] = str(voice).strip() or None
    save(project, tag, data)
    return data


def names(project, tag: str) -> dict[str, str]:
    return dict(load(project, tag).get("names") or {})


def offsets(project, tag: str) -> dict[tuple[str, int], float]:
    """A szinkronlánc számára: ``(uid, feliratsor index) -> elcsúszás``."""
    out: dict[tuple[str, int], float] = {}
    for pair, entry in _entries(load(project, tag), project, tag):
        try:
            value = float((entry or {}).get("offset") or 0.0)
        except (TypeError, ValueError):
            continue
        if not value:
            continue
        out[pair] = value
    return out


def _live_pairs(project, tag: str) -> tuple[set, dict[int, list]]:
    """A mostani ``(uid, feliratsor)`` párok és a sorszám szerinti nézet."""
    path = project.ep_dir(tag.upper()) / "utterances.json"
    pairs: set[tuple[str, int]] = set()
    by_cue: dict[int, list[tuple[str, int]]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return pairs, by_cue
    if not isinstance(data, list):
        return pairs, by_cue
    for row in data:
        uid = str((row or {}).get("uid") or "")
        for cue in (row or {}).get("cues") or []:
            try:
                idx = int(cue.get("index"))
            except (TypeError, ValueError):
                continue
            pair = (uid, idx)
            pairs.add(pair)
            by_cue.setdefault(idx, []).append(pair)
    return pairs, by_cue


def _entries(data: dict, project, tag: str):
    """``(élő kulcs, bejegyzés)`` párok.

    Újracsoportosítás után az ``uid`` megváltozhat: ha a mentett kulcs már nem
    él, de a feliratsor sorszáma egyértelműen egy mostani sorra mutat, arra
    képezzük le — így a felületen látszó döntés és a szinkron ugyanazt a sort
    használja. Kétértelmű (kettévágott) sorszámnál nem tippelünk.
    """
    live, by_cue = _live_pairs(project, tag)
    out = []
    for k, entry in (data.get("lines") or {}).items():
        pair = split_key(k)
        if pair is None:
            continue
        if live and pair not in live:
            options = by_cue.get(pair[1]) or []
            if len(options) != 1:
                continue
            pair = options[0]
        out.append((pair, entry or {}))
    return out


def labels(project, tag: str) -> dict[tuple[str, int], list[dict]]:
    """A szinkronlánc számára: ``(uid, feliratsor) -> kézi szereplő-megjelölések``.

    Csak a ``char`` megjelölés rendel hangot (a „képen kívül” és a „nem tudom”
    emberi jegyzet). A szótartomány nélküli mark a **teljes sor** hangját írja
    felül; a szótartományos mark a sort a szavak arányában vágja ketté, és a
    kijelölt rész más hangon szólal meg. Így a felülvizsgáló döntése nem marad
    jegyzet: a következő szinkronfutás tényleg azt a hangot használja.
    """
    out: dict[tuple[str, int], list[dict]] = {}
    for pair, entry in _entries(load(project, tag), project, tag):
        marks = [m for m in (entry.get("marks") or [])
                 if m.get("kind") == "char" and str(m.get("char") or "")]
        if marks:
            out[pair] = [dict(m) for m in marks]
    return out


def summary(project, tag: str) -> dict:
    data = load(project, tag)
    rows = list((data.get("lines") or {}).values())
    marks = [m for row in rows for m in (row.get("marks") or [])]
    return {
        "lines": len(rows),
        "marks": len(marks),
        "char": sum(1 for m in marks if m.get("kind") == "char"),
        "offscreen": sum(1 for m in marks if m.get("kind") == "offscreen"),
        "unknown": sum(1 for m in marks if m.get("kind") == "unknown"),
        "offsets": sum(1 for row in rows if float(row.get("offset") or 0.0)),
        "names": len(data.get("names") or {}),
        "updated_at": data.get("updated_at"),
    }
