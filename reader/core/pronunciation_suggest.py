"""Kiejtési szabályjavaslatok a könyv szövegéből.

A kiejtési szótár eddig kézzel készült: a felhasználó elolvasott egy hosszú
szöveget, megkereste a rosszul ejtett szavakat, és begépelt egy-egy szabályt.
Ez a modul azt a munkát végzi el: átvizsgálja a könyv szövegét, és
**javaslatot** tesz, mely szavakat érdemes felolvasási alakra cserélni.

Amit a javasló SZÁNDÉKOSAN NEM csinál: nem írja felül, amit az Auris már
magától jól csinál. A ``hungarian_numbers.normalize_hungarian`` kezeli a
számokat, sorszámokat, évszámokat, dátumokat, időket, pénznemeket,
mértékegységeket, római számokat, kezdőbetűket és az ismert rövidítéseket;
a ``hungarian_lexicon`` azokat. Ezért itt ezekből **egyetlen javaslat sem
keletkezik** — a normalizáló ugyanis már jobban tudja nálunk.

Amit felderít: azokat a szavakat, amelyeket a normalizáló érintetlenül hagy,
mert nem szám, nem rövidítés és nem kezdőbetű — nevezetesen **tulajdonneveket
és idegen szavakat**. Ezeket a neuralis beszédmotor magyar ortográfiából nem
találja biztosan, ezért felolvasási alak nélkül kiejtési hibát produkál.

Minden javaslat ``needs_review=True``, mert a felolvasási alak helyességét
ember dönti el — a rendszer nem tudja, hogy a „Szecső”-t „Szeccső”-nek vagy
„Sze-cső”-nek ejtik-e. A szabályt az ``experience.save_rule`` menti, a
Beállítások → Kiejtési szótár képernyőn.

Bemenet: a könyv bekezdései, vagy bármilyen szöveg. A fájlok kinyerését az
Auris parser csomagja végzi, ezért ez a modul csak szöveggel dolgozik.

Futtatás:
  .venv\\Scripts\\python.exe -m core.pronunciation_suggest <könyv-azonosító>
  .venv\\Scripts\\python.exe -m core.pronunciation_suggest --text „Szecső Gatsby…”
"""

from __future__ import annotations

import os
import re
from collections import Counter

from core import hungarian_lexicon, hungarian_numbers, hu_pronunciation

# A magyar szókincs (CharsiuG2P, ~55 000 szó). A fájl NEM a gitben van, mert
# az adat forrásfüggő licencű (a kód MIT, az adat nem); az első használatkor
# letöltjük a CharsiuG2P nyilvános tárházéról, és a `paths.user_path("lexicon")
# könyvtárba puffereljük. A VOICEAI_HUN_LEXICON változóval vagy a
# core/data/hun.tsv fájllal felülírható (fejlesztő/lefelé kompatibilitás).
# A hun.tsv formátuma „szó\talapalak” — mindkét oszlopot betöltjük, hogy a
# ragozott és az alapalak is ismert legyen.
LEXICON_ENV = "VOICEAI_HUN_LEXICON"
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LEXICON_DEFAULT = os.path.join(_DATA_DIR, "hun.tsv")

# A szótár hiteles forrása (CharsiuG2P, dicts/hun.tsv). A SHA-256 azonos a
# korábban repóban tárolt másolatéval, így a letöltés sértetlensége ellenőrizhető.
LEXICON_URL = (
    "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/hun.tsv"
)
LEXICON_SHA256 = (
    "168c4880a44e9f3d13225cc7e60bd3989ce7f52fb9444a3530640286ee55bf57"
)

# A hun.tsv sorai: felületi alak és alapalak, tabulátorral elválasztva.
_LEXICON_CACHE: set[str] | None = None
# A letöltést egyszer próbáljuk ki folyamat indulásakor; a `None` jelenti a
# még meg nem kísérelt állapotot, a `False` a sikertelen (offline) próbát.
_LEXICON_TRIED: bool | None = None


def _lexicon_candidates() -> list[str]:
    """A szótár lehetséges helyei, elsőbbségi sorrendben.

    A `paths.user_path("lexicon")` a letöltött (pufferelt) másolat helye —
    a repóban nincs benne, így ez az első saját hely, ahová kerülhet.
    """
    from core import paths

    root = str(paths.APP_DIR.parent.parent)
    return [
        os.environ.get(LEXICON_ENV, ""),
        LEXICON_DEFAULT,
        str(paths.user_path("lexicon", "hun.tsv")),
        os.path.join(root, "hub", "data", "hun.tsv"),
    ]


_LEXICON_LOCK = __import__("threading").Lock()


def _download_lexicon() -> str | None:
    """Letölti a szótárt a hiteles forrásból a pufferhelyre. A célút vagy None.

    A sértetlenséget SHA-256 ellenőrzi; hiba (offline, rossz tartalom) esetén
    None-t ad vissza, és a kiejtési javaslatok szóismeretes lépése üresen marad —
    az app nem bukik el. Az írás atomikus (tmp + os.replace), hogy egy
    félig írt fájl se legyen soha látható.
    """
    import hashlib
    import tempfile
    import urllib.request

    from core import paths

    target_dir = paths.user_path("lexicon")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    try:
        with urllib.request.urlopen(LEXICON_URL, timeout=30) as response:
            data = response.read()
    except Exception:
        return None
    if hashlib.sha256(data).hexdigest() != LEXICON_SHA256:
        # A forrás megváltozott vagy a letöltés sérült — nem használjuk fel.
        return None
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=str(target_dir), prefix=".hun.", suffix=".tsv.tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(data)
        os.replace(tmp_name, str(target_dir / "hun.tsv"))
    except OSError:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        return None
    return str(target_dir / "hun.tsv")


def _load_lexicon() -> set[str]:
    """A magyar szókincs betöltése, gyorsítótárazva. Üres halmaz, ha nincs meg.

    Ha egyik helyi példány sincs meg, egyszer megpróbáljuk letölteni (lásd
    `_download_lexicon`); offline gépen csendben üres marad.
    """
    global _LEXICON_CACHE, _LEXICON_TRIED
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    words: set[str] = set()
    for path in _lexicon_candidates():
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    for field in line.split("\t", 2)[:2]:
                        word = field.strip().lower()
                        if word and word.isalpha():
                            words.add(word)
        except OSError:
            continue
        if words:
            break
    if not words and not _LEXICON_TRIED:
        with _LEXICON_LOCK:
            if not _LEXICON_TRIED:
                # Első használatkor letöltjük; a `_LEXICON_TRIED` jegyzi, hogy
                # már próbáltuk (offline gépen ne próbálkozzon minden kérés).
                _LEXICON_TRIED = True
                downloaded = _download_lexicon()
                if downloaded:
                    try:
                        with open(downloaded, encoding="utf-8") as handle:
                            for line in handle:
                                for field in line.split("\t", 2)[:2]:
                                    word = field.strip().lower()
                                    if word and word.isalpha():
                                        words.add(word)
                    except OSError:
                        pass
    _LEXICON_CACHE = words
    return words


def has_lexicon() -> bool:
    """Van-e magyar szókincs a szóismeretes döntésekhez."""
    return bool(_load_lexicon())


# ------------------------------------------------------------ szóismeret

# Magyar toldalékok: a levágásuk után a tő ismert szó-e. A hub generátorból
# átvett lista, a ragozott alakok felismerésére („dédszüleink” -> „dédszül”).
_SUFFIXES = (
    "nak", "nek", "val", "vel", "ban", "ben", "ból", "ből", "ra", "re",
    "ról", "ről", "hoz", "hez", "höz", "ig", "ig", "ul", "ül", "os", "es",
    "ös", "an", "en", "in", "on", "un", "ön", "at", "et", "öt", "ot", "et",
    "ja", "je", "ük", "ük", "unk", "tok", "tek", "tak", "t", "k", "i", "a",
    "e", "é", "s", "n", "st",
)


def known_hungarian(word: str, lex: set[str], depth: int = 3) -> bool:
    """Szó (vagy legfeljebb ``depth`` toldalék levonása után) ismert magyar szó.

    A szókincsre támaszkodik, ezért hiánya esetén ``False``-sal tér vissza —
    a hívó ilyenkor inkább ne döntsön a szóismeret alapján.
    """
    if not lex:
        return False
    w = word.lower()
    if not w:
        return False
    frontier = [w]
    seen = {w}
    for _ in range(depth):
        nxt: list[str] = []
        for cand in frontier:
            if cand in lex:
                return True
            for suf in _SUFFIXES:
                if cand.endswith(suf) and len(cand) - len(suf) >= 3:
                    stub = cand[: -len(suf)]
                    if stub not in seen:
                        seen.add(stub)
                        nxt.append(stub)
        if not nxt:
            break
        frontier = nxt
    return any(c in lex for c in frontier)


# ------------------------------------------------------------ tokenizálás

_TOKEN_RE = re.compile(r"[A-ZÁÉÍÓÖŐÚÜŰ][A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű']*[.]?|[a-záéíóöőúüű][A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű']*[.]?|\d+(?:[.,]\d+)*\.?")
# A magyar y csak gy/ly/ny/ty-ban szabályos („gyors”, „nyelv”, „katona”);
# minden más y idegen betű („Gatsby”, „Rosemary”).
_NON_DIGRAPH_Y = re.compile(r"[^glnt]y", re.IGNORECASE)


def _foreign_letters(word: str) -> bool:
    """Idegen betűt (q/w/x vagy nem-digraph y) tartalmaz-e a szó."""
    w = word.lower()
    if "q" in w or "w" in w or "x" in w:
        return True
    return bool(_NON_DIGRAPH_Y.search(w))


def _covered(word: str) -> bool:
    """Már a normalizáló vagy a rövidítés-szótár kezeli-e ezt az alakot.

    Ez a függvény felel a nem-duplikációért: ha a normalizáló magától átírja
    a szót, vagy a lexikon rövidítésként ismeri, nincs mit javasolni — a
    számok, sorszámok, évszámok, dátumok, idők, pénznemek, mértékegységek és
    kezdőbetűk mind ide tartoznak.
    """
    stripped = word.rstrip(".")
    if not stripped:
        return True
    if len(stripped) == 1:
        # Kezdőbetű („J.”, „A.”): a normalizáló kontextusban ejti ki.
        return True
    if stripped.lower() in hungarian_lexicon.ABBREVIATIONS:
        return True
    if re.fullmatch(r"[IVXLCDM]+", stripped):
        # Római szám: önmagában a normalizáló nem ejti ki („XX.” -> "XX."),
        # mert szám és római szám között csak a mondat kontextusa dönt —
        # például „XX. század” vs. „a XX. fejezet”. A motor is tudja,
        # ezért itt nem hozunk rá szabályt.
        return True
    if hungarian_numbers.normalize_hungarian(word) != word:
        return True
    return False


def _mid_sentence_capital(paragraphs: list[str]) -> set[str]:
    """Mondat KÖZEPÉN nagybetűvel előforduló szavak — tulajdonnév-jel.

    A mondat eleji nagybetű önmagában nem jelent semmit, ezért azt nem
    jelöljük: a mondatkezdeti „A” vagy „Az” semmiképp sem tulajdonnév.
    """
    found: set[str] = set()
    cap_re = re.compile(
        r"\b([A-ZÁÉÍÓÖŐÚÜŰ][A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű']{1,})\b"
    )
    for para in paragraphs:
        for match in cap_re.finditer(para):
            before = para[: match.start(1)].rstrip()
            if before and before[-1] not in ".!?…":
                found.add(match.group(1))
    return found


# ------------------------------------------------------------ felderítés


def suggest(text: str | list[str], *, min_count: int = 1,
            limit: int = 300,
            extra_rules: dict[str, str] | None = None) -> list[dict]:
    """Kiejtési szabályjavaslatok a szövegből.

    ``text``: teljes szöveg vagy bekezdéslista. A visszatérő lista elemei a
    ``{source, replacement, category, note, needs_review, count, priority}``
    alakúak, gyakoriság szerint csökkenő sorrendben — a gyakori szavak a
    legfontosabbak, mert azok hallatszanak legtöbbször.

    ``extra_rules``: már mentett felhasználói szabályok (``source`` →
    ``replacement``), hogy azokból se keletkezzen új javaslat. A beépített
    szabályok mindig szerepelnek a szűrőben.

    A ``needs_review`` minden javaslatnál ``True``: a felolvasási alak
    helyességét ember dönti el. A ``priority`` 1 (biztosan kell szabály:
    idegen betű, lexikonban nem szereplő tulajdonnév) és 3 (magyar szó,
    lehet hogy a motor jól ejti) között van.
    """
    if isinstance(text, str):
        text = [text]
    paragraphs = [p for p in text if p and p.strip()]
    if not paragraphs:
        return []

    lex = _load_lexicon()
    # A már megoldott esetek: a beépített szótár + a felhasználó mentett
    # szabályai. Ezekből semmiképp ne javasoljunk újat, mert azzal a
    # felhasználó munkáját írnánk felül.
    solved = dict(hu_pronunciation.builtin_map())
    if extra_rules:
        solved.update({str(k): str(v) for k, v in extra_rules.items()})

    full = "\n".join(paragraphs)
    counts: Counter[str] = Counter(
        m.group(0) for m in _TOKEN_RE.finditer(full)
    )
    # A pont nélküli alakhoz tartozó összesített gyakoriság: a „Shakespeare”
    # és a „Shakespeare.” ugyanazt a szót jelöli, ezért a megjelenített
    # gyakoriság ezek összege. Ezt egyszer számoljuk ki, mert a számlálók
    # bejárása tokenenként négyzetes lenne egy nagy könyvnél.
    groups: Counter[str] = Counter()
    for token, n in counts.items():
        groups[token.rstrip(".")] += n
    caps = _mid_sentence_capital(paragraphs)

    proposals: dict[str, dict] = {}

    def add(source: str, category: str, note: str, priority: int) -> None:
        if not source or len(source) < 2 or source in proposals:
            return
        proposals[source] = {
            "source": source,
            # A felolvasási alak üres: a javaslat csak jelzi, hogy ezzel a
            # szóval kezdeni kell valamit; a felhasználó tölti ki.
            "replacement": "",
            "category": category,
            "note": note,
            "needs_review": True,
            "priority": priority,
        }

    for token, n in counts.items():
        if n < min_count or token in solved or _covered(token):
            continue

        word = token.rstrip(".")
        if len(word) < 2 or word in solved:
            continue

        occurrences = groups[word]
        in_lexicon = known_hungarian(word, lex)

        if _foreign_letters(word):
            if in_lexicon:
                # Idegen betűs, de ismert magyar szó („konyha”, „nyelv",
                # „gyors”): a motorral már jó, nincs teendő.
                continue
            add(
                word, "idegen szó",
                f"Idegen betűt tartalmaz, de nem ismert magyar szó: {word}. "
                f"Adj meg felolvasási alakot (pl. „Shakespeare” → „Sakszpér”).",
                1,
            )
        elif token[0].isupper() and not token.endswith(".") and token in caps:
            # Mondat közepén nagybetűs: vagy tulajdonnév, vagy elgépelt szó.
            if in_lexicon:
                add(
                    word, "név (magyar)",
                    f"Ismerősen hangzó magyar szó, mondat közepén nagybetűvel: "
                    f"{word}. Ha tulajdonnév és néma betűt tartalmaz (Thököly, "
                    f"Móricz), adj hozzá felolvasási alakot.",
                    3,
                )
            else:
                add(
                    word, "név",
                    f"Nem ismert szó, mondat közepén nagybetűvel: {word}. "
                    f"Ha a motor rosszul ejti, adj meg felolvasási alakot.",
                    2,
                )
        # A csak mondat elején nagybetűs, ismeretlen alakokat nem jelöljük:
        # a mondatkezdeti nagybetű önmagában nem utal tulajdonnévre.

    ranked = sorted(
        proposals.values(),
        key=lambda item: (item["priority"], -item.get("count", 1), item["source"]),
    )
    for item in ranked:
        item["count"] = groups[item["source"]]
    return ranked[: max(0, int(limit))]


def suggest_for_book(book_id: int, *, min_count: int = 1,
                     limit: int = 300) -> list[dict]:
    """Szabályjavaslatok egy, az adatbázisban tárolt könyv szövegéből.

    A könyv és az általános felhasználói szabályok is a szűrőben vannak,
    így a javaslatok csak a valóban új eseteket fedezik fel.
    """
    from core import database, experience

    with database.get_conn() as conn:
        chapters = conn.execute(
            "SELECT content FROM chapters WHERE book_id=? ORDER BY order_num",
            (book_id,),
        ).fetchall()
    text = "\n".join(row["content"] or "" for row in chapters).strip()
    if not text:
        raise ValueError(
            "A könyv szövege nem található — először importáld vagy töltsd be."
        )
    rules = {
        rule["source"]: rule["replacement"]
        for rule in experience.list_rules(book_id)
    }
    return suggest(text, min_count=min_count, limit=limit,
                   extra_rules=rules)


# ------------------------------------------------------------ CLI


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m core.pronunciation_suggest",
        description="Kiejtési szabályjavaslatok a könyv szövegéből.",
    )
    parser.add_argument("book_id", nargs="?", type=int,
                        help="a könyv azonosítója az adatbázisban")
    parser.add_argument("--text", help="szabad szöveg elemzése könyv nélkül")
    parser.add_argument("--min-count", type=int, default=1,
                        help="legalább ennyiszer forduljon elő (alap: 1)")
    parser.add_argument("--limit", type=int, default=100,
                        help="legfeljebb ennyi javaslat (alap: 100)")
    parser.add_argument("--json", action="store_true",
                        help="JSON-ban írja ki (menthető, feldolgozható)")
    args = parser.parse_args(argv)

    if args.text:
        items = suggest(args.text, min_count=args.min_count, limit=args.limit)
    elif args.book_id is not None:
        items = suggest_for_book(
            args.book_id, min_count=args.min_count, limit=args.limit
        )
    else:
        parser.error("adj meg könyv-azonosítót vagy --text-et")

    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0

    print(f"{len(items)} javaslat, szókincs: "
          f"{'betöltve' if has_lexicon() else 'NINCS (a szóismeretes döntések kihagyva)'}")
    print()
    for item in items:
        print(f"  ? [{item['category']:<13}] ×{item.get('count', 1):<4} "
              f"prioritás {item['priority']}: {item['source']}")
        print(f"      {item['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
