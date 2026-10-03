"""Beépített magyar kiejtések.

A kiejtési szótár a szöveg írásmódját felolvasási alakra cseréli. Ez a modul
olyan szavakat gyűjt, amelyeket a neuralis beszédmotor a magyar
ortográfiából még nem találja biztosan: idegen eredetű nevek, néma betűt
tartalmazó magyar nevek, és szövegben gyakori rövidítések.

Fontos: a felhasználó által mentett szabály **mindig** felülírja a beépítettet.
A beépített lista tehát nem foglal helyet a könyvtárban, és nem is állítható
be kapcsolóval: csak alapértéket ad, amit a Beállítások → Kiejtési szótár
képernyőn lehet egy kattintással saját szabályra cserélni.

Minden bejegyzés ellenőrizhető a „Felolvasandó szöveg ellenőrzése" gombbal,
ezért a lista bővíthető anélkül, hogy a kódot módosítani kellene.

Ragozás: a felolvasási alak csak teljes szóként illeszkedik, toldalék nélkül.
Ennek oka, hogy a toldó a *helyes* alakhoz tartozik, nem a szótőhöz: a
„Móricz" → „Móric" átírás toldalékaiból a „Móriczot" helyett „Móricot" keletkezne,
ami rossz magyar. Ha egy szót ragozva is alkalmazni akarsz, mentsd el saját
szabályként — azoknál a ragozás változatlanul működik.
"""

import unicodedata

# (szövegben így, így olvassa, kategória, megjegyzés)
BUILTIN_RULES: tuple[tuple[str, str, str, str], ...] = (
    # ── Magyar nevek: néma h (külföldi név magyar hanglekése) ──────────────
    ("Bethlen", "Betlen", "név", "Bethlen Gábor: a h néma magyarban."),
    ("Thököly", "Tököly", "név", "Thököly Imre: a h néma magyarban."),
    ("Kossuth", "Kosut", "név", "Kossuth Lajos: a sh magyar sz hang."),
    ("Móricz", "Móric", "név", "Móricz Zsigmond: a cz magyar sz."),
    ("Kőrösfői-Móricz", "Kőrösfői-Móric", "név", "Móricz Zsigmond így írja magát."),
    ("Mikszáth", "Mikszát", "név", "Mikszáth Kálmán: a th magyar t."),

    # ── Magyar nevek: néma betű a szó végén ───────────────────────────────
    ("Zrínyi", "Zrinyi", "név", "Ilona királyné neve magyar írásmód szerint."),

    # ── Idegen nevek magyar alakban ───────────────────────────────────────
    ("Shakespeare", "Sakszpér", "idegen név", "William Shakespeare."),
    ("Napoleon", "Napóleon", "idegen név", "Bonaparte: magyar írásmód."),
    ("Columbus", "Kolumbus", "idegen név", "Kolumbus Kristóf."),
    ("Salome", "Szalome", "idegen név", "Salome, a bibliai alak magyar alakja."),
    ("Casanova", "Kaszanova", "idegen név", "Casanova, az alak nevét magyarul írjuk."),
    ("Rasputin", "Raszputyin", "idegen név", "Az orosz írásmód szerinti alak."),
    ("Copernicus", "Kopernikus", "idegen név", "Kopernikusz."),
    ("Tchaikovsky", "Csajkovszkij", "idegen név", "Csaikovszkij magyar alakja."),

    # ── Helynevek ─────────────────────────────────────────────────────────
    ("Zsigetvár", "Szigetvár", "helynév", "Zsiget és vár két szó."),

    # ── Rövidítések ────────────────────────────────────────────────────────
    ("stb.", "s t b", "rövidítés", "és így tovább."),
    ("pl.", "például", "rövidítés", "Az előző mondat folytatása."),
    ("u.i.", "úgy így", "rövidítés", "Úgy is."),
    ("ill.", "illetve", "rövidítés", "vagyis."),
    ("öv.", "óv", "rövidítés", "óvatos, óvatosság rövidítése."),
    ("Kft.", "Kft", "rövidítés", "korlátolt felelősségű társaság."),
    ("Zrt.", "Zrt", "rövidítés", "zártkörűen működő részvénytársaság."),
    ("Bp.", "Budapest", "rövidítés", "Budapest rövidítése."),
    ("kr.", "kor", "rövidítés", "Század, korábbi évszázad utána."),

    # ── Szövegben gyakori idegen betűszavak ───────────────────────────────
    ("vs.", "versusz", "rövidítés", "Összehasonlítás jele."),
    ("kb.", "körülbelül", "rövidítés", "Mennyiség megadásánál."),
    ("max.", "maximum", "rövidítés", "Legnagyobb érték."),
    ("min.", "minimum", "rövidítés", "Legkisebb érték."),
)

# Gyorsítótárazott szótár: írásmód -> felolvasási alak.
_MAP = {source: replacement for source, replacement, _, _ in BUILTIN_RULES}

CATEGORIES: tuple[str, ...] = ("név", "idegen név", "helynév", "rövidítés")

# Csak az egész szóra illeszkedő beépített szabályok (lásd a modul docstringjét).
EXACT_ONLY: frozenset[str] = frozenset(_MAP)


def builtin_map() -> dict[str, str]:
    """Beépített szabályok írásmód -> felolvasási alak formájában."""
    return dict(_MAP)


def builtin_count() -> int:
    return len(BUILTIN_RULES)


def _fold(text: str) -> str:
    """Kisbetűsít és ékezetet bont, hogy a keresés „sakszper" alakban is találjon."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def search(query: str = "", category: str = "") -> list[dict]:
    """Beépített szabályok keresésre szűkítve.

    A keresés a forrásszóra, a felolvasási alakra és a megjegyzésre is
    illeszkedik, kis- és nagybetűtől és ékezetektől függetlenül, hogy
    magyarul és latinul is megtalálható legyen (pl. „sakszper" és
    „Shakespeare" ugyanazt a szabályt adja).
    """
    needle = _fold(query or "")
    out = []
    for source, replacement, kind, note in BUILTIN_RULES:
        if category and kind != category:
            continue
        if needle and not any(
            needle in _fold(text) for text in (source, replacement, note)
        ):
            continue
        out.append(
            {"source": source, "replacement": replacement, "category": kind, "note": note}
        )
    return out