"""SRT beolvasás, szövegtisztítás és magyar TTS-szöveg előkészítés."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # pragma: no cover
        pass

#: ezek a sorok nem dialógusok (stáblista-fordítás)
SKIP_PATTERNS = ("feliratot fordította", "fordította:", "felirat készítése",
                 "subtitle by", "translated by")

_TAG = re.compile(r"<[^>]+>")
_ASSDIR = re.compile(r"^\{\\[^}]*\}")
_BRACKET_EMPTY = re.compile(r"\[\s*\]")
_BRACKET_LANG = re.compile(r"\[[A-ZÁÉÍÓÖŐÚÜŰ0-9]{2,}\]")
_PAREN = re.compile(r"\([^)]{1,40}\)")
#: a fordítás-prompt időtartam-jelzése (``[2.0s]``) — soha nem része a szövegnek
_SLOT_META = re.compile(r"\[\d+(?:[.,]\d+)?\s*s\]", re.IGNORECASE)
#: lógó fordulójel: a felirat a következő sor beszélőváltását a sor végén
#: jelöli (``- A? -`` / ``B.``) — a jel nem szöveg, a határ viszont valódi
_DANGLING_DASH = re.compile(r"(?:^|\s)-\s*$")

#: hangjegy-jelölés: a felirat így jelöli, hogy a sor **ének/zene**, nem
#: beszélt dialógus (mért példa: ``♪ Ez az álom... ♪``)
NOTES = "♪♫♬♩🎵🎶"

#: zenei/háttérhang-leírások: egymagukban nem dialógusok, ezért nem kapnak
#: szereplőt és nem mennek a TTS-be
SOUND_WORDS = frozenset({
    "music", "music playing", "music plays", "upbeat music", "sad music",
    "instrumental", "song", "singing", "humming", "whistling",
    "applause", "laughs", "laughter", "sighs", "sigh", "breathing",
    "silence", "static", "noise", "background noise", "wind", "rain",
    "thunder", "door slams", "phone rings", "gunshot", "explosion",
    "zene", "zene szól", "zenei aláfestés", "instrumentális", "ének",
    "dúdolás", "fütyülés", "taps", "nevetés", "sóhaj", "csend", "zaj",
    "háttérzaj", "szél", "eső", "mennydörgés", "lövés", "robbanás",
})

_BARE_PUNCT = re.compile(r"[.,!?…:;\-–—\"'“”„»«]+")


@dataclass
class Cue:
    """Egy feliratsor."""

    index: int
    start: float
    end: float
    text: str
    en: str = ""
    #: zene/dalszöveg/háttérhang sor: nem dialógus, ezért nincs szereplője,
    #: hangja és TTS-sora (lásd ``is_music``)
    music: bool = False

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def _t(sec: str) -> float:
    h, m, rest = sec.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def parse_srt(path: str | Path) -> list[Cue]:
    """SRT beolvasása (több sort összefűz, a formázást eldobja)."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n"))
    cues: list[Cue] = []
    for b in blocks:
        lines = [ln for ln in b.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        i = 0
        if lines[0].strip().isdigit():
            i = 1
        if i >= len(lines) or "-->" not in lines[i]:
            continue
        a, bb = lines[i].split("-->")[:2]
        text = " ".join(ln.strip() for ln in lines[i + 1:]).strip()
        cues.append(Cue(index=len(cues) + 1, start=_t(a.strip()),
                        end=_t(bb.strip().split()[0]), text=text))
    return cues


def one_line(text: str) -> str:
    """Egy sorba rendezett, formázás nélküli feliratszöveg.

    **Olvasáskor tisztít**: a fordítás-prompt időtartam-jelzése (``[2.0s]``)
    és a lógó fordulójel (sor végi ``-``) nem beszélt szöveg. A szűrés itt,
    a központi beolvasón van, hogy egy korábbi/idegen felirat se vihesse be
    őket a tervbe, a TTS-be vagy a QA elvárt szövegébe.
    """
    t = _TAG.sub(" ", text)
    t = _ASSDIR.sub("", t)
    t = t.replace("\\N", " ").replace("\\n", " ")
    t = _SLOT_META.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return _DANGLING_DASH.sub("", t).strip()


def ends_with_turn_dash(text: str) -> bool:
    """A sor végén lógó fordulójel áll-e (a következő sor új beszélőé)?"""
    t = _TAG.sub(" ", text)
    t = t.replace("\\N", " ").replace("\\n", " ").strip()
    return bool(_DANGLING_DASH.search(t))


def _bare(text: str) -> str:
    """A sor szövege jelölések (hangjegy, zárójel, írásjel) nélkül, kisbetűvel."""
    t = one_line(text)
    for n in NOTES:
        t = t.replace(n, " ")
    t = re.sub(r"[\[\](){}]", " ", t)
    t = _BARE_PUNCT.sub(" ", t)
    return re.sub(r"\s{2,}", " ", t).strip().lower()


def is_music(text: str) -> bool:
    """Zene/dalszöveg/háttérhang sor-e (tehát **nem** beszélt dialógus)?

    Három alak számít:

      * a sor két hangjegyjel között áll (``♪ Ez az álom... ♪``) — ez a
        felirat dalszöveg-jelölése;
      * a sor csak hangjegy/pont (``♪``, ``♪♪``);
      * a sor csak zenei vagy hangutánzó leírás (``[Music]``, ``(zene)``),
        feltéve, hogy a zárójeleken kívül nincs más szó — különben a
        ``[kiáltás] [válasz]`` típusú kétbeszélős sor elveszne.

    A mért eset (E04): két dalszöveg-sor kapott szereplőt és hangot, és ment
    volna a TTS-be; ezek innen kimaradnak.
    """
    t = one_line(text)
    if not t:
        return False
    notes = [i for i, ch in enumerate(t) if ch in NOTES]
    if notes:
        head = t[:notes[0]].strip(" -–—.,!?…")
        tail = t[notes[-1] + 1:].strip(" -–—.,!?…")
        if not head and not tail:
            return True
        if not _bare(t):
            return True
    bare = _bare(t)
    if not bare:
        return bool(notes) or t.strip("[](){}-–—. ") == ""
    if bare in SOUND_WORDS:
        return True
    words = bare.split()
    return len(words) <= 4 and all(w in SOUND_WORDS for w in words)


def strip_notes(text: str) -> str:
    """A hangjegy-jelek kivétele (a ♪ sosem része a kimondott szövegnek)."""
    t = one_line(text)
    for n in NOTES:
        t = t.replace(n, " ")
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip(" -–—").strip()


def tts_text(text: str) -> str:
    """A TTS-nek átadandó szöveg: jelölések, zárójeles megjegyzések nélkül."""
    t = strip_notes(text)
    t = _BRACKET_EMPTY.sub(" ", t)
    t = _BRACKET_LANG.sub(" ", t)      # [ARAB] nyelvjelölő
    t = _PAREN.sub(" ", t)             # (mosolyog) rendezői megjegyzés
    t = re.sub(r"[\[\]]", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def is_screen_text(text: str) -> bool:
    """Képernyőszöveg/helyszín-felirat: csupa nagybetű, és nincs beszéd."""
    t = one_line(text)
    t = re.sub(r"^\{[^}]*\}", "", t).strip(" -\t")
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 3 and not any(c.islower() for c in letters)


#: megszólítás-felismerés: a sor megszólítottja (név vagy rokonsági
#: megszólítás). A magyar felirat a megszólítást vesszővel és nagybetűs
#: névvel jelöli („Nem tudod a teljes történetet, Mark.”), illetve magában
#: álló szólítással („Kicsim.”). A döntés ezen a szöveges bizonyítékon áll:
#: a sor nem a megszólítotté.
_CAP = r"[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+"
_TITLE = r"(?:Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+"
#: záró/hangsúlyos megszólítás: „…, Mark.” / „…, Dr. Lewis.” — a név után
#: mondatvég is állhat („Már megtettem, Claire. A legjobb barátnőd…”)
_VOC_END = re.compile(r",\s*(?:" + _TITLE + r")?(" + _CAP + r")\s*(?:[.!?…](?=\s|$)|$)")
#: mondatközi megszólítás: „Hé, Claire, ne mondd ezt!”
_VOC_MID = re.compile(r",\s*(?:" + _TITLE + r")?(" + _CAP + r")\s*,")
#: mondat eleji megszólítás: „Claire, ne mondd ezt!”
_VOC_HEAD = re.compile(r"^(" + _CAP + r")\s*,")
#: név + cím/rokonságszó: „Garner nővér?”, „Fenwick úr!”
_VOC_TITLE = re.compile(r"(" + _CAP + r")\s+(nővér|úr|asszony|tanárnő|doktor|doki)\b")
#: záró rokonsági megszólítás: „Claire, lányom.” — a név nagybetűs, a
#: rokonságszó nem, ezért külön minta kell
_VOC_KIN_END = re.compile(r",\s*(kicsim|drágám|kedvesem|lányom|fiam|anyu|apu|"
                          r"húgom|bátyám|nővérem|testvérem|uram|asszonyom|"
                          r"hölgyem|barátom|barátnőm)\s*[.!?…]*$", re.IGNORECASE)
#: magában álló rokonsági szólítás: „Kicsim.”
_KINSHIP = frozenset({"kicsim", "drágám", "kedvesem", "lányom", "fiam",
                      "anyu", "anya", "apu", "apa", "húgom", "bátyám",
                      "nővérem", "testvérem", "uram", "asszonyom",
                      "hölgyem", "barátom", "barátnőm"})
#: nagybetűs szavak, amelyek magukban/mondat elején **nem** megszólítások
_NOT_NAMES = frozenset({
    "igen", "nem", "oké", "nos", "de", "és", "hát", "ó", "jaj", "persze",
    "talán", "semmi", "rendben", "szia", "köszönöm", "kérem", "sajnálom",
    "vége", "mi", "ki", "hol", "hogy", "ez", "az", "már", "még", "most",
    "csak", "itt", "ott", "mindig", "soha", "hé", "hahó", "figyelj",
    "halló", "nézd", "bocsánat", "istenem", "ugye", "minden", "valami",
})


def _name_ok(word: str, known: set[str] | None) -> bool:
    """Elfogadható-e a szó névként (nagybetűs, nem tiltott, ismert)?"""
    if not re.fullmatch(_CAP, word) or word.lower() in _NOT_NAMES:
        return False
    return known is None or word in known


def continues(prev_text: str, text: str) -> bool:
    """A sor az előző **félbemaradt** mondat folytatása-e?

    A diarizáció mondat közben is válthat címkét („…, amikor még mindketten”
    / „nagyon fiatalok voltatok.”). Ilyenkor az előző sor végén nincs
    mondatvégi írásjel, a folytatás pedig kisbetűvel indul — ez a két jel
    együtt bizonyítja, hogy ugyanaz a beszélő; egy önálló kérdés („Hol
    vagy?”) nem olvad össze a félbemaradt sorral.
    """
    p = one_line(prev_text).rstrip().rstrip("\"”»'")
    t = one_line(text).lstrip()
    return bool(p) and p[-1] not in ".!?…" and bool(t) and t[0].islower()


def vocative_names(text: str) -> set[str]:
    """A sor **biztos** megszólításai (vesszős név, név + cím).

    Ezek alkotják az epizód névkészletét: a gyenge alakok (mondat eleji szó,
    magában álló szó) csak akkor számítanak megszólításnak, ha a név máshol
    ilyen erős alakban is szerepel — különben az „Emlékszel,”, „Öhm,”,
    „Gyere!” típusú mondatkezdés tévesen névnek látszana.
    """
    t = one_line(text)
    out: set[str] = set()
    for rx in (_VOC_MID, _VOC_END):
        for m in rx.finditer(t):
            if m.group(1).lower() not in _NOT_NAMES:
                out.add(m.group(1))
    m = _VOC_TITLE.search(t)
    if m and m.group(1).lower() not in _NOT_NAMES:
        out.add(m.group(1))
    return out


def vocative(text: str, known: set[str] | None = None) -> str | None:
    """A sor **megszólítottja**, ha a szöveg megszólít valakit.

    A magyar felirat három alakban szólít meg:

      * záró/mondatközi vesszős név — ``…, Mark.`` / ``Hé, Claire, ne!``;
      * magában álló szólítás — ``Mark?`` / ``Kicsim.``;
      * név + cím vagy rokonságszó — ``Garner nővér?`` / ``Claire, lányom.``

    A visszaadott név a beszélő **partnere**, nem a beszélő: ezt használja a
    klaszterezés a rövid jelenet-töredékek hozzárendeléséhez. A bizonytalan
    eseteket („Nos,”, „Vége.”) kihagyjuk — a téves megszólítás rosszabb,
    mint a hiányzó. A ``known`` az epizód bizonyított névkészlete
    (``vocative_names``): ha meg van adva, a gyenge alakok csak abból
    választhatnak.
    """
    t = one_line(text)
    if not t:
        return None
    for rx in (_VOC_MID, _VOC_END):
        for m in rx.finditer(t):
            word = m.group(1)
            if word.lower() not in _NOT_NAMES:
                return word
    m = _VOC_TITLE.search(t)
    if m and m.group(1).lower() not in _NOT_NAMES:
        return m.group(1)
    m = _VOC_HEAD.match(t)
    if m and _name_ok(m.group(1), known):
        return m.group(1)
    m = _VOC_KIN_END.search(t)
    if m:
        return m.group(1)
    word = t.strip(" .,!?…:;\"'„”")
    if word and " " not in word:
        low = word.lower()
        if low in _KINSHIP:
            return word
        # Magában álló nagybetűs szó csak akkor szólítás, ha kérdeznek vagy
        # kiáltanak vele („Mark?”, „Claire!”); a puszta pont („Vége.”,
        # „Beszélek.”) egy egyszavas mondat, nem megszólítás.
        if t.rstrip()[-1:] in "?!" and _name_ok(word, known):
            return word
    return None


_BRACKET_SEG = re.compile(r"\[[^\]]+\]")


def _dialogue_brackets(t: str):
    """A zárójeles blokkok, amelyek valódi beszédet tartalmaznak.

    A ``[ARAB]``-hoz hasonló nyelvjelölők (csupa nagybetű/digit) nem
    beszéd-részek, azok nem választhatnak szét egy soron.
    """
    out = []
    for m in _BRACKET_SEG.finditer(t):
        inner = m.group(0)[1:-1].strip()
        if not inner or _BRACKET_LANG.fullmatch(f"[{inner}]"):
            continue
        out.append(m)
    return out


def has_bracket_dialogue(text: str) -> bool:
    """Két zárójeles blokkra bontott kétbeszélős sor:
    „[kiáltás] [válasz]” — a feliratkészítő így választja el a beszélőket.
    """
    return len(_dialogue_brackets(one_line(text))) >= 2


def split_bracket_dialogue(text: str) -> tuple[str, str] | None:
    """A „[A] [B.]” sort a zárójelek határán ketté; ha nem ilyen, None."""
    t = one_line(text)
    segs = _dialogue_brackets(t)
    if len(segs) < 2:
        return None
    left, right = t[:segs[1].start()].strip(), t[segs[1].start():].strip()
    return (left, right) if left and right else None


def starts_new_turn(text: str) -> bool:
    """Új beszélő fordulója kezdődik-e a sor elején?

    A felirat a forduló elejét „- ” jellel vagy zárójeles blokkal jelöli.
    Két ilyen sor **nem** tartozhat ugyanahhoz a megnyilatkozáshoz (kivéve a
    kisbetűs folytatást, amit az ``utterances._continuation`` enged).
    """
    t = one_line(text).lstrip()
    return t.startswith(("-", "["))


def has_dash_dialogue(text: str) -> bool:
    """Kétbeszélős feliratsor: két vagy több gondolat-rész.

    A feliratkészítő a „- ” jellel választja el a fordulókat; ez a **szöveges
    bizonyíték** önmagában elég a szétvágáshoz (lásd ``split_dialogue_parts``).
    """
    parts = split_dash_dialogue(text)
    return bool(parts and parts[0] and parts[1])


#: fordulóhatár: „- ” után nagybetű, zárójel, idézőjel vagy szám következik.
#: A „Hát - nem tudom.” típusú gondolatjelet ez nem vágja szét. A szóköz a
#: jel után **nem kötelező**: a magyar feliratok többsége „-Szia!” alakban ír,
#: és a szóköz nélküli fordulatjel kihagyása két színészt tett egy hangra
#: (mért hiba: a 911-jelenet és az iskolai jelenet párbeszédei).
_TURN_DASH = re.compile(r"(?:^|\s)[-–—]\s*(?=[A-ZÁÉÍÓÖŐÚÜŰ„\"\[]|\d)")


def split_dialogue_parts(text: str) -> list[str] | None:
    """Egy feliratsor **összes** beszélő-fordulója, szöveges jelölés alapján.

    Két alakot ismer:
      * zárójeles blokkok — ``[kiáltás] [válasz]`` (a felirat így jelöli a
        kiáltást és a választ), akár több fordulóval;
      * gondolatjel — ``- A? - B.`` (a „- ” után nagybetű/zárójel/szám áll).

    A régi lánc a vágást az akusztikus hasonlóságtól tette függővé: rövid,
    zenével kevert ablakokon ez a kapu rendszeresen elutasította a valódi
    fordulóváltást, és a két szereplő (gyakran ellentétes nemű) sora egy
    hangon szólalt meg. A szöveges jelölés erősebb bizonyíték, ezért a vágás
    elsődleges forrása; az akusztika csak a nem-mérést adja fordulónként.
    """
    t = one_line(text)
    segs = _dialogue_brackets(t)
    if len(segs) >= 2:
        parts = [m.group(0)[1:-1].strip(" -") for m in segs]
        head = t[:segs[0].start()].strip(" -")
        tail = t[segs[-1].end():].strip(" -")
        if head:
            parts[0] = f"{head} {parts[0]}".strip()
        if tail:
            parts[-1] = f"{parts[-1]} {tail}".strip()
        parts = [p for p in parts if p]
        return parts if len(parts) >= 2 else None
    parts = [p.strip(" -") for p in _TURN_DASH.split(t) if p.strip(" -")]
    return parts if len(parts) >= 2 else None


def split_dash_dialogue(text: str) -> tuple[str, str] | None:
    """A „- A? - B.” sort két részre bontja; ha nem ilyen, None.

    Ugyanazt a fordulatjel-felismerőt használja, mint a teljes szétvágás,
    különben a szóköz nélküli „-Szia!” alak itt is elveszne.
    """
    t = one_line(text)
    parts = [p.strip(" -–—") for p in _TURN_DASH.split(t)
             if p.strip(" -–—")]
    if len(parts) < 2:
        return None
    return parts[0], " ".join(parts[1:])


def cut_words(text: str, frac: float) -> tuple[str, str] | None:
    """Szöveg vágása szóhatáron, írásjelet előnyben részesítve."""
    punct = ".!?…,;:-–"
    words = text.split()
    if len(words) < 2:
        return None
    target = max(1, min(len(words) - 1, int(round(frac * len(words)))))
    best = target
    for d in range(0, 4):
        hit = next((c for c in (target - d, target + d)
                    if 1 <= c <= len(words) - 1 and words[c - 1][-1] in punct), None)
        if hit is not None:
            best = hit
            break
    left, right = " ".join(words[:best]).strip(), " ".join(words[best:]).strip()
    if not left or not right:
        return None
    return left, right


def ends_sentence(text: str) -> bool:
    return one_line(text).rstrip().endswith((".", "!", "?", "…", ":", "-"))


def starts_lower(text: str) -> bool:
    t = one_line(text).lstrip("- ").strip()
    return bool(t) and t[0].islower()


def prepare_hu(text: str) -> str:
    """Magyar TTS-előkészítés: számok, rövidítések, kiejtés (Auris szótár).

    Az Auris ``prepare_text``-jét használjuk, hogy a magyar szám- és
    rövidítés-kiejtés egységes legyen (pl. „12 óra” -> „tizenkét óra”).
    Ha az Auris nincs elérhető útvonalon, a szöveget változatlanul adjuk
    vissza — a szintézis így is működik, csak a számok kiejtése lesz nyers.
    """
    if not text:
        return ""
    reader = os.environ.get("DUBSTUDIO_AURIS_READER", "")
    cands = [Path(reader)] if reader else []
    cands += [Path(__file__).resolve().parents[2] / "reader"]
    for c in cands:
        if (c / "core" / "local_engines.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            try:
                from core.local_engines import prepare_text  # type: ignore

                out = prepare_text(text, "hu", True)
                if out:
                    return str(out)
            except Exception as e:      # pragma: no cover
                print(f"! prepare_text nem elérhető ({e}); nyers szöveg")
            break
    return text


@dataclass
class SubtitlePair:
    """Magyar + eredeti felirat páronként (időátfedés szerint)."""

    hun: list[Cue] = field(default_factory=list)
    eng: list[Cue] = field(default_factory=list)

    def match(self, cue: Cue) -> Cue | None:
        best, best_ov = None, 0.0
        for c in self.eng:
            if c.end <= cue.start or c.start >= cue.end:
                continue
            ov = min(c.end, cue.end) - max(c.start, cue.start)
            if ov > best_ov:
                best, best_ov = c, ov
        return best
