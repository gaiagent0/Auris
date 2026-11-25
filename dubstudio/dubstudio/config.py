"""Projekt- és epizód-konfiguráció.

Egy projekt egy film/sorozat-evad. A ``project.json`` adja meg, hol vannak a
nyers epizódok, a feliratok, a leválasztott M&E sáv és a TTS-modellek.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # pragma: no cover
        pass

# a csomag gyökere (dubstudio/dubstudio/config.py -> dubstudio/)
APP_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Project:
    """Egy szinkronprojekt minden útvonala és beállítása."""

    name: str
    season: int = 1
    root: Path = field(default_factory=lambda: APP_ROOT)
    data_dir: Path = field(default_factory=lambda: APP_ROOT / "data")
    subs_dir: Path = field(default_factory=lambda: APP_ROOT / "subs")
    # a leválasztott énekbeszéd-sáv mintái; {tag} = E05, {ep} = 5
    vocals_globs: list[str] = field(default_factory=list)
    # a zene+effekt (M&E) sáv mintái; ide kerül rá a magyar dialógus
    me_globs: list[str] = field(default_factory=list)
    source_lang: str = "heb"
    original_audio: int = 0
    tts_sr: int = 44100
    models: dict = field(default_factory=dict)
    tuning: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    #: szövegminőség-ellenőrző pont a szintézis ELŐTT (helyi LLM + szabályok;
    #: lásd dubstudio/textcheck.py) — pl. {"enabled": true, "base":
    #: "http://127.0.0.1:11435", "model": "qwen3.5:9b", "required": true}
    llm: dict = field(default_factory=dict)
    #: ASR-visszaellenőrzés beállítása, pl. {"mode": "hybrid"} — a módok:
    #: whisper (teljes sáv a magyar finetune-nal), parakeet (gyors ONNX),
    #: hybrid (Parakeet elsődleges + Whisper-HU megerősítés a gyanús sorokon)
    asr: dict = field(default_factory=dict)
    #: ide kerül minden közbenső és kimeneti fájl (a médiaforrás könyvtárát
    #: nem írjuk tele a saját artefaktjainkkal)
    cache_dir: Path = field(default_factory=lambda: APP_ROOT / "work")
    #: "series" vagy "movie" — a film egyetlen E01 egységként fut, a sorozat
    #: epizódokból áll és növekményesen bővíthető
    kind: str = "series"
    #: sorozatfigyelés beállításai: {"auto_process": bool, "poll_s": int}
    series: dict = field(default_factory=dict)
    #: nyers epizód előkészítése (felirat fordítás + hangsáv-szétválasztás);
    #: lásd dubstudio/intake.py és a README „Intake” szakaszát
    intake: dict = field(default_factory=dict)
    #: forduló-alapú beszélő-azonosítás a teljes énekbeszéd-sávon; lásd
    #: dubstudio/diarization.py (pyannote szegmentáció + CAM++ embedding)
    diarization: dict = field(default_factory=dict)
    #: médiaszerver-modul: {"path": "\\\\NAS\\Media", "deliver": true} — a
    #: forrás innen másolódik a projektbe, a kész szinkron pedig vissza
    media_server: dict = field(default_factory=dict)

    @property
    def work(self) -> Path:
        return self.cache_dir

    @property
    def out(self) -> Path:
        return self.cache_dir / "out"

    @property
    def cast_file(self) -> Path:
        return self.cache_dir / f"cast-{self.name.replace(' ', '_')}.json"

    def model(self, key: str, default: str | Path = "") -> Path:
        v = self.models.get(key) or default
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def ep_dir(self, tag: str) -> Path:
        d = self.work / tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def vocals(self, tag: str) -> Path:
        """A leválasztott énekbeszéd-sáv (a beszélő-azonosítás alapja)."""
        try:
            return self._find(self.vocals_globs, tag, "leválasztott énekbeszéd-sáv")
        except FileNotFoundError:
            # az intake alapból ide ír; a projekt saját glob-mintái elsőbbséget
            # kapnak, de egy új projektnél nem kell külön mintát beállítani
            fallback = self.work / tag.upper() / "vocals.wav"
            if fallback.is_file():
                return fallback
            raise

    def _find(self, globs: list[str], tag: str, what: str) -> Path:
        ep = int("".join(c for c in tag if c.isdigit()) or 0)
        for pat in globs:
            q = pat.format(tag=tag.upper(), ep=ep, ep2=f"{ep:02d}")
            base = Path(q)
            cands = ([base] if base.is_absolute() else
                     [self.work / base, self.root / base])
            for c in cands:
                hits = sorted(c.parent.glob(c.name)) if any(ch in c.name for ch in "*?[") else ([c] if c.is_file() else [])
                if hits:
                    return hits[0]
        raise FileNotFoundError(
            f"nincs {what} ehhez: {tag} (keresett minták: {globs})")

    def me_bed(self, tag: str) -> Path:
        """A zene+effekt sáv (ebbe kerül a magyar dialógus)."""
        try:
            return self._find(self.me_globs, tag, "M&E sáv")
        except FileNotFoundError:
            fallback = self.work / tag.upper() / "me.wav"
            if fallback.is_file():
                return fallback
            raise

    def subtitle(self, tag: str, lang: str) -> Path:
        return self.subs_dir / f"{tag.upper()}_{lang}.srt"

    def episode_media(self, tag: str) -> Path:
        if self.kind == "movie":
            # Filmnél nincs SxxExx kötelező névforma: a data_dir első
            # támogatott médiafájlja a film. A tag (E01) csak a work-útvonalhoz
            # kell, a felhasználói felület egyetlen filmként kezeli.
            suffixes = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts",
                        ".ts", ".m4v"}
            try:
                hits = sorted(p for p in self.data_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in suffixes)
            except OSError:
                hits = []
            if hits:
                return hits[0]
            raise FileNotFoundError(
                f"nincs filmfájl ({self.data_dir}); tölts be egy videót")
        ep = int("".join(c for c in tag if c.isdigit()) or 0)
        # a könyvtárnézet és a médiafeltöltés több konténert is elfogad; a
        # keresés ugyanezeket a kiterjesztéseket használja
        suffixes = (".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts",
                    ".ts", ".m4v")
        pats = [f"*S{self.season:02d}E{ep:02d}*{suffix}" for suffix in suffixes]
        pats += [f"*E{ep:02d}*{suffix}" for suffix in suffixes]
        for p in pats:
            hits = sorted(self.data_dir.glob(p))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"nincs epizódfájl ({self.data_dir}, S{self.season:02d}E{ep:02d})")


def resolve_project(value: str | Path | None = None) -> Path:
    """A használandó projektfájl feloldása (a repó nem tartalmaz készt).

    Sorrend: a megadott érték → ``DUBSTUDIO_PROJECT`` → az egyetlen
    ``projects/*.json`` a ``DUBSTUDIO_ROOT`` (vagy a csomag) alatt. Ha nincs
    vagy több van, ``FileNotFoundError``-t ad használható üzenettel.
    """
    for raw in (value, os.environ.get("DUBSTUDIO_PROJECT")):
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"nincs ilyen projektfájl: {path}")
        return path.resolve()
    root = Path(os.environ.get("DUBSTUDIO_ROOT") or APP_ROOT)
    folder = root / "projects"
    found = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if len(found) == 1:
        return found[0].resolve()
    if not found:
        init = APP_ROOT.parent / "scripts" / "check_dubstudio.py"
        raise FileNotFoundError(
            "nincs projekt a projects/ mappában. Hozz létre egyet:\n"
            f"  {sys.executable} {init} --init \"Film címe\"\n"
            "vagy a felületen: /szinkron → új film")
    names = ", ".join(p.stem for p in found)
    raise FileNotFoundError(
        f"több projekt van ({names}); add meg a --project opcióval.")


def load_project(path: str | Path) -> Project:
    """``project.json`` betöltése; minden relatív út a fájl mappájához képest."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"nincs projektfájl: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    base = p.parent
    root = (base / raw.get("root", ".")).resolve()
    cache = (base / raw["cache_dir"]).resolve() if raw.get("cache_dir") \
        else (APP_ROOT / "work").resolve()

    def rp(key: str, default: Path) -> Path:
        """A médiakönyvtárak a **root**-hoz képest értendők (nem a JSON-hoz)."""
        v = raw.get(key)
        if not v:
            return default
        p = Path(str(v)).expanduser()
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    return Project(
        name=raw.get("name", p.stem),
        season=int(raw.get("season", 1)),
        root=root,
        data_dir=rp("data_dir", root / "data"),
        subs_dir=rp("subs_dir", root / "subs"),
        vocals_globs=[str(g) for g in raw.get("vocals_globs", [])],
        me_globs=[str(g) for g in raw.get("me_globs", [])],
        source_lang=raw.get("source_lang", "heb"),
        original_audio=int(raw.get("original_audio", 0)),
        tts_sr=int(raw.get("tts_sr", 44100)),
        models=dict(raw.get("models", {})),
        tuning=dict(raw.get("tuning", {})),
        gate=dict(raw.get("gate", {})),
        llm=dict(raw.get("llm", {})),
        asr=dict(raw.get("asr", {})),
        cache_dir=cache,
        kind=(str(raw.get("kind", "series")).strip().lower()
              if str(raw.get("kind", "series")).strip().lower() in ("series", "movie")
              else "series"),
        series=dict(raw.get("series", {})),
        intake=dict(raw.get("intake", {})),
        diarization=dict(raw.get("diarization", {})),
        media_server=dict(raw.get("media_server", {})),
    )
