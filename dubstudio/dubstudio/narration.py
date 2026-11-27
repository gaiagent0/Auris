"""Egyszereplős videó narráció — **önálló lánc**, a film/sorozat pipeline nélkül.

A szinkronlánc (``pipeline.py``) szereplőket keres, diarizál, hangot oszt ki,
képi bizonyítékot gyűjt és utómunkát (M&E, ducking) végez. Egy egyszereplős
videóhoz — előadás, bemutató, saját felvétel — ebből semmi nem kell: a beszélő
végig ugyanaz, a hangot a felhasználó választja (kész hang vagy klón).

Ez a modul ezért a saját, rövid láncát járja:

    média (fájl vagy YouTube) → szöveg (felirat vagy ASR) → folyamatos blokkok
    → szintézis egy hangon → kész hangsáv → mux

A **folyamatosság** a lényeg: a szöveget nem feliratsoronként, hanem összefüggő
blokkokban (mondathatárig, jellemzően 15–30 s) szintetizáljuk. Az XTTS a blokk
teljes szövegét egyetlen generálásban mondja el, így a hang lejtése nem
nullázódik minden sor elején. A blokk a saját időablakához igazodik (legfeljebb
``max_tempo`` gyorsítással), ezért a narráció végig együtt marad a képpel.

Nincs benne: diarizáció, szereplő-azonosítás, hangkiosztás, képi ASD,
felülvizsgálat, M&E-szétválasztás, ducking, QA-kapu. A kimenet egy hangsáv,
a riport csak a méréseket tartalmazza.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import asrcheck, audio, media, mix, subs, youtube
from .config import APP_ROOT
from .voices import BY_ID, PiperEngine, SupertonicEngine, XTTSEngine

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # pragma: no cover
        pass

MEDIA_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".m4v"}

#: egy blokk ennyi karakterig / másodpercig nő (a mondathatár a kedvező vágás)
BLOCK_CHARS = 320
BLOCK_MAX_S = 26.0
#: ennél nagyobb szünetnél új blokk kezdődik (a természetes szünet megmarad)
GAP_SPLIT_S = 1.0
#: a blokk csak akkor gyorsítható, hogy beférjen az ablakába
MAX_TEMPO = 1.35
#: a kész hangsáv célja (ugyanaz, mint a szinkronláncé)
OUT_SR = 44100


class NarrationError(RuntimeError):
    """A narrációs lánc használható hibaüzenete."""


@dataclass
class Block:
    """Egy folyamatos megszólalás: több feliratsor egy TTS-hívásban."""

    i: int
    start: float
    end: float
    text: str
    wav: Path | None = None
    fitted: float = 0.0
    tempo: float = 1.0
    cached: bool = False

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"i": self.i, "start": round(self.start, 3),
                "end": round(self.end, 3), "text": self.text,
                "wav": None if self.wav is None else str(self.wav),
                "fitted": round(self.fitted, 3), "tempo": round(self.tempo, 3),
                "cached": self.cached}


@dataclass
class Options:
    """A narráció hangolható értékei (mind megadható a CLI-ból is)."""

    language: str = "hu"
    block_chars: int = BLOCK_CHARS
    block_max_s: float = BLOCK_MAX_S
    gap_split: float = GAP_SPLIT_S
    max_tempo: float = MAX_TEMPO
    out_sr: int = OUT_SR
    target_lufs: float = mix.TARGET_LUFS
    peak_db: float = mix.TARGET_PEAK_DB
    limit: int = 0
    audio_stream: int = 0
    keep_original: bool = True
    title: str = "Magyar narráció"


# ── segédek ─────────────────────────────────────────────────────────────────
def slugify(text: str, fallback: str = "narráció") -> str:
    """Fájlnév-barát azonosító (ékezet nélkül, kötőjeles, max 48 karakter)."""
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:48] or fallback


def _run(cmd: list[str], timeout: float | None = 300) -> subprocess.CompletedProcess:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise NarrationError(
            f"{Path(str(cmd[0])).name} hiba: "
            f"{p.stderr.decode('utf-8', 'replace')[-500:]}")
    return p


def duration(path: str | Path) -> float:
    """A média hossza másodpercben (ffprobe; hiba esetén 0)."""
    try:
        p = _run([media.FFPROBE, "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nw=1:nk=1", path], timeout=60)
        return max(0.0, float(p.stdout.decode("utf-8", "replace").strip() or 0))
    except (NarrationError, ValueError):
        return 0.0


def model_paths(overrides: dict | None = None) -> dict:
    """A hangmotorok útvonalai: alapérték + ``narration.json`` + felülírás.

    A Supertonic/Piper az Auris közös modellmappájából jön (mint a
    szinkronláncnál); az XTTS klónmodell helye a ``NARRATION_XTTS``
    környezeti változóból, a ``narration.json`` ``models.xtts`` kulcsából vagy
    a szokásos munkamappából. A klónhoz referenciafelvétel is kell
    (``NARRATION_XTTS_REF`` / ``models.xtts_ref`` / ``--ref``).
    """
    from .newproject import default_models

    models = default_models()
    doc = {}
    cfg = APP_ROOT / "narration.json"
    if cfg.is_file():
        try:
            doc = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {}
    models.update({k: str(v) for k, v in (doc.get("models") or {}).items() if v})
    models.update({k: str(v) for k, v in (overrides or {}).items() if v})
    if not models.get("xtts"):
        xtts = (os.environ.get("NARRATION_XTTS")
                or os.environ.get("DUBSTUDIO_XTTS") or "")
        if not xtts:
            for cand in (APP_ROOT / "work" / "models" / "xtts",
                         APP_ROOT / "models" / "xtts"):
                if (cand / "metadata.json").is_file():
                    xtts = str(cand)
                    break
        if not xtts:
            # utolsó fogódzó: egy meglévő projekt már beállított XTTS-útja
            # (csak az útvonalat vesszük át, a filmes logikát nem)
            for doc_path in sorted((APP_ROOT / "projects").glob("*.json")):
                try:
                    doc_models = json.loads(doc_path.read_text(encoding="utf-8")) \
                        .get("models") or {}
                except (OSError, ValueError):
                    continue
                cand = str(doc_models.get("xtts") or "")
                if cand and (Path(cand) / "metadata.json").is_file():
                    xtts = cand
                    if not models.get("xtts_ref") and doc_models.get("xtts_ref"):
                        models["xtts_ref"] = str(doc_models["xtts_ref"])
                    if not models.get("xtts_stubs") and doc_models.get("xtts_stubs"):
                        models["xtts_stubs"] = str(doc_models["xtts_stubs"])
                    break
        if xtts:
            models["xtts"] = xtts
    if not models.get("xtts_ref"):
        ref = os.environ.get("NARRATION_XTTS_REF") or ""
        if ref:
            models["xtts_ref"] = ref
    if not models.get("xtts_stubs"):
        stubs = os.environ.get("NARRATION_XTTS_STUBS") or ""
        if stubs:
            models["xtts_stubs"] = stubs
    return models


def clone_available(models: dict | None = None) -> bool:
    """Van-e használható XTTS klónmodell a gépen?"""
    models = models or model_paths()
    return bool(models.get("xtts")) and (Path(models["xtts"]) / "metadata.json").is_file()


def resolve_voice(voice: str | None, ref: str | Path | None,
                  models: dict) -> tuple:
    """A hang és a motor kiválasztása; klónhoz referenciafelvétel kell.

    ``ref`` nélkül kész hang szólal meg (a klónhang nem automatikus, mert az
    egy konkrét ember hangja), ``ref``-fel az XTTS klón.
    """
    vid = str(voice or "").strip()
    if ref:
        v = BY_ID.get(vid or "XTTS_istvan01")
        if v is None or v.engine != "xtts":
            raise NarrationError(f"A klónhanghoz XTTS hang kell, nem {vid or '?'}.")
        sample = Path(str(ref)).expanduser()
        if not sample.is_file():
            raise NarrationError(f"Nincs ilyen referenciafelvétel: {sample}")
        if not clone_available(models):
            raise NarrationError(
                "Nincs XTTS klónmodell. Állítsd be a NARRATION_XTTS környezeti "
                "változót vagy a narration.json models.xtts kulcsát.")
        return v, sample
    v = BY_ID.get(vid or "SUP_M1")
    if v is None:
        raise NarrationError(f"Ismeretlen hang: {vid}")
    if v.clone:
        raise NarrationError(f"A(z) {v.vid} klónhang — add meg a referenciafelvételt (--ref).")
    return v, None


# ── 1. média ────────────────────────────────────────────────────────────────
def prepare_media(source: str, work: Path, opts: Options,
                  force: bool = False) -> dict:
    """A forrás előkészítése: YouTube-letöltés vagy helyi fájl.

    Visszatérés: ``{"media", "title", "subtitle", "duration", "url"}``.
    """
    source = str(source or "").strip().strip('"')
    if not source:
        raise NarrationError("Add meg a videót (fájl vagy YouTube URL).")
    work.mkdir(parents=True, exist_ok=True)
    info: dict = {"media": None, "title": Path(source).stem,
                  "subtitle": None, "duration": 0.0, "url": ""}
    if youtube.is_youtube_url(source):
        stage = work / "download"
        marker = work / "youtube.json"
        if marker.is_file() and not force:
            try:
                old = json.loads(marker.read_text(encoding="utf-8"))
                if (old.get("url") == source and old.get("media")
                        and Path(old["media"]).is_file()):
                    return old
            except (OSError, ValueError):
                pass
        print(f"[narráció] YouTube letöltés: {source}", flush=True)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        info = youtube.fetch(source, stage, language=opts.language)
        info["url"] = source
        marker.write_text(json.dumps(info, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"[narráció] kész: {info.get('title') or info['media']}", flush=True)
        return info
    path = Path(source).expanduser()
    if not path.is_file():
        raise NarrationError(f"Nincs ilyen médiafájl: {path}")
    if path.suffix.lower() not in MEDIA_SUFFIXES:
        raise NarrationError(f"Nem támogatott videó: {path.suffix} "
                             f"({', '.join(sorted(MEDIA_SUFFIXES))})")
    info.update(media=str(path), title=path.stem, duration=duration(path))
    return info


def _sidecar_subtitle(src: Path) -> Path | None:
    """A videó mellett lévő magyar/általános felirat (ha van)."""
    for name in (f"{src.stem}.hu.srt", f"{src.stem}.hun.srt",
                 f"{src.stem}.srt", f"{src.stem}.hu.vtt", f"{src.stem}.vtt"):
        cand = src.parent / name
        if cand.is_file():
            return cand
    return None


def extract_audio(src: Path, out: Path, stream: int = 0, sr: int = 16000) -> Path:
    """A forrás hangsávja mono WAV-ként (az ASR-hez 16 kHz)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([media.FFMPEG, "-v", "error", "-y", "-i", src,
          "-map", f"0:a:{stream}", "-ac", "1", "-ar", str(sr),
          "-c:a", "pcm_s16le", out], timeout=3600)
    return out


def _sentence(text: str) -> bool:
    return subs.one_line(text).rstrip().endswith((".", "!", "?", "…"))


def _words_to_cues(words: list[dict], max_s: float = 12.0,
                   max_chars: int = 160) -> list[subs.Cue]:
    """ASR-szavakból feliratszerű, de beszédközpontú szegmensek.

    A szegmens akkor zárul, ha a szó mondatot zár, ha elérte a hosszkorlátot,
    vagy ha a következő szó előtt szünet van. Így a későbbi blokképítés valódi
    mondatokra tud vágni, nem szóközönként.
    """
    cues: list[subs.Cue] = []
    cur: list[dict] = []

    def close() -> None:
        nonlocal cur
        if not cur:
            return
        text = " ".join(str(w.get("word") or "").strip() for w in cur).strip()
        text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        start = float(cur[0]["start"])
        end = max(float(w.get("end") if w.get("end") is not None
                        else w["start"] + 0.3) for w in cur)
        if text and end > start:
            cues.append(subs.Cue(index=len(cues) + 1, start=round(start, 3),
                                 end=round(end, 3), text=text))
        cur = []

    for k, w in enumerate(words):
        if w.get("start") is None or not str(w.get("word") or "").strip():
            continue
        cur.append(w)
        text = " ".join(str(x.get("word") or "") for x in cur)
        end = float(cur[-1].get("end") if cur[-1].get("end") is not None
                    else cur[-1]["start"] + 0.3)
        nxt = words[k + 1] if k + 1 < len(words) else None
        gap = (float(nxt["start"]) - end) if (nxt and nxt.get("start") is not None) else 0.0
        if _sentence(text) or end - float(cur[0]["start"]) >= max_s \
                or len(text) >= max_chars or gap >= 0.9:
            close()
    close()
    return cues


def load_cues(info: dict, work: Path, opts: Options, *,
              subtitle: str | Path | None = None,
              srt: str | Path | None = None, force: bool = False) -> list[subs.Cue]:
    """A narráció szövege: megadott felirat → YouTube-felirat → mellékelt SRT
    → (ha nincs felirat) helyi ASR. Az eredmény a munkamappában gyorsítótárazva.
    """
    cache = work / "transcript.json"
    key = {"media": info.get("media"), "srt": str(srt or ""),
           "language": opts.language}
    if cache.is_file() and not force:
        try:
            old = json.loads(cache.read_text(encoding="utf-8"))
            if old.get("key") == key:
                return [subs.Cue(index=c.get("index", i + 1),
                                 start=float(c["start"]), end=float(c["end"]),
                                 text=str(c.get("text") or ""))
                        for i, c in enumerate(old.get("cues") or [])]
        except (OSError, ValueError):
            pass
    src: Path | None = None
    if srt:
        src = Path(str(srt)).expanduser()
    elif subtitle:
        src = Path(str(subtitle)).expanduser()
    elif info.get("subtitle"):
        src = Path(str(info["subtitle"])).expanduser()
    elif info.get("media"):
        src = _sidecar_subtitle(Path(str(info["media"])))
    cues: list[subs.Cue] = []
    if src is not None:
        if not src.is_file():
            raise NarrationError(f"Nincs ilyen felirat: {src}")
        if src.suffix.lower() == ".vtt":
            from .youtube import _vtt_to_srt
            tmp = work / "source_from_vtt.srt"
            tmp.write_text(_vtt_to_srt(src.read_text(encoding="utf-8", errors="replace")),
                           encoding="utf-8")
            src = tmp
        cues = subs.parse_srt(src)
        print(f"[narráció] felirat: {src.name} ({len(cues)} sor)", flush=True)
    if not cues:
        src_media = Path(str(info["media"]))
        wav = work / "source16k.wav"
        if force or not wav.is_file():
            print(f"[narráció] hang kinyerve: {src_media.name}", flush=True)
            extract_audio(src_media, wav, stream=opts.audio_stream)
        print(f"[narráció] nincs felirat — helyi ASR ({opts.language})…", flush=True)
        words, _text = asrcheck.transcribe_words(wav, language=opts.language,
                                                 verbose=True)
        cues = _words_to_cues(words)
        print(f"[narráció] ASR: {len(words)} szó, {len(cues)} szegmens", flush=True)
    if not cues:
        raise NarrationError("Nem sikerült szöveget nyerni a videóból "
                             "(nincs felirat és az ASR sem adott szöveget).")
    cache.write_text(json.dumps({
        "key": key,
        "cues": [{"index": c.index, "start": round(c.start, 3),
                  "end": round(c.end, 3), "text": c.text} for c in cues],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return cues


# ── 2. folyamatos blokkok ───────────────────────────────────────────────────
def build_blocks(cues: list[subs.Cue], opts: Options | None = None) -> list[Block]:
    """Feliratsorok → folyamatos TTS-blokkok.

    A blokk addig nő, amíg a következő sor még ugyanabba a folyamatba illik:
    nincs nagy szünet, nem lóg túl a hosszkorláton, és a blokk már mondatot
    zárt (akkor érdemes vágni). Így a TTS nem soronként indul újra.
    """
    o = opts or Options()
    blocks: list[Block] = []
    cur: list[subs.Cue] = []
    chars = 0

    def flush() -> None:
        nonlocal cur, chars
        if not cur:
            return
        parts = [subs.tts_text(subs.one_line(c.text)).strip() for c in cur]
        text = " ".join(p for p in parts if p).strip()
        if text:
            blocks.append(Block(i=len(blocks) + 1, start=round(cur[0].start, 3),
                                end=round(max(c.end for c in cur), 3), text=text))
        cur, chars = [], 0

    for cue in cues:
        if subs.is_music(cue.text) or any(
                k in cue.text.lower() for k in subs.SKIP_PATTERNS):
            continue
        text = subs.tts_text(subs.one_line(cue.text)).strip()
        if not text:
            continue
        if cur:
            gap = cue.start - cur[-1].end
            length = cue.end - cur[0].start
            too_long = (chars + 1 + len(text) > o.block_chars
                        or length > o.block_max_s or gap > o.gap_split)
            # kedvező vágás: a blokk már mondatot zárt és elég tartalmas
            natural = (chars >= o.block_chars * 0.55
                       and _sentence(cur[-1].text))
            if too_long or natural:
                flush()
        cur.append(cue)
        chars += len(text) + (1 if chars else 0)
    flush()
    return blocks


# ── 3. szintézis egy hangon ─────────────────────────────────────────────────
def _engine_for(voice, models: dict, ref: Path | None, language: str):
    if voice.engine == "xtts":
        return XTTSEngine(models.get("xtts"), ref, language=language,
                          num_threads=int(models.get("threads") or 4),
                          stubs=models.get("xtts_stubs") or None)
    if voice.engine == "piper":
        return PiperEngine(models.get("piper"), models.get("espeak"))
    return SupertonicEngine(models.get("supertonic"),
                            num_threads=int(models.get("threads") or 4))


def _cache_key(text: str, voice_id: str, params: dict, target: float) -> str:
    import hashlib

    h = hashlib.sha1(
        f"{voice_id}|{json.dumps(params, sort_keys=True)}|{text}|"
        f"{target:.2f}".encode("utf-8"))
    return h.hexdigest()[:16]


def synthesize(blocks: list[Block], voice, ref: Path | None, models: dict,
               out_dir: Path, opts: Options, *, force: bool = False) -> dict:
    """Minden blokk egyetlen TTS-hívásban, a saját ablakához igazítva."""
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = _engine_for(voice, models, ref, opts.language)
    params = dict(voice.params)
    if ref is not None:
        params["ref"] = str(ref)
        params["ref_mtime"] = int(ref.stat().st_mtime)
    meta_path = out_dir / "_meta.json"
    meta: dict[str, str] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    todo = [b for b in blocks if b.i <= opts.limit] if opts.limit else blocks
    by_i = {b.i: b for b in blocks}
    t0 = time.time()
    stats = {"done": 0, "cached": 0, "failed": [], "tempos": []}
    for k, b in enumerate(todo):
        nxt = by_i.get(b.i + 1)
        # a rendelkezésre álló idő: a blokk saját ablaka + a következő blokkig
        # tartó szünet (legfeljebb 0,8 s), hogy a természetes szünet ne vesszen el
        room = max(b.dur, (nxt.start - b.start - 0.06) if nxt else b.dur + 0.8)
        text = subs.prepare_hu(b.text)
        key = _cache_key(text, voice.vid, params, room)
        wav_path = out_dir / f"{b.i:04d}.wav"
        stored = str(meta.get(str(b.i)) or "")
        if not force and wav_path.is_file() and stored.split("|", 1)[0] == key:
            b.wav, b.cached = wav_path, True
            b.fitted = float(stored.split("|", 1)[1]) if "|" in stored else 0.0
            b.tempo = 1.0
            stats["cached"] += 1
            continue
        t = time.time()
        x, sr = engine.synth(text, voice)
        if len(x) == 0:
            stats["failed"].append({"i": b.i, "text": text[:80]})
            continue
        x = audio.trim_silence(x, sr)
        if sr != opts.out_sr:
            x = audio._resample_poly(x, sr, opts.out_sr)
            sr = opts.out_sr
        dur = len(x) / sr
        target = min(room, dur) if dur > room else dur
        tempo = 1.0
        if dur > room:
            x, tempo = audio.fit_duration(x, sr, dur / room, 1.0, opts.max_tempo)
            if len(x) / sr > room + 0.01:
                x = audio.trim_tail(x, sr, room, fade=0.03)
        audio.write_wav(wav_path, x, sr)
        b.wav, b.fitted, b.tempo = wav_path, len(x) / sr, tempo
        meta[str(b.i)] = f"{key}|{b.fitted:.3f}"
        stats["done"] += 1
        stats["tempos"].append(tempo)
        if stats["done"] % 5 == 0:
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            print(f"  {stats['done']}/{len(todo)} blokk, {time.time() - t0:.0f}s",
                  flush=True)
        if os.environ.get("NARRATION_DEBUG_SYNTH"):
            print(f"[narráció] blokk {b.i}: {dur:.2f}s → {b.fitted:.2f}s "
                  f"(x{tempo:.2f}) {time.time() - t:.1f}s", flush=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    if stats["failed"]:
        print(f"[narráció] ! {len(stats['failed'])} blokk szintézise nem sikerült",
              file=sys.stderr, flush=True)
    return stats


# ── 4. hangsáv + mux ────────────────────────────────────────────────────────
def assemble(blocks: list[Block], total_s: float, out_wav: Path,
             opts: Options) -> dict:
    """A blokkok elhelyezése a saját idejükben, majd cél-hangerőre masterelés."""
    total = max(total_s, max((b.start + max(b.dur, b.fitted) for b in blocks),
                             default=0.0)) + 0.5
    sr = opts.out_sr
    track = np.zeros(int(round(total * sr)) + sr, dtype=np.float32)
    placed = 0
    for b in blocks:
        if b.wav is None or not b.wav.is_file():
            continue
        x, xsr = audio.load_mono(b.wav, sr=sr)
        if len(x) == 0:
            continue
        i0 = max(0, int(round(b.start * sr)))
        n = min(len(x), len(track) - i0)
        if n <= 0:
            continue
        track[i0:i0 + n] += x[:n]
        placed += 1
    # a forrás hossza a videóhoz igazodik; a rövid sáv csenddel végződik
    if total_s > 0:
        track = track[:int(round((total_s + 0.2) * sr))]
    master = mix.master(track, sr, opts.target_lufs, opts.peak_db)
    audio.write_wav(out_wav, track, sr)
    peak = float(np.abs(track).max()) if track.size else 0.0
    return {"blocks_placed": placed, "seconds": round(len(track) / sr, 2),
            "lufs": round(audio.integrated_lufs(track, sr), 2),
            "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-9)), 2),
            "rms_db": round(audio.rms_db(track), 2), **master}


def mux(src: Path, narration: Path, out: Path, opts: Options,
        duration: float | None = None) -> Path:
    """Videó + narráció (+ eredeti hangok) egy MKV-be.

    ``duration``: a kimenet hossza másodpercben (próbafutásnál a szintetizált
    blokkok végéig vágva); ``None`` esetén a teljes videó marad.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    auds = media.audio_streams(src)
    subs_streams = media.subtitle_streams(src)
    cmd = [media.FFMPEG, "-v", "error", "-y", "-i", src, "-i", narration,
           "-map", "0:v:0", "-map", "1:a:0"]
    if opts.keep_original:
        for s in auds:
            cmd += ["-map", f"0:{s['index']}"]
    picked = [s for s in subs_streams
              if (s.get("tags") or {}).get("language") in ("hun", "eng")]
    for s in picked:
        cmd += ["-map", f"0:{s['index']}"]
    cmd += ["-c:v", "copy", "-c:a:0", "aac", "-b:a:0", "192k", "-ac:a:0", "2",
            "-metadata:s:a:0", f"language={opts.language[:2]}",
            "-metadata:s:a:0", f"title={opts.title}"]
    if opts.keep_original:
        for k, s in enumerate(auds, start=1):
            tags = s.get("tags") or {}
            lang = tags.get("language") or "und"
            cmd += [f"-c:a:{k}", "copy",
                    f"-metadata:s:a:{k}", f"language={lang}"]
            if tags.get("title"):
                cmd += [f"-metadata:s:a:{k}", f"title={tags['title']}"]
    if picked:
        cmd += ["-c:s", "srt"]
        for k, s in enumerate(picked):
            tags = s.get("tags") or {}
            cmd += [f"-metadata:s:s:{k}",
                    f"language={tags.get('language') or 'und'}"]
    cmd += ["-disposition:a:0", "default"]
    if opts.keep_original:
        for k in range(1, len(auds) + 1):
            cmd += [f"-disposition:a:{k}", "0"]
    if duration:
        cmd += ["-t", f"{float(duration):.3f}"]
    cmd += [out]
    _run(cmd, timeout=7200)
    return out


# ── 5. a teljes lánc ────────────────────────────────────────────────────────
def run(source: str, *, out: str | Path | None = None,
        work: str | Path | None = None, voice: str | None = None,
        ref: str | Path | None = None, subtitle: str | Path | None = None,
        srt: str | Path | None = None, opts: Options | None = None,
        models: dict | None = None, force: bool = False,
        no_mux: bool = False, verbose: bool = True) -> dict:
    """Egy egyszereplős videó teljes narrációja.

    ``source``: helyi videó vagy YouTube URL. ``ref``: klónozandó
    hangminta (enélkül kész hang szólal meg). Visszatérés: a riport dict-je.
    """
    o = opts or Options()
    t0 = time.time()
    if work is None:
        stem = slugify(Path(str(source)).stem if not youtube.is_youtube_url(source)
                       else source.rsplit("v=", 1)[-1])
        work = APP_ROOT / "work" / "narration" / stem
    work = Path(work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    models = {**model_paths(models), "threads": (models or {}).get("threads")}
    chosen, sample = resolve_voice(voice, ref, models)
    if verbose:
        print(f"[narráció] hang: {chosen.vid}"
              + (f" (klón: {sample.name})" if sample else " (kész hang)"), flush=True)
    info = prepare_media(str(source), work, o, force=force)
    cues = load_cues(info, work, o, subtitle=subtitle, srt=srt, force=force)
    blocks = build_blocks(cues, o)
    if not blocks:
        raise NarrationError("Nincs megszólalás a videóban.")
    # csak akkor próba, ha a limit ténylegesen le is vágja a blokksort
    partial = bool(o.limit) and o.limit < len(blocks)
    if verbose:
        print(f"[narráció] {len(blocks)} folyamatos blokk "
              f"({sum(len(b.text) for b in blocks)} karakter, "
              f"medián {np.median([b.dur for b in blocks]):.1f}s)", flush=True)
        if partial:
            # egy limitált futás kimenete a videó végéig csendet tartalmaz,
            # ezért ezt a logban és a riportban is félreérthetetlenül jelezzük
            print(f"[narráció] ! PRÓBAFUTÁS: csak az első {min(o.limit, len(blocks))}"
                  f"/{len(blocks)} blokk készül el, a többi csend lesz "
                  f"(a teljes változathoz --limit nélkül futtasd)", flush=True)
        else:
            print(f"[narráció] limit: nincs — mind a {len(blocks)} blokk "
                  f"szintetizálódik", flush=True)
    tts_dir = work / "blocks"
    stats = synthesize(blocks, chosen, sample, models, tts_dir, o, force=force)
    done = [b for b in blocks if b.wav is not None]
    total = float(info.get("duration") or 0.0) or duration(info["media"])
    trimmed_to = None
    if partial and done:
        # egy próbafutás ne maradjon csendben a videó végéig: a hangsávot és a
        # kész videót is a szintetizált blokkok végéig vágjuk
        trimmed_to = round(max(b.start + max(b.dur, b.fitted or 0.0)
                               for b in done) + 0.4, 3)
        total = min(total, trimmed_to) if total > 0 else trimmed_to
    wav = work / "narration.wav"
    mix_info = assemble(done, total, wav, o)
    result = {
        "source": str(source),
        "media": info.get("media"),
        "title": info.get("title"),
        "voice": chosen.vid,
        "clone_ref": None if sample is None else str(sample),
        "language": o.language,
        "limited": partial,
        "limit": int(o.limit),
        "blocks_total": len(blocks),
        "trimmed": trimmed_to is not None,
        "trimmed_to": trimmed_to,
        "blocks": [b.to_dict() for b in done],
        "synth": {k: v for k, v in stats.items() if k != "tempos"},
        "mix": mix_info,
        "audio": str(wav),
        "output": None,
        "seconds": round(time.time() - t0, 1),
    }
    if not no_mux:
        stem = re.sub(r"\s*-\s*HUN.*$", "", Path(info["media"]).stem)
        # a próbafutás külön fájlba megy, hogy egy kész, teljes változatot
        # ne írjon felül egy fél perces vágás
        suffix = " - HUN NARRACIO PROBA.mkv" if partial else " - HUN NARRACIO.mkv"
        out_path = (Path(out).expanduser() if out
                    else work / f"{stem}{suffix}")
        result["output"] = str(mux(Path(info["media"]), wav, out_path, o,
                                   duration=trimmed_to))
    (work / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        m = result["mix"]
        note = (f" [PRÓBA: {len(done)}/{len(blocks)} blokk, "
                f"{m['seconds']:.0f}s-re vágva]" if partial else "")
        print(f"[narráció] kész: {m['seconds']:.0f}s, {m['lufs']:.2f} LUFS, "
              f"{m['peak_dbfs']:.2f} dBFS{note} — "
              f"{result['output'] or result['audio']}", flush=True)
        print(f"[narráció] riport: {work / 'report.json'}", flush=True)
    return result
