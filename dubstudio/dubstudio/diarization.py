"""Forduló-alapú beszélő-azonosítás (pyannote szegmentáció + CAM++).

A feliratsoronkénti klaszterezés rövid, zenés ablakokon dolgozik, ezért a
szomszédos sorok könnyen más-más klaszterbe kerülnek: a magyar hang mondatról
mondatra vált. A profi szinkronban előbb **beszélőfordulókat** kell találni a
teljes sávon, és a feliratsort ahhoz a fordulóhoz rendelni, amelyik alatt
ténylegesen elhangzik.

Ez a modul a sherpa-onnx csomagban elérhető pyannote 3.0 szegmentálót és a
CAM++ hanglenyomat-kinyerőt használja. A szegmentáló 10 ms-os felbontással
jelzi a beszédet és a beszélőváltásokat; a klaszterezés a teljes epizód
fordulóin történik, nem feliratsoronként. A kimenet stabil, időrendi
fordulólista, amelyet a ``utterances`` és a ``clustering`` használ.

A modell nem kötelező: ha a projektben nincs bekapcsolva, a régi lánc marad.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .config import Project

CACHE_VERSION = "v3"


@dataclass
class Segment:
    start: float
    end: float
    speaker: int

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "speaker": int(self.speaker)}


def settings(project: Project, tag: str | None = None) -> dict:
    """A projekt diarizációs beállításai, ésszerű alapértékekkel.

    ``num_speakers`` > 0 esetén a klaszterezés **fix** beszélőszámmal fut
    (nincs küszöb-alapú klaszterszám-ingadozás futásról futásra). Epizódonként
    felülírható a ``num_speakers_by_ep`` térképpel (``{"E04": 12}``).
    """
    cfg = dict(project.diarization or {})
    if not cfg.get("enabled"):
        return {"enabled": False}
    cfg.setdefault("num_speakers", 0)
    if tag:
        per = (cfg.get("num_speakers_by_ep") or {}).get(str(tag).upper())
        if per:
            cfg["num_speakers"] = int(per)
    cfg.setdefault("threshold", 0.50)
    cfg.setdefault("min_duration_on", 0.15)
    cfg.setdefault("min_duration_off", 0.25)
    cfg.setdefault("min_turn", 0.30)
    cfg.setdefault("merge_gap", 0.60)
    cfg.setdefault("island_max", 0.80)
    cfg.setdefault("attach_gap", 2.0)
    cfg.setdefault("num_threads", 4)
    return cfg


def _audio16(vocals: Path):
    import numpy as np

    from . import audio

    x, sr = audio.load_mono(vocals)
    x16 = audio.to_16k(x, sr)
    return np.ascontiguousarray(x16, dtype=np.float32)


def _smooth(segments: list[Segment], cfg: dict) -> list[Segment]:
    """Rövid törmelék eldobása, azonos beszélő összevonása, sziget-javítás."""
    min_turn = float(cfg["min_turn"])
    merge_gap = float(cfg["merge_gap"])
    island_max = float(cfg["island_max"])
    out: list[Segment] = []
    for s in sorted(segments, key=lambda z: z.start):
        if s.end - s.start < min_turn:
            continue
        if out and out[-1].speaker == s.speaker and s.start - out[-1].end <= merge_gap:
            out[-1].end = max(out[-1].end, s.end)
        else:
            out.append(Segment(s.start, s.end, s.speaker))
    # Rövid „sziget” két azonos beszélő között: a szomszédok bizonyítéka
    # erősebb, mint egy 0,3–0,8 s-os, zenés ablak.
    for _ in range(4):
        changed = False
        for i in range(1, len(out) - 1):
            cur = out[i]
            if cur.end - cur.start > island_max:
                continue
            if out[i - 1].speaker == out[i + 1].speaker \
                    and cur.speaker != out[i - 1].speaker \
                    and out[i - 1].end >= cur.start - merge_gap \
                    and out[i + 1].start <= cur.end + merge_gap:
                cur.speaker = out[i - 1].speaker
                changed = True
        merged: list[Segment] = []
        for s in out:
            if merged and merged[-1].speaker == s.speaker \
                    and s.start - merged[-1].end <= merge_gap:
                merged[-1].end = max(merged[-1].end, s.end)
            else:
                merged.append(Segment(s.start, s.end, s.speaker))
        out = merged
        if not changed:
            break
    return out


def run(project: Project, tag: str, verbose: bool = True,
        num_speakers: int | None = None,
        cache: bool = True) -> list[Segment]:
    """A teljes énekbeszéd-sáv diarizálása; gyorsítótárazva az epizód mappában.

    ``num_speakers`` megadásával felülírja a projekt beállítását (kalibráláshoz);
    ``cache=False`` esetén nem írja a gyorsítótárat.
    """
    cfg = settings(project, tag)
    if num_speakers is not None:
        cfg["num_speakers"] = int(num_speakers)
    if not cfg.get("enabled"):
        return []
    vocals = project.vocals(tag)
    seg_model = project.model("diar_segmentation")
    emb_model = project.model("diar_embedding")
    if not seg_model.is_file():
        raise FileNotFoundError(f"nincs pyannote szegmentáló modell: {seg_model}")
    if not emb_model.is_file():
        raise FileNotFoundError(f"nincs CAM++ hanglenyomat modell: {emb_model}")
    key = hashlib.sha256(json.dumps({
        "version": CACHE_VERSION,
        "vocals": [str(vocals.resolve()), vocals.stat().st_size,
                   vocals.stat().st_mtime_ns],
        "seg": [str(seg_model.resolve()), seg_model.stat().st_size,
                seg_model.stat().st_mtime_ns],
        "emb": [str(emb_model.resolve()), emb_model.stat().st_size,
                emb_model.stat().st_mtime_ns],
        "cfg": {k: cfg[k] for k in ("threshold", "min_duration_on",
                                    "min_duration_off", "min_turn",
                                    "merge_gap", "island_max")},
        "n": int(cfg.get("num_speakers") or 0),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    out = project.ep_dir(tag) / "diarization.json"
    if cache and out.is_file():
        try:
            cached = json.loads(out.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                if verbose:
                    print(f"[diarizáció] gyorsítótár: {len(cached['segments'])} forduló, "
                          f"{cached.get('speakers', '?')} nyers címke")
                return [Segment(float(s["start"]), float(s["end"]),
                                int(s["speaker"])) for s in cached["segments"]]
        except (OSError, ValueError, KeyError):
            pass

    import sherpa_onnx

    scfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model), window_shift_ratio=0.1),
            num_threads=int(cfg["num_threads"]), provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=int(cfg["num_threads"]),
            provider="cpu"),
        clustering=(sherpa_onnx.FastClusteringConfig(
                        num_clusters=int(cfg["num_speakers"]),
                        compute_confidence=True)
                    if int(cfg.get("num_speakers") or 0) > 0 else
                    sherpa_onnx.FastClusteringConfig(
                        num_clusters=-1, threshold=float(cfg["threshold"]),
                        compute_confidence=True)),
        min_duration_on=float(cfg["min_duration_on"]),
        min_duration_off=float(cfg["min_duration_off"]))
    if not scfg.validate():
        raise RuntimeError("a diarizációs modell/konfiguráció érvénytelen")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(scfg)
    x = _audio16(vocals)
    fixed = int(cfg.get("num_speakers") or 0)
    if verbose:
        mode = (f"fix {fixed} beszélő" if fixed > 0 else
                f"küszöb {cfg['threshold']}")
        print(f"[diarizáció] {len(x) / 16000 / 60:.1f} perc, "
              f"pyannote + {emb_model.stem.split('_')[0]} ({mode})")
    result = diarizer.process(x, None)
    raw = [Segment(float(s.start), float(s.end), int(s.speaker))
           for s in result.sort_by_start_time()]
    segments = _smooth(raw, cfg)
    if cache:
        out.write_text(json.dumps({
            "key": key, "settings": cfg, "speakers": int(result.num_speakers),
            "raw_segments": len(raw), "segments": [s.to_dict() for s in segments],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        used = len({s.speaker for s in segments})
        print(f"[diarizáció] {len(raw)} nyers szegmens → {len(segments)} forduló "
              f"({used} címke)")
    return segments


def label_span(segments: list[Segment], start: float, end: float,
               attach_gap: float = 2.0) -> int | None:
    """Az adott időablakhoz legjobban illeszkedő beszélő címkéje."""
    if not segments:
        return None
    best: Segment | None = None
    best_overlap = 0.0
    for s in segments:
        ov = min(end, s.end) - max(start, s.start)
        if ov > best_overlap:
            best_overlap = ov
            best = s
    if best is not None and best_overlap >= 0.02:
        return best.speaker
    # Nincs beszéd a feliratsor alatt (halk sor, zene alatti suttogás): a
    # legközelebbi fordulóhoz csatoljuk, de csak rövid távolságon belül.
    best = min(segments, key=lambda s: min(abs(s.start - end), abs(start - s.end)))
    gap = min(abs(best.start - end), abs(start - best.end))
    return best.speaker if gap <= attach_gap else None


def assign(cues, segments: list[Segment], attach_gap: float = 2.0
           ) -> list[int | None]:
    """Feliratsorok -> beszélőcímkék (a legnagyobb átfedés alapján)."""
    return [label_span(segments, c.start, c.end, attach_gap) for c in cues]


def validity(segments: list[Segment]) -> dict:
    """Rövid diagnosztika a riporthoz."""
    total = sum(max(0.0, s.end - s.start) for s in segments)
    return {
        "segments": len(segments),
        "speakers": len({s.speaker for s in segments}),
        "seconds": round(total, 1),
        "mean_turn_s": round(total / max(len(segments), 1), 2),
    }
