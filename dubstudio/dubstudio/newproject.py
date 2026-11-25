"""Új projekt alapértékei — a friss klón első projektje is ebből jön.

A dubstudio nem tartalmaz beégetett projektet: a ``projects/`` mappa üresen
érkezik a repóval. A felület „új film” varázslója és a
``scripts/check_dubstudio.py --init`` ugyanezt a gyárat használja, hogy az
első projekt is működőképes legyen (modellek, küszöbök, kapuk, fordítás).

A mért alapértékek projektfüggetlenek: a klaszterezés és a QA kapuk ugyanazok
egy filmnél és egy sorozatnál. A média- és munkakönyvtár viszont projektenként
külön van (``movies/<slug>``, illetve ``work/<slug>``).
"""
from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

from .config import APP_ROOT

#: a QA kapuk alapértékei (a mérés a dubstudio README-jében)
GATE_DEFAULTS = {
    "voice_switch_max": 0,
    "voice_switch_ratio_max": 0.0,
    "one_voice_per_utt_max": 0,
    "gender_conflict_max": 0,
    "coverage_min": 1.0,
    "scene_conflict_max": 0,
    "main_shared_max": 0,
    "fit_over_ratio_max": 0.02,
    "fit_max_over_s": 0.6,
    "asr_cer_max": {"parakeet": 0.3, "whisper": 0.6, "default": 0.6},
    "voice_overlap_max": 2,
    "scene_minor_ratio_max": 0.01,
}

#: a klaszterezés/hangzár alapértékei
TUNING_DEFAULTS = {
    "gap_merge": 0.9,
    "gap_cont": 2.2,
    "cos_same": 0.24,
    "cos_diff": 0.12,
    "sim_same": 0.24,
    "link_cos": 0.25,
    "viterbi_pen": 0.3,
    "main_share_min": 0.04,
    "sim_gender": 0.17,
    "recall_cos": 0.5,
}

#: a `models` szakasz fájljai a közös modellmappán belül
MODEL_FILES = {
    "embedding": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "diar_embedding": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "diar_segmentation": "diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "piper": "piper",
}


def slugify(name: str) -> str:
    """Projekt-azonosító: ékezet nélküli, kisbetűs, kötőjeles (max 48)."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:48] or "film"


def default_models_dir() -> Path:
    """A szinkron modellek helye: ``<dubstudio>/work/models``."""
    return APP_ROOT / "work" / "models"


def default_auris_models_dir() -> Path:
    """A hangoskönyv és a szinkron **közös** modelljei: ``<Auris>/reader/models``."""
    return APP_ROOT.parent / "reader" / "models"


def default_models(models_dir: Path | None = None,
                   auris_models: Path | None = None) -> dict:
    """A ``models`` szakasz: a közös modellek útvonalai abszolút formában.

    A Supertonic (TTS) az Auris saját modellmappájában van, mert a hangoskönyv
    is onnan használja — ez a két modul közös motorja.
    """
    models = Path(models_dir) if models_dir else default_models_dir()
    auris = Path(auris_models) if auris_models else default_auris_models_dir()
    out = {
        "embedding": str(models / MODEL_FILES["embedding"]),
        "diar_embedding": str(models / MODEL_FILES["diar_embedding"]),
        "diar_segmentation": str(models / MODEL_FILES["diar_segmentation"]),
        "supertonic": str(auris / "supertonic-sherpa-int8"),
        "piper": str(models / MODEL_FILES["piper"]),
    }
    espeak = shutil.which("espeak-ng") or shutil.which("espeak") or ""
    if espeak:
        out["espeak"] = espeak
    return out


#: Feliratfordítás: a repó **csak helyi** LLM-et használ (Ollama OpenAI-
#: kompatibilis végpontja), külső szolgáltatás szándékosan nincs támogatva
#: (lásd ``intake._endpoint``). A modell a szerveren telepített helyi modell
#: neve; a kimenet JSON, ezért nem „gondolkodó” modellt érdemes választani.
TRANSLATE_DEFAULTS = {
    "enabled": True,
    "base": "http://127.0.0.1:11434/v1",
    "model": "qwen3.5:9b",
    "api_key": "",
    "api_key_env": "",
    "batch": 20,
    "timeout": 300,
    "max_tokens": 4096,
    "source_lang": "",
}


def project_doc(name: str, kind: str = "movie", season: int = 1,
                source_lang: str = "eng", models_dir: Path | None = None,
                auris_models: Path | None = None) -> dict:
    """Egy új projekt JSON-ja, minden alapértékkel kitöltve.

    ``kind``: ``movie`` (egyetlen E01) vagy ``series`` (SxxExx epizódok).
    A média a ``movies/<slug>`` (sorozatnál ``series/<slug>``) mappába kerül,
    a munkafájlok a ``work/<slug>`` alá — így két projekt nem írja felül
    egymást.
    """
    kind = "series" if str(kind).strip().lower() == "series" else "movie"
    slug = slugify(name)
    folder = "series" if kind == "series" else "movies"
    models = Path(models_dir) if models_dir else default_models_dir()
    return {
        "name": name,
        "season": int(season),
        "kind": kind,
        "root": f"../{folder}/{slug}",
        "cache_dir": f"../work/{slug}",
        "data_dir": "data",
        "subs_dir": "subs",
        "vocals_globs": ["{tag}/vocals.wav"],
        "me_globs": ["{tag}/me.wav"],
        "source_lang": source_lang,
        "original_audio": 0,
        "tts_sr": 44100,
        "models": default_models(models, auris_models),
        "tuning": dict(TUNING_DEFAULTS),
        "gate": dict(GATE_DEFAULTS),
        # szövegellenőrző pont a szintézis előtt (helyi LLM; nem kötelező)
        "llm": {"enabled": False, "base": "http://127.0.0.1:11434",
                "model": "qwen3.5:9b", "batch": 10, "timeout": 180,
                "required": False},
        "asr": {"mode": "parakeet", "batch": 1, "confirm_max": 30},
        "intake": {
            "translate": {**TRANSLATE_DEFAULTS, "source_lang": source_lang},
            "separate": {"enabled": True, "model": "UVR_MDXNET_Main.onnx",
                         "model_dir": str(models / "separate"),
                         "audio_stream": 0, "second_pass": False,
                         "sample_rate": 44100, "timeout": 3600},
        },
        "diarization": {"enabled": True, "num_speakers": 0, "threshold": 0.5,
                        "min_turn": 0.3, "merge_gap": 0.6, "island_max": 0.8,
                        "min_duration_on": 0.15, "min_duration_off": 0.25,
                        "attach_gap": 2.0, "num_threads": 4},
        "series": {"auto_process": False, "poll_s": 30},
        "media_server": {"path": "", "deliver": False},
    }
