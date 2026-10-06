r"""Magyar Piper-hangok mérése (x86_64 emulált venv, D:\hu-voice-ai\venv-x64).

Cél: RTF + a natív ARM64-es ONNX-futtatáshoz phonem-ID-k kiírása.
Kimenet: out_piper/*.wav, out_piper/phonemes.json, out_piper/cases.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out_piper")
os.makedirs(OUT, exist_ok=True)

PIPER_DIR = r"D:\hu-voice-ai\models\piper"
VOICES = ["anna", "berta", "imre"]

TEXTS = [
    "A tengerparton reggel még köd lebegett a fák között.",
    "Ebben a könyvben egy magyar nyelvű mesét találhatsz.",
]


def log(m):
    print(m, flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from piper import PiperVoice, SynthesisConfig
    from piper.voice import phonemes_to_ids

    cases = {}
    ph = {}
    report = []
    for v in VOICES:
        model = os.path.join(PIPER_DIR, f"hu_HU-{v}-medium.onnx")
        cfg = os.path.join(PIPER_DIR, f"hu_HU-{v}-medium.onnx.json")
        if not os.path.isfile(model):
            log(f"SKIP {v}: nincs {model}")
            continue
        t0 = time.perf_counter()
        voice = PiperVoice.load(model, config_path=cfg)
        load_s = time.perf_counter() - t0
        log(f"[{v}] betöltve {load_s:.2f}s  sr={voice.config.sample_rate}")

        for ti, text in enumerate(TEXTS, 1):
            cfg_s = SynthesisConfig(length_scale=1.0, normalize_audio=False)
            t0 = time.perf_counter()
            chunks = list(voice.synthesize(text, cfg_s))
            wall = time.perf_counter() - t0
            audio = np.concatenate([c.audio_float_array for c in chunks], axis=0)
            sr = chunks[0].sample_rate
            dur = len(audio) / sr
            out = os.path.join(OUT, f"{v}_{ti:02d}.wav")
            sf.write(out, audio, sr, subtype="PCM_16")
            cases[out] = text
            rtf_speed = dur / wall  # >1 = gyorsabb valós időnél
            log(f"[{v}] #{ti} wall={wall:6.2f}s audio={dur:5.2f}s "
                f"sebesség={rtf_speed:6.2f}x (idő/audio={wall / dur:5.2f})")
            report.append({"voice": v, "case": ti, "text": text, "wall_s": round(wall, 3),
                           "audio_s": round(dur, 3), "speed_x": round(rtf_speed, 3),
                           "sample_rate": sr, "wav": out})

            # phonem ID-k a natív ARM64-es ONNX futtatáshoz
            if v == VOICES[0]:
                try:
                    sents = voice.phonemize(text)  # list[list[str]] / mondatonként
                    flat_ph: list[str] = []
                    for s in sents:  # mindkét mérőszöveg egy mondat
                        flat_ph.extend(s)
                    flat = [int(x) for x in phonemes_to_ids(flat_ph,
                                                             voice.config.phoneme_id_map)]
                    ph[text] = {"ids": flat,
                                "scales": [0.667, 1.0, 0.8],  # noise, length, noise_w
                                "scales_det": [0.0, 1.0, 0.0],  # determinisztikus
                                "phoneme_type": str(getattr(voice.config, "phoneme_type", "?"))}
                    log(f"  {len(flat)} phonem-id kiment")
                except Exception as e:
                    log(f"  phonemize hiba: {type(e).__name__} {e}")

    # Determinisztikus ellenőrzés: noise_scale=0, noise_w=0 -> két futás
    # bit-gyen azonosnak kell lennie (a VITS szó zajából nem marad).
    v = VOICES[0]
    model = os.path.join(PIPER_DIR, f"hu_HU-{v}-medium.onnx")
    cfgp = os.path.join(PIPER_DIR, f"hu_HU-{v}-medium.onnx.json")
    voice = PiperVoice.load(model, config_path=cfgp)
    det_cfg = SynthesisConfig(noise_scale=0.0, length_scale=1.0, noise_w_scale=0.0,
                              normalize_audio=False)
    for ti, text in enumerate(TEXTS, 1):
        chunks = list(voice.synthesize(text, det_cfg))
        audio = np.concatenate([c.audio_float_array for c in chunks], axis=0)
        sf.write(os.path.join(OUT, f"{v}_det_{ti:02d}.wav"), audio,
                 chunks[0].sample_rate, subtype="PCM_16")
        ph.setdefault(text, {})
        log(f"[det {v}] #{ti} {len(audio)} minta @ {chunks[0].sample_rate}")

    with open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "phonemes.json"), "w", encoding="utf-8") as f:
        json.dump(ph, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"kész: {len(cases)} wav -> {OUT}")


if __name__ == "__main__":
    main()
