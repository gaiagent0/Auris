"""F5-TTS magyar klón **natív ARM64** Pythonnal, torchaudio-shimmel.

A shimet a repóból importáljuk; a f5_tts importja előtt telepítjük, mert
a f5_tts.model a modul import-időben is beimportálja a torchaudio-t.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

REPO = os.path.join("C:\\", "Users", "istva", "Dev", "portfolio", "Projects",
                    "audiobook narrator", "auris", "reader")
sys.path.insert(0, REPO)

WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out_native")
os.makedirs(OUT, exist_ok=True)

SAMPLES = os.path.join(REPO, "exports", "test-samples")
REFS = [
    ("F1", "02-Az_öreg_kőház_udvarán_csende__serial.wav",
     "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban."),
]
TEXTS = [
    "A tengerparton reggel még köd lebegett a fák között.",
    "Ebben a könyvben egy magyar nyelvű mesét találhatsz.",
]
STEPS = [32, 8]
REPO_ID = "sarpba/F5-TTS_V1_hun_v2"
CKPT = "model_309300.safetensors"
VOCAB = "vocab.txt"


def log(msg):
    print(msg, flush=True)


def main():
    import platform

    log(f"python {platform.python_version()} | {platform.machine()} | "
        f"proc: {platform.processor()[:40]}")
    torch.set_num_threads(os.cpu_count() or 8)
    log(f"torch {torch.__version__} | szálak {torch.get_num_threads()}")

    from core.torchaudio_compat import install

    install()
    import torchaudio

    log(f"torchaudio -> {getattr(torchaudio, '__version__', 'shim')}")

    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(REPO_ID, CKPT)
    vocab = hf_hub_download(REPO_ID, VOCAB)

    from f5_tts.api import F5TTS

    t0 = time.perf_counter()
    model = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab,
                  device="cpu")
    log(f"modell betöltve {time.perf_counter() - t0:.1f} s")

    cases, rows = {}, []
    for voice, wav_name, ref_text in REFS:
        ref_wav = os.path.join(SAMPLES, voice, wav_name)
        info = sf.info(ref_wav)
        log(f"ref {voice}: {info.duration:.2f} s @ {info.samplerate} Hz")
        for steps in STEPS:
            for ti, text in enumerate(TEXTS, 1):
                t0 = time.perf_counter()
                wav, sr, _ = model.infer(
                    ref_file=ref_wav, ref_text=ref_text, gen_text=text,
                    nfe_step=steps, speed=1.0, remove_silence=False,
                    show_info=lambda *a, **k: None, progress=None,
                )
                dt = time.perf_counter() - t0
                audio = np.asarray(wav, dtype=np.float32).reshape(-1)
                dur = len(audio) / sr
                path = os.path.join(OUT, f"{voice}_nfe{steps}_{ti:02d}.wav")
                sf.write(path, audio, sr)
                nan = bool(np.isnan(audio).any() or np.isinf(audio).any())
                cases[path] = text
                rtf = dur / dt if dt else 0.0
                rows.append({"voice": voice, "nfe_step": steps, "idx": ti,
                             "dur": dur, "time": dt, "rtf": rtf, "nan": nan})
                log(f"[{voice} nfe={steps:2d} #{ti}] {dur:5.2f} s hang / {dt:6.2f} s "
                    f"-> RTF {rtf:6.4f} nan={nan}")

    with open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8") as fh:
        json.dump(cases, fh, ensure_ascii=False, indent=1)

    log("")
    for steps in STEPS:
        sel = [r for r in rows if r["nfe_step"] == steps]
        if sel:
            at = sum(r["dur"] for r in sel)
            tt = sum(r["time"] for r in sel)
            # at/tt = hang_idő/idő, tehát 1 óra hangra 3600/(at/tt) másodperc.
            # A korábbi "3600 * (at/tt) / 60" képlet fordított volt (600x eltérés).
            log(f"nfe={steps:2d}: RTF {at / tt:.4f}  ({tt:.1f} s / {at:.2f} s hang) "
                f"-> 1 óra hangra {3600 * (tt / at) / 60:.1f} perc")
    return 2 if any(r["nan"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())