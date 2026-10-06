"""F5-TTS magyar klónozás: hang x lépésszám mátrix, RTF-vel.

Referenciahangok: a repo Supertonic-mintái (F1, M3), pontos átirattal.
Kimenet: WAV-ok + JSON riport az asr_check_wavs.py-hoz.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out_matrix")
os.makedirs(OUT, exist_ok=True)

REPO = os.path.join("C:\\", "Users", "istva", "Dev", "portfolio", "Projects",
                    "audiobook narrator", "auris", "reader")
SAMPLES = os.path.join(REPO, "exports", "test-samples")

# (fálynév-szár, pontos átirat) — a F5-TTS-nél a ref_textnek egyeznie kell.
REFS = [
    ("F1", "02-Az_öreg_kőház_udvarán_csende__serial.wav",
     "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban."),
    ("M3", "02-Az_öreg_kőház_udvarán_csende__serial.wav",
     "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban."),
]

TEXTS = [
    "A tengerparton reggel még köd lebegett a fák között.",
    "Ebben a könyvben egy magyar nyelvű mesét találhatsz.",
]

REPO_ID = "sarpba/F5-TTS_V1_hun_v2"
CKPT = "model_309300.safetensors"
VOCAB = "vocab.txt"

# nfe_step: a F5 flow-matching lépéseinek száma (alap 32).
STEPS = [32, 16, 8]


def log(msg):
    print(msg, flush=True)


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    log(f"torch {torch.__version__} | szálak {torch.get_num_threads()}")
    import torchaudio  # betölti a torchaudio_compat DLL-útvonalat
    log(f"torchaudio {torchaudio.__version__}")

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(REPO_ID, CKPT)
    vocab_path = hf_hub_download(REPO_ID, VOCAB)

    from f5_tts.api import F5TTS

    t0 = time.perf_counter()
    model = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_path,
                  vocab_file=vocab_path, device="cpu")
    log(f"modell betöltve {time.perf_counter() - t0:.1f} s")

    cases = {}
    rows = []
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
                name = f"{voice}_nfe{steps}_{ti:02d}.wav"
                path = os.path.join(OUT, name)
                sf.write(path, audio, sr)
                peak = float(np.abs(audio).max())
                rms = float(np.sqrt((audio ** 2).mean()))
                nan = bool(np.isnan(audio).any() or np.isinf(audio).any())
                cases[path] = text
                rows.append({"voice": voice, "nfe_step": steps, "idx": ti,
                             "dur": dur, "time": dt, "rtf": dur / dt if dt else 0.0,
                             "peak": peak, "rms": rms, "nan": nan, "wav": path})
                log(f"[{voice} nfe={steps:2d} #{ti}] {dur:5.2f} s hang / {dt:6.2f} s "
                    f"-> RTF {rows[-1]['rtf']:6.4f} | peak {peak:.3f} rms {rms:.4f} nan={nan}")

    with open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8") as fh:
        json.dump(cases, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "bench.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    log("")
    log("=== hang x nfe_step összesítés ===")
    log("  (a 'valós idő hányosa' = hang_idő/idő; 0,01 → 100× lassabb a valósnál)")
    for voice, _, _ in REFS:
        for steps in STEPS:
            sel = [r for r in rows if r["voice"] == voice and r["nfe_step"] == steps]
            if not sel:
                continue
            at = sum(r["dur"] for r in sel)
            tt = sum(r["time"] for r in sel)
            log(f"{voice} nfe={steps:2d}: {at / tt:.4f}  ({tt:.1f} s idő / {at:.2f} s hang) "
                f"-> 1 óra hangra {3600 * (tt / at) / 60 / 60:.1f} óra")
    return 2 if any(r["nan"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())