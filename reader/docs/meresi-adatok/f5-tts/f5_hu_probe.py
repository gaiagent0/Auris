"""Magyar F5-TTS hangklónozás próba ARM64-en, natív torchaudio-val.

A referenciahang a repo saját Supertonic-mintája (F1, serial), a pontos
átirattal — F5-TTS-nél a ref_textnek EXAKTan egyeznie kell a ref_audio-val.

Kimenet: WAV-ok + RTF + az első mondat ASR-ellenőrzéshez.
"""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)

REPO = os.path.join("C:\\", "Users", "istva", "Dev", "portfolio", "Projects",
                    "audiobook narrator", "auris", "reader")
SAMPLES = os.path.join(REPO, "exports", "test-samples", "F1")

REF_WAV = os.path.join(SAMPLES, "02-Az_öreg_kőház_udvarán_csende__serial.wav")
REF_TEXT = "Az öreg kőház udvarán csendesen gólyák bújtak a nádasban."

GEN_TEXTS = [
    "A tengerparton reggel még köd lebegett a fák között.",
    "Ebben a könyvben egy magyar nyelvű mesét találhatsz.",
    "A kis falu utcáin a macskák éjszaka is járkáltak.",
]

REPO_ID = os.environ.get("F5_HU_REPO", "sarpba/F5-TTS_V1_hun_v2")
CKPT = os.environ.get("F5_HU_CKPT", "model_309300.safetensors")
VOCAB = os.environ.get("F5_HU_VOCAB", "vocab.txt")


def log(msg):
    print(msg, flush=True)


def main():
    log(f"torch {torch.__version__} | threads {torch.get_num_threads()}")
    import torchaudio
    log(f"torchaudio {torchaudio.__version__} (natív ARM64)")
    log(f"threads: {os.cpu_count()} logikai mag")

    from huggingface_hub import hf_hub_download

    log(f"súlyok letöltése: {REPO_ID}")
    ckpt_path = hf_hub_download(REPO_ID, CKPT)
    vocab_path = hf_hub_download(REPO_ID, VOCAB)
    log(f"  ckpt: {ckpt_path}")

    from f5_tts.api import F5TTS

    t0 = time.perf_counter()
    model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=ckpt_path,
        vocab_file=vocab_path,
        device="cpu",
    )
    log(f"modell betöltve {time.perf_counter() - t0:.1f} s")

    ref_sr = sf.info(REF_WAV).samplerate
    ref_dur = sf.info(REF_WAV).duration
    log(f"referencia: {os.path.basename(REF_WAV)} {ref_dur:.2f} s @ {ref_sr} Hz")

    results = []
    for idx, text in enumerate(GEN_TEXTS, 1):
        t0 = time.perf_counter()
        wav, sr, _ = model.infer(
            ref_file=REF_WAV,
            ref_text=REF_TEXT,
            gen_text=text,
            speed=1.0,
            remove_silence=False,
        )
        dt = time.perf_counter() - t0
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        dur = len(audio) / sr
        path = os.path.join(OUT, f"f5hu_{idx:02d}.wav")
        sf.write(path, audio, sr)
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt((audio ** 2).mean()))
        nan = bool(np.isnan(audio).any() or np.isinf(audio).any())
        results.append((idx, text, dur, dt, dur / dt if dt else 0.0, peak, rms, nan, path))
        log(
            f"[{idx}] {dur:5.2f} s hang / {dt:6.2f} s -> RTF {dur / dt if dt else 0:6.3f} "
            f"| peak {peak:.3f} rms {rms:.4f} nan={nan}"
        )
        log(f"     {text}")

    total_audio = sum(r[2] for r in results)
    total_time = sum(r[3] for r in results)
    log("")
    log(f"összes: {total_audio:.2f} s hang / {total_time:.2f} s -> RTF {total_audio / total_time:.4f}")
    rtf = total_audio / total_time
    log(f"1 óra hangra becsült idő: {3600 * rtf / 60:.1f} perc")
    if any(r[7] for r in results):
        log("FIGYELMEZTETES: NaN/Inf a kimenetben")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())