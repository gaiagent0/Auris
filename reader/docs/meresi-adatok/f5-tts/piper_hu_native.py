"""Piper-hangok futtatása a NATÍV ARM64-es ONNX Runtime-tel (Auris venv).

Bemenet: out_piper/phonemes.json (a piper CLI által kiírt phonem-ID-k).
Így nem kell espeak/piper_phonemize: csak a tiszta ONNX-inferencia mérődik.
Kimenet: out_piper_native/*.wav + cases.json + report.json (időzítés + bit-egyezés).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

WORK = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(WORK, "out_piper")
OUT = os.path.join(WORK, "out_piper_native")
os.makedirs(OUT, exist_ok=True)

PIPER_DIR = r"D:\hu-voice-ai\models\piper"
VOICES = ["anna", "berta", "imre"]


def log(m):
    print(m, flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(os.path.join(SRC, "phonemes.json"), encoding="utf-8") as f:
        ph = json.load(f)
    if not ph:
        log("NINCS phonemes.json — futtasd előbb a piper_hu_probe.py-t")
        return

    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    log(f"python {platform.python_version()} {platform.machine()} | "
        f"ort {ort.__version__} | szálak {so.intra_op_num_threads}")

    cases, report = {}, []
    for v in VOICES:
        model = os.path.join(PIPER_DIR, f"hu_HU-{v}-medium.onnx")
        if not os.path.isfile(model):
            continue
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        load_s = time.perf_counter() - t0
        sr = json.load(open(os.path.join(PIPER_DIR, f"hu_HU-{v}-medium.onnx.json"),
                            encoding="utf-8"))["audio"]["sample_rate"]
        log(f"[{v}] betöltve {load_s:.2f}s sr={sr} bemenet="
            f"{[i.name for i in sess.get_inputs()]}")

        for ti, (text, pdata) in enumerate(ph.items(), 1):
          for mode, scales_key, suffix in (("zaj", "scales", "native"),
                                           ("determ", "scales_det", "native_det")):
            ids = np.array([pdata["ids"]], dtype=np.int64)
            args = {"input": ids,
                    "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                    "scales": np.array(pdata[scales_key], dtype=np.float32)}
            t0 = time.perf_counter()
            audio = sess.run(None, args)[0].squeeze()
            wall = time.perf_counter() - t0
            dur = len(audio) / sr
            out = os.path.join(OUT, f"{v}_{suffix}_{ti:02d}.wav")
            sf.write(out, audio, sr, subtype="PCM_16")
            cases[out] = text

            # numerikus egyezés az emulált futással (azonos bemenet, másik ORT/arm)
            ref = os.path.join(SRC, f"{v}_{'det' if mode == 'determ' else ''}_{ti:02d}.wav")
            diff = corr = None
            if os.path.isfile(ref):
                a_ref, _ = sf.read(ref, dtype="float32")
                n = min(len(a_ref), len(audio))
                d = np.abs(a_ref[:n] - audio[:n])
                diff = round(float(d.max()), 6)
                corr = round(float(np.corrcoef(a_ref[:n], audio[:n])[0, 1]), 6)
            log(f"[{v}] #{ti} {mode:6s} wall={wall:6.2f}s audio={dur:5.2f}s "
                f"sebesség={dur / wall:6.2f}x  max|Δ|={diff} corr={corr}")
            report.append({"voice": v, "case": ti, "mode": mode, "text": text,
                           "wall_s": round(wall, 3), "audio_s": round(dur, 3),
                           "speed_x": round(dur / wall, 3),
                           "max_abs_diff_vs_emulated": diff,
                           "corr_vs_emulated": corr, "wav": out})

    with open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"kész: {len(cases)} wav -> {OUT}")


if __name__ == "__main__":
    main()
