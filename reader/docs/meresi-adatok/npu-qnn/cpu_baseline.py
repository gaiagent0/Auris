"""Clean CPU baseline for vector_estimator at the shapes used by the DLC."""
import os
import time

import numpy as np
import onnxruntime as ort

MODELS = "C:/Users/istva/Dev/portfolio/Projects/audiobook narrator/auris/reader/models"
NAMES = ["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask",
         "current_step", "total_step"]

SHAPES = {
    1: {"noisy_latent": (1, 144, 64), "text_emb": (1, 256, 96), "style_ttl": (1, 50, 256),
        "latent_mask": (1, 1, 64), "text_mask": (1, 1, 96), "current_step": (1,),
        "total_step": (1,)},
    4: {"noisy_latent": (4, 144, 64), "text_emb": (4, 256, 96), "style_ttl": (4, 50, 256),
        "latent_mask": (4, 1, 64), "text_mask": (4, 1, 96), "current_step": (4,),
        "total_step": (4,)},
}


def make_feed(bsz):
    rng = np.random.default_rng(7 + bsz)
    feed = {}
    for n in NAMES:
        shape = SHAPES[bsz][n]
        if n in ("latent_mask", "text_mask"):
            arr = np.ones(shape, dtype=np.float32)
        elif n == "current_step":
            arr = np.zeros(shape, dtype=np.float32)
        elif n == "total_step":
            arr = np.full(shape, 10.0, dtype=np.float32)
        elif n == "noisy_latent":
            arr = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        else:
            arr = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        feed[n] = arr
    return feed


def bench(sess, feed, reps=10):
    for _ in range(3):
        sess.run(None, feed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return ts.min() * 1000, np.median(ts) * 1000


MODELS_ONNX = {
    "fp32": MODELS + "/supertonic-3/onnx/vector_estimator.onnx",
    "int8": MODELS + "/supertonic-3-int8/onnx/vector_estimator.int8.onnx",
}

print("CPU baseline, vector_estimator, ONNX Runtime %s" % ort.__version__)
print("%-6s %-5s %-8s %10s %10s" % ("model", "bsz", "threads", "min ms", "median ms"))
for label, path in MODELS_ONNX.items():
    if not os.path.exists(path):
        print("missing", path)
        continue
    for bsz in (1, 4):
        for threads in (1, 12):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            sess = ort.InferenceSession(path, sess_options=so,
                                        providers=["CPUExecutionProvider"])
            mn, md = bench(sess, make_feed(bsz))
            print("%-6s %-5d %-8d %10.2f %10.2f" % (label, bsz, threads, mn, md))