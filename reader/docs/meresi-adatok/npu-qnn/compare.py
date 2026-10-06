"""Compare the HTP DLC output against CPU ONNX references and time both paths."""
import os
import time

import numpy as np
import onnxruntime as ort

WORK = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(WORK, "inputs")
MODELS = os.path.join(
    "C:\\", "Users", "istva", "Dev", "portfolio", "Projects",
    "audiobook narrator", "auris", "reader", "models",
)

NAMES = [
    "noisy_latent", "text_emb", "style_ttl", "latent_mask",
    "text_mask", "current_step", "total_step",
]
SHAPES = {
    "noisy_latent": (1, 144, 64),
    "text_emb": (1, 256, 96),
    "style_ttl": (1, 50, 256),
    "latent_mask": (1, 1, 64),
    "text_mask": (1, 1, 96),
    "current_step": (1,),
    "total_step": (1,),
}

feed = {}
for n in NAMES:
    feed[n] = np.fromfile(os.path.join(IN_DIR, "s0_%s.raw" % n), dtype=np.float32).reshape(
        SHAPES[n]
    )


def time_session(sess, reps=10):
    for _ in range(3):
        sess.run(None, feed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return ts.min() * 1000, np.median(ts) * 1000, ts.mean() * 1000


def report(label, sess, ref=None):
    out = sess.run(None, feed)[0]
    mn, md, mean = time_session(sess)
    print("%-26s min %7.2f ms  median %7.2f ms  mean %7.2f ms" % (label, mn, md, mean))
    if ref is not None:
        d = out.astype(np.float64) - ref.astype(np.float64)
        denom = np.abs(ref).mean()
        corr = np.corrcoef(out.ravel(), ref.ravel())[0, 1]
        print(
            "    vs ref: corr %.5f  MAE %.5f  rel_MAE %.2f%%  ref_std %.4f"
            % (corr, np.abs(d).mean(), 100 * np.abs(d).mean() / denom, ref.std())
        )
    return out


so = ort.SessionOptions()
so.intra_op_num_threads = 12

ref32 = report(
    "CPU fp32",
    ort.InferenceSession(
        os.path.join(MODELS, "supertonic-3", "onnx", "vector_estimator.onnx"),
        sess_options=so, providers=["CPUExecutionProvider"],
    ),
)

int8_path = os.path.join(MODELS, "supertonic-3-int8", "onnx", "vector_estimator.int8.onnx")
if os.path.exists(int8_path):
    report(
        "CPU int8",
        ort.InferenceSession(int8_path, sess_options=so, providers=["CPUExecutionProvider"]),
        ref=ref32,
    )

htp = np.load(os.path.join(WORK, "ort_htp_out.npy"))
d = htp.astype(np.float64) - ref32.astype(np.float64)
print(
    "HTP int8 DLC vs CPU fp32: corr %.5f  MAE %.5f  rel_MAE %.2f%%"
    % (
        np.corrcoef(htp.ravel(), ref32.ravel())[0, 1],
        np.abs(d).mean(),
        100 * np.abs(d).mean() / np.abs(ref32).mean(),
    )
)
print("ref32 stats: std %.4f  absmax %.4f" % (ref32.std(), np.abs(ref32).max()))