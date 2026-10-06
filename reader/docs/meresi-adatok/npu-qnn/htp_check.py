"""Is the QNN EP really executing the DLC, or silently falling back to CPU?"""
import collections
import json
import os
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as q

MODELS = "C:/Users/istva/Dev/portfolio/Projects/audiobook narrator/auris/reader/models"
ONNX = MODELS + "/supertonic-3/onnx/vector_estimator.onnx"
DLC = os.path.abspath("vector_estimator_int8.dlc")

NAMES = ["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask",
         "current_step", "total_step"]
S = {"noisy_latent": (1, 144, 64), "text_emb": (1, 256, 96), "style_ttl": (1, 50, 256),
     "latent_mask": (1, 1, 64), "text_mask": (1, 1, 96), "current_step": (1,),
     "total_step": (1,)}


def load_feed(tag="s0"):
    return {n: np.fromfile("inputs/%s_%s.raw" % (tag, n), dtype=np.float32).reshape(S[n])
            for n in NAMES}


ort.register_execution_provider_library(q.get_ep_name(), q.get_library_path())
devs = [d for d in ort.get_ep_devices()
        if d.ep_name == q.get_ep_name() and "NPU" in str(d.device.type)]

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.enable_profiling = True
so.log_severity_level = 3
so.add_provider_for_devices(devs, {
    "backend_path": os.path.dirname(q.__file__),
    "graph_path": DLC,
    "profiling_level": "basic",
})
sess = ort.InferenceSession(ONNX, sess_options=so)
print("providers:", sess.get_providers())

feed = load_feed()
o1 = sess.run(None, feed)[0]
feed2 = dict(feed)
feed2["noisy_latent"] = feed["noisy_latent"] * 3.0
o2 = sess.run(None, feed2)[0]

so2 = ort.SessionOptions()
so2.intra_op_num_threads = 12
cpu = ort.InferenceSession(ONNX, sess_options=so2, providers=["CPUExecutionProvider"])
c1 = cpu.run(None, feed)[0]
c2 = cpu.run(None, feed2)[0]

print("qnn(out1) == qnn(out2) :", np.array_equal(o1, o2))
print("qnn(out1) == cpu(out1) :", np.array_equal(o1, c1), "maxd", np.abs(o1 - c1).max())
print("qnn(out2) == cpu(out2) :", np.array_equal(o2, c2), "maxd", np.abs(o2 - c2).max())

prof = sess.end_profiling()
counts = collections.Counter()
with open(prof) as fh:
    for line in fh:
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        counts[e.get("provider")] += 1
print("node counts by provider:", dict(counts))
print("profile:", prof)