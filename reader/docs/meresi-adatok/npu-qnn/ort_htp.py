"""Run the static vector_estimator DLC on the Hexagon HTP through the ORT QNN EP."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

WORK = os.path.dirname(os.path.abspath(__file__))
QNN_LIB = os.path.join(os.path.dirname(qnn_ep.__file__))
IN_DIR = os.path.join(WORK, "inputs")

MODELS = os.path.join(
    "C:\\", "Users", "istva", "Dev", "portfolio", "Projects",
    "audiobook narrator", "auris", "reader", "models", "supertonic-3", "onnx",
)

NAMES = [
    "noisy_latent",
    "text_emb",
    "style_ttl",
    "latent_mask",
    "text_mask",
    "current_step",
    "total_step",
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

print("ort", ort.__version__, "| qnn ep lib:", qnn_ep.get_library_path())
ort.register_execution_provider_library(qnn_ep.get_ep_name(), qnn_ep.get_library_path())

dlc = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "vector_estimator_int8.dlc")
onnx_model = (
    sys.argv[2] if len(sys.argv) > 2 else os.path.join(MODELS, "vector_estimator.onnx")
)

devs = [
    d
    for d in ort.get_ep_devices()
    if d.ep_name == qnn_ep.get_ep_name() and "NPU" in str(d.device.type)
]
print("using devices:", [(str(d.device.type), d.device.device_id) for d in devs])

opts = {
    "backend_path": QNN_LIB,
    "graph_path": dlc,
    "profiling_level": "basic",
}
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.log_severity_level = 2
so.add_provider_for_devices(devs, opts)

feed = {}
for n in NAMES:
    a = np.fromfile(os.path.join(IN_DIR, "s0_%s.raw" % n), dtype=np.float32)
    feed[n] = a.reshape(SHAPES[n])

sess = ort.InferenceSession(onnx_model, sess_options=so)
print("providers:", sess.get_providers())
print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

out = sess.run(None, feed)
print("outputs returned:", len(out), [np.asarray(o).shape for o in out])
np.save(os.path.join(WORK, "ort_htp_out.npy"), np.asarray(out[0]).astype(np.float32))

for _ in range(3):
    sess.run(None, feed)
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    sess.run(None, feed)
    ts.append(time.perf_counter() - t0)
ts = np.array(ts)
print(
    "HTP exec: min %.2f ms  median %.2f ms  mean %.2f ms"
    % (ts.min() * 1000, np.median(ts) * 1000, ts.mean() * 1000)
)