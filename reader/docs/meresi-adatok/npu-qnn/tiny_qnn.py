"""Can the ORT QNN EP load its HTP backend and run a tiny static model?"""
import collections
import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn as q
from onnx import TensorProto, helper, numpy_helper

BACKEND = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(q.__file__)

os.makedirs("tiny", exist_ok=True)
path = "tiny/tiny_conv.onnx"

rng = np.random.default_rng(0)
w = numpy_helper.from_array(rng.standard_normal((32, 16, 3, 3)).astype(np.float32), "w")
b = numpy_helper.from_array(rng.standard_normal((32,)).astype(np.float32), "b")
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16, 32, 32])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32, 30, 30])
node = helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
graph = helper.make_graph([node], "tiny", [x], [y], [w, b])
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, path)
print("model:", path, os.path.getsize(path), "bytes | backend:", BACKEND)

ort.register_execution_provider_library(q.get_ep_name(), q.get_library_path())
npu = [d for d in ort.get_ep_devices() if d.ep_name == q.get_ep_name() and "NPU" in str(d.device.type)]
print("npu devices:", len(npu))

feed = {"x": np.random.default_rng(1).standard_normal((1, 16, 32, 32)).astype(np.float32)}

so = ort.SessionOptions()
so.enable_profiling = True
so.log_severity_level = 1
so.add_provider_for_devices(npu, {"backend_path": BACKEND, "profiling_level": "basic"})
sess = ort.InferenceSession(path, sess_options=so)
print("session providers:", sess.get_providers())

sess.run(None, feed)
for _ in range(5):
    sess.run(None, feed)
t = []
for _ in range(20):
    t0 = time.perf_counter()
    sess.run(None, feed)
    t.append(time.perf_counter() - t0)
print("exec median %.3f ms" % (np.median(t) * 1000))

prof = sess.end_profiling()
d = json.load(open(prof))
c = collections.Counter(e.get("args", {}).get("provider") for e in d)
print("by provider:", dict(c))
os.remove(prof)