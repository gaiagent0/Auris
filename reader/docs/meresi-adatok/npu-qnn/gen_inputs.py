"""Random (deterministic) HTP input files + input_list for vector_estimator.dlc.

qairt-net-run / qairt-quantizer expect ONE line per inference sample, with the
input files of that sample separated by spaces, in the DLC's input order.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
os.makedirs(OUT, exist_ok=True)

# (name, native NHWC shape) in DLC input order
SPECS = [
    ("noisy_latent", (1, 144, 64)),
    ("text_emb", (1, 256, 96)),
    ("style_ttl", (1, 50, 256)),
    ("latent_mask", (1, 1, 64)),
    ("text_mask", (1, 1, 96)),
    ("current_step", (1,)),
    ("total_step", (1,)),
]


def make_sample(seed: int, out_dir: str, tag: str) -> list[str]:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, shape in SPECS:
        if name == "noisy_latent":
            arr = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        elif name in ("latent_mask", "text_mask"):
            arr = np.ones(shape, dtype=np.float32)
        elif name == "current_step":
            arr = np.array([0.0], dtype=np.float32)
        elif name == "total_step":
            arr = np.array([10.0], dtype=np.float32)
        else:
            arr = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        p = os.path.join(out_dir, f"{tag}_{name}.raw")
        arr.tofile(p)
        paths.append(p)
    return paths


single = make_sample(1234, OUT, "s0")
with open(os.path.join(OUT, "input_list.txt"), "w") as fh:
    fh.write(" ".join(single) + "\n")

calib_lines = []
for i in range(8):
    calib_lines.append(" ".join(make_sample(1000 + i, os.path.join(OUT, "calib"), f"c{i}")))
with open(os.path.join(OUT, "calib_list.txt"), "w") as fh:
    fh.write("\n".join(calib_lines) + "\n")

print("input_list.txt :", single[0])
print("calib_list.txt :", len(calib_lines), "samples")