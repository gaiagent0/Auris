"""Download the gated pltobing/XTTSv2-Streaming-ONNX snapshot (HF_TOKEN must have access)."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download

REPO = "pltobing/XTTSv2-Streaming-ONNX"
DEST = r"C:\AI\xtts-onnx"

t0 = time.time()
path = snapshot_download(
    REPO,
    local_dir=DEST,
    allow_patterns=[
        "*.py",
        "*.txt",
        "*.json",
        "*.npy",
        "xtts_onnx/*.onnx",
        "audio_ref/*",
    ],
    max_workers=4,
)
print("snapshot:", path, f"{time.time() - t0:.1f}s")

total = 0
for root, _dirs, files in os.walk(DEST):
    for f in files:
        p = os.path.join(root, f)
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
print(f"total on disk: {total / 1e6:.1f} MB")
for root, _dirs, files in os.walk(os.path.join(DEST, "xtts_onnx")):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, DEST):55s} {os.path.getsize(p) / 1e6:9.2f} MB")
sys.exit(0)
