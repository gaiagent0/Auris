"""Egy kiválasztott arc nagyított időbeli csíkja (kézi ellenőrzéshez).

A ``asd_bench``/``asd.json`` koordinátái 960×540-re vonatkoznak; itt 1920×1080-en
dekódolunk (2× nagyítás), így a száj tényleg látszik.

  python -m dubstudio.tools.zoom 1032.42 0 420 198 69 1037.62
        idő  arc#  cx  cy  méret  sor vége
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dubstudio import lightasd  # noqa: E402
from dubstudio.asd_visual import frames  # noqa: E402
from dubstudio.config import load_project  # noqa: E402


def main(argv: list[str]) -> int:
    t, face, cx, cy, size, end = (float(a) for a in argv[:6])
    p = load_project("projects/biohackers.json")
    src = sorted(Path(p.data_dir).glob("*E04*.mkv"))[0]
    dur = end - t
    cols = 9
    x, fps = frames(src, t, dur, size="1920x1080", fps=cols / max(dur, 0.5))
    cell = 200
    out = Image.new("RGB", (cell * cols, cell + 18), (12, 12, 14))
    d = ImageDraw.Draw(out)
    for i in range(min(cols, len(x))):
        c = lightasd.crop(x[i], cx * 2, cy * 2, size * 2, size * 2, out=cell,
                          expand=2.0)
        out.paste(Image.fromarray(c).convert("RGB"), (i * cell, 18))
        d.text((i * cell + 3, 3), f"{t + i / fps:.2f}s", fill=(240, 240, 245))
    path = p.ep_dir("E04") / "visual" / "labels" / f"zoom_{t:.2f}_{int(face)}.png"
    out.save(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
