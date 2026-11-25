"""Kontaktlapok a kézi címkézéshez: soronként, arconként időbeli arckivágások.

A 15 hosszú gyanús sor mindegyikéről egy PNG készül: soronként egy arc,
oszloponként egy időminta (a sor elejétől a végéig egyenletesen). Így a
szájmozgás (nyit-zár ciklusok) szabad szemmel eldönthető: melyik arc beszél,
vagy egyik sem (képen kívüli beszélő).

Használat:
  python -m dubstudio.tools.label_sheets            # mind a 15 sor
  python -m dubstudio.tools.label_sheets 0 7 13     # csak ezek az indexek
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dubstudio import lightasd  # noqa: E402
from dubstudio.asd_visual import MODEL, Yunet, frames, track_faces  # noqa: E402
from dubstudio.config import load_project  # noqa: E402

SIZE = "1280x720"
COLS = 9            # időminta soronként
CELL = 190          # cella mérete a lapon
PAD = 0.15


def sheet(net: Yunet, src: Path, start: float, dur: float, tag: str) -> Image.Image:
    fps = COLS / max(dur, 0.5)
    x, _ = frames(src, start, dur, size=SIZE, fps=fps)
    tracks = track_faces(net, x)
    if not tracks:
        img = Image.new("RGB", (CELL * 3, CELL), (20, 20, 24))
        ImageDraw.Draw(img).text((10, 10), "nincs arc", fill=(230, 90, 90))
        return img
    rows = []
    for ti, tr in enumerate(tracks):
        cells = []
        for i in range(len(x)):
            box = tr["boxes"].get(i)
            if box is None:
                cells.append(Image.new("RGB", (CELL, CELL), (16, 16, 18)))
                continue
            cx, cy, bw, bh, _ = box
            c = lightasd.crop(x[i], cx, cy, bw, bh, out=CELL, expand=1.35)
            cells.append(Image.fromarray(c).convert("RGB"))
        rows.append((ti, tr, cells))
    w = CELL * COLS
    h = CELL * len(rows) + 22
    out = Image.new("RGB", (w, h + 22 * len(rows)), (12, 12, 14))
    d = ImageDraw.Draw(out)
    y = 0
    for ti, tr, cells in rows:
        d.text((4, y + 4),
               f"{tag}  arc#{ti}  méret {round(tr['w'])}px  "
               f"közép ({round(tr['cx'])},{round(tr['cy'])})  "
               f"{(tr['last'] - tr['first']) / max(len(x) - 1, 1) * 100:.0f}% követve",
               fill=(240, 240, 245))
        y += 22
        for i, c in enumerate(cells):
            out.paste(c, (i * CELL, y))
        y += CELL
    return out


def filmstrip(net: Yunet, src: Path, start: float, dur: float, tag: str,
              cols: int = 4, rows_n: int = 3) -> Image.Image:
    """Teljes képkockák időrendben, a detektált arcok bekeretezve.

    A kivágás-lap (``sheet``) a követő hibáit is mutatja; ez a lap a valós
    jelenetet: látszik, kik vannak a képen és mozog-e a szájuk.
    """
    n = cols * rows_n
    fps = n / max(dur, 0.5)
    x, _ = frames(src, start, dur, size=SIZE, fps=fps)
    cw, ch = 470, 264
    out = Image.new("RGB", (cw * cols, ch * rows_n + 18 * rows_n), (10, 10, 12))
    d = ImageDraw.Draw(out)
    for i in range(min(n, len(x))):
        fr = np.stack([x[i]] * 3, axis=2)
        im = Image.fromarray(fr).resize((cw, ch))
        r, c = divmod(i, cols)
        y = r * (ch + 18) + 18
        dd = ImageDraw.Draw(im)
        for d_ in net.detect(x[i]):
            cx, cy, bw, bh, sc = d_
            k = cw / float(x.shape[2])
            box = [(cx - bw / 2) * k, (cy - bh / 2) * k,
                   (cx + bw / 2) * k, (cy + bh / 2) * k]
            dd.rectangle(box, outline=(60, 220, 120), width=2)
            dd.text((box[0] + 3, box[1] + 2), f"{round(bw)}px", fill=(60, 220, 120))
        out.paste(im, (c * cw, y))
        d.text((c * cw + 4, y - 15),
               f"{tag}  +{i / fps:.2f}s = {start + i / fps:.1f}s",
               fill=(235, 235, 240))
    return out


def main(argv: list[str]) -> int:
    p = load_project("projects/biohackers.json")
    ep = p.ep_dir("E04")
    rows = json.loads((ep / "visual" / "asd.json").read_text(encoding="utf-8"))
    want = {int(a) for a in argv} if argv else set(range(len(rows)))
    hits = sorted(Path(p.data_dir).glob("*E04*.mkv"))
    if not hits:
        raise FileNotFoundError(p.data_dir)
    net = Yunet(MODEL)
    out_dir = ep / "visual" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        if i not in want:
            continue
        start = max(0.0, r["t"] - PAD)
        dur = (r["end"] - r["t"]) + 2 * PAD
        img = sheet(net, hits[0], start, dur, f"#{i} {r['t']:.1f}s")
        img.save(out_dir / f"row{i:02d}.png")
        fs = filmstrip(net, hits[0], start, dur, f"#{i} {r['t']:.1f}s")
        fs.save(out_dir / f"strip{i:02d}.png")
        print(f"row{i:02d}.png + strip{i:02d}.png  {dur:.1f}s  {img.width}x{img.height}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
