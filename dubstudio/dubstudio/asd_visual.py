"""Aktív beszélő detektálás (ASD) — melyik **arc** beszél, és mettől meddig.

Két pontozó fut ugyanazon a képkockákon, ugyanazon arcokon:

* **light** (alapértelmezett): tanított Light-ASD (CVPR 2023, 1 M paraméter)
  25 fps-es, 112×112-es arckivágásokon + MFCC-n. Ez ad valódi
  valószínűséget, és bizonytalan esetben „nem eldönthető”-t mond.
* **corr**: a korábbi, tanítás nélküli út — a száj-sáv mozgásának
  korrelációja a hang-burkológéppel. **Mérés:** a jel a zajszinten van
  (2 s-on a véletlen korreláció is r≈0,8), ezért ez már csak
  összehasonlítási alap és negatív ellenőrzés, nem dönt.

A kép sosem dönt egyedül a hang helyett: a kimenet háromféle lehet —
„arc #k beszél”, „nincs beszélő arc a képen (képen kívüli beszélő)”,
„nem eldönthető”. Utóbbi kettőnél a hang-alapú azonosítás a helyes.

Használat:
  python work/visual_asd.py E04 --suspicious 15           # 15 gyanús sor
  python work/visual_asd.py E04 341.12 718.67 1498.92     # kézi időpontok
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import audio, lightasd
from .asd import corr_sig, decide_light, mouth_spans, speak_spans
from .config import APP_ROOT, load_project

#: YuNet arcdetektor (~230 KB ONNX) — a ``work/`` gitignore-olt, így a
#: modell letöltése kézi lépés (lásd docs/asd.md).
MODEL = APP_ROOT / "work" / "models" / "face-yunet.onnx"
FPS = 25.0          # a Light-ASD 25 fps-en tanult
PAD = 0.15          # a sor előtti/utáni ráhagyás
MIN_WIN_S = 3.5     # ennél rövidebb soron nincs statisztikai erő
LAG = 6             # ±0,24 s a korrelációs úton (25 fps-en 6 képkocka)
# A képkocka mérete: kicsi (távoli) arcnál 640×360-on a szájmozgás elveszik.
SIZE = os.environ.get("ASD_SIZE", "960x540")
#: A YuNet pontszám-küszöbe. 0,6 helyett 0,35: sötét jelenetben a valódi arc
#: 0,28–0,5 pontot kap (lásd ``_stretch``), és a 0,6-os küszöbön a beszélő
#: eltűnt a mérésből (E04 447 s: „nincs arc a képen”, pedig 355 px-es arc).
#: A döntést nem ez a pontszám hozza, hanem a Light-ASD valószínűsége.
SCORE_MIN = float(os.environ.get("ASD_SCORE_MIN", "0.35"))
#: alapból az AVA-n tanult súly (ez a cikk mért benchmarkja); a TalkSet
#: finomhangolás is mérve van, lásd ``docs/asd.md`` (ASD_WEIGHT env).
WEIGHT = Path(__file__).resolve().parents[1] / "work" / "models" / os.environ.get(
    "ASD_WEIGHT", "light-asd-ava.model")


def _stretch(img: np.ndarray) -> np.ndarray:
    """Kontraszt-nyújtás a 2–98 percentilisre (sötét jelenet miatt).

    Mért példa (E04, 447,2 s): a nyers kép átlaga 27/255, a valódi, 355 px-es
    arcra a detektor csak 0,28–0,64 pontszámot ad → a 0,6-os küszöbön a fél
    sor elveszik. Nyújtás után ugyanaz az arc 0,62–0,64.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # A percentilist ritkított mintán számoljuk: 950×540-en a teljes kép
    # rendezése képkockánként ~50 ms lenne, a 6×-ra ritkítotté ~1 ms.
    sub = img[::6, ::6]
    lo, hi = np.percentile(sub, 2), np.percentile(sub, 98)
    if hi - lo < 8:                     # majdnem konstans kép: ne nyújtsuk fel
        return img
    return np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                   0, 255).astype(np.uint8)


class Yunet:
    """YuNet arcdetektor (640×640, anchor-free), onnxruntime-mal."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path),
                                         providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.sess.get_outputs()]

    def detect(self, bgr: np.ndarray, score_min: float = None
               ) -> list[tuple[float, float, float, float, float]]:
        if score_min is None:
            score_min = SCORE_MIN
        if bgr.ndim == 2:                 # szürke képkocka: 3 csatornásnak álcázzuk
            bgr = np.stack([bgr] * 3, axis=2)
        bgr = _stretch(bgr)
        h0, w0 = bgr.shape[:2]
        img = np.zeros((640, 640, 3), dtype=np.float32)
        scale = min(640 / w0, 640 / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        # egyszerű átméretezés numpy-val (nincs cv2)
        ys = (np.arange(nh) / scale).astype(np.int32).clip(0, h0 - 1)
        xs = (np.arange(nw) / scale).astype(np.int32).clip(0, w0 - 1)
        small = bgr[ys][:, xs].astype(np.float32)
        img[:nh, :nw] = small
        blob = img.transpose(2, 0, 1)[None]
        out = dict(zip(self.names, self.sess.run(None, {"input": blob})))
        faces = []
        for stride in (8, 16, 32):
            cls = out[f"cls_{stride}"][0][:, 0]
            obj = out[f"obj_{stride}"][0][:, 0]
            bbox = out[f"bbox_{stride}"][0]
            sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            keep = np.where(sc >= score_min)[0]
            if not len(keep):
                continue
            side = 640 // stride
            cols = keep % side
            rows = keep // side
            cx = (cols + bbox[keep, 0]) * stride
            cy = (rows + bbox[keep, 1]) * stride
            bw = np.exp(bbox[keep, 2]) * stride
            bh = np.exp(bbox[keep, 3]) * stride
            for i, k in enumerate(keep):
                faces.append((float(cx[i] / scale), float(cy[i] / scale),
                              float(bw[i] / scale), float(bh[i] / scale),
                              float(sc[k])))
        # nem-maximum kiszűrés (átfedő ablakok)
        faces.sort(key=lambda f: -f[4])
        out_faces: list[tuple[float, float, float, float, float]] = []
        for f in faces:
            if all(abs(f[0] - g[0]) > 0.5 * f[2] or abs(f[1] - g[1]) > 0.5 * f[3]
                   for g in out_faces):
                out_faces.append(f)
        return out_faces[:6]


def frames(src: Path, start: float, dur: float, size: str = "640x360",
           fps: float | None = None) -> tuple[np.ndarray, float]:
    """Szürke képkockák (T, H, W) egy időablakból, ffmpeg-gel.

    ``fps``: a kért mintavétel; alapból a modul ``FPS``-e. A hívónak
    ugyanazt az értéket kell használnia az idő-tengely számításához, különben
    a „másodpercek” csúsznak (ez volt az ASD egyik hibája: 8 fps-en dekódolt,
    de 12 fps-nek hitte).
    """
    rate = float(fps if fps else FPS)
    w, h = (int(v) for v in size.split("x"))
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
           "-i", str(src), "-vf", f"fps={rate},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = len(arr) // (w * h)
    return arr[: n * w * h].reshape(n, h, w), rate


#: Ennyi átlagos eltérés (0–255) a 16×16 kivágáson még ugyanaz az arc.
#: Mért: ugyanaz az arc 2 képkocka múlva 16, beszélő arc mozgás közben akár
#: 62–69, vágás után 76+ — laza küszöb, mert a vágást külön kezeljük.
THUMB_MAX = 90.0
#: Jelenetvágás: a 8×-ra ritkított képkockák átlagos eltérése. Mért: normál
#: képkockapár < 4, vágás 21–43 → 10 biztonságos. Vágásnál **nem** párosítunk
#: (különben a követő identitást vált: E04 965,5 s és 180,2 s).
CUT_DIFF = 10.0


def _thumb(img: np.ndarray, box: tuple, out: int = 16) -> np.ndarray:
    """Kicsi, megvilágítás-független arckivágás a követő azonosság-ellenőrzéséhez."""
    cx, cy, bw, bh = box[0], box[1], box[2], box[3]
    t = lightasd.crop(img, cx, cy, bw, bh, out=out, expand=1.2).astype(np.float32)
    t -= t.mean()
    s = t.std()
    return t / s if s > 1e-6 else t


def _thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Két thumb átlagos eltérése 0–255 skálán (normalizált képek)."""
    if a.shape != b.shape:
        return 1e9
    return float(np.abs(a - b).mean() * 64.0)


def track_faces(net: Yunet, x: np.ndarray) -> list[dict]:
    """Arcpályák: képkockánként a **kinézetre is** illeszkedő korábbi archoz.

    Csak távolsággal párosítva a követő identitást váltott: egy vágás után az
    új arc a régi pályára került, így a Light-ASD egy soron belül két különböző
    embert pontozott (E04 965 s, 1077 s, 180 s). Ezért a 16×16-os kivágás
    különbsége is számít: ``_thumb_diff`` < ``THUMB_MAX``.
    """
    tracks: list[dict] = []
    for i in range(len(x)):
        cut = False
        if i:
            a = x[i - 1, ::8, ::8].astype(np.float32)
            b = x[i, ::8, ::8].astype(np.float32)
            cut = float(np.abs(b - a).mean()) > CUT_DIFF
        dets = net.detect(x[i])
        for d in dets:
            th = _thumb(x[i], d)
            best, bd = None, 1e9
            for tr in tracks:
                if cut or tr["last"] < i - 4:
                    continue
                dist = abs(tr["cx"] - d[0]) + abs(tr["cy"] - d[1])
                if dist >= bd or dist >= 0.8 * max(d[2], tr["w"]):
                    continue
                if _thumb_diff(th, tr["thumb"]) > THUMB_MAX:
                    continue
                best, bd = tr, dist
            if best is None:
                tracks.append({"cx": d[0], "cy": d[1], "w": d[2], "h": d[3],
                               "first": i, "last": i, "score": d[4],
                               "thumb": th, "boxes": {i: d}})
            else:
                best.update(cx=d[0], cy=d[1], w=d[2], h=d[3], last=i,
                            score=d[4], thumb=th)
                best["boxes"][i] = d
    # Rövid pálya is lehet a **beszélő** (egy 1,5 s-os közeli vágás): ezért a
    # szűrő enyhe, a döntésnél úgyis számít a pálya hossza (``n_min``).
    return [t for t in tracks if t["last"] - t["first"] >= 5]


def mouth_signal(x: np.ndarray, tr: dict) -> np.ndarray:
    """Szájmozgás-jel képkockánként (a száj-sáv változása)."""
    _, h, w = x.shape
    vals = np.zeros(len(x), dtype=np.float32)
    for i, box in tr["boxes"].items():
        cx, cy, bw, bh, _ = box
        x0, x1 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
        y0, y1 = int(max(0, cy + bh * 0.10)), int(min(h, cy + bh * 0.50))
        if x1 - x0 < 6 or y1 - y0 < 4 or i == 0:
            continue
        a = x[i - 1, y0:y1, x0:x1].astype(np.float32)
        b = x[i, y0:y1, x0:x1].astype(np.float32)
        vals[i] = float(np.abs(b - a).mean())
    return vals


def audio_env(x16: np.ndarray, start: float, n: int, fps: float) -> np.ndarray:
    """Hang-burkológép ugyanazon az időskálán, mint a képkockák."""
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = int((start + i / fps) * audio.SR16)
        b = int((start + (i + 1) / fps) * audio.SR16)
        seg = x16[max(0, a):max(0, b)]
        out[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return out


def audio_slice(x16: np.ndarray, start: float, n_frames: int,
                fps: float = FPS) -> np.ndarray:
    """A képkockákhoz tartozó hang, pontosan 4 MFCC-sorral képkockánként."""
    s = int(round(start * audio.SR16))
    n = int(round(n_frames * audio.SR16 / fps))
    seg = np.asarray(x16[s:s + n], dtype=np.float32)
    # a framesig a végén 240 mintát veszít: ezt nullákkal pótoljuk
    return np.concatenate([seg, np.zeros(int(0.015 * audio.SR16), dtype=np.float32)])


def face_frames(x: np.ndarray, tr: dict, n: int) -> np.ndarray:
    """A követett arc 112×112-es kivágásai képkockánként (lyuk = előző kép)."""
    faces = np.zeros((n, lightasd.FACE, lightasd.FACE), dtype=np.uint8)
    have = np.zeros(n, dtype=bool)
    for i, (cx, cy, bw, bh, _) in tr["boxes"].items():
        if 0 <= i < n:
            faces[i] = lightasd.crop(x[i], cx, cy, bw, bh)
            have[i] = True
    last = None
    for i in range(n):
        if have[i]:
            last = i
        elif last is not None:
            faces[i] = faces[last]
    return faces


def analyze(net: Yunet, model, src: Path, x16: np.ndarray, start: float,
            dur: float, net_kind: str = "light") -> dict:
    x, fps = frames(src, start, dur, size=SIZE, fps=FPS)
    if len(x) < 4:
        return {"faces": 0, "speaking": None, "verdict": "nincs képkocka"}
    env = audio_env(x16, start, len(x), fps)
    mf = lightasd.mfcc(audio_slice(x16, start, len(x), fps), fps=fps)
    tracks = track_faces(net, x)
    corr_c, light_c = [], []
    for ti, tr in enumerate(tracks):
        center = [round(tr["cx"]), round(tr["cy"])]
        size = round(tr["w"])
        # --- korrelációs út (összehasonlítási alap) ---
        sig = mouth_signal(x, tr)
        r, z = corr_sig(sig, env, max_lag=LAG)
        spans = mouth_spans(sig, fps)
        corr_c.append({"face": ti, "corr": round(r, 3), "z": round(z, 2),
                       "n": int(len(sig)), "peak": round(float(sig.max()), 1),
                       "active_s": round(sum(b - a for a, b in spans), 2),
                       "center": center, "size": size,
                       "spans": [(round(start + a, 2), round(start + b, 2))
                                 for a, b in spans]})
        # --- Light-ASD út ---
        faces = face_frames(x, tr, len(x))
        # Csak azokon a képkockákon pontozunk, ahol ez az arc **ott van**:
        # különben egy 1,5 s-os közeli vágás beszélőjét a 7 s-os sor többi
        # képkockája (más arc / üres kép) lehúzza (E04 1365 s: p 0,26).
        present = np.zeros(len(x), dtype=bool)
        for i in tr["boxes"]:
            if 0 <= i < len(x):
                present[i] = True
        sc = lightasd.score_line(model, faces, mf, present=present)
        sp = speak_spans(sc["frames"], fps)
        light_c.append({"face": ti, "prob": round(sc["prob"], 3),
                        "n": int(sc["n"]), "size": size, "center": center,
                        "present_s": round(float(present.sum()) / fps, 2),
                        "speech_s": round(sc["speech_s"], 2),
                        "spans": [(round(start + a, 2), round(start + b, 2))
                                  for a, b in sp]})
    light_c, light_sp, light_why = decide_light(light_c)
    # A korreláció már csak bizonyíték (r, z, szájaktív idő), nem dönt.
    corr_c.sort(key=lambda c: -float(c.get("corr", 0.0)))
    out = {"faces": len(tracks),
           "light": {"speaking": light_sp, "verdict": light_why, "cands": light_c},
           "corr": {"speaking": None, "verdict": "mérés (nem dönt)",
                    "cands": corr_c}}
    chosen = out[net_kind]
    out.update(speaking=chosen["speaking"], verdict=f"[{net_kind}] {chosen['verdict']}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    tag = argv[0]
    rest = list(argv[1:])
    net_kind = "light"
    if "--net" in rest:
        i = rest.index("--net")
        net_kind = rest[i + 1]
        del rest[i:i + 2]
    if net_kind not in ("light", "corr"):
        raise SystemExit(f"ismeretlen --net: {net_kind}")
    p = load_project("projects/biohackers.json")
    ep = p.ep_dir(tag)
    utt = json.loads((ep / "utterances.json").read_text(encoding="utf-8"))
    if rest and rest[0] == "--suspicious":
        n = int(rest[1]) if len(rest) > 1 else 10
        rep_path = ep / "report.json"
        if not rep_path.is_file():
            raise FileNotFoundError(f"nincs riport (előbb futtasd a láncot): {rep_path}")
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        # A riport az ASR hibás sorait itt tartja: ``rows`` = bukó (CER >=),
        # ``watch_rows`` = figyelendő.
        asr = rep.get("asr") or {}
        bad = list(asr.get("rows") or []) + list(asr.get("watch_rows") or [])
        if not bad:
            raise RuntimeError("a riportban nincs gyanús ASR-sor (rows/watch_rows üres)")
        starts = [float(r["start"]) for r in bad if isinstance(r, dict) and "start" in r]
        rows, used = [], set()
        for s in starts:
            cand = min(utt, key=lambda u: abs(u["start"] - s))
            if abs(cand["start"] - s) <= 0.25 and cand["uid"] not in used:
                rows.append(cand)
                used.add(cand["uid"])
        long_rows = [u for u in rows if u["dur"] >= MIN_WIN_S]
        rows = sorted(long_rows or rows, key=lambda u: -u["dur"])[:n]
        print(f"[gyanús] {len(bad)} ASR-sor → {len(rows)} megnyilatkozás "
              f"(ebből {len(long_rows)} elég hosszú, {n} kért)", flush=True)
        times = [(u["start"], u["end"]) for u in rows]
    else:
        times = [(float(t), float(t) + MIN_WIN_S) for t in rest]
    hits = sorted(Path(p.data_dir).glob(f"*{tag[-3:]}*.mkv"))
    if not hits:
        raise FileNotFoundError(f"nincs forrás MKV: {p.data_dir}")
    src = hits[0]
    if not WEIGHT.is_file():
        raise FileNotFoundError(
            f"nincs Light-ASD súly: {WEIGHT}\n"
            "  git clone --depth 1 https://github.com/Junhua-Liao/Light-ASD && "
            "copy weight\\pretrain_AVA_CVPR.model work/models/light-asd-ava.model")
    net = Yunet(MODEL)
    model = lightasd.load(WEIGHT)
    x, sr = audio.load_mono(p.vocals(tag))
    x16 = audio.to_16k(x, sr)
    out_dir = ep / "visual"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in times:
        start = max(0.0, a - PAD)
        dur = (b - a) + 2 * PAD
        r = analyze(net, model, src, x16, start, dur, net_kind)
        r.update(t=round(a, 2), end=round(b, 2), uid=next(
            (u["uid"] for u in utt if abs(u["start"] - a) < 0.05), None))
        rows.append(r)
        light = r["light"]
        print(f"{a:8.2f}s  {light['verdict']}", flush=True)
        for c in light["cands"]:
            print(f"          arc#{c['face']} közép({c['center'][0]},{c['center'][1]}) "
                  f"p={c['prob']:.2f} beszéd {c['speech_s']}s "
                  f"méret={c['size']}px {c['spans']}")
        cc = r["corr"]["cands"]
        print("          [corr/mérés]"
              + (f" legjobb r {cc[0]['corr']:+.2f} z {cc[0]['z']:+.1f}"
                 f" aktív {cc[0]['active_s']}s (nem dönt)" if cc else " nincs arc"),
              flush=True)
    (out_dir / "asd.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"→ {out_dir / 'asd.json'}")
    # A riport JSON-ba bizonyítékként (nem kapuként) bekerül a képi ellenőrzés.
    rep_path = ep / "report.json"
    if rep_path.is_file():
        from . import report as report_mod

        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        ev = report_mod.asd_evidence(p, tag)
        if ev:
            ev["weight"] = WEIGHT.name
            rep["asd"] = ev
            rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            print(f"→ {rep_path} (asd: {ev['n']} sor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

