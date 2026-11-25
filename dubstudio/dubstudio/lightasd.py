"""Light-ASD: tanított aktív-beszélő modell (CVPR 2023) — CPU, PyTorch.

A korábbi megoldás (``dubstudio/asd.py``) a száj-sáv mozgását korrelálta a
hang-burkológéppel. Az mérés szerint ez a jel a zajszinten van: 2 s-os
ablakon a véletlen korreláció is eléri r≈0,8-at, ezért 15 hosszú soron
**0** megerősítést adott. Ez a modul valódi, tanított ASD-t használ.

Eredet: Junhua Liao et al., *A Light Weight Model for Active Speaker
Detection* (CVPR 2023), https://github.com/Junhua-Liao/Light-ASD (MIT).
A háló 1,0 M paraméter, 0,6 GFLOP; a súlyok a repó ``weight/`` mappájában
vannak (``pretrain_AVA_CVPR.model`` = AVA, ``finetuning_TalkSet.model`` =
TalkSet finomhangolás). A kód változatlan architektúra, csak CPU-ra és a mi
képcsőnkre igazítva.

Bemenet (a tanítással egyezően):
  * kép: 25 fps, szürke, **112×112**, arcra vágva (fej ~ a kép közepe);
  * hang: MFCC 13 együttható **100 fps**-en (25 ms ablak, 10 ms lépés),
    azaz képkockánként pontosan 4 MFCC-sor.

Kimenet: képkockánkénti „ez az arc beszél” valószínűség (softmax, 0–1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FACE = 112           # a háló bemeneti arcmérete
VIDEO_FPS = 25.0     # a tanítás 25 fps-en történt
AUDIO_FPS = 100.0    # MFCC ráta: 4 sor / képkocka
SR = 16000
MIN_FRAMES = 25      # 1 s alatt nincs értelme dönteni


class _AudioBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(cin, cout, (3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(cout, cout, (1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.m_5 = nn.Conv2d(cin, cout, (5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(cout, cout, (1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.last = nn.Conv2d(cout, cout, (1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.relu(self.bn_m_3(self.m_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_m_5(self.m_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class _VisualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, down: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        if down:
            self.s_3 = nn.Conv3d(cin, cout, (1, 3, 3), stride=(1, 2, 2),
                                 padding=(0, 1, 1), bias=False)
            self.bn_s_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.s_5 = nn.Conv3d(cin, cout, (1, 5, 5), stride=(1, 2, 2),
                                 padding=(0, 2, 2), bias=False)
            self.bn_s_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(cout, cout, (5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        else:
            self.s_3 = nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1), bias=False)
            self.bn_s_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.s_5 = nn.Conv3d(cin, cout, (1, 5, 5), padding=(0, 2, 2), bias=False)
            self.bn_s_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(cout, cout, (5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.last = nn.Conv3d(cout, cout, (1, 1, 1), bias=False)
        self.bn_last = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.relu(self.bn_s_3(self.s_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_s_5(self.s_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class LightASD(nn.Module):
    """A CVPR 2023-as Light-ASD háló (vizualitás + hang + BGRU + 2 osztály)."""

    def __init__(self) -> None:
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()            # type: ignore[assignment]
        self.GRU = _BGRU(128)                          # type: ignore[assignment]
        self.FC = nn.Linear(128, 2)

    def forward_visual_frontend(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w, h = x.shape
        x = x.view(b, 1, t, w, h)
        x = (x / 255 - 0.4161) / 0.1688
        return self.visualEncoder(x)

    def forward_audio_frontend(self, x: torch.Tensor) -> torch.Tensor:
        return self.audioEncoder(x.unsqueeze(1).transpose(2, 3))

    def forward_audio_visual_backend(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.GRU(a + v).reshape(-1, 128)

    @torch.no_grad()
    def probs(self, faces: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
        """Képkockánkénti beszél-valószínűség.

        ``faces``: (T, 112, 112) uint8 szürke arckivágások 25 fps-en.
        ``mfcc``: (4T, 13) MFCC 100 fps-en, ugyanarra az időablakra.
        """
        faces = np.asarray(faces)
        mfcc = np.asarray(mfcc, dtype=np.float32)
        n = min(len(faces), len(mfcc) // 4)
        if n < MIN_FRAMES:
            return np.zeros(0, dtype=np.float32)
        v = torch.from_numpy(faces[:n].astype(np.float32)).unsqueeze(0)
        a = torch.from_numpy(np.ascontiguousarray(mfcc[: 4 * n])).unsqueeze(0)
        logits = self.FC(self.forward_audio_visual_backend(
            self.forward_audio_frontend(a), self.forward_visual_frontend(v)))
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


class _VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _VisualBlock(1, 32, down=True)
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64)
        self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        b, c, t, w, h = x.shape
        x = self.maxpool(x.transpose(1, 2).reshape(b * t, c, w, h))
        return x.view(b, t, c)


class _AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _AudioBlock(1, 32)
        self.pool1 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block2 = _AudioBlock(32, 64)
        self.pool2 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block3 = _AudioBlock(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        return x.mean(dim=2, keepdim=True).squeeze(2).transpose(1, 2)


class _BGRU(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.gru_forward = nn.GRU(channel, channel, batch_first=True)
        self.gru_backward = nn.GRU(channel, channel, batch_first=True)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru_forward(x)
        x = torch.flip(self.gelu(x), dims=[1])
        x, _ = self.gru_backward(x)
        return self.gelu(torch.flip(x, dims=[1]))


def speech_weight(mfcc: np.ndarray, rel_db: float = 25.0) -> np.ndarray:
    """Beszéd-súly MFCC-soronként 1/0 (a 0. oszlop a log-energia).

    Enélkül a soron belüli csend lehúzza az átlagot: egy 8 s-os sorban a
    beszéd lehet 3 s, és a maradék 5 s csendben a háló nyugodtan ad 0,05-öt.
    A „beszél-e ez az arc **a hallható beszéd alatt**” kérdésre így
    válaszolunk. A küszöb a leghangosabb MFCC-sor −25 dB-je.
    """
    e = np.asarray(mfcc, dtype=np.float64)[:, 0]
    if len(e) == 0:
        return np.zeros(0, dtype=np.float32)
    w = (e >= float(e.max()) - 0.23026 * rel_db).astype(np.float32)
    if w.sum() < 5:            # túl halk/egyenletes jel: ne dobjunk el mindent
        w[:] = 1.0
    return w


def score_line(model: LightASD, faces: np.ndarray, mfcc: np.ndarray,
               rel_db: float = 25.0, smooth: int = 4,
               present: np.ndarray | None = None) -> dict:
    """Egy arc pontszáma egy sorra: beszéd alatti átlagos valószínűség.

    ``prob`` a döntés alapja (0–1), ``frames`` a simított képkocka-sor a
    beszéd-intervallumokhoz, ``speech_s`` a tényleges beszéd hossza.

    ``present``: képkockánkénti maszk, hogy az arc tényleg **látszik-e** azon
    a képkockán. Enélkül a követő lyukait kitöltő (elavult) kivágás is
    beleszámít, és egy rövid közeli vágás beszélőjét a sor többi része lehúzza.
    """
    n = min(len(faces), len(mfcc) // 4)
    if n < MIN_FRAMES:
        return {"prob": 0.0, "frames": np.zeros(0), "speech_s": 0.0, "n": n}
    if present is not None:
        pres = np.asarray(present, dtype=bool)[:n]
    else:
        pres = np.ones(n, dtype=bool)
    p = model.probs(faces[:n], mfcc[: 4 * n])
    wf = speech_weight(mfcc[: 4 * n], rel_db).reshape(n, 4).mean(axis=1)
    if smooth > 0 and n > 2 * smooth:
        # szélen ismételt képkocka: így a konstans jel átlaga nem csökken
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        ps = np.convolve(np.pad(p, smooth, mode="edge"), k, mode="valid")
    else:
        ps = p
    m = (wf > 0.5) & pres
    if not m.any():                     # nincs beszéd- és jelenlét-átfedés
        m = pres
    prob = float(ps[m].mean()) if m.any() else float(ps.mean())
    return {"prob": prob, "frames": ps, "speech_s": float(m.sum() / VIDEO_FPS),
            "n": int(pres.sum()), "window": n}


def load(weight: Path | str) -> LightASD:
    """Súlyok betöltése (a checkpoint ``model.*`` + ``lossAV.FC.*`` kulcsai)."""
    model = LightASD()
    sd = torch.load(str(weight), map_location="cpu", weights_only=True)
    new = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new[k[6:]] = v
        elif k.startswith("lossAV.FC."):
            new[k[len("lossAV."):]] = v
    model.load_state_dict(new, strict=True)
    model.eval()
    return model


def mfcc(audio16: np.ndarray, fps: float = VIDEO_FPS) -> np.ndarray:
    """MFCC 13 (100 fps), a tanítás beállításaival (python_speech_features)."""
    import python_speech_features as psf

    scale = VIDEO_FPS / float(fps)      # nem 25 fps videónál az ablakot nyújtjuk
    m = psf.mfcc(np.asarray(audio16, dtype=np.float64), SR, numcep=13,
                 winlen=0.025 * scale, winstep=0.010 * scale)
    return np.asarray(m, dtype=np.float32)


def square_box(cx: float, cy: float, bw: float, bh: float,
               expand: float = 1.3) -> tuple[float, float, float]:
    """(x0, y0, oldalhossz): az arcboxból négyzetes kivágás, kis ráhagyással."""
    side = max(float(bw), float(bh)) * float(expand)
    return float(cx) - side / 2, float(cy) - side / 2, side


def _bilinear(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = img.shape
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    fy = (ys - y0).astype(np.float32)[:, None]
    fx = (xs - x0).astype(np.float32)[None, :]

    def at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        return np.where(ok, img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)], 0.0)

    top = at(y0[:, None], x0[None, :]) * (1 - fx) + at(y0[:, None], x0[None, :] + 1) * fx
    bot = at(y0[:, None] + 1, x0[None, :]) * (1 - fx) + at(y0[:, None] + 1, x0[None, :] + 1) * fx
    return top * (1 - fy) + bot * fy


def crop(img: np.ndarray, cx: float, cy: float, bw: float, bh: float,
         out: int = FACE, expand: float = 1.3) -> np.ndarray:
    """Négyzetes arckivágás ``out×out``-ra (bilineáris, a kép szélén 0-val)."""
    x0, y0, side = square_box(cx, cy, bw, bh, expand)
    step = side / out
    xs = x0 + (np.arange(out) + 0.5) * step - 0.5
    ys = y0 + (np.arange(out) + 0.5) * step - 0.5
    return np.clip(_bilinear(img.astype(np.float32), ys, xs), 0, 255).astype(np.uint8)
