"""Egységtesztek az aktív beszélő (ASD) pontozásához — kép és ffmpeg nélkül.

A korábbi hiba tanulsága: a nyers Pearson-korreláció rövid ablakon
véletlenből is ~0,5, ezért „mindenki beszél” lett. Ezek a tesztek azt
rögzítik, hogy a null-eloszlásos z-érték ezt kiszűri, és hogy a döntés
(„melyik arc beszél”) a szájaktív időt is megköveteli.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import asd  # noqa: E402


def _rhythm(n: int = 60, seed: int = 0) -> np.ndarray:
    """Szótag-ritmus: gyors nyitás-zárás, mint egy beszélő száj."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 3.0) > 0).astype(float) + 0.15 * rng.random(n)


class CorrSigTest(unittest.TestCase):
    def test_aligned_signal_is_significant(self):
        """Ha a száj és a hang ugyanazt a ritmust követi, z nagy — 24 mintán is."""
        for n in (24, 60):
            mouth = _rhythm(n, seed=1)
            env = mouth + 0.05 * np.random.default_rng(2).random(n)
            r, z = asd.corr_sig(mouth, env)
            self.assertGreater(r, 0.9)
            self.assertGreater(z, 2.5)

    def test_independent_noise_is_not_significant(self):
        """Független jelekre a Fisher-z közel N(0,1): 30 párból legfeljebb 1
        csúszhat át a p<0,01 küszöbön (korábban a lag-keresés miatt bármelyik
        pár átcsúszott, akár z=11-gyel is)."""
        rng = np.random.default_rng(7)
        out = [asd.corr_sig(rng.random(60), rng.random(60)) for _ in range(30)]
        rs = [o[0] for o in out]
        zs = [o[1] for o in out]
        # mért: zaj max r ≈ 0,41 (n=60) · z medián ≈ 0,1, max ≈ 4,6
        # → a **hatásméret** (r) a szétválasztó, a z csak kísérő feltétel
        self.assertLess(max(rs), asd.R_MIN)
        self.assertLess(float(np.median(zs)), 1.5)

    def test_constant_signal_gives_zero(self):
        """Lapos jel (álló arc, csend): nincs mit korrelálni → (0, 0)."""
        self.assertEqual(asd.corr_sig(np.ones(40), np.arange(40, dtype=float)), (0.0, 0.0))
        self.assertEqual(asd.corr_sig(np.zeros(40), np.zeros(40)), (0.0, 0.0))

    def test_too_short_returns_zero(self):
        """Néhány minta alatt nem döntünk."""
        self.assertEqual(asd.corr_sig([1.0, 2.0, 1.0], [1.0, 3.0, 2.0]), (0.0, 0.0))

    def test_z_is_scale_invariant(self):
        """A döntés ne függjön a jel erősségétől (fényerő, hangerő)."""
        mouth = _rhythm(48, seed=3)
        env = _rhythm(48, seed=3) * 0.01
        _, z1 = asd.corr_sig(mouth, env)
        _, z2 = asd.corr_sig(mouth * 50, env * 5000)
        self.assertAlmostEqual(z1, z2, places=6)


class MouthSpansTest(unittest.TestCase):
    def test_two_movements_two_intervals(self):
        sig = np.zeros(24)
        sig[4:8] = 10.0        # 0,33–0,67 s
        sig[14:18] = 9.0       # 1,17–1,50 s
        spans = asd.mouth_spans(sig, fps=12.0)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][0], 4 / 12, places=3)
        self.assertAlmostEqual(spans[1][1], 18 / 12, places=3)

    def test_short_flash_is_dropped(self):
        sig = np.zeros(24)
        sig[10] = 20.0         # 1 képkocka = 0,08 s < 0,12 s
        self.assertEqual(asd.mouth_spans(sig, fps=12.0), [])

    def test_empty_signal(self):
        self.assertEqual(asd.mouth_spans(np.array([]), fps=12.0), [])


class ChanceTest(unittest.TestCase):
    """A korreláció csak mérés: a döntést a Light-ASD hozza (lásd test_lightasd).

    A ``chance_r`` görbe maradt, mert megmagyarázza, miért nem dönthet a
    nyers r: rövid ablakon a véletlen is magas korrelációt ad.
    """

    def test_chance_r_grows_with_short_window(self):
        """A véletlen szint rövidebb ablakon magasabb — ezt a döntés tükrözi.

        Mért értékek: 24 mintán 0,80 · 60 mintán 0,41 (véletlen párok,
        ±3 képkockás maximum). A szükséges r mindig efölött van."""
        self.assertAlmostEqual(asd.chance_r(24), 0.80, delta=0.02)
        self.assertAlmostEqual(asd.chance_r(60), 0.41, delta=0.05)
        self.assertGreater(asd.chance_r(24), asd.chance_r(60))
        self.assertGreater(asd.r_needed(24), 0.85)
        self.assertLess(asd.r_needed(60), 0.6)

    def test_no_correlation_decision_left(self):
        """A korrelációs döntést kivettük: nincs ilyen függvény a magban."""
        self.assertFalse(hasattr(asd, "score_candidates"))


if __name__ == "__main__":
    unittest.main()
