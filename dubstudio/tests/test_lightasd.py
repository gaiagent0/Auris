"""Light-ASD port tesztek — háló nélkül (fake modellel) és egy betöltéssel.

A hangsúly a képi előkészítésen és a döntésen van: az arckivágás geometriája,
a beszéd-súly (csend ne húzza le az átlagot), és a „nem eldönthető” kimenet.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import asd, lightasd  # noqa: E402

WEIGHT = Path(__file__).resolve().parents[1] / "work" / "models" / "light-asd-talkset.model"


class _Fake:
    """Állandó (vagy képkocka-függő) valószínűséget adó modell."""

    def __init__(self, value: float = 0.9) -> None:
        self.value = value

    def probs(self, faces, mfcc):
        n = min(len(faces), len(mfcc) // 4)
        return np.full(n, self.value, dtype=np.float32)


class CropTest(unittest.TestCase):
    def test_constant_image_stays_constant(self):
        img = np.full((360, 960), 128, dtype=np.uint8)
        c = lightasd.crop(img, 480, 180, 100, 100)
        self.assertEqual(c.shape, (112, 112))
        self.assertEqual(c.dtype, np.uint8)
        self.assertLessEqual(abs(int(c.mean()) - 128), 1)

    def test_outside_frame_is_zero_padded(self):
        img = np.full((360, 960), 200, dtype=np.uint8)
        c = lightasd.crop(img, 5, 5, 100, 100)      # a bal felső sarokban
        self.assertEqual(int(c[:40, :40].max()), 0)
        self.assertGreater(int(c[80:, 80:].mean()), 150)

    def test_square_box_uses_larger_side(self):
        x0, y0, side = lightasd.square_box(100, 50, 40, 80, expand=1.0)
        self.assertEqual(side, 80.0)
        self.assertEqual((x0, y0), (60.0, 10.0))


class SpeechWeightTest(unittest.TestCase):
    def test_quiet_frames_are_dropped(self):
        mf = np.zeros((40, 13), dtype=np.float32)
        mf[:, 0] = -10.0
        mf[10:20, 0] = 0.0                          # 10 dB-erősebb rész
        w = lightasd.speech_weight(mf, rel_db=25.0)
        self.assertEqual(float(w.sum()), 10.0)

    def test_flat_signal_keeps_everything(self):
        """Egyenletes hangerőn nem dobhatunk el mindent (nincs beszéd-súly)."""
        mf = np.zeros((40, 13), dtype=np.float32)
        w = lightasd.speech_weight(mf)
        self.assertEqual(float(w.sum()), 40.0)

    def test_empty(self):
        self.assertEqual(len(lightasd.speech_weight(np.zeros((0, 13)))), 0)


class ScoreLineTest(unittest.TestCase):
    def test_prob_follows_model(self):
        """Konstans modellkimenetre a pontszám ugyanaz — a simítás nem húzza le."""
        faces = np.zeros((60, 112, 112), dtype=np.uint8)
        mf = np.zeros((240, 13), dtype=np.float32)
        out = lightasd.score_line(_Fake(0.8), faces, mf)
        self.assertAlmostEqual(out["prob"], 0.8, places=4)
        self.assertAlmostEqual(out["speech_s"], 2.4, places=3)
        self.assertEqual(out["n"], 60)

    def test_silence_does_not_dilute(self):
        """A csend képkockái nem húzzák le az átlagot: ez volt a 0,62-es eset."""
        n = 100
        faces = np.zeros((n, 112, 112), dtype=np.uint8)
        mf = np.zeros((4 * n, 13), dtype=np.float32)
        mf[:, 0] = -20.0
        mf[4 * 50:, 0] = 0.0                        # a sor második fele beszéd

        class _Ramp(_Fake):
            def probs(self, f, m):
                k = min(len(f), len(m) // 4)
                p = np.full(k, 0.02, dtype=np.float32)
                p[50:50 + k] = 0.95
                return p[:k]

        out = lightasd.score_line(_Ramp(), faces, mf, smooth=0)
        self.assertGreater(out["prob"], 0.9)
        self.assertLess(out["speech_s"], float(n) / lightasd.VIDEO_FPS)

    def test_absent_frames_do_not_dilute(self):
        """Csak a jelenlévő képkockák számítanak: egy rövid közeli vágás
        beszélőjét a sor többi képkockája (más arc) nem húzhatja le."""
        n = 100
        faces = np.zeros((n, 112, 112), dtype=np.uint8)
        mf = np.zeros((4 * n, 13), dtype=np.float32)

        class _Seg(_Fake):
            def probs(self, f, m):
                k = min(len(f), len(m) // 4)
                p = np.full(k, 0.05, dtype=np.float32)
                p[40:60] = 0.9               # ebben a 20 képkockában látszik
                return p[:k]

        present = np.zeros(n, dtype=bool)
        present[40:60] = True
        out = lightasd.score_line(_Seg(), faces, mf, smooth=0, present=present)
        self.assertGreater(out["prob"], 0.85)
        self.assertEqual(out["n"], 20)

    def test_too_short_line(self):
        out = lightasd.score_line(_Fake(1.0), np.zeros((10, 112, 112), np.uint8),
                                  np.zeros((40, 13), np.float32))
        self.assertEqual(out["prob"], 0.0)


class DecideLightTest(unittest.TestCase):
    def _c(self, face, prob, size=100, n=75):
        return {"face": face, "prob": prob, "size": size, "n": n}

    def test_clear_winner_speaks(self):
        _, sp, why = asd.decide_light([self._c(0, 0.12), self._c(1, 0.91)])
        self.assertEqual(sp, 1)
        self.assertIn("beszél", why)

    def test_offscreen_speaker(self):
        _, sp, why = asd.decide_light([self._c(0, 0.10), self._c(1, 0.22)])
        self.assertIsNone(sp)
        self.assertIn("képen kívüli", why)

    def test_ambiguous_stays_undecided(self):
        # 0,42 a mért „nem eldönthető” sávban van (OFFSCREEN_P 0,35 – SPEAK_P 0,45).
        _, sp, why = asd.decide_light([self._c(0, 0.42), self._c(1, 0.10)])
        self.assertIsNone(sp)
        self.assertIn("nem eldönthető", why)

    def test_two_faces_without_margin(self):
        """Mindkét arc magas: nem lehet eldönteni, melyik beszél."""
        _, sp, why = asd.decide_light([self._c(0, 0.72), self._c(1, 0.66)])
        self.assertIsNone(sp)
        self.assertIn("nem eldönthető", why)

    def test_tiny_face_is_not_used(self):
        _, sp, why = asd.decide_light([self._c(0, 0.95, size=30)])
        self.assertIsNone(sp)
        self.assertIn("túl kicsi", why)

    def test_no_faces(self):
        self.assertEqual(asd.decide_light([]), ([], None, "nincs arc a képen"))

    def test_speaking_spans(self):
        p = np.zeros(50)
        p[10:30] = 0.9
        spans = asd.speak_spans(p, fps=25.0)
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], 0.4, places=3)
        self.assertAlmostEqual(spans[0][1], 1.2, places=3)


@unittest.skipUnless(WEIGHT.is_file(), "nincs Light-ASD súly")
class ModelTest(unittest.TestCase):
    def test_load_and_forward(self):
        m = lightasd.load(WEIGHT)
        self.assertLess(sum(p.numel() for p in m.parameters()) / 1e6, 1.1)
        rng = np.random.default_rng(0)
        faces = (rng.random((50, 112, 112)) * 255).astype(np.uint8)
        mf = lightasd.mfcc((rng.random(32000) * 0.1).astype(np.float32))[:200]
        p = m.probs(faces, mf)
        # a hangág 2×2-es időbeli poolt végez: a hossz a rövidebb oldal
        self.assertEqual(p.shape, (min(len(faces), len(mf) // 4),))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))


if __name__ == "__main__":
    unittest.main()
