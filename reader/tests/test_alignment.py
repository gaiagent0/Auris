import unittest

from core import alignment


class AlignmentTest(unittest.TestCase):
    def test_estimate_is_syllable_weighted_and_covers_duration(self):
        timings = alignment.estimate_words("A tükörfúrógép működik.", 6.0)
        self.assertEqual(len(timings), 3)
        self.assertAlmostEqual(timings[-1][1], 6.0, places=2)
        self.assertGreater(timings[1][1] - timings[1][0], (timings[0][1] - timings[0][0]) * 3)

    def test_asr_words_map_to_displayed_words_including_numbers(self):
        text = "Béla 1241. március 15-én kelt útra."
        heard = [
            {"word": "Béla", "start": 0.0, "end": 0.6},
            {"word": "ezerkétszáznegyvenegy", "start": 0.6, "end": 2.0},
            {"word": "március", "start": 2.0, "end": 2.6},
            {"word": "tizenötödikén", "start": 2.6, "end": 3.5},
            {"word": "kelt", "start": 3.5, "end": 3.8},
            {"word": "útra.", "start": 3.8, "end": 4.3},
        ]
        timings = alignment.align_words(text, heard, 4.5, "hu")
        self.assertEqual(len(timings), 6)
        self.assertEqual(timings[1], [0.6, 2.0])
        self.assertEqual(timings[3], [2.6, 3.5])

    def test_unmatched_words_are_interpolated_monotonically(self):
        text = "Egy kettő három négy öt"
        heard = [{"word": "egy", "start": 0.0, "end": 0.5}, {"word": "öt", "start": 3.0, "end": 3.5}]
        timings = alignment.align_words(text, heard, 4.0, "hu")
        starts = [t[0] for t in timings]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(timings[0], [0.0, 0.5])
        self.assertEqual(timings[-1], [3.0, 3.5])
        self.assertTrue(0.5 <= timings[2][0] <= 3.0)

    def test_no_asr_words_falls_back_to_estimate(self):
        self.assertEqual(alignment.align_words("Két szó", [], 2.0, "hu"),
                         alignment.estimate_words("Két szó", 2.0))

    def test_prefix_match_stays_inside_its_own_gap(self):
        # Regesszió a M-54-es méréshez: a fejezet eleji „sír” NE a később
        # (139,9 s) elhangzó azonos alakú szóra ugrjon — a globális prefix-
        # keresés a teljes hátra lévő idővonalat arra torlódta (minden mondat
        # 140 s-os lett, majd a túlhosszú szelettől az ONNX encoder is elbukott).
        text = "Miért sír olyasmiért, ami később újra elhangzik: sír."
        heard = [
            {"word": "miért", "start": 0.0, "end": 0.3},
            {"word": "sírvas", "start": 0.3, "end": 0.8},
            {"word": "olyasmiért,", "start": 0.8, "end": 1.4},
            {"word": "ami", "start": 1.4, "end": 1.6},
            {"word": "később", "start": 1.6, "end": 2.0},
            {"word": "újra", "start": 139.0, "end": 139.4},
            {"word": "elhangzik:", "start": 139.4, "end": 139.9},
            {"word": "sír.", "start": 139.9, "end": 140.2},
        ]
        timings = alignment.align_words(text, heard, 141.0, "hu")
        self.assertEqual(len(timings), 8)
        self.assertLess(timings[1][0], 10.0, "a 2. szó a saját szakaszában marad")
        starts = [t[0] for t in timings]
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(timings[2][0], timings[1][0])
        self.assertAlmostEqual(timings[-1][1], 140.2, places=2)


if __name__ == "__main__":
    unittest.main()
