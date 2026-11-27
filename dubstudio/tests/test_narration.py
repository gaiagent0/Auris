"""Az egyszereplős narrációs modul tiszta (motor nélküli) egységei."""
from __future__ import annotations

import unittest

from dubstudio import narration, subs


def cue(i, start, end, text):
    return subs.Cue(index=i, start=start, end=end, text=text)


class WordsToCuesTest(unittest.TestCase):
    def test_sentence_punctuation_closes_segment(self):
        words = [{"word": "Ez", "start": 0.0, "end": 0.2},
                 {"word": "egy", "start": 0.2, "end": 0.4},
                 {"word": "mondat.", "start": 0.4, "end": 0.9},
                 {"word": "Ez", "start": 1.2, "end": 1.4},
                 {"word": "a", "start": 1.4, "end": 1.5},
                 {"word": "másik.", "start": 1.5, "end": 2.0}]
        cues = narration._words_to_cues(words, max_s=30, max_chars=200)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "Ez egy mondat.")
        self.assertAlmostEqual(cues[0].start, 0.0)
        self.assertAlmostEqual(cues[1].start, 1.2)

    def test_long_silence_closes_segment(self):
        words = [{"word": "alma", "start": 0.0, "end": 0.3},
                 {"word": "korte", "start": 3.0, "end": 3.4}]
        cues = narration._words_to_cues(words, max_s=60, max_chars=300)
        self.assertEqual([c.text for c in cues], ["alma", "korte"])


class BuildBlocksTest(unittest.TestCase):
    def test_continuous_lines_merge_until_sentence_end(self):
        cues = [cue(1, 0.0, 2.0, "Első gondolat,"),
                cue(2, 2.0, 5.0, "ami egy mondatban folytatódik."),
                cue(3, 5.0, 8.0, "Második mondat."),
                cue(4, 8.0, 11.0, "Harmadik mondat.")]
        blocks = narration.build_blocks(cues, narration.Options(block_chars=80))
        # az első két sor egy blokk; utána a hosszkorlát miatt vált
        self.assertGreaterEqual(len(blocks), 2)
        self.assertIn("Első gondolat, ami egy mondatban folytatódik.",
                      blocks[0].text)
        self.assertTrue(all(b.text for b in blocks))

    def test_big_gap_starts_new_block(self):
        cues = [cue(1, 0.0, 3.0, "Rövid mondat."),
                cue(2, 5.0, 8.0, "Későbbi mondat.")]
        blocks = narration.build_blocks(cues, narration.Options(gap_split=1.0))
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(blocks[0].end, 3.0)
        self.assertAlmostEqual(blocks[1].start, 5.0)

    def test_music_and_empty_lines_are_skipped(self):
        cues = [cue(1, 0.0, 2.0, "♪ zene ♪"),
                cue(2, 2.0, 5.0, "Beszéd."),
                cue(3, 5.0, 6.0, "[Music]")]
        blocks = narration.build_blocks(cues)
        self.assertEqual([b.text for b in blocks], ["Beszéd."])


class VoiceTest(unittest.TestCase):
    def test_default_voice_is_ready_made(self):
        models = narration.model_paths()
        voice, ref = narration.resolve_voice(None, None, models)
        self.assertFalse(voice.clone)
        self.assertIsNone(ref)

    def test_clone_without_reference_is_refused(self):
        with self.assertRaises(narration.NarrationError):
            narration.resolve_voice("XTTS_istvan01", None, narration.model_paths())

    def test_unknown_voice_is_refused(self):
        with self.assertRaises(narration.NarrationError):
            narration.resolve_voice("NOPE", None, narration.model_paths())


if __name__ == "__main__":
    unittest.main()
