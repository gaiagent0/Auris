import unittest

from core import hu_wer
from core.hu_wer import Case, run_bench


class CorpusTest(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = hu_wer.corpus_ids()
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_case_has_text_and_note(self):
        for case in hu_wer.CORPUS:
            self.assertTrue(case.text.strip(), f"{case.id} üres szöveg")
            self.assertTrue(case.note.strip(), f"{case.id} megjegyzés nélkül")

    def test_sentences_end_with_punctuation(self):
        for case in hu_wer.CORPUS:
            self.assertIn(case.text[-1], ".?!", case.id)

    def test_get_case_by_id(self):
        self.assertEqual(hu_wer.get_case("bethlen").id, "bethlen")
        with self.assertRaises(KeyError):
            hu_wer.get_case("nincs-ilyen")

    def test_corpus_covers_the_dictionary_names(self):
        text = " ".join(case.text for case in hu_wer.CORPUS)
        for word in ("Shakespeare", "Bethlen", "Móricz"):
            self.assertIn(word, text, f"{word} nincs a korpuszban")


class RunBenchTest(unittest.TestCase):
    """A mérési ciklus modellek nélkül, hamis szintetizálóval és visszaíróval."""

    CASES = [Case("a", "Egy mondat a levegőben.", "teszt"),
             Case("b", "Második mondat Hangos szóval.", "teszt")]

    def test_perfect_transcription_scores_zero(self):
        report = run_bench(
            self.CASES,
            synthesize=lambda case: f"/tmp/{case.id}.wav",
            transcribe=lambda path: {
                "/tmp/a.wav": "egy mondat a levegőben",
                "/tmp/b.wav": "második mondat hangos szóval",
            }[path],
        )
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["scored"], 2)
        self.assertEqual(report["wer"], 0.0)
        self.assertEqual(report["cer"], 0.0)

    def test_missing_words_raise_wer(self):
        report = run_bench(
            self.CASES,
            synthesize=lambda case: f"/tmp/{case.id}.wav",
            transcribe=lambda path: "egy mondat levegőben második hangos szóval",
        )
        self.assertGreater(report["wer"], 0)
        # Az első mondatból a "a" cikk hiányzik, a "levegőben" megmaradt.
        self.assertIn("a", report["results"][0]["missing_words"])
        self.assertNotIn("levegőben", report["results"][0]["missing_words"])

    def test_one_failing_case_does_not_stop_the_bench(self):
        def synthesize(case):
            if case.id == "a":
                raise RuntimeError("modellhiba")
            return f"/tmp/{case.id}.wav"

        report = run_bench(
            self.CASES, synthesize, lambda path: "második mondat hangos szóval"
        )
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["scored"], 1)
        self.assertIn("modellhiba", report["results"][0]["error"])

    def test_aggregate_weighs_by_length_not_by_average(self):
        # A hosszú, hibátlan mondat mellett egy egyetlen szavas, hibás mondat:
        # súlyozott átlagban ez 1/8, a mondatonkénti átlagban 0,5 lenne.
        cases = [Case("long", "egy két három négy öt hat hét", "teszt"),
                 Case("short", "nyolc.", "teszt")]
        heard = {"/tmp/long.wav": "egy két három négy öt hat hét",
                 "/tmp/short.wav": "kilenc"}
        report = run_bench(cases, lambda c: f"/tmp/{c.id}.wav", heard.__getitem__)
        self.assertAlmostEqual(report["wer"], 1 / 8, places=4)
        self.assertLess(report["wer"], 0.2)

    def test_progress_callback_sees_every_case(self):
        seen = []
        run_bench(self.CASES, lambda c: f"/tmp/{c.id}.wav",
                  lambda p: "egy mondat a levegőben második mondat hangos szóval",
                  progress=lambda case, row: seen.append(case.id))
        self.assertEqual(seen, ["a", "b"])

    def test_report_renders_without_error(self):
        report = run_bench(self.CASES, lambda c: f"/tmp/{c.id}.wav",
                           lambda p: "egy mondat a levegőben második mondat hangos szóval")
        text = hu_wer.format_report(report)
        self.assertIn("Magyar érthetőségi mérés", text)
        self.assertIn("WER:", text)


if __name__ == "__main__":
    unittest.main()