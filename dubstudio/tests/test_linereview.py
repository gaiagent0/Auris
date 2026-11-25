"""Soronkénti emberi címkézés: nevek, szó-megjelölés, elcsúszás a tervben."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import linereview, plan, subs  # noqa: E402
from dubstudio.utterances import Utterance  # noqa: E402


class _Project:
    """Csak a munkakönyvtár kell (``ep_dir``)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def ep_dir(self, tag: str) -> Path:
        d = self.root / tag.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d


class NamingTest(unittest.TestCase):
    def test_hungarian_names_by_gender_and_rank(self):
        names = linereview.suggest_names([
            {"char_id": "C001", "gender": "F", "rank": 5},
            {"char_id": "C002", "gender": "M", "rank": 3},
            {"char_id": "C003", "gender": "F", "rank": 1},
        ])
        # a rangsor 1 = a legtöbbet beszélő szereplő, ő kapja az első nevet
        self.assertEqual(names["C003"], "Anna")
        self.assertEqual(names["C001"], "Éva")
        self.assertEqual(names["C002"], "Ottó")

    def test_existing_name_wins_and_is_not_reused(self):
        names = linereview.suggest_names([
            {"char_id": "C001", "gender": "F", "rank": 9, "name": "Panna"},
            {"char_id": "C002", "gender": "F", "rank": 1},
        ])
        self.assertEqual(names["C001"], "Panna")
        self.assertEqual(names["C002"], "Anna")

    def test_mark_of_vanished_character_is_dropped(self):
        """A megszűnt C### mark ne maradjon név nélkül a felületen."""
        data = {"names": {}, "people": {}, "lines": {
            "E01:0013#15": {"marks": [{"words": [1, 1], "kind": "char",
                                       "char": "C034", "gender": "F"}]},
            "E01:0008#10": {"marks": [{"words": None, "kind": "char",
                                       "char": "N001", "gender": "F"}]},
            "E01:0011#13": {"marks": [{"words": None, "kind": "char",
                                       "char": "C001", "gender": "F"}],
                            "offset": 1.5},
        }}
        linereview.ensure_names(data, [{"char_id": "C001", "gender": "F",
                                        "rank": 1}])
        self.assertNotIn("E01:0013#15", data["lines"])
        self.assertIn("E01:0008#10", data["lines"])   # felületi N### marad
        self.assertIn("E01:0011#13", data["lines"])   # élő C001 marad


class CarryNameTest(unittest.TestCase):
    def test_name_and_mark_follow_the_same_lines(self):
        """Újraszámozás után a név/mark a legtöbb közös sorú szereplőre kerül."""
        data = {"names": {"C003": "Anna", "C007": "Peti"},
                "people": {}, "lines": {
                    "E01:0001#1": {"marks": [{"words": None, "kind": "char",
                                               "char": "C003", "gender": "F"}]},
                    "E01:0002#2": {"marks": [{"words": [1, 1], "kind": "char",
                                               "char": "C007", "gender": "M"}]},
                }}
        before = {"E01:0001": "C003", "E01:0003": "C003",
                  "E01:0002": "C007", "E01:0004": "C007"}
        after = {"E01:0001": "C001", "E01:0003": "C001",
                 "E01:0002": "C002", "E01:0004": "C002"}
        remap = linereview.carry(data, before, after,
                                 genders={"C001": "F", "C002": "M"})
        self.assertEqual(remap, {"C003": "C001", "C007": "C002"})
        self.assertEqual(data["names"], {"C001": "Anna", "C002": "Peti"})
        self.assertEqual(data["lines"]["E01:0001#1"]["marks"][0]["char"], "C001")
        self.assertEqual(data["lines"]["E01:0002#2"]["marks"][0]["char"], "C002")

    def test_existing_name_of_the_target_is_never_overwritten(self):
        """Egy eltűnt töredék neve nem írhatja felül az élő szereplőét."""
        data = {"names": {"C001": "Anna", "C015": "Tímea"},
                "people": {}, "lines": {}}
        before = {"E01:0001": "C001", "E01:0002": "C001",
                  "E01:0003": "C015", "E01:0004": "C015"}
        after = {"E01:0001": "C001", "E01:0002": "C001",
                 "E01:0003": "C001", "E01:0004": "C001"}
        remap = linereview.carry(data, before, after, genders={"C001": "F"})
        self.assertEqual(remap, {})
        self.assertEqual(data["names"]["C001"], "Anna")

    def test_weak_overlap_does_not_move_a_name(self):
        data = {"names": {"C015": "Tímea"}, "people": {}, "lines": {}}
        before = {"E01:0001": "C015", "E01:0002": "C015",
                  "E01:0003": "C015", "E01:0004": "C015"}
        after = {"E01:0001": "C015", "E01:0002": "C015",
                 "E01:0003": "C015", "E01:0004": "C002"}
        self.assertEqual(linereview.carry(data, before, after,
                                          genders={"C002": "F"}), {})

    def test_name_does_not_cross_gender(self):
        data = {"names": {"C003": "Anna"}, "people": {}, "lines": {}}
        remap = linereview.carry(data, {"E01:0001": "C003"},
                                 {"E01:0001": "C002"}, genders={"C002": "M"})
        self.assertEqual(remap, {})
        self.assertEqual(data["names"], {"C003": "Anna"})


class LabelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.p = _Project(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_word_range_then_whole_line(self):
        linereview.set_label(self.p, "E01", "E01:0007", 2, kind="char",
                            char_id="C004", words=[1, 2])
        data = linereview.load(self.p, "E01")
        marks = data["lines"]["E01:0007#2"]["marks"]
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["words"], [1, 2])
        self.assertEqual(marks[0]["char"], "C004")

        linereview.set_label(self.p, "E01", "E01:0007", 2, kind="offscreen")
        marks = linereview.load(self.p, "E01")["lines"]["E01:0007#2"]["marks"]
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["kind"], "offscreen")
        self.assertIsNone(marks[0]["words"])

    def test_char_label_needs_character(self):
        with self.assertRaises(ValueError):
            linereview.set_label(self.p, "E01", "E01:0001", 1, kind="char")

    def test_new_person_gets_hungarian_name(self):
        linereview.set_label(self.p, "E01", "E01:0002", 3, kind="new",
                            gender="M", name="Józsi")
        data = linereview.load(self.p, "E01")
        cid = data["lines"]["E01:0002#3"]["marks"][0]["char"]
        self.assertTrue(cid.startswith("N"))
        self.assertEqual(data["people"][cid]["name"], "Józsi")
        self.assertEqual(data["people"][cid]["gender"], "M")

        renamed = linereview.set_name(self.p, "E01", cid, "Laci")
        self.assertEqual(renamed["people"][cid]["name"], "Laci")


class OffsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.p = _Project(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_offset_is_stored_and_feeds_plan(self):
        linereview.set_offset(self.p, "E01", "E01:0001", 2, 0.5)
        self.assertEqual(linereview.offsets(self.p, "E01"),
                         {("E01:0001", 2): 0.5})

        u = Utterance(uid="E01:0001", ep="E01", start=10.0, end=12.0)
        u.cues = [subs.Cue(index=1, start=10.0, end=11.0, text="Első."),
                  subs.Cue(index=2, start=11.5, end=12.0, text="Második.")]
        u.char, u.voice = "C001", "SUP_F1"
        entries = plan.build([u], verbose=False,
                             offsets=linereview.offsets(self.p, "E01"))
        self.assertEqual([e.cue_index for e in entries], [1, 2])
        self.assertAlmostEqual(entries[0].cue_start, 10.0, places=3)
        self.assertAlmostEqual(entries[1].cue_start, 12.0, places=3)

    def test_offset_is_bounded(self):
        with self.assertRaises(ValueError):
            linereview.set_offset(self.p, "E01", "E01:0001", 1, 42.0)


class LabelsTest(unittest.TestCase):
    """A kézi szereplő-döntés bekerül a szinkronláncba (nem csak jegyzet)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.p = _Project(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_labels_feed_the_plan(self):
        linereview.set_label(self.p, "E01", "E01:0003", 4, kind="char",
                            char_id="C002")
        linereview.set_label(self.p, "E01", "E01:0003", 5, kind="char",
                            char_id="C003", words=[0, 1])
        # a „képen kívül” jegyzet nem rendel hangot
        linereview.set_label(self.p, "E01", "E01:0003", 6, kind="offscreen")
        got = linereview.labels(self.p, "E01")
        self.assertEqual(set(got), {("E01:0003", 4), ("E01:0003", 5)})
        self.assertEqual(got[("E01:0003", 4)][0]["char"], "C002")
        self.assertIsNone(got[("E01:0003", 4)][0]["words"])
        self.assertEqual(got[("E01:0003", 5)][0]["words"], [0, 1])

    def test_bad_offset_does_not_break_name_cleanup(self):
        data = {"names": {"C001": "Anna"}, "people": {},
                "lines": {"E01:0001#1": {"marks": [], "offset": "nem szám"}}}
        linereview.ensure_names(data, [{"char_id": "C001", "gender": "F",
                                        "rank": 1}])
        self.assertEqual(data["names"]["C001"], "Anna")

    def test_stale_uid_uses_the_cue_number_when_unambiguous(self):
        """Újracsoportosítás után a felületen látszó mark ne vesszen el."""
        linereview.set_label(self.p, "E01", "E01:0013", 15, kind="char",
                            char_id="C001")
        (self.p.ep_dir("E01") / "utterances.json").write_text(
            '[{"uid": "E01:0014", "cues": [{"index": 15, "start": 1.0, '
            '"end": 2.0, "text": "Szia!"}]}]', encoding="utf-8")
        self.assertEqual(set(linereview.labels(self.p, "E01")),
                         {("E01:0014", 15)})

    def test_ambiguous_cue_number_is_not_guessed(self):
        linereview.set_label(self.p, "E01", "E01:0013", 13, kind="char",
                            char_id="C001")
        (self.p.ep_dir("E01") / "utterances.json").write_text(
            '[{"uid": "E01:0010", "cues": [{"index": 13}]},'
            ' {"uid": "E01:0011", "cues": [{"index": 13}]}]',
            encoding="utf-8")
        self.assertEqual(linereview.labels(self.p, "E01"), {})


if __name__ == "__main__":
    unittest.main()
