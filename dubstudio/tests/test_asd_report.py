"""A képi beszélő-ellenőrzés bizonyítéka a riportban (nem kapu)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import report  # noqa: E402


class _Project:
    """Csak annyi kell a riportnak: ``ep_dir(tag)``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def ep_dir(self, tag: str) -> Path:
        return self.root / tag


ROWS = [
    {"t": 1445.96, "end": 1450.92, "uid": "E04:0232", "faces": 2,
     "light": {"speaking": 0, "verdict": "arc #0 beszél (p 0.89)",
               "cands": [{"face": 0, "prob": 0.887, "size": 156,
                          "spans": [[1445.81, 1447.49]]},
                         {"face": 1, "prob": 0.025, "size": 49, "spans": []}]}},
    {"t": 1764.17, "end": 1769.12, "uid": "E04:0299", "faces": 0,
     "light": {"speaking": None,
               "verdict": "nincs beszélő arc a képen (legjobb p 0.00 ≤ 0.35)",
               "cands": []}},
    {"t": 965.54, "end": 982.42, "uid": "E04:0155", "faces": 3,
     "light": {"speaking": None, "verdict": "nem eldönthető (p 0.84)",
               "cands": [{"face": 1, "prob": 0.841, "size": 118,
                          "spans": [[965.71, 968.07]]}]}},
]


class EvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.p = _Project(self.root)
        (self.root / "E04" / "visual").mkdir(parents=True)
        (self.root / "E04" / "visual" / "asd.json").write_text(
            json.dumps(ROWS, ensure_ascii=False), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_counts_and_best_face(self):
        ev = report.asd_evidence(self.p, "E04")
        self.assertIsNotNone(ev)
        self.assertEqual(ev["n"], 3)
        self.assertEqual(ev["on_screen"], 1)
        self.assertEqual(ev["offscreen"], 1)
        self.assertEqual(ev["undecided"], 1)
        self.assertEqual(ev["rows"][0]["best_prob"], 0.887)
        self.assertEqual(ev["rows"][1]["best_prob"], None)

    def test_missing_file_is_none(self):
        self.assertIsNone(report.asd_evidence(self.p, "E05"))

    def test_html_renders_row_and_is_not_a_gate(self):
        html = report.asd_html(report.asd_evidence(self.p, "E04"))
        self.assertIn("Képi beszélő-ellenőrzés", html)
        self.assertIn("E04:0232", html)
        self.assertIn("bizonyíték, nem kapu", html)
        self.assertEqual(report.asd_html(None), "")


if __name__ == "__main__":
    unittest.main()
