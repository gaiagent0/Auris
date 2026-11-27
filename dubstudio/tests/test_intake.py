"""Intake-tesztek: feliratfordítás, fallback sávok, M&E-kivonás.

A valódi LLM-, ffmpeg- és audio-separator-hívásokat helyettesítjük; a
fájlrendszeri viselkedést és az időzítés megőrzését valós temp-fájlokkal
ellenőrizzük.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import intake, newproject  # noqa: E402
from dubstudio.config import Project  # noqa: E402

SRT = """1
00:00:01,000 --> 00:00:03,000
Hello, how are you?

2
00:00:04,500 --> 00:00:07,250
I am fine, thank you.
"""


class IntakeTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        for name in ("data", "subs", "work"):
            (self.base / name).mkdir()
        self.project = Project(
            name="Minta", season=1, root=self.base,
            data_dir=self.base / "data", subs_dir=self.base / "subs",
            cache_dir=self.base / "work", source_lang="eng",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _source(self, lang="eng") -> Path:
        path = self.base / "subs" / f"E01_{lang}.srt"
        path.write_text(SRT, encoding="utf-8")
        return path

    def test_translate_writes_hungarian_srt_and_keeps_timing(self):
        self._source()
        calls = []

        def fake_chat(cfg, system, prompt):
            calls.append(prompt)
            return {"lines": [
                {"id": 1, "text": "Üdv, hogy vagy?"},
                {"id": 2, "text": "Jól vagyok, köszönöm."},
            ]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertFalse(result["skipped"])
        out = (self.base / "subs" / "E01_hun.srt").read_text(encoding="utf-8")
        self.assertIn("00:00:01,000 --> 00:00:03,000", out)
        self.assertIn("00:00:04,500 --> 00:00:07,250", out)
        self.assertIn("Üdv, hogy vagy?", out)
        self.assertIn("Jól vagyok, köszönöm.", out)
        # időzítés nem csúszik el, a sorszámok 1-től jönnek
        self.assertTrue(out.startswith("1\n00:00:01,000"))
        self.assertEqual(len(calls), 1)

        # a meglévő magyar feliratot nem fordítja újra
        with patch.object(intake, "_chat_json") as again:
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
            again.assert_not_called()
        self.assertTrue(result["skipped"])

    def test_translate_prompt_carries_cue_durations(self):
        self._source()
        prompts = []

        def fake_chat(cfg, system, prompt):
            prompts.append(prompt)
            return {"lines": [{"id": 1, "text": "Üdv"},
                              {"id": 2, "text": "Jól"}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertIn("1. [2.0s] Hello, how are you?", prompts[0])
        self.assertIn("2. [2.8s] I am fine, thank you.", prompts[0])
        self.assertEqual(prompts[0].count("[2."), 2)
        self.assertEqual(result["tight_lines"], 0)

    def test_translate_reports_over_long_lines(self):
        self._source()
        long_text = "Ez egy szandekosan tul hosszu magyar forditas mondat."

        def fake_chat(cfg, system, prompt):
            return {"lines": [{"id": 1, "text": long_text},
                              {"id": 2, "text": "Jól"}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertEqual(result["tight_lines"], 1)
        self.assertEqual(result["tight_ids"], [1])

    def test_translate_strips_echoed_slot_hint(self):
        self._source()

        def fake_chat(cfg, system, prompt):
            return {"lines": [{"id": 1, "text": "[2.0s] Üdv"},
                              {"id": 2, "text": "[2.8s] Jól"}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            intake.translate_subtitles(self.project, "E01", cfg=cfg)
        out = (self.base / "subs" / "E01_hun.srt").read_text(encoding="utf-8")
        self.assertIn("Üdv", out)
        self.assertNotIn("[2.0s]", out)
        self.assertNotIn("[2.8s]", out)

    def test_translate_retry_prompt_keeps_slot_hints(self):
        self._source()
        prompts = []

        def fake_chat(cfg, system, prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return {"lines": [{"id": 1, "text": "Üdv"}]}
            return {"lines": [{"id": 2, "text": "Jól"}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertEqual(result["translated"], 2)
        self.assertEqual(len(prompts), 2)
        self.assertIn("[2.8s] I am fine, thank you.", prompts[1])

    def test_translate_retries_a_stubborn_line_on_its_own(self):
        self._source()
        prompts = []

        def fake_chat(cfg, system, prompt):
            prompts.append(prompt)
            if len(prompts) <= 2:      # első kör és a hiányzókat kérő második
                return {"lines": [{"id": 1, "text": "Üdv"}]}
            return {"lines": [{"id": 2, "text": "Jól vagyok"}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertEqual(result["translated"], 2)
        self.assertEqual(len(prompts), 3)
        self.assertIn("single subtitle line", prompts[2])
        self.assertIn("2. [2.8s] I am fine, thank you.", prompts[2])

    def test_single_line_retry_accepts_a_renumbered_answer(self):
        """Egyetlen sort kérve a modell néha 1-re számozza a választ — ez is jó."""
        self._source()
        prompts = []

        def fake_chat(cfg, system, prompt):
            prompts.append(prompt)
            if len(prompts) <= 2:      # első kör és a hiányzókat kérő második
                return {"lines": [{"id": 1, "text": "Üdv"}]}
            return {"lines": [{"id": 1, "text": "Jól vagyok"}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertEqual(result["translated"], 2)
        out = (self.base / "subs" / "E01_hun.srt").read_text(encoding="utf-8")
        self.assertIn("Jól vagyok", out)

    def test_translate_keeps_sound_only_cues_without_translation(self):
        path = self.base / "subs" / "E01_eng.srt"
        path.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello.\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\n[music]\n", encoding="utf-8")

        def fake_chat(cfg, system, prompt):
            # a zörejjelzős sort a modell üresen hagyja
            return {"lines": [{"id": 1, "text": "Szia."}]}

        cfg = intake.settings(self.project)["translate"]
        with patch.object(intake, "_chat_json", side_effect=fake_chat):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        out = (self.base / "subs" / "E01_hun.srt").read_text(encoding="utf-8")
        self.assertEqual(result["translated"], 2)
        self.assertIn("[music]", out)
        self.assertIn("Szia.", out)

    def test_translate_uses_english_external_before_hebrew(self):
        self._source("eng")
        self._source("heb")
        cfg = intake.settings(self.project)["translate"]
        cfg["source_lang"] = "eng"
        with patch.object(intake, "_chat_json", return_value={"lines": [
                {"id": 1, "text": "Egy"}, {"id": 2, "text": "Kettő"}]}):
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertTrue(result["source"].endswith("E01_eng.srt"))
        self.assertEqual(result["source_lang"], "eng")

    def test_hungarian_source_is_adopted_without_translation(self):
        self.project.source_lang = "hun"
        (self.base / "subs" / "E01_hu.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nSzia, hogy vagy?\n",
            encoding="utf-8")
        # a projekt forrásnyelve magyar (a felirat neve ezért _hu.srt is lehet)
        cfg = dict(intake.settings(self.project)["translate"], source_lang="hu")
        with patch.object(intake, "_chat_json") as chat:
            result = intake.translate_subtitles(self.project, "E01", cfg=cfg)
        chat.assert_not_called()
        self.assertTrue(result["passthrough"])
        out = (self.base / "subs" / "E01_hun.srt").read_text(encoding="utf-8")
        self.assertIn("00:00:01,000 --> 00:00:03,000", out)
        self.assertIn("Szia, hogy vagy?", out)

    def test_hungarian_language_aliases(self):
        self.assertTrue(intake.is_hungarian("hu"))
        self.assertTrue(intake.is_hungarian("magyar"))
        self.assertFalse(intake.is_hungarian("eng"))
        self.assertTrue(intake._language_matches("hu", "hun"))
        self.assertTrue(intake._language_matches("hun", "hun"))
        self.assertFalse(intake._language_matches("eng", "hun"))

    def test_missing_source_subtitle_is_a_clear_error(self):
        cfg = intake.settings(self.project)["translate"]
        with self.assertRaises(intake.IntakeError) as ctx:
            intake.translate_subtitles(self.project, "E01", cfg=cfg)
        self.assertIn("Nincs fordítható felirat", str(ctx.exception))

    def test_status_falls_back_to_intake_work_paths(self):
        (self.base / "subs" / "E01_hun.srt").write_text("1\n", encoding="utf-8")
        ep = self.base / "work" / "E01"
        ep.mkdir()
        (ep / "vocals.wav").write_bytes(b"voc")
        (ep / "me.wav").write_bytes(b"me")
        self.assertTrue(self.project.vocals("E01").is_file())
        self.assertTrue(self.project.me_bed("E01").is_file())
        state = intake.status(self.project, "E01")
        self.assertTrue(state["subtitle_ready"])
        self.assertTrue(state["separated"])
        self.assertEqual(state["state"], "ready")

    def test_retry_is_blocked_after_recent_error(self):
        intake._write_marker(self.project, "E01", last_error="LLM nem elérhető")
        state = intake.status(self.project, "E01")
        self.assertEqual(state["state"], "error")
        self.assertFalse(state["retry_allowed"])
        self.assertFalse(intake.should_retry(self.project, "E01"))

    def test_subtract_beds_removes_vocals(self):
        import numpy as np
        import soundfile as sf

        sr = 8000
        t = np.arange(sr, dtype=np.float32) / sr
        bed = (0.7 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        voc = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        inst = (bed + voc).astype(np.float32)
        a = self.base / "inst.wav"
        b = self.base / "voc.wav"
        sf.write(str(a), inst, sr)
        sf.write(str(b), voc, sr)
        out = self.base / "me.wav"
        intake._subtract_beds(a, b, out)
        result, got_sr = sf.read(str(out), dtype="float32")
        self.assertEqual(got_sr, sr)
        self.assertLess(float(np.max(np.abs(result - bed))), 1e-4)

    def test_legacy_project_globs_are_not_re_separated(self):
        legacy = self.base / "legacy"
        (legacy / "me1").mkdir(parents=True)
        (legacy / "me1" / "E01_(Vocals).wav").write_bytes(b"voc")
        (legacy / "E01_me.wav").write_bytes(b"me")
        project = Project(
            name="Régi", season=1, root=self.base,
            data_dir=self.base / "data", subs_dir=self.base / "subs",
            cache_dir=self.base / "work",
            vocals_globs=[str(legacy / "me1" / "*Vocals*.wav")],
            me_globs=[str(legacy / "E01_me.wav")],
        )
        with patch.object(intake, "_run_separator") as separator:
            result = intake.separate_audio(project, "E01")
        self.assertTrue(result["skipped"])
        self.assertTrue(result["vocals"].endswith("E01_(Vocals).wav"))
        separator.assert_not_called()

    def test_ensure_ready_collects_both_steps_and_marks_error(self):
        with patch.object(intake, "translate_subtitles", side_effect=intake.IntakeError("fordító hiba")), \
             patch.object(intake, "separate_audio", side_effect=intake.IntakeError("separátor hiba")):
            with self.assertRaises(intake.IntakeError) as ctx:
                intake.ensure_ready(self.project, "E01")
        message = str(ctx.exception)
        self.assertIn("fordító hiba", message)
        self.assertIn("separátor hiba", message)
        self.assertEqual(intake.status(self.project, "E01")["state"], "error")

    def test_translate_endpoint_only_accepts_local_llm(self):
        """A repó nem küld feliratszöveget külső szolgáltatásnak."""
        with self.assertRaises(intake.IntakeError) as ctx:
            intake._endpoint("https://openrouter.ai/api/v1")
        self.assertIn("helyi", str(ctx.exception))
        self.assertEqual(intake._endpoint("http://127.0.0.1:11434/v1"),
                         "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(intake._endpoint("http://localhost:1234"),
                         "http://localhost:1234/v1/chat/completions")

    def test_new_project_translate_defaults_are_local(self):
        doc = newproject.project_doc("Teszt film", "movie", 1, "eng")
        tr = doc["intake"]["translate"]
        self.assertTrue(tr["base"].startswith("http://127.0.0.1"))
        self.assertEqual(tr["api_key"], "")
        self.assertEqual(tr["api_key_env"], "")
        self.assertEqual(tr["source_lang"], "eng")


if __name__ == "__main__":
    unittest.main()
