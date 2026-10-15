import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import qa, qa_api


class BackendChoiceTest(unittest.TestCase):
    def _choose(self, requested, *, gpu, parakeet=True, whisper=True, language="hu"):
        with patch("core.settings.get", side_effect=lambda key, default=None:
                   requested if key == "asr_backend" else default), \
             patch.object(qa, "parakeet_available", return_value=parakeet), \
             patch.object(qa, "whisper_available", return_value=whisper), \
             patch("torch.cuda.is_available", return_value=gpu):
            return qa.asr_backend_for(language)

    def test_auto_keeps_whisper_on_gpu_and_uses_hybrid_on_cpu(self):
        self.assertEqual(self._choose("auto", gpu=True), "whisper")
        self.assertEqual(self._choose("auto", gpu=False), "hybrid")

    def test_falls_back_to_whisper_without_package_or_language(self):
        self.assertEqual(self._choose("auto", gpu=False, parakeet=False), "whisper")
        self.assertEqual(self._choose("parakeet", gpu=False, language="ja"), "whisper")
        self.assertEqual(self._choose("hybrid", gpu=True, parakeet=False), "whisper")

    def test_explicit_choices_are_respected(self):
        self.assertEqual(self._choose("parakeet", gpu=True), "parakeet")
        self.assertEqual(self._choose("whisper", gpu=False), "whisper")
        self.assertEqual(self._choose("nonsense", gpu=True), "whisper")

    def test_without_whisper_runtime_falls_back_to_parakeet(self):
        # On ARM64 Windows the transformers-based Whisper backend may not be
        # installed; every path must fall back to the ONNX Parakeet backend.
        self.assertEqual(self._choose("auto", gpu=True, whisper=False), "parakeet")
        self.assertEqual(self._choose("auto", gpu=False, whisper=False), "parakeet")
        self.assertEqual(self._choose("whisper", gpu=False, whisper=False), "parakeet")
        self.assertEqual(self._choose("hybrid", gpu=False, whisper=False), "parakeet")
        # ...and to whatever works when even Parakeet is missing.
        self.assertEqual(self._choose("auto", gpu=False, parakeet=False, whisper=False), "whisper")


class TranscriberSwitchTest(unittest.TestCase):
    """A Beállításokban váltott `asr_backend` tényleg másik futtatót ad.

    Ez a felhasználói "váltás a módszerek közt" beépített bizonyítéka:
    ugyanaz a nyelv más-más transcriber-osztályt kap, és a visszaváltás
    ugyanazt a már betöltött példányt adja (nem tölt újra modellt).
    """

    def _for(self, requested):
        fake = {"asr_backend": requested}
        with patch("core.settings.get", side_effect=lambda key, default=None:
                   fake.get(key, default)), \
             patch.object(qa, "parakeet_available", return_value=True), \
             patch.object(qa, "whisper_available", return_value=True):
            return qa.Transcriber.for_language("hu")

    def test_switching_setting_switches_the_transcriber(self):
        saved = qa.Transcriber._instances
        qa.Transcriber._instances = {}
        try:
            hybrid = self._for("hybrid")
            parakeet = self._for("parakeet")
            whisper = self._for("whisper")
            self.assertIsInstance(hybrid, qa.HybridTranscriber)
            self.assertIsInstance(parakeet, qa.ParakeetTranscriber)
            self.assertNotIsInstance(parakeet, qa.HybridTranscriber)
            self.assertIsInstance(whisper, qa.Transcriber)
            self.assertNotIsInstance(whisper, qa.ParakeetTranscriber)
            # a három mód három külön példány, nem ugyanaz az egyetlen
            self.assertEqual(len({id(hybrid), id(parakeet), id(whisper)}), 3)
            # visszaváltás ugyanazt a betöltött példányt adja
            self.assertIs(self._for("hybrid"), hybrid)
            self.assertIs(self._for("parakeet"), parakeet)
        finally:
            qa.Transcriber._instances = saved


class ParakeetWordsTest(unittest.TestCase):
    def test_subword_tokens_become_timed_words(self):
        tokens = [" Á", "r", "víz", " tü", "kör", ".", " ", "8", " már", "cius"]
        stamps = [0.0, 0.1, 0.2, 0.5, 0.6, 0.8, 1.0, 1.1, 1.4, 1.6]
        words = qa._parakeet_words(tokens, stamps, 2.0)
        self.assertEqual([w["word"] for w in words], ["Árvíz", "tükör.", "8", "március"])
        self.assertEqual((words[0]["start"], words[0]["end"]), (0.0, 0.5))
        self.assertEqual(words[-1]["end"], 2.0)


class _Fake:
    single_pass_words = True
    confirms = True
    model_id = "fake"

    def __init__(self, first, second):
        self.first, self.second, self.confirmed = first, second, 0

    def transcribe(self, path, language=None, *, prompt="", word_timestamps=False):
        return {"text": self.first, "words": [{"word": "x", "start": 0.0, "end": 0.5}]}

    def confirm(self, path, language=None, *, prompt=""):
        self.confirmed += 1
        return {"text": self.second, "words": []}


class HybridEvaluationTest(unittest.TestCase):
    def _evaluate(self, fake, text):
        metrics = {"flags": [], "duration_sec": 1.0}
        with patch.object(qa, "analyze_audio", return_value=metrics):
            return qa_api._evaluate("a.wav", text, "hu", transcriber=fake, prompt="", align=False,
                                    confirm_above=0.05)

    def test_clean_sentence_is_not_rechecked(self):
        fake = _Fake("Gyere ide, mondta Anna.", "whatever")
        result = self._evaluate(fake, "– Gyere ide! – mondta Anna.")
        self.assertEqual(fake.confirmed, 0)
        self.assertEqual(result["cer"], 0.0)

    def test_suspicious_sentence_uses_better_whisper_reading(self):
        fake = _Fake("Gyerejde mondta Kovácsan a.", "Gyere ide, mondta Kovács Anna.")
        result = self._evaluate(fake, "– Gyere ide! – mondta Kovács Anna.")
        self.assertEqual(fake.confirmed, 1)
        self.assertEqual(result["cer"], 0.0)
        self.assertEqual(result["heard"], "Gyere ide, mondta Kovács Anna.")

    def test_worse_confirmation_keeps_first_reading(self):
        fake = _Fake("Gyere ide mondta Kovácsan a.", "teljesen más")
        result = self._evaluate(fake, "– Gyere ide! – mondta Kovács Anna.")
        self.assertEqual(result["heard"], "Gyere ide mondta Kovácsan a.")


class AsrBackendApiTest(unittest.TestCase):
    """Az Ellenőrzés oldalról lekérdezhető és állítható a felismerő mód."""

    def setUp(self):
        import app as application
        from core import database, settings

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            patch.object(database, "DB_PATH", str(root / "reader.db")),
            patch.object(settings, "SETTINGS_FILE", root / "settings.json"),
            patch.object(application, "_startup_complete", True),
        ]
        for p in self.patches:
            p.start()
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type, language) "
                "VALUES (1, 'Tesztkönyv', 'k.txt', 'txt', 'hu')"
            )
        application.app.config["TESTING"] = True
        self.client = application.app.test_client()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _get(self):
        with patch.object(qa, "parakeet_available", return_value=True), \
             patch.object(qa, "whisper_available", return_value=True), \
             patch("torch.cuda.is_available", return_value=False):
            return self.client.get("/api/books/1/qa/asr-backend")

    def test_reports_requested_and_effective_backend_with_options(self):
        response = self._get()
        self.assertEqual(response.status_code, 200)
        state = response.get_json()
        # Az alapértelmezett mód a hibrid (settings.DEFAULTS['asr_backend']).
        self.assertEqual(state["requested"], "hybrid")
        # CUDA nélkül, elérhető Parakeet+Whisper mellett a hibrid az effektív.
        self.assertEqual(state["effective"], "hybrid")
        # Az oldal automatikus indítása is kérhető az API-tól.
        self.assertIn("qa_autostart", state)
        self.assertIsInstance(state["qa_autostart"], bool)
        self.assertEqual(state["whisper_model"], qa.HUNGARIAN_ASR_MODEL)
        self.assertEqual(state["parakeet_model"], qa.PARAKEET_MODEL)
        values = {o["value"]: o for o in state["options"]}
        self.assertEqual(set(values), set(qa.ASR_BACKENDS))
        self.assertTrue(all(o["available"] for o in state["options"]))
        self.assertTrue(all(o["label"] for o in state["options"]))

    def test_switch_is_saved_and_reported_back(self):
        with patch.object(qa, "parakeet_available", return_value=True), \
             patch.object(qa, "whisper_available", return_value=True):
            response = self.client.post(
                "/api/books/1/qa/asr-backend", json={"backend": "parakeet"})
        self.assertEqual(response.status_code, 200)
        state = response.get_json()
        self.assertEqual(state["requested"], "parakeet")
        self.assertEqual(state["effective"], "parakeet")
        from core import settings
        self.assertEqual(settings.get("asr_backend"), "parakeet")
        # A következő lekérdezés is ezt az állapotot mutatja.
        state2 = self._get().get_json()
        self.assertEqual(state2["requested"], "parakeet")

    def test_unknown_backend_is_rejected(self):
        response = self.client.post(
            "/api/books/1/qa/asr-backend", json={"backend": "gpt-4o"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_missing_runtime_marks_options_unavailable(self):
        # ARM64 Windows: nincs transformers -> a Whisper opciók kikapcsolva.
        with patch.object(qa, "parakeet_available", return_value=True), \
             patch.object(qa, "whisper_available", return_value=False):
            state = self.client.get("/api/books/1/qa/asr-backend").get_json()
        values = {o["value"]: o["available"] for o in state["options"]}
        self.assertFalse(values["whisper"])
        self.assertFalse(values["hybrid"])
        self.assertTrue(values["parakeet"])
        self.assertEqual(state["effective"], "parakeet")

    def test_unknown_book_is_404(self):
        response = self.client.get("/api/books/99/qa/asr-backend")
        self.assertEqual(response.status_code, 404)

    def test_autostart_setting_is_saved_and_reported(self):
        """A Beállítások automatikus-indítás kapcsolója eljut az Ellenőrzésig."""
        from core import settings

        # Alapból bekapcsolva, és a mentett állapot nem felülírt.
        self.assertIs(settings.get("qa_autostart"), True)
        self.assertTrue(self._get().get_json()["qa_autostart"])

        response = self.client.post("/api/settings", json={"qa_autostart": False})
        self.assertEqual(response.status_code, 200)
        self.assertIs(settings.get("qa_autostart"), False)
        self.assertFalse(self._get().get_json()["qa_autostart"])
        # A kapcsoló kikapcsolása sem írja felül a választható módokat.
        self.assertEqual(self._get().get_json()["requested"], "hybrid")


if __name__ == "__main__":
    unittest.main()
