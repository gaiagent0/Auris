import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database, local_engines, settings
from core.local_engines import SUPERTONIC_FILES, SUPERTONIC_VARIANT_DIRS, SupertonicEngine

VENDOR = Path(local_engines.__file__).resolve().parent / "vendor" / "supertonic"


def _vendor_helper():
    sys.path.insert(0, str(VENDOR))
    try:
        import helper

        return helper
    finally:
        sys.path.pop(0)


class VariantFileMapTest(unittest.TestCase):
    def test_local_map_matches_the_vendored_helper(self):
        # local_engines mirrors the names so the presence check works without
        # the vendor on sys.path; the two must never drift apart.
        helper = _vendor_helper()
        for variant in SUPERTONIC_FILES:
            self.assertEqual(
                tuple(helper.variant_files(variant).values()),
                SUPERTONIC_FILES[variant],
                variant,
            )

    def test_every_variant_has_a_folder(self):
        self.assertEqual(set(SUPERTONIC_FILES), set(SUPERTONIC_VARIANT_DIRS))

    def test_int8_uses_the_binary_unicode_table(self):
        self.assertTrue(SUPERTONIC_FILES["int8"][-1].endswith(".bin"))
        self.assertTrue(SUPERTONIC_FILES["fp32"][-1].endswith(".json"))


class VariantSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patch = patch.object(settings, "SETTINGS_FILE", root / "settings.json")
        self.patch.start()
        self.root = root

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _engine(self, **stored):
        if stored:
            settings.save(stored)
        return SupertonicEngine()

    def test_defaults_to_fp32(self):
        self.assertEqual(self._engine().variant, "fp32")

    def test_stored_variant_selects_the_folder(self):
        engine = self._engine(supertonic_variant="int8")
        self.assertEqual(engine.variant, "int8")
        self.assertEqual(engine.folder.name, SUPERTONIC_VARIANT_DIRS["int8"])

    def test_unknown_variant_falls_back_to_fp32(self):
        self.assertEqual(self._engine(supertonic_variant="int4").variant, "fp32")

    def test_both_variants_share_one_voice_folder(self):
        engine = self._engine(supertonic_variant="int8")
        self.assertEqual(engine.voice_dir.name, "voice_styles")
        self.assertEqual(
            engine.voice_dir, self._engine(supertonic_variant="fp32").voice_dir
        )

    def test_cache_key_separates_the_variants(self):
        # A variáns a beállításból olvasódik, ezért külön-külön kell mérni.
        def identity_for(variant):
            stored = {**settings.DEFAULTS, "supertonic_variant": variant}
            with patch.object(settings, "load", return_value=stored):
                return SupertonicEngine().settings_identity()

        fp32, int8 = identity_for("fp32"), identity_for("int8")
        self.assertNotEqual(fp32, int8)
        self.assertIn("variant=fp32", fp32)
        self.assertIn("variant=int8", int8)

    def test_variant_reaches_the_cache_payload(self):
        stored = {**settings.DEFAULTS, "supertonic_variant": "int8"}
        with patch.object(settings, "load", return_value=stored):
            key = SupertonicEngine().cache_key("Azonos szöveg", None, None, 1.0,
                                               language="hu")
        with patch.object(settings, "load",
                          return_value={**stored, "supertonic_variant": "fp32"}):
            other = SupertonicEngine().cache_key("Azonos szöveg", None, None, 1.0,
                                                 language="hu")
        self.assertNotEqual(key, other)

    def test_presence_check_fails_when_a_graph_is_missing(self):
        engine = self._engine(supertonic_variant="int8")
        with patch.object(local_engines, "MODELS_DIR", self.root / "models"):
            self.assertFalse(engine.model_present())

    def test_presence_check_passes_with_every_graph_present(self):
        engine = self._engine(supertonic_variant="int8")
        models = self.root / "models"
        onnx = models / SUPERTONIC_VARIANT_DIRS["int8"] / "onnx"
        voices = models / SUPERTONIC_VARIANT_DIRS["fp32"] / "voice_styles"
        onnx.mkdir(parents=True)
        voices.mkdir(parents=True)
        for name in SUPERTONIC_FILES["int8"]:
            (onnx / name).write_bytes(b"x")
        (onnx / "tts.json").write_text("{}")
        (voices / "F1.json").write_text("{}")
        with patch.object(local_engines, "MODELS_DIR", models):
            self.assertTrue(engine.model_present())


class VariantApiTest(unittest.TestCase):
    """A beállítási felület csak a két ismert variánst fogadja el."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            patch.object(settings, "SETTINGS_FILE", root / "settings.json"),
            patch.object(database, "DB_PATH", str(root / "reader.db")),
        ]
        for p in self.patches:
            p.start()
        database.init_db()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _save(self, value):
        return self.client.post("/api/settings", json={"supertonic_variant": value})

    @property
    def client(self):
        import app as application

        # app.py regisztrálja a settings_blueprint-ot; csak használjuk.
        self.assertIn("settings_api", application.app.blueprints)
        return application.app.test_client()

    def test_a_known_variant_is_stored(self):
        self.assertEqual(self._save("int8").status_code, 200)
        self.assertEqual(settings.get("supertonic_variant"), "int8")

    def test_an_unknown_variant_falls_back_to_fp32(self):
        self._save("int4")
        self.assertEqual(settings.get("supertonic_variant"), "fp32")


if __name__ == "__main__":
    unittest.main()