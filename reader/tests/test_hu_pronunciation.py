import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as application
from core import database, experience, hu_pronunciation, settings
from core.assist_api import bp as assist_bp


if "assist" not in application.app.blueprints:
    application.app.register_blueprint(assist_bp)


class BuiltinDictionaryTest(unittest.TestCase):
    def test_no_rule_maps_a_word_onto_itself(self):
        # Önmagára cserélő szabály nem javít semmit, csak zavar.
        for source, replacement, _, _ in hu_pronunciation.BUILTIN_RULES:
            self.assertNotEqual(source, replacement, source)

    def test_every_category_is_declared(self):
        declared = set(hu_pronunciation.CATEGORIES)
        for _, _, kind, _ in hu_pronunciation.BUILTIN_RULES:
            self.assertIn(kind, declared)

    def test_every_rule_has_a_note(self):
        for source, _, _, note in hu_pronunciation.BUILTIN_RULES:
            self.assertTrue(note.strip(), f"{source} megjegyzés nélkül")

    def test_sources_are_unique(self):
        sources = [r[0] for r in hu_pronunciation.BUILTIN_RULES]
        self.assertEqual(len(sources), len(set(sources)))

    def test_search_matches_written_and_spoken_form(self):
        self.assertEqual(
            [r["source"] for r in hu_pronunciation.search("shakespeare")], ["Shakespeare"]
        )
        self.assertEqual(
            [r["replacement"] for r in hu_pronunciation.search("sakszper")], ["Sakszpér"]
        )

    def test_search_filters_by_category(self):
        hits = hu_pronunciation.search("", "rövidítés")
        self.assertTrue(hits)
        self.assertTrue(all(r["category"] == "rövidítés" for r in hits))

    def test_search_without_match_is_empty(self):
        self.assertEqual(hu_pronunciation.search("nincsilyen"), [])


class ApplyPronunciationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            patch.object(database, "DB_PATH", str(root / "reader.db")),
            patch.object(settings, "SETTINGS_FILE", root / "settings.json"),
        ]
        for p in self.patches:
            p.start()
        database.init_db()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_builtin_rule_applies_without_any_saved_rule(self):
        out = experience.apply_pronunciation("Shakespeare megírta a drámát.")
        self.assertEqual(out, "Sakszpér megírta a drámát.")

    def test_saved_rule_overrides_the_builtin(self):
        experience.save_rule(None, "Shakespeare", "Sakszpere")
        out = experience.apply_pronunciation("Shakespeare így mondta.")
        self.assertEqual(out, "Sakszpere így mondta.")

    def test_builtin_does_not_swallow_an_inflected_form(self):
        # A toldó a helyes alakhoz tartozik, nem a cserélt szótőhöz.
        out = experience.apply_pronunciation("Móriczot olvasta.")
        self.assertEqual(out, "Móriczot olvasta.")

    def test_saved_rule_still_inflates(self):
        experience.save_rule(None, "Gorcsev", "Gorcsef")
        self.assertEqual(
            experience.apply_pronunciation("Gorcsevék ott ültek."),
            "Gorcsefék ott ültek.",
        )

    def test_builtin_shortcut_applies(self):
        self.assertEqual(
            experience.apply_pronunciation("Igen, stb. ennyi volt."),
            "Igen, s t b ennyi volt.",
        )


class BuiltinApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            patch.object(database, "DB_PATH", str(root / "reader.db")),
            patch.object(settings, "SETTINGS_FILE", root / "settings.json"),
            patch.object(application, "UPLOAD_DIR", str(root / "uploads")),
            patch.object(application, "_startup_complete", True),
        ]
        for p in self.patches:
            p.start()
        database.init_db()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_endpoint_lists_builtins_with_categories(self):
        response = application.app.test_client().get("/api/pronunciation/builtin")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["total"], hu_pronunciation.builtin_count())
        self.assertEqual(data["categories"], list(hu_pronunciation.CATEGORIES))
        self.assertEqual(len(data["rules"]), data["total"])

    def test_endpoint_search_narrows_the_list(self):
        response = application.app.test_client().get(
            "/api/pronunciation/builtin?q=bethlen"
        )
        data = response.get_json()
        self.assertEqual([r["source"] for r in data["rules"]], ["Bethlen"])

    def test_endpoint_marks_overridden_rules(self):
        experience.save_rule(None, "Bethlen", "Saját")
        data = (
            application.app.test_client()
            .get("/api/pronunciation/builtin?q=bethlen")
            .get_json()
        )
        self.assertTrue(data["rules"][0]["overridden"])


if __name__ == "__main__":
    unittest.main()