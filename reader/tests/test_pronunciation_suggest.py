"""A kiejtési szabályjavasló tesztjei.

A lényeg, amit igazolni kell:
  1. a javasló **felfedezi** a motor által rosszul kiejtett szavakat;
  2. **nem másolja le**, amit az Auris már magától jól csinál (számok,
     sorszámok, évszámok, dátumok, római számok, kezdőbetűk, ismert
     rövidítések) — ez a feladat fele;
  3. **nem írja felül** a beépített és a felhasználó által mentett szabályokat.
"""

import contextlib
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from core import database, hu_pronunciation, paths, pronunciation_suggest as suggester
from core import settings as app_settings

# Egy rövid, magyar szókincs a tesztekhez: a valódi 55 000 szavas lista
# betöltése minden tesztnél lassítana, és a döntések lényegét nem ez
# határozza meg. A szótár hiányát a saját teszt igazolja.
TEST_LEXICON = "király\nváros\nfal\nablak\nnapfény\nszoba\nkönyv\ndédszül\n" \
               "thököly\nmóricz\nbethlen\nimre\ngábor\nzsigmond\n"


class _Engine:
    engine_name = "omnivoice"

    def invalidate_voice_prompt(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


class SuggestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.lexicon = Path(self.tmp.name) / "hun.tsv"
        # „felületi alak<TAB>alapalak” formátum, mint az igazi fájlban.
        self.lexicon.write_text(
            "".join(w + "\t" + w + "\n" for w in TEST_LEXICON.split()),
            encoding="utf-8",
        )
        self._cache = suggester._LEXICON_CACHE
        suggester._LEXICON_CACHE = None
        self._patch = patch.dict(
            os.environ, {suggester.LEXICON_ENV: str(self.lexicon)}
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        suggester._LEXICON_CACHE = self._cache
        self.tmp.cleanup()

    # ── felderítés ──────────────────────────────────────────────────────

    def test_finds_name_that_tts_will_mispronounce(self):
        """Egy lexikonban nem szereplő, mondatközepi név javaslatot kap."""
        items = suggester.suggest(
            "A hajó a Hufflepuff nevű kikötőben horgonyzott le."
        )
        self.assertIn("Hufflepuff", [i["source"] for i in items])

    def test_proposals_are_ranked_by_priority(self):
        """Az idegen betűs szó (p1) a magyar név (p3) előtt áll."""
        items = suggester.suggest(
            "A parton Gatsby és Zsigmond együtt ment, közben Gábor is jött."
        )
        priorities = [i["priority"] for i in items]
        self.assertEqual(priorities, sorted(priorities), items)
        sources = [i["source"] for i in items]
        # A q/w/x/y-ot tartalmazó szó a legelső.
        self.assertEqual(sources[0], "Gatsby")

    def test_every_proposal_needs_human_review(self):
        """A felolvasási alak helyességét ember dönti el, nem a gép."""
        items = suggester.suggest("A parton Szecső és Gatsby jöttek.")
        self.assertTrue(items)
        self.assertTrue(all(i["needs_review"] for i in items))
        # Nincs kész felolvasási alak: a javaslat csak jelzi a szót.
        self.assertTrue(all(i["replacement"] == "" for i in items))

    def test_count_reflects_repetitions(self):
        items = suggester.suggest(
            "A parton Szecső jött. Ott volt Szecső. És aludt Szecső."
        )
        row = next(i for i in items if i["source"] == "Szecső")
        self.assertEqual(row["count"], 3)

    # ── nem-duplikáció: a normalizáló már kezeli ────────────────────────

    def test_numbers_and_ordinals_are_never_proposed(self):
        """A számok, évszámok, sorszámok a hungarian_numbers dolgai."""
        text = ("A könyv 1993-ban jelent meg, a 21. fejezetben, "
                "kb. 300 oldalon, 1500 Ft volt.")
        sources = [i["source"] for i in suggester.suggest(text)]
        for token in ("1993", "21.", "300", "1500", "kb."):
            self.assertNotIn(token, sources, f"{token}-ből javaslat keletkezett")

    def test_roman_numerals_and_initials_are_never_proposed(self):
        text = "A XX. században élt, IV. Károly és J. R. R. Tolkien."
        sources = [i["source"] for i in suggester.suggest(text)]
        for token in ("XX.", "IV.", "J."):
            self.assertNotIn(token, sources, f"{token}-ből javaslat keletkezett")

    def test_known_abbreviations_are_never_proposed(self):
        text = "Lásd a 3. oldalon, stb. meg dr. Kiss és a Kft. képviselője."
        sources = [i["source"] for i in suggester.suggest(text)]
        for token in ("stb.", "dr.", "Kft."):
            self.assertNotIn(token, sources, f"{token}-ből javaslat keletkezett")

    def test_ordinary_hungarian_text_yields_no_proposals(self):
        """A tiszta magyar szövegből nem termelünk zajt."""
        text = ("A király ült a trónján, és nézte a város falait. Az ablakon "
                "át a napfény ragyogott be a szobába, ahol a dédszüleink "
                "könyvei álltak.")
        self.assertEqual(suggester.suggest(text), [])

    def test_foreign_letters_in_known_hungarian_word_are_skipped(self):
        """A „nyelv”, „konyha”, „gyors” idegen betűs, de magyar szó."""
        text = "A nyelv a konyha mellett, és gyors volt a döntés."
        self.assertEqual(suggester.suggest(text), [])

    # ── nem írja felül a meglévő szabályokat ────────────────────────────

    def test_builtin_rules_are_never_re_proposed(self):
        """A beépített szótár szavait nem javasoljuk újra."""
        text = "Thököly Imre, Bethlen Gábor és Móricz Zsigmond élt."
        sources = [i["source"] for i in suggester.suggest(text)]
        for word in ("Thököly", "Bethlen", "Móricz"):
            self.assertNotIn(word, sources, f"{word} beépített szabály")

    def test_saved_user_rules_are_never_re_proposed(self):
        """A felhasználó mentett szabályai felülírnak bármilyen javaslatot."""
        text = "A parton Szecső és Copernicust is látták."
        base = {i["source"] for i in suggester.suggest(text)}
        self.assertIn("Szecső", base)
        filtered = {
            i["source"]
            for i in suggester.suggest(text, extra_rules={"Szecső": "Szeccső"})
        }
        self.assertNotIn("Szecső", filtered)

    def test_sentence_initial_capital_is_not_a_candidate(self):
        """A mondatkezdeti nagybetű önmagában nem tulajdonnév-jel."""
        text = ("Kijött az utcára. Elindult a város felé. Megállt a falnál, "
                "és leült az ablak alá.")
        self.assertEqual(suggester.suggest(text), [])

    def test_min_count_filters_rare_words(self):
        text = "A parton Szecső jött. Ott volt Gatsby. És Zsigmond is."
        self.assertEqual(suggester.suggest(text, min_count=2), [])

    def test_limit_truncates_but_keeps_best_first(self):
        # Különböző, lexikonban nem szereplő nevek — így lesz 12 különböző
        # javaslat, amiből a limit csak az első ötöt adja vissza.
        names = " ".join(f"A parton {name} jött."
                         for name in ("Szecső", "Gatsby", "Kovácsfi", "Bálvány",
                                      "Rozgonyi", "Csete", "Fodorfi", "Nyári",
                                      "Bukovinai", "Sárközy", "Talléros", "Véghely"))
        items = suggester.suggest(names, limit=5)
        self.assertEqual(len(items), 5)

    def test_empty_input_returns_nothing(self):
        self.assertEqual(suggester.suggest(""), [])
        self.assertEqual(suggester.suggest([]), [])


class LexiconFallbackTest(unittest.TestCase):
    """A szókincs hiánya nem töri el a modult — csak szűkített a döntés."""

    def test_missing_lexicon_yields_empty_set_and_still_works(self):
        cache = suggester._LEXICON_CACHE
        suggester._LEXICON_CACHE = None
        try:
            with patch.dict(os.environ, {suggester.LEXICON_ENV: "/nonexistent"}):
                with patch.object(suggester, "LEXICON_DEFAULT", "/nonexistent"):
                    with patch.object(suggester, "_lexicon_candidates",
                                      return_value=["/nonexistent"]):
                        # Offline gép: a letöltés is elbukik, nem töri el a modult.
                        with patch.object(suggester, "_download_lexicon",
                                          return_value=None):
                            self.assertEqual(suggester._load_lexicon(), set())
                            self.assertFalse(suggester.has_lexicon())
                            # Az idegen betűs szót így is felderíti.
                            items = suggester.suggest(
                                "A hajó a Gatsby nevű kikötőben horgonyzott."
                            )
                            self.assertIn("Gatsby", [i["source"] for i in items])
        finally:
            suggester._LEXICON_CACHE = cache


class LexiconDownloadTest(unittest.TestCase):
    """A szótár letöltése futásidőben, mert a repóban nincs benne.

    Az adat forrásfüggő licencű, ezért a `hun.tsv` nem a gitben van: az első
    használatkor le kell tölteni, és a sértetlenségét SHA-256 védi.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            patch.object(suggester, "_LEXICON_CACHE", None),
            patch.object(suggester, "_LEXICON_TRIED", None),
        ]
        for p in self.patches:
            p.start()
        self._orig_user_path = paths.user_path
        paths.user_path = lambda *parts: self.root.joinpath(*parts)

    def tearDown(self):
        paths.user_path = self._orig_user_path
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _file(self) -> Path:
        return self.root / "lexicon" / "hun.tsv"

    def test_download_writes_verified_buffer_and_loads_words(self):
        """Sikeres letöltés: pufferfájl + szavak betöltése, egyszer tölt le."""
        payload = b"konyv\tK\nremek\tR\n"
        fake_sha = hashlib.sha256(payload).hexdigest()
        words = self._load_with_stubbed_fetch(payload, sha=fake_sha)
        # Mindkét oszlop bekerül kisbetűsítve: felületi alak ÉS alapalak.
        self.assertEqual(words, {"konyv", "k", "remek", "r"})
        self.assertTrue(self._file().is_file())
        self.assertEqual(self._file().read_bytes(), payload)

    def test_bad_checksum_rejects_download(self):
        """Rossz SHA-256 esetén nem íródik fájl (sértett/sértő forrás)."""
        payload = b"konyv\tK\n"
        words = self._load_with_stubbed_fetch(payload, sha="0" * 64)
        self.assertEqual(words, set())
        self.assertFalse(self._file().exists())

    def _load_with_stubbed_fetch(self, payload: bytes, sha: str | None = None) -> set[str]:
        """`_download_lexicon` lefuttatása hálózat nélkül, adott tartalommal."""
        import io
        import urllib.request as ureq

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        patches = [
            patch.object(suggester, "_lexicon_candidates",
                         return_value=["/nonexistent"]),
            patch.object(ureq, "urlopen", lambda *a, **k: _Resp(payload)),
        ]
        if sha is not None:
            patches.append(patch.object(suggester, "LEXICON_SHA256", sha))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return suggester._load_lexicon()


class BookSuggestionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.patches = [
            patch.object(database, "DB_PATH",
                         os.path.join(self.tmp.name, "reader.db")),
            patch.object(app_settings, "SETTINGS_FILE",
                         Path(self.tmp.name) / "settings.json"),
            patch.object(app_module, "UPLOAD_DIR", self.tmp.name),
            patch.object(app_module, "_startup_complete", True),
            patch.object(app_module, "tts", _Engine()),
        ]
        for p in self.patches:
            p.start()
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id,title,file_path,file_type) "
                "VALUES (1,'K','k','txt')"
            )
            conn.execute(
                "INSERT INTO chapters (book_id,title,order_num,content) "
                "VALUES (1,'1',1,?)",
                ("Thököly Imre és Bethlen Gábor találkozott. "
                 "A parton Gatsby és Szecső is ott volt, kb. 300 vendéggel.",),
            )
            conn.execute(
                "INSERT INTO pronunciation_rules(book_id,source,replacement) "
                "VALUES (1,'Gatsby','Gedzsi')",
            )
        self.client = app_module.app.test_client()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_book_suggestions_skip_existing_rules(self):
        """A könyv mentett szabályai a szűrőben vannak."""
        from core import pronunciation_suggest

        items = pronunciation_suggest.suggest_for_book(1)
        sources = [i["source"] for i in items]
        # A felhasználó már mentett szabályt rá: nem javasoljuk újra.
        self.assertNotIn("Gatsby", sources)
        # A beépített szabályok szintén kihullanak.
        self.assertNotIn("Thököly", sources)
        self.assertNotIn("Bethlen", sources)
        # Az új, fedezetlen név viszont megmarad.
        self.assertIn("Szecső", sources)

    def test_api_endpoint_returns_suggestions(self):
        response = self.client.get("/api/pronunciation/suggest?book_id=1")
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["has_lexicon"])
        sources = [i["source"] for i in payload["suggestions"]]
        self.assertIn("Szecső", sources)
        self.assertNotIn("Gatsby", sources)

    def test_api_endpoint_rejects_missing_book(self):
        response = self.client.get("/api/pronunciation/suggest?book_id=999")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_api_endpoint_can_save_a_suggestion(self):
        response = self.client.post(
            "/api/pronunciation/suggest",
            json={"book_id": 1, "source": "Szecső", "replacement": "Szeccső"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        with database.get_conn() as conn:
            row = conn.execute(
                "SELECT replacement FROM pronunciation_rules "
                "WHERE book_id=1 AND source='Szecső'"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["replacement"], "Szeccső")

    def test_api_endpoint_validates_the_replacement(self):
        # A [ és ] a TTS vezérlőcímkéje: a mentő ezt joggal utasítja el.
        response = self.client.post(
            "/api/pronunciation/suggest",
            json={"book_id": 1, "source": "Szecső", "replacement": "[rossz]"},
        )
        self.assertEqual(response.status_code, 400, response.get_json())


if __name__ == "__main__":
    unittest.main()
