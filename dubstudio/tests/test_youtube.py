"""A YouTube-letöltő tesztjei — hálózat nélkül.

A valódi yt-dlp- és HTTP-hívásokat helyettesítjük: a feliratok jelöltjeit,
az újrapróbálkozást és a VTT/JSON3 → SRT átalakítást ellenőrizzük. A cél az a
viselkedés, ami miatt korábban elhalt a letöltés: egy 429-cel elutasított
felirat-cím nem viszi el az egészet, a következő jelölt (vagy a következő kör)
sikerrel jár.

Futtatás:
  ../reader/.venv/Scripts/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import youtube  # noqa: E402


def _entry(url: str, ext: str) -> dict:
    return {"url": url, "ext": ext, "http_headers": {}}


class UrlTest(unittest.TestCase):

    def test_public_youtube_urls_are_accepted(self):
        for url in ("https://www.youtube.com/watch?v=abc123",
                    "https://youtu.be/abc123",
                    "http://m.youtube.com/watch?v=abc123"):
            self.assertTrue(youtube.is_youtube_url(url), url)

    def test_other_hosts_and_schemes_are_rejected(self):
        for url in ("https://example.com/watch?v=abc",
                    "file:///etc/passwd",
                    "https://youtube.com.evil.example/watch?v=abc",
                    "https://user:pass@youtube.com/watch?v=abc",
                    ""):
            self.assertFalse(youtube.is_youtube_url(url), url)


class LanguageTest(unittest.TestCase):

    def test_project_language_maps_to_youtube_code_and_back(self):
        self.assertEqual(youtube._lang_code("eng"), "en")
        self.assertEqual(youtube._lang_code("heb"), "he")
        self.assertEqual(youtube._lang_code("ismeretlen"), "en")
        self.assertEqual(youtube._subtitle_lang("en"), "eng")
        self.assertEqual(youtube._subtitle_lang("he"), "heb")


class CandidateTest(unittest.TestCase):

    def test_manual_captions_come_before_automatic(self):
        info = {
            "subtitles": {"en": [_entry("manual", "srt")]},
            "automatic_captions": {"en": [_entry("auto", "srt")]},
        }
        got = youtube._caption_candidates(info, "en")
        self.assertEqual([e["url"] for e in got], ["manual", "auto"])

    def test_language_variants_are_accepted(self):
        info = {"automatic_captions": {"en-orig": [_entry("orig", "vtt")],
                                       "en-US": [_entry("us", "vtt")]}}
        got = youtube._caption_candidates(info, "en")
        self.assertEqual([e["url"] for e in got], ["orig", "us"])

    def test_ordering_prefers_srt_and_drops_duplicates(self):
        entries = [_entry("a", "vtt"), _entry("b", "srt"),
                   _entry("a", "vtt"), _entry("c", "json3"),
                   _entry("d", "ttml")]
        got = youtube._ordered_candidates(entries)
        self.assertEqual([ext for ext, _ in got], ["srt", "vtt", "json3"])
        self.assertEqual([e["url"] for _, e in got], ["b", "a", "c"])


class ConvertTest(unittest.TestCase):

    def test_vtt_is_converted_and_srt_untouched(self):
        vtt = ("WEBVTT\nKind: captions\nLanguage: en\n\n"
               "00:00:01.000 --> 00:00:03.500 align:start position:0%\n"
               "Hello<00:00:01.500><c> world</c>\n")
        srt = youtube._convert(vtt.encode("utf-8"), "vtt")
        self.assertIn("00:00:01,000 --> 00:00:03,500", srt)
        self.assertIn("Hello world", srt)
        self.assertTrue(youtube._convert(srt.encode("utf-8"), "srt").startswith("1\n"))

    def test_vtt_rolling_captions_are_not_duplicated(self):
        vtt = ("WEBVTT\n\n"
               "00:00:00.000 --> 00:00:02.000\nFirst line\n\n"
               "00:00:02.000 --> 00:00:04.000\nFirst line\nSecond line\n\n"
               "00:00:04.000 --> 00:00:06.000\nSecond line\nThird line\n")
        srt = youtube._convert(vtt.encode("utf-8"), "vtt")
        self.assertEqual(srt.count("First line"), 1)
        self.assertEqual(srt.count("Second line"), 1)
        self.assertIn("Third line", srt)

    def test_json3_is_converted(self):
        raw = ('{"events": [{"tStartMs": 1000, "dDurationMs": 2000, '
               '"segs": [{"utf8": "Hello "}, {"utf8": "there"}]}, '
               '{"tStartMs": 4000, "dDurationMs": 1000, "segs": [{"utf8": "\\n"}]}]}')
        srt = youtube._convert(raw.encode("utf-8"), "json3")
        self.assertIn("00:00:01,000 --> 00:00:03,000", srt)
        self.assertIn("Hello there", srt)

    def test_unknown_format_is_empty(self):
        self.assertEqual(youtube._convert(b"<xml/>", "ttml"), "")


class MergeFragmentsTest(unittest.TestCase):

    def test_fragments_are_joined_into_sentences(self):
        srt = ("1\n00:00:00,000 --> 00:00:02,400\nGoogle just turned Gemini Notebook into\n\n"
               "2\n00:00:02,400 --> 00:00:04,300\na website builder. You drop in\n\n"
               "3\n00:00:04,300 --> 00:00:09,300\nyour notes. Then it builds it.\n")
        merged = youtube._merge_fragments(srt)
        self.assertEqual(merged.count("--> "), 2)
        self.assertIn("Google just turned Gemini Notebook into a website builder.", merged)

    def test_merged_cues_do_not_overlap(self):
        srt = ("1\n00:00:00,000 --> 00:00:04,300\nfirst fragment\n\n"
               "2\n00:00:02,400 --> 00:00:06,199\nsecond fragment.\n\n"
               "3\n00:00:06,000 --> 00:00:09,300\nthird one.\n")
        merged = youtube._merge_fragments(srt)
        stamps = [line for line in merged.splitlines() if "-->" in line]
        self.assertEqual(len(stamps), 2)
        for previous, following in zip(stamps, stamps[1:]):
            end = previous.split("--> ")[1]
            start = following.split(" -->")[0]
            self.assertLessEqual(end, start)

    def test_empty_input_stays_empty(self):
        self.assertEqual(youtube._merge_fragments(""), "")


class RetryTest(unittest.TestCase):

    def test_rate_limited_candidate_is_skipped_and_next_wins(self):
        calls = []

        def fake_fetch(url, headers=None):
            calls.append(url)
            if url == "first":
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
            return b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"

        entries = [_entry("first", "srt"), _entry("second", "srt")]
        srt = youtube._download_subtitle(entries, "en", fetch=fake_fetch, delays=(0,))
        self.assertIn("Hello", srt)
        self.assertEqual(calls, ["first", "second"])

    def test_transient_429_is_retried_in_a_later_round(self):
        state = {"calls": 0}

        def fake_fetch(url, headers=None):
            state["calls"] += 1
            if state["calls"] == 1:
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
            return b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"

        entries = [_entry("only", "srt")]
        srt = youtube._download_subtitle(entries, "en", fetch=fake_fetch,
                                         delays=(0, 0))
        self.assertIn("Hello", srt)
        self.assertEqual(state["calls"], 2)

    def test_all_candidates_failing_is_a_clear_error(self):
        def fake_fetch(url, headers=None):
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

        with self.assertRaises(youtube.YouTubeError) as ctx:
            youtube._download_subtitle([_entry("a", "srt")], "en",
                                       fetch=fake_fetch, delays=(0,))
        self.assertIn("429", str(ctx.exception))
        self.assertIn("SRT", str(ctx.exception))

    def test_missing_subtitle_track_is_a_clear_error(self):
        with self.assertRaises(youtube.YouTubeError) as ctx:
            youtube._download_subtitle([], "en", fetch=lambda *a, **k: b"")
        self.assertIn("nincs en nyelvű felirat", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
