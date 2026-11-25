"""Könyvtár- és lejátszómodul tesztek (ffmpeg nélkül, temp fájlokkal)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import library, player  # noqa: E402
from dubstudio.config import Project, load_project  # noqa: E402


class LibraryTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        (self.base / "data").mkdir()
        (self.base / "subs").mkdir()
        (self.base / "work").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _project(self, kind="series"):
        return Project(
            name="Teszt S05", season=5, kind=kind,
            root=self.base, data_dir=self.base / "data",
            subs_dir=self.base / "subs", cache_dir=self.base / "work",
            vocals_globs=["vocals/E{ep2}.wav"],
            me_globs=["me/E{ep2}.wav"],
        )

    def _episode_assets(self, tag="E06"):
        ep = int(tag[1:])
        data = self.base / "data" / f"Show.S05E{ep:02d}.mkv"
        data.write_bytes(b"media")
        (self.base / "subs" / f"{tag}_hun.srt").write_text("1\n", encoding="utf-8")
        (self.base / "work" / "vocals").mkdir(parents=True, exist_ok=True)
        (self.base / "work" / "vocals" / f"E{ep:02d}.wav").write_bytes(b"voc")
        (self.base / "work" / "me").mkdir(parents=True, exist_ok=True)
        (self.base / "work" / "me" / f"E{ep:02d}.wav").write_bytes(b"me")
        return data

    def test_tag_from_name_variants(self):
        self.assertEqual(library.tag_from_name("Show.S05E07.1080p.mkv"), "E07")
        self.assertEqual(library.tag_from_name("Show - E12 - cím.mkv"), "E12")
        self.assertEqual(library.tag_from_name("epizód 3.mkv"), "E03")
        self.assertIsNone(library.tag_from_name("film.mp4"))

    def test_series_library_ready_and_dubbed(self):
        data = self._episode_assets("E06")
        p = self._project()
        lib = library.library(p)
        self.assertEqual(lib["kind"], "series")
        self.assertEqual([e["tag"] for e in lib["episodes"]], ["E06"])
        row = lib["episodes"][0]
        self.assertEqual(row["media"], str(data))
        self.assertTrue(row["ready"])
        self.assertEqual(row["status"], "ready")
        self.assertEqual(lib["next_tag"], "E07")

        # előkészítés + kész kimenet: dubbed, a következő rész E07
        ep = self.base / "work" / "E06"
        ep.mkdir(exist_ok=True)
        (ep / "utterances.json").write_text("[]", encoding="utf-8")
        (ep / "report.json").write_text('{"passed": true}', encoding="utf-8")
        (self.base / "work" / "out").mkdir(exist_ok=True)
        (self.base / "work" / "out" / "Show.S05E06.HUN.mkv").write_bytes(b"out")
        row = library.library(p)["episodes"][0]
        self.assertEqual(row["status"], "dubbed")
        self.assertTrue(row["report_passed"])
        self.assertIsNotNone(row["output"])

    def test_series_missing_assets_and_next_tag(self):
        p = self._project()
        (self.base / "data" / "Show.S05E08.mkv").write_bytes(b"media")
        lib = library.library(p)
        row = lib["episodes"][0]
        self.assertFalse(row["ready"])
        self.assertIn("magyar felirat", row["missing"])
        self.assertIn("leválasztott énekbeszéd", row["missing"])
        self.assertEqual(lib["next_tag"], "E09")

    def test_movie_has_single_e01_and_any_suffix(self):
        (self.base / "data" / "film.avi").write_bytes(b"media")
        p = self._project(kind="movie")
        lib = library.library(p)
        self.assertTrue(lib["is_movie"])
        self.assertEqual([e["tag"] for e in lib["episodes"]], ["E01"])
        self.assertEqual(lib["episodes"][0]["media_name"], "film.avi")
        self.assertEqual(p.episode_media("E01").name, "film.avi")

    def test_scan_marks_known_and_new(self):
        self._episode_assets("E06")
        p = self._project()
        incoming = self.base / "incoming"
        incoming.mkdir()
        (incoming / "Show.S05E06.mkv").write_bytes(b"media")
        (incoming / "Show.S05E07.mkv").write_bytes(b"media")
        rows = library.scan_sources(p, incoming)
        by_tag = {r["tag"]: r for r in rows}
        self.assertTrue(by_tag["E06"]["known"])
        self.assertFalse(by_tag["E07"]["known"])

    def test_project_json_kind_and_series_settings(self):
        path = self.base / "projects" / "film.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "name": "Teszt film", "kind": "movie", "season": 1,
            "root": ".", "data_dir": "../data", "cache_dir": "../work",
            "series": {"auto_process": True, "poll_s": 45},
        }), encoding="utf-8")
        project = load_project(path)
        self.assertEqual(project.kind, "movie")
        self.assertTrue(project.series["auto_process"])
        self.assertEqual(project.series["poll_s"], 45)

        # ismeretlen kind visszaesik a sorozatra
        path.write_text(json.dumps({
            "name": "Teszt", "kind": "ismeretlen", "root": ".",
            "data_dir": "../data", "cache_dir": "../work",
        }), encoding="utf-8")
        self.assertEqual(load_project(path).kind, "series")


class PlayerManifestTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        for d in ("data", "subs", "work/E06/review"):
            (self.base / d).mkdir(parents=True, exist_ok=True)
        self.src = self.base / "data" / "Show.S05E06.mkv"
        self.src.write_bytes(b"media")
        (self.base / "subs" / "E06_hun.srt").write_text("1\n", encoding="utf-8")
        (self.base / "work" / "E06" / "dub.wav").write_bytes(b"dub")
        self.project = Project(
            name="Teszt S05", season=5, root=self.base,
            data_dir=self.base / "data", subs_dir=self.base / "subs",
            cache_dir=self.base / "work",
        )
        self.probe_json = json.dumps({
            "format": {"duration": "120.5"},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264",
                 "width": 1920, "height": 1080},
                {"index": 1, "codec_type": "audio", "codec_name": "aac",
                 "tags": {"language": "eng", "title": "Original"}},
                {"index": 2, "codec_type": "audio", "codec_name": "ac3",
                 "tags": {"language": "hun", "title": "Dub"}},
                {"index": 3, "codec_type": "subtitle", "codec_name": "subrip",
                 "tags": {"language": "eng"}},
                {"index": 4, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle",
                 "tags": {"language": "hun"}},
            ],
        })

    def tearDown(self):
        self._tmp.cleanup()

    def _manifest(self):
        proc = SimpleNamespace(returncode=0, stdout=self.probe_json, stderr="")
        with patch("dubstudio.player.subprocess.run", return_value=proc):
            return player.manifest(self.project, "E06")

    def test_manifest_lists_audio_files_and_text_subtitles(self):
        info = self._manifest()
        self.assertEqual([a["id"] for a in info["audio"]], ["a0", "a1", "dub"])
        self.assertEqual(info["audio"][2]["label"], "Magyar szinkron (kész)")
        # a PGS képfelirat kimarad, a szöveges beágyazott és a fájl bekerül
        ids = [s["id"] for s in info["subtitles"]]
        self.assertIn("s0", ids)
        self.assertIn("file_hun", ids)
        self.assertEqual(len(ids), 2)

    def test_cached_proxy_and_webvtt_are_reused(self):
        info = self._manifest()
        proxy = self.base / "work" / "E06" / "review" / "player_a0.mp4"
        proxy.write_bytes(b"proxy")
        vtt = self.base / "work" / "E06" / "review" / "sub_file_hun.vtt"
        vtt.write_text("WEBVTT\n", encoding="utf-8")
        with patch("dubstudio.player.subprocess.run") as run:
            self.assertEqual(player.media_for(self.project, "E06", "a0", info), proxy)
            self.assertEqual(player.subtitle_for(self.project, "E06", "file_hun", info), vtt)
            run.assert_not_called()

    def test_manifest_marks_cached_proxy_and_offers_dub_sample(self):
        import json

        (self.base / "work" / "E06" / "utterances.json").write_text(json.dumps([
            {"start": 2.0, "end": 6.0},
            {"start": 20.0, "end": 22.0},
        ]), encoding="utf-8")
        proxy = self.base / "work" / "E06" / "review" / "player_a0.mp4"
        proxy.write_bytes(b"proxy")
        info = self._manifest()
        by_id = {a["id"]: a for a in info["audio"]}
        self.assertTrue(by_id["a0"]["cached"])
        self.assertFalse(by_id["dub"]["cached"])
        self.assertEqual(info["dub_sample"]["start"], 1.85)
        self.assertEqual(info["dub_sample"]["duration"], 4.35)

    def test_proxy_status_and_cached_prepare_do_not_start_a_thread(self):
        proxy = self.base / "work" / "E06" / "review" / "player_a0.mp4"
        proxy.write_bytes(b"proxy")
        info = self._manifest()
        st = player.proxy_status(self.project, "E06", "a0", info)
        self.assertEqual(st["state"], "ready")
        self.assertTrue(st["cached"])
        st = player.prepare(self.project, "E06", "a0", info)
        self.assertEqual(st["state"], "ready")
        missing = player.proxy_status(self.project, "E06", "dub", info)
        self.assertEqual(missing["state"], "missing")


if __name__ == "__main__":
    unittest.main()
