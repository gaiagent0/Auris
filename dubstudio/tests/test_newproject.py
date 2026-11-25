"""A projektgyár tesztjei — a repóban nincs beégetett projekt.

A friss klón ``projects/`` mappája üres, ezért az első projektet a
``newproject.project_doc`` állítja elő (a felület varázslója és a
``scripts/check_dubstudio.py --init`` is ezt használja). Itt azt ellenőrizzük,
hogy a dokumentum önmagában működőképes: a ``config.load_project`` betölti,
a modellútvonalak abszolútak, és a mért alapértékek benne vannak.

Futtatás:
  ../reader/.venv/Scripts/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubstudio import config, newproject  # noqa: E402


class TestProjectDoc(unittest.TestCase):
    """A gyár kimenete betölthető, projektfüggetlen alapértékekkel."""

    def test_movie_and_series_paths(self):
        """A média- és munkakönyvtár projektenként külön van."""
        movie = newproject.project_doc("Teszt Film", "movie", 1, "eng")
        self.assertEqual(movie["kind"], "movie")
        self.assertEqual(movie["root"], "../movies/teszt-film")
        self.assertEqual(movie["cache_dir"], "../work/teszt-film")
        series = newproject.project_doc("Teszt Sorozat", "series", 2, "heb")
        self.assertEqual(series["kind"], "series")
        self.assertEqual(series["season"], 2)
        self.assertEqual(series["root"], "../series/teszt-sorozat")
        self.assertEqual(series["source_lang"], "heb")

    def test_defaults_are_present(self):
        """A kapuk, a hangzár és az intake beállításai mind benne vannak."""
        doc = newproject.project_doc("Teszt Film")
        self.assertEqual(doc["gate"], newproject.GATE_DEFAULTS)
        self.assertEqual(doc["tuning"], newproject.TUNING_DEFAULTS)
        self.assertTrue(doc["intake"]["translate"]["enabled"])
        self.assertTrue(doc["intake"]["separate"]["enabled"])
        # a sorozatra kalibrált fix beszélőszám nem kerülhet bele
        self.assertEqual(doc["diarization"]["num_speakers"], 0)
        # a kulcslistát ne bővítse csendben senki
        self.assertEqual(
            set(doc), {"name", "season", "kind", "root", "cache_dir", "data_dir",
                       "subs_dir", "vocals_globs", "me_globs", "source_lang",
                       "original_audio", "tts_sr", "models", "tuning", "gate",
                       "llm", "asr", "intake", "diarization", "series",
                       "media_server"})

    def test_models_are_absolute(self):
        """A modellek abszolút útvonalat kapnak (a projekt mappája máshol van)."""
        doc = newproject.project_doc("Teszt Film")
        for key, value in doc["models"].items():
            self.assertTrue(Path(value).is_absolute(), f"{key}: {value}")
        self.assertTrue(doc["models"]["supertonic"].endswith("supertonic-sherpa-int8"))
        self.assertTrue(Path(doc["intake"]["separate"]["model_dir"]).is_absolute())

    def test_load_project_roundtrip(self):
        """A kiírt JSON a config.load_project-tal betölthető és konzisztens."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projects" / "teszt-film.json"
            path.parent.mkdir(parents=True)
            doc = newproject.project_doc("Teszt Film", "movie", 1, "eng")
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            project = config.load_project(path)
        self.assertEqual(project.name, "Teszt Film")
        self.assertEqual(project.kind, "movie")
        self.assertEqual(project.model("diar_segmentation"), Path(
            doc["models"]["diar_segmentation"]))
        self.assertTrue(project.root.is_absolute())

    def test_slugify(self):
        """Az azonosító ékezet nélküli, kötőjeles és nem üres."""
        self.assertEqual(newproject.slugify("Így írjuk: Ő meg Ű!"), "igy-irjuk-o-meg-u")
        self.assertEqual(newproject.slugify("!!!"), "film")
        self.assertLessEqual(len(newproject.slugify("a" * 80)), 48)


class TestResolveProject(unittest.TestCase):
    """A CLI projektfeloldása: --project → DUBSTUDIO_PROJECT → egyetlen JSON."""

    def _env(self, **values):
        old = {k: os.environ.get(k) for k in values}
        os.environ.update({k: v for k, v in values.items() if v is not None})
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None
                                 else os.environ.__setitem__(k, v)
                                 for k, v in old.items()])

    def test_env_wins_over_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sajat.json"
            path.write_text("{}", encoding="utf-8")
            self._env(DUBSTUDIO_PROJECT=str(path), DUBSTUDIO_ROOT=None)
            self.assertEqual(config.resolve_project(None), path.resolve())

    def test_single_project_in_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "projects"
            folder.mkdir()
            (folder / "csak-egy.json").write_text("{}", encoding="utf-8")
            self._env(DUBSTUDIO_PROJECT=None, DUBSTUDIO_ROOT=tmp)
            self.assertEqual(config.resolve_project(None),
                             (folder / "csak-egy.json").resolve())

    def test_empty_folder_raises_with_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._env(DUBSTUDIO_PROJECT=None, DUBSTUDIO_ROOT=tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                config.resolve_project(None)
        self.assertIn("--init", str(ctx.exception))

    def test_two_projects_need_explicit_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "projects"
            folder.mkdir()
            for name in ("egy", "ketto"):
                (folder / f"{name}.json").write_text("{}", encoding="utf-8")
            self._env(DUBSTUDIO_PROJECT=None, DUBSTUDIO_ROOT=tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                config.resolve_project(None)
            self.assertIn("egy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
