"""A klónreferencia-mintavételező tesztjei.

A lényeg, amit igazolni kell: a mintavételező **a tisztább szakaszt választja**,
nem az elsőt. Ezért a fő teszt mesterségesen zajos és tiszta szakaszt kever,
és azt várja, hogy a tiszta nyer.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import app as app_module
from core import database, reference_sampler as sampler
from core import settings as app_settings

HAS_FFMPEG = shutil.which("ffmpeg") is not None
SR = 24000


def _speech(seconds, sr=SR, rng=None):
    """Beszédszerű jel: hangmagasság-váltakozó, ritkított harmonikus tartomány."""
    rng = rng or np.random.default_rng(1)
    t = np.arange(int(sr * seconds)) / sr
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tone = 0.25 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)
    # szótagok: ~4 Hz-es kapcsolás, hogy legyenek szünetek, mint az élő beszédnél
    gate = (np.sin(2 * np.pi * 4.0 * t) > -0.3).astype(float)
    gate *= np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t), 0, 1)
    return (tone * gate).astype(np.float32)


def _write_mixed(path):
    """5 mp csend + 12 mp ZAJOS beszéd + 3 mp csend + 12 mp TISZTA + 5 mp csend.

    A zajos szakasz hangosabb is, hogy a naív „leghangosabb nyer" logika is
    rossz választ adna — a pontozásnak a tisztát kell választania.
    """
    rng = np.random.default_rng(7)
    silence = np.zeros(int(SR * 5), dtype=np.float32)
    noisy = _speech(12, rng=rng)
    noisy = noisy * 1.6 + 0.05 * rng.standard_normal(len(noisy)).astype(np.float32)
    noisy += 0.04 * np.sin(2 * np.pi * 440 * np.arange(len(noisy)) / SR)  # háttérzene
    clean = _speech(12, rng=rng) * 0.8
    audio = np.concatenate([silence, noisy, silence, clean, silence])
    sf.write(path, audio, SR)
    return 5.0, 17.0, 20.0, 32.0            # noisy start/end, clean start/end


class SpeechSpanTest(unittest.TestCase):
    def test_rms_fallback_finds_speech(self):
        path = os.path.join(tempfile.mkdtemp(), "s.wav")
        sf.write(path, _speech(6), SR)
        audio, sr = sampler.load_audio(path)
        spans, method = sampler.speech_spans(audio, sr)
        self.assertIn(method, ("silero", "rms"))
        self.assertTrue(spans, "nem talált beszédet")
        covered = sum(e - s for s, e in spans)
        self.assertGreater(covered, 2.0, f"túl kevés beszéd: {covered:.1f} mp")

    def test_pure_silence_has_no_speech(self):
        path = os.path.join(tempfile.mkdtemp(), "z.wav")
        sf.write(path, np.zeros(SR * 5, dtype=np.float32), SR)
        audio, sr = sampler.load_audio(path)
        spans, _ = sampler.speech_spans(audio, sr)
        self.assertLess(sum(e - s for s, e in spans), 0.5)


class ScoreWindowTest(unittest.TestCase):
    def test_clipping_is_penalised(self):
        audio = _speech(10)
        clean = sampler.score_window(audio, SR, 0, 10, [(0, 10)], 10)
        clipped = audio.copy()
        clipped[::7] = 1.0
        bad = sampler.score_window(clipped, SR, 0, 10, [(0, 10)], 10)
        self.assertLess(bad["score"], clean["score"])
        self.assertEqual(bad["parts"]["no_clipping"], 0.0)

    def test_noise_lowers_snr_score(self):
        rng = np.random.default_rng(5)
        clean = sampler.score_window(_speech(10), SR, 0, 10, [(0, 10)], 10)
        noisy = _speech(10, rng=rng) + 0.08 * rng.standard_normal(SR * 10).astype(np.float32)
        bad = sampler.score_window(noisy, SR, 0, 10, [(0, 10)], 10)
        self.assertLess(bad["parts"]["snr"], clean["parts"]["snr"])

    def test_silence_window_scores_near_zero(self):
        # A csendnek nincs beszéd-spanja — a hívó így kapja meg a VAD-től.
        # (Korábban itt [(0, 10)] span volt, ami önmagának ellentmondó
        # előfeltétel: csendet jelöltünk ki beszédnek.)
        quiet = sampler.score_window(np.zeros(SR * 10, dtype=np.float32), SR,
                                     0, 10, [], 10)
        self.assertLess(quiet["score"], 40.0, quiet)
        self.assertEqual(quiet["parts"]["speech_density"], 0.0)

    def test_silence_is_not_credited_as_stable_level(self):
        # Még ha a hívó tévesen beszédnek is jelölné a csendet, a
        # hangerő-stabilitás nem adhat teljes pontot nulla jel szinten.
        quiet = sampler.score_window(np.zeros(SR * 10, dtype=np.float32), SR,
                                     0, 10, [(0, 10)], 10)
        self.assertEqual(quiet["parts"]["level_stability"], 0.0, quiet)


class SampleReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "bemenet.wav")
        self.dst = os.path.join(self.tmp, "ref.wav")
        self.noisy_s, self.noisy_e, self.clean_s, self.clean_e = _write_mixed(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_picks_the_clean_segment_not_the_loud_one(self):
        report = sampler.sample_reference(self.src, self.dst)
        self.assertTrue(report["sampled"], report.get("error"))
        start = report["start_sec"]
        # a tiszta szakasz 20-32 mp között van
        self.assertGreaterEqual(start, self.clean_s - 1.0,
                                f"nem a tiszta szakaszt választotta: {start:.1f} mp")
        self.assertLessEqual(start, self.clean_e,
                             f"a tiszta szakaszon túlra nyúlt: {start:.1f} mp")

    def test_output_is_24k_mono_and_cleaned(self):
        report = sampler.sample_reference(self.src, self.dst)
        self.assertTrue(report["sampled"], report.get("error"))
        info = sf.info(self.dst)
        self.assertEqual((info.samplerate, info.channels), (24000, 1))
        after = report["after"]
        self.assertGreater(after["seconds"], 4.0)
        self.assertLess(after["peak_dbfs"], 0.0, "a kimenet csúcsa 0 dB felett van")
        self.assertEqual(after["clipped_samples"], 0)

    def test_report_shape(self):
        report = sampler.sample_reference(self.src, self.dst)
        for key in ("source", "output", "sampled", "error", "vad", "candidates",
                    "start_sec", "end_sec", "score", "grade", "parts", "after",
                    "top", "recommendation"):
            self.assertIn(key, report)
        self.assertIn(report["grade"], [g for g, _ in sampler.GRADE_THRESHOLD] + ["gyenge"])
        self.assertGreaterEqual(report["candidates"], 1)
        self.assertTrue(report["top"])
        self.assertTrue(report["recommendation"])
        # a jelentés legyen JSON-serializálható (az API ezt adja vissza)
        json.dumps(report)

    def test_no_clean_skips_the_chain(self):
        report = sampler.sample_reference(self.src, self.dst, clean=False)
        self.assertTrue(report["sampled"], report.get("error"))
        self.assertFalse(report["cleaned"])
        self.assertEqual(sf.info(self.dst).samplerate, SR)

    def test_short_speech_is_reported_not_crashed(self):
        short = os.path.join(self.tmp, "rovid.wav")
        sf.write(short, _speech(1.5), SR)
        report = sampler.sample_reference(short, os.path.join(self.tmp, "ki.wav"))
        self.assertFalse(report["sampled"])
        self.assertTrue(report["error"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "ki.wav")))

    def test_silence_only_is_reported(self):
        z = os.path.join(self.tmp, "z.wav")
        sf.write(z, np.zeros(SR * 8, dtype=np.float32), SR)
        report = sampler.sample_reference(z, os.path.join(self.tmp, "zki.wav"))
        self.assertFalse(report["sampled"])
        self.assertIn("beszéd", report["error"].lower())

    def test_missing_file_is_reported(self):
        report = sampler.sample_reference(os.path.join(self.tmp, "nincs.wav"), self.dst)
        self.assertFalse(report["sampled"])
        self.assertIn("nem található", report["error"].lower())


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required")
class FormatTest(unittest.TestCase):
    """A nem-WAV bemenetet az ffmpeg útján kell kezelnie."""

    def test_mp3_input_is_decoded(self):
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, "in.wav")
        mp3 = os.path.join(tmp, "in.mp3")
        sf.write(wav, _speech(8), SR)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", wav, "-b:a", "128k", mp3], check=True)
        report = sampler.sample_reference(mp3, os.path.join(tmp, "ref.wav"))
        self.assertTrue(report["sampled"], report.get("error"))
        self.assertEqual(sf.info(os.path.join(tmp, "ref.wav")).samplerate, SR)
        shutil.rmtree(tmp, ignore_errors=True)


class CliTest(unittest.TestCase):
    def test_cli_writes_json_and_exit_code(self):
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "in.wav")
        sf.write(src, _speech(8), SR)
        out = os.path.join(tmp, "ref.wav")
        js = os.path.join(tmp, "jelentes.json")
        code = sampler.main([src, "--out", out, "--json", js, "--quiet"])
        self.assertEqual(code, 0)
        with open(js, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertTrue(report["sampled"])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_returns_one_on_failure(self):
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "z.wav")
        sf.write(src, np.zeros(SR * 5, dtype=np.float32), SR)
        self.assertEqual(sampler.main([src, "--out", os.path.join(tmp, "o.wav"),
                                       "--quiet"]), 1)
        shutil.rmtree(tmp, ignore_errors=True)


class _Engine:
    engine_name = "omnivoice"

    def invalidate_voice_prompt(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required")
class SampleUploadTest(unittest.TestCase):
    """A végpont szintű mintavétel: `sample=1` a ref-audio feltöltésen."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.patches = [
            patch.object(database, "DB_PATH", os.path.join(self.tmp.name, "reader.db")),
            patch.object(app_settings, "SETTINGS_FILE", Path(self.tmp.name) / "settings.json"),
            patch.object(app_module, "UPLOAD_DIR", self.tmp.name),
            patch.object(app_module, "_startup_complete", True),
            patch.object(app_module, "tts", _Engine()),
        ]
        for p in self.patches:
            p.start()
        database.init_db()
        with database.get_conn() as conn:
            conn.execute("INSERT INTO books (id,title,file_path,file_type) "
                         "VALUES (1,'K','k','txt')")
            conn.execute("INSERT INTO characters (id,book_id,name) VALUES (7,1,'Anna')")
        self.client = app_module.app.test_client()
        self.wav = os.path.join(self.tmp.name, "hosszu.wav")
        sf.write(self.wav, _speech(20), SR)          # 20 mp beszéd

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _post(self, path, filename, payload, data):
        buf = io.BytesIO(data)
        return self.client.post(path,
                                data={"file": (buf, filename), **payload},
                                content_type="multipart/form-data")

    def _stored(self, query):
        with database.get_conn() as conn:
            return conn.execute(query).fetchone()[0]

    def test_sample_picks_a_segment_and_reports_quality(self):
        with open(self.wav, "rb") as handle:
            response = self._post("/api/characters/7/ref-audio", "eloszo.wav",
                                  {"sample": "1", "ref_text": "Szia"}, handle.read())
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("sample", body, "a jelentés hiányzik a válaszból")
        report = body["sample"]
        self.assertTrue(report["sampled"])
        self.assertGreater(report["score"], 0)
        self.assertIn("parts", report)
        self.assertIn("recommendation", report)
        stored = self._stored("SELECT ref_audio_path FROM characters WHERE id=7")
        self.assertEqual(sf.info(stored).samplerate, 24000)
        self.assertEqual(sf.info(stored).channels, 1)

    def test_sample_accepts_mp3_not_only_wav(self):
        mp3 = os.path.join(self.tmp.name, "hang.mp3")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", self.wav, "-b:a", "128k", mp3], check=True)
        with open(mp3, "rb") as handle:
            response = self._post("/api/characters/7/ref-audio", "hang.mp3",
                                  {"sample": "1"}, handle.read())
        self.assertEqual(response.status_code, 200, response.get_json())
        stored = self._stored("SELECT ref_audio_path FROM characters WHERE id=7")
        self.assertEqual(sf.info(stored).samplerate, 24000)

    def test_sample_of_silence_is_rejected_with_422(self):
        silent = os.path.join(self.tmp.name, "csend.wav")
        sf.write(silent, np.zeros(SR * 8, dtype=np.float32), SR)
        with open(silent, "rb") as handle:
            response = self._post("/api/characters/7/ref-audio", "csend.wav",
                                  {"sample": "1"}, handle.read())
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.get_json())
        self.assertIsNone(self._stored("SELECT ref_audio_path FROM characters WHERE id=7"))

    def test_without_sample_wav_is_still_required(self):
        """A `sample` nélkül változatlan a régi viselkedés: csak WAV."""
        with open(self.wav, "rb") as handle:
            response = self._post("/api/characters/7/ref-audio", "hang.wav",
                                  {"clean": "0"}, handle.read())
        self.assertEqual(response.status_code, 200, response.get_json())
        # a nem-WAV feltöltés továbbra is elutasítva
        with open(self.wav, "rb") as handle:
            bad = self._post("/api/characters/7/ref-audio", "hang.flac",
                             {"clean": "0"}, handle.read())
        self.assertEqual(bad.status_code, 400)
        self.assertIn("WAV", bad.get_json()["error"])

    def test_narrator_endpoint_also_samples(self):
        with open(self.wav, "rb") as handle:
            response = self._post("/api/books/1/narrator-ref-audio", "eloszo.wav",
                                  {"sample": "1", "ref_text": "Helló"}, handle.read())
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertIn("sample", body)
        stored = self._stored(
            "SELECT narrator_ref_audio_path FROM books WHERE id=1")
        self.assertEqual(sf.info(stored).samplerate, 24000)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required")
class ResampleNarratorRefTest(unittest.TestCase):
    """A „Legjobb szakasz kiválasztása" gomb végpontja a Voice Stúdióban.

    A lényeg: a végpont **a már aktív** narrátorhangot mintavételezi, nem vár
    új fájlt, és a kiválasztás után a könyv új referenciaútvonalat kap.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.patches = [
            patch.object(database, "DB_PATH", os.path.join(self.tmp.name, "reader.db")),
            patch.object(app_settings, "SETTINGS_FILE", Path(self.tmp.name) / "settings.json"),
            patch.object(app_module, "UPLOAD_DIR", self.tmp.name),
            patch.object(app_module, "_startup_complete", True),
            patch.object(app_module, "tts", _Engine()),
        ]
        for p in self.patches:
            p.start()
        database.init_db()
        with database.get_conn() as conn:
            conn.execute("INSERT INTO books (id,title,file_path,file_type) "
                         "VALUES (1,'K','k','txt')")
        self.client = app_module.app.test_client()
        # Egy hosszú, zajos narrátorhang — a mintavételezőnek ebből kell
        # kiválasztania a tiszta szakaszt.
        self.src = os.path.join(self.tmp.name, "narrator_hosszu.wav")
        _write_mixed(self.src)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _active_ref(self):
        with database.get_conn() as conn:
            return conn.execute(
                'SELECT narrator_ref_audio_path, narrator_ref_audio_name, narrator_ref_text '
                'FROM books WHERE id=1').fetchone()

    def _activate(self):
        """Egy régi, hosszú narrátorhangot tesz aktívvá, ahogy a feltöltés tenné."""
        old = os.path.join(self.tmp.name, "narrator_ref_regi.wav")
        with open(self.src, "rb") as src, open(old, "wb") as dst:
            dst.write(src.read())
        with database.get_conn() as conn:
            conn.execute(
                'UPDATE books SET narrator_ref_audio_path=?, narrator_ref_audio_name=?, '
                'narrator_ref_text=? WHERE id=?',
                (old, "regi_narrator.wav", "Szia, ez a régi átirat.", 1))
        return old

    def test_resamples_the_active_reference_without_a_new_upload(self):
        old = self._activate()
        response = self.client.post("/api/books/1/narrator-ref-sample")
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("sample", body, "a minőségi jelentés hiányzik")
        report = body["sample"]
        self.assertTrue(report["sampled"])
        self.assertGreater(report["score"], 0)
        self.assertIn("parts", report)
        self.assertIn("recommendation", report)
        row = self._active_ref()
        # Az új referencia más fájl, a név a mentett fájlnév (ahogy a
        # feltöltési útnál is), az átirat változatlanul megmarad.
        self.assertNotEqual(row["narrator_ref_audio_path"], old)
        self.assertEqual(row["narrator_ref_audio_name"], os.path.basename(old))
        self.assertEqual(row["narrator_ref_text"], "Szia, ez a régi átirat.")
        info = sf.info(row["narrator_ref_audio_path"])
        self.assertEqual((info.samplerate, info.channels), (24000, 1))
        # A kiválasztott szakasz rövidebb, mint a teljes zajos anyag.
        self.assertLess(info.frames / info.samplerate, 20.0)

    def test_rejects_when_no_reference_is_active(self):
        response = self.client.post("/api/books/1/narrator-ref-sample")
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.get_json())
        self.assertIsNone(self._active_ref()["narrator_ref_audio_path"])

    def test_rejects_on_silent_reference(self):
        silent = os.path.join(self.tmp.name, "narrator_csend.wav")
        sf.write(silent, np.zeros(SR * 8, dtype=np.float32), SR)
        with database.get_conn() as conn:
            conn.execute("UPDATE books SET narrator_ref_audio_path=?, "
                         "narrator_ref_audio_name=? WHERE id=?",
                         (silent, "csend.wav", 1))
        before = self._active_ref()["narrator_ref_audio_path"]
        response = self.client.post("/api/books/1/narrator-ref-sample")
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.get_json())
        # Sikertelen mintavétel után a régi referencia marad aktív.
        self.assertEqual(self._active_ref()["narrator_ref_audio_path"], before)

    def test_rejects_on_missing_book(self):
        response = self.client.post("/api/books/99/narrator-ref-sample")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
