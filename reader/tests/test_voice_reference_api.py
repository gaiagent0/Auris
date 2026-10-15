import io
import os
import tempfile
import unittest
from unittest.mock import patch

import app as app_module
from core import database


class VoiceReferenceApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        self.original_upload_dir = app_module.UPLOAD_DIR
        self.original_startup = app_module._startup_complete
        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        app_module.UPLOAD_DIR = self.tmp.name
        app_module._startup_complete = True
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type) VALUES (1, 'Test', 'test.txt', 'txt')"
            )
            conn.execute(
                "INSERT INTO characters (id, book_id, name) VALUES (2, 1, 'Alice')"
            )
            conn.execute(
                "INSERT INTO characters (id, book_id, name) VALUES (3, 1, 'Bob')"
            )
            conn.execute(
                "INSERT INTO chapters "
                "(id, book_id, title, order_num, content, word_count) "
                "VALUES (4, 1, 'Opening', 0, 'Test chapter.', 2)"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        app_module.UPLOAD_DIR = self.original_upload_dir
        app_module._startup_complete = self.original_startup
        self.tmp.cleanup()

    def test_narrator_upload_is_visible_and_removable(self):
        response = self.client.post(
            '/api/books/1/narrator-ref-audio',
            data={
                'file': (io.BytesIO(b'RIFF-test'), 'narrator voice.wav'),
                'ref_text': 'The exact narrator transcript.',
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['ref_audio_name'], 'narrator voice.wav')

        page = self.client.get('/voice-studio/1')
        self.assertIn(b'narrator voice.wav', page.data)
        self.assertIn(b'The exact narrator transcript.', page.data)

        response = self.client.delete('/api/books/1/narrator-ref-audio')
        self.assertEqual(response.status_code, 200)
        with database.get_conn() as conn:
            book = conn.execute('SELECT * FROM books WHERE id=1').fetchone()
        self.assertIsNone(book['narrator_ref_audio_path'])
        self.assertIsNone(book['narrator_ref_audio_name'])
        self.assertIsNone(book['narrator_ref_text'])

    def test_replacing_reference_changes_path_and_removes_old_file(self):
        def upload(payload):
            response = self.client.post(
                '/api/characters/2/ref-audio',
                data={'file': (io.BytesIO(payload), 'alice.wav'), 'ref_text': 'Same words.'},
                content_type='multipart/form-data',
            )
            self.assertEqual(response.status_code, 200)
            with database.get_conn() as conn:
                return conn.execute('SELECT ref_audio_path FROM characters WHERE id=2').fetchone()[0]

        first = upload(b'RIFF-first-voice')
        second = upload(b'RIFF-second-voice')
        # Cache keys include the reference path, so a new voice needs a new path.
        self.assertNotEqual(first, second)
        self.assertFalse(os.path.exists(first))
        self.assertTrue(os.path.exists(second))
        self.assertEqual(upload(b'RIFF-second-voice'), second)

        narrator = []
        for payload in (b'RIFF-n1', b'RIFF-n2'):
            self.client.post(
                '/api/books/1/narrator-ref-audio',
                data={'file': (io.BytesIO(payload), 'n.wav'), 'ref_text': 'x'},
                content_type='multipart/form-data',
            )
            with database.get_conn() as conn:
                narrator.append(conn.execute('SELECT narrator_ref_audio_path FROM books WHERE id=1').fetchone()[0])
        self.assertNotEqual(narrator[0], narrator[1])
        self.assertFalse(os.path.exists(narrator[0]))

    def test_character_upload_and_api_return_filename_and_transcript(self):
        response = self.client.post(
            '/api/characters/2/ref-audio',
            data={
                'file': (io.BytesIO(b'RIFF-test'), 'alice.wav'),
                'ref_text': 'Alice speaks these words.',
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 200)

        characters = self.client.get('/api/books/1/characters').get_json()
        self.assertEqual(characters[0]['ref_audio_name'], 'alice.wav')
        self.assertEqual(characters[0]['ref_text'], 'Alice speaks these words.')

        response = self.client.delete('/api/characters/2/ref-audio')
        self.assertEqual(response.status_code, 200)
        characters = self.client.get('/api/books/1/characters').get_json()
        self.assertIsNone(characters[0]['ref_audio_path'])
        self.assertIsNone(characters[0]['ref_audio_name'])
        self.assertIsNone(characters[0]['ref_text'])

    def test_characters_can_be_filtered_to_current_chapter(self):
        chapter_segments = [
            {'character_name': None},
            {'character_name': 'alice'},
        ]
        with patch.object(
            app_module,
            '_compute_segments_for_chapter',
            return_value=chapter_segments,
        ) as compute_segments:
            response = self.client.get('/api/books/1/characters?chapter_id=4')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [character['name'] for character in response.get_json()],
            ['Alice'],
        )
        compute_segments.assert_called_once_with(
            1,
            4,
            single_narrator_mode=False,
        )

    def test_voice_studio_exposes_chapter_filter_for_requested_chapter(self):
        response = self.client.get('/voice-studio/1?chapter_id=4')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="chapter-character-filter"', response.data)
        self.assertIn('Csak az aktuális fejezet szereplői'.encode(), response.data)
        self.assertIn(b'Opening', response.data)
        self.assertIn(b'window.CURRENT_CHAPTER_ID = 4', response.data)

    def test_voice_studio_renders_profile_search_and_neutral_accent_controls(self):
        response = self.client.get('/voice-studio/1')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'/static/css/voice-experience.css', response.data)
        self.assertIn(b'id="profile-narrator"', response.data)
        self.assertIn(b'id="profile-name-narrator"', response.data)
        self.assertIn(b'id="character-search"', response.data)
        self.assertIn('Semleges / nincs akcentus'.encode(), response.data)
        self.assertIn('aria-label="Mentett narrátorhang"'.encode(), response.data)


class NarratorTranscribeApiTest(unittest.TestCase):
    """A POST /narrator-ref-transcribe felismeri az átiratot, de nem menti."""

    class _FakeTranscriber:
        model_id = 'fake-asr'

        def __init__(self, text):
            self._text = text
            self.calls = []

        def transcribe(self, path, language=None, **kw):
            self.calls.append((path, language))
            return {'text': self._text, 'words': []}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        self.original_upload_dir = app_module.UPLOAD_DIR
        self.original_startup = app_module._startup_complete
        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        app_module.UPLOAD_DIR = self.tmp.name
        app_module._startup_complete = True
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type) VALUES (1, 'Test', 'test.txt', 'txt')"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        app_module.UPLOAD_DIR = self.original_upload_dir
        app_module._startup_complete = self.original_startup
        self.tmp.cleanup()

    def _attach_reference(self, text=None):
        path = os.path.join(self.tmp.name, 'narrator_ref.wav')
        with open(path, 'wb') as handle:
            handle.write(b'RIFF-fake-pcm')
        with database.get_conn() as conn:
            conn.execute(
                'UPDATE books SET narrator_ref_audio_path=?, narrator_ref_text=? WHERE id=1',
                (path, text),
            )
        return path

    def _patch_transcriber(self, transcriber):
        from core import qa

        return patch.object(qa.Transcriber, 'for_language', return_value=transcriber)

    def test_transcribe_returns_text_and_model_without_saving(self):
        path = self._attach_reference(text='Meglévő átirat.')
        transcriber = self._FakeTranscriber('Felismert szöveg a felvételről.')
        with self._patch_transcriber(transcriber):
            response = self.client.post('/api/books/1/narrator-ref-transcribe')

        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body['ok'])
        self.assertEqual(body['text'], 'Felismert szöveg a felvételről.')
        self.assertEqual(body['model'], 'fake-asr')
        self.assertEqual(transcriber.calls, [(path, 'hu')])
        # A felismerés nem ment: az átirat változatlan a listában.
        with database.get_conn() as conn:
            row = conn.execute(
                'SELECT narrator_ref_text FROM books WHERE id=1'
            ).fetchone()
        self.assertEqual(row['narrator_ref_text'], 'Meglévő átirat.')

    def test_transcribe_requires_active_reference(self):
        with self._patch_transcriber(self._FakeTranscriber('X')):
            response = self.client.post('/api/books/1/narrator-ref-transcribe')
        self.assertEqual(response.status_code, 422)
        self.assertIn('error', response.get_json())

    def test_transcribe_rejects_missing_book(self):
        response = self.client.post('/api/books/99/narrator-ref-transcribe')
        self.assertEqual(response.status_code, 404)

    def test_transcribe_rejects_empty_recognition(self):
        self._attach_reference()
        with self._patch_transcriber(self._FakeTranscriber('   ')):
            response = self.client.post('/api/books/1/narrator-ref-transcribe')
        self.assertEqual(response.status_code, 422)
        self.assertIn('error', response.get_json())

    def test_transcribe_reports_asr_failure(self):
        self._attach_reference()

        class _Broken:
            model_id = 'broken'

            def transcribe(self, *args, **kw):
                raise RuntimeError('model missing')

        with self._patch_transcriber(_Broken()):
            response = self.client.post('/api/books/1/narrator-ref-transcribe')
        self.assertEqual(response.status_code, 500)
        self.assertIn('model missing', response.get_json()['error'])

    def test_voice_studio_has_transcribe_button(self):
        response = self.client.get('/voice-studio/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="transcribe-narrator-ref-btn"', response.data)
        self.assertIn(b'transcribeNarratorRef(this)', response.data)


class ReferenceSampleUploadTest(unittest.TestCase):
    """A `sample=1` feltöltés a legjobb klónozási szakaszt menti.

    A felület alapból ezt küldi (a „legjobb szakasz kiválasztása” jelölő),
    így a klónozás a tiszta, folyamatos beszédet kapja — nem az intro
    köszönését, a hosszú szüneteket vagy a szobazajt. A jelentés (pontszám,
    részpontok, ajánlás) visszajut a felületre, hogy látszódjon, miért ez a
    szakasz lett a kiválasztott.
    """

    SR = 24000

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        self.original_upload_dir = app_module.UPLOAD_DIR
        self.original_startup = app_module._startup_complete
        database.DB_PATH = os.path.join(self.tmp.name, 'reader.db')
        app_module.UPLOAD_DIR = self.tmp.name
        app_module._startup_complete = True
        database.init_db()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT INTO books (id, title, file_path, file_type) VALUES (1, 'Test', 'test.txt', 'txt')"
            )
            conn.execute("INSERT INTO characters (id, book_id, name) VALUES (2, 1, 'Alice')")
            conn.execute(
                "INSERT INTO chapters (id, book_id, title, order_num, content, word_count) "
                "VALUES (4, 1, 'Opening', 0, 'Test chapter.', 2)"
            )
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        app_module.UPLOAD_DIR = self.original_upload_dir
        app_module._startup_complete = self.original_startup
        self.tmp.cleanup()

    def _speech_file(self, seconds=10.0, name='eloszo.wav'):
        """Beszédszerű, szótagolva kapcsoló hang — a mintavételező ezt keresi."""
        import numpy as np
        import soundfile as sf

        t = np.arange(int(self.SR * seconds)) / self.SR
        f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(f0) / self.SR
        tone = (0.25 * np.sin(phase) + 0.12 * np.sin(2 * phase)
                + 0.06 * np.sin(3 * phase))
        gate = (np.sin(2 * np.pi * 4.0 * t) > -0.3).astype(float)
        path = os.path.join(self.tmp.name, name)
        sf.write(path, (tone * gate).astype('float32'), self.SR)
        return path

    def _upload(self, endpoint, path, filename, **extra):
        with open(path, 'rb') as handle:
            data = {'file': (handle, filename)}
            data.update(extra)
            return self.client.post(endpoint, data=data,
                                    content_type='multipart/form-data')

    def test_character_sample_upload_returns_report_without_path(self):
        response = self._upload(
            '/api/characters/2/ref-audio', self._speech_file(), 'eloszo.wav',
            sample='1', ref_text='Ez a mondat hangzik a felvételen.', clean='1')
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body['ok'])
        report = body['sample']
        self.assertTrue(report['sampled'])
        self.assertGreater(report['score'], 0)
        self.assertTrue(report['recommendation'])
        # a belső útvonal nem szivárog ki a kliensnek
        self.assertNotIn('stored_path', report)
        # a mentett fájl 24 kHz mono WAV, ahogy a klónlánc elvárja
        import soundfile as sf

        stored = self.client.get('/api/books/1/characters').get_json()[0]
        self.assertTrue(stored['ref_audio_path'].endswith('.wav'))
        info = sf.info(stored['ref_audio_path'])
        self.assertEqual(info.samplerate, self.SR)
        self.assertEqual(info.channels, 1)

    def test_narrator_sample_upload_returns_report(self):
        response = self._upload(
            '/api/books/1/narrator-ref-audio',
            self._speech_file(name='narrator.mp3'), 'narrator.mp3',
            sample='1', clean='1')
        self.assertEqual(response.status_code, 200, response.get_json())
        report = response.get_json()['sample']
        self.assertTrue(report['sampled'])
        # nem-WAV is feldolgozható, mert a mintavételező maga dekódol
        with database.get_conn() as conn:
            row = conn.execute(
                "SELECT narrator_ref_audio_path FROM books WHERE id=1").fetchone()
        self.assertTrue(row['narrator_ref_audio_path'].endswith('.wav'))

    def test_plain_upload_still_requires_wav(self):
        # Mintavétel nélkül továbbra is csak WAV fogadható el.
        bogus = os.path.join(self.tmp.name, 'x.mp3')
        with open(bogus, 'wb') as handle:
            handle.write(b'ID3notawave')
        response = self._upload('/api/characters/2/ref-audio', bogus, 'x.mp3')
        self.assertEqual(response.status_code, 400)
        self.assertIn('WAV', response.get_json()['error'])

    def test_voice_studio_offers_sampling_controls(self):
        response = self.client.get('/voice-studio/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="ref-sample-narrator"', response.data)
        self.assertIn(b'sampleNarratorRef(this)', response.data)
        self.assertIn(b'transcribeNarratorRef(this)', response.data)

    def test_settings_page_offers_autostart_and_backend_choice(self):
        response = self.client.get('/settings')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="qa-autostart"', response.data)
        self.assertIn(b'id="asr-backend"', response.data)

    def test_qa_page_offers_all_four_backends_and_autostart_script(self):
        response = self.client.get('/qa/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="qa-asr-backend"', response.data)
        for value in (b'auto', b'hybrid', b'parakeet', b'whisper'):
            self.assertIn(b'value="' + value + b'"', response.data)
        script = self.client.get('/static/js/qa.js')
        self.assertEqual(script.status_code, 200)
        self.assertIn(b'maybeAutoStart', script.data)
        self.assertIn(b'qa_autostart', script.data)


if __name__ == '__main__':
    unittest.main()
