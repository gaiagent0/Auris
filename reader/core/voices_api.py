"""Character and narrator voices: listing, editing, previews and reference audio."""

import os
from flask import Blueprint, jsonify, request
from core.database import get_conn

import app as application

bp = Blueprint('voices', __name__)


@bp.route('/api/books/<int:book_id>/characters')
def list_characters(book_id):
    chapter_id = request.args.get('chapter_id', type=int)
    chapter_character_names = None
    if chapter_id is not None:
        with get_conn() as conn:
            chapter = conn.execute(
                'SELECT id FROM chapters WHERE id=? AND book_id=?',
                (chapter_id, book_id),
            ).fetchone()
        if not chapter:
            return jsonify({'error': 'A fejezet nem található'}), 404
        chapter_character_names = {
            str(segment['character_name']).casefold()
            for segment in application._compute_segments_for_chapter(
                book_id,
                chapter_id,
                single_narrator_mode=False,
            )
            if segment['character_name']
        }

    with get_conn() as conn:
        rows = conn.execute(
            'SELECT * FROM characters WHERE book_id=? ORDER BY frequency DESC',
            (book_id,)
        ).fetchall()
        # Rule-based attribution (no language model) lives in the segments,
        # not in speaker_annotations; count whichever is larger.
        segment_lines = {
            row[0]: row[1] for row in conn.execute(
                'SELECT character_name, COUNT(*) FROM tts_segments '
                'WHERE book_id=? AND character_name IS NOT NULL GROUP BY character_name',
                (book_id,))
        }
    characters = [dict(r) for r in rows]
    for character in characters:
        character['line_count'] = max(int(character.get('frequency') or 0),
                                      int(segment_lines.get(character['name'], 0)))
    characters.sort(key=lambda c: -c['line_count'])
    if chapter_character_names is not None:
        characters = [
            character
            for character in characters
            if str(character['name']).casefold() in chapter_character_names
        ]
    return jsonify(characters)


@bp.route('/api/books/<int:book_id>/characters/<int:char_id>', methods=['PUT'])
def update_character(book_id, char_id):
    body = request.get_json(force=True)
    allowed = {'instruct', 'gender', 'color_hex', 'ref_text'}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        return jsonify({'error': 'Nincs mit módosítani'}), 400
    set_clause = ', '.join(f'{k}=?' for k in updates)
    with get_conn() as conn:
        prev = conn.execute(
            'SELECT ref_audio_path, ref_text FROM characters WHERE id=? AND book_id=?',
            (char_id, book_id),
        ).fetchone()
        conn.execute(
            f'UPDATE characters SET {set_clause} WHERE id=? AND book_id=?',
            (*updates.values(), char_id, book_id)
        )
    if prev and prev['ref_audio_path'] and 'ref_text' in updates:
        application.tts.invalidate_voice_prompt(prev['ref_audio_path'], prev['ref_text'])
        application.tts.invalidate_voice_prompt(prev['ref_audio_path'], updates.get('ref_text'))
    application._clear_book_tts_segments(book_id)
    return jsonify({'ok': True, 'segments_cleared': True})


@bp.route('/api/books/<int:book_id>/characters/<int:char_id>/preview', methods=['POST'])
def preview_character(book_id, char_id):
    body = request.get_json(silent=True) or {}
    with get_conn() as conn:
        row = conn.execute('SELECT * FROM characters WHERE id=? AND book_id=?',
                           (char_id, book_id)).fetchone()
    if not row:
        return jsonify({'error': 'Nem található'}), 404

    status = application.tts.status()
    if status['state'] != 'ready':
        return jsonify({'error': 'A beszédmotor még nem áll készen', 'status': status}), 503

    instruct = (body.get('instruct') or row['instruct'] or '').strip()
    ref_audio = row['ref_audio_path'] if row['ref_audio_path'] else None
    requested_ref_text = body.get('ref_text', row['ref_text'])
    ref_text = requested_ref_text.strip() if ref_audio and isinstance(requested_ref_text, str) and requested_ref_text.strip() else None
    book = application._load_book(book_id)
    language = book['language'] or 'hu'
    sample_text = str(body.get('text') or (
        'A délutáni fényben csendesen lapoztam a könyvet. Új történet kezdődik.'
        if language == 'hu' else application.VOICE_PREVIEW_TEXT
    )).strip()[:1500]

    try:
        result = application.tts.generate_preview(
            instruct=instruct,
            sample_text=sample_text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return jsonify({'audio_url': f'/api/audio/{result["cache_key"]}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/books/<int:book_id>/narrator', methods=['GET'])
def get_narrator(book_id):
    book = application._load_book(book_id)
    if not book:
        return jsonify({'error': 'Nem található'}), 404
    book_data = dict(book)
    return jsonify({
        'instruct': application._book_narrator_instruct(book_data),
        'single_narrator_mode': application._book_single_narrator_mode(book_data),
        'ref_audio_name': book_data.get('narrator_ref_audio_name'),
        'ref_text': book_data.get('narrator_ref_text') or '',
    })


@bp.route('/api/books/<int:book_id>/narrator', methods=['PUT'])
def update_narrator(book_id):
    body = request.get_json(force=True) or {}
    book = application._load_book(book_id)
    if not book:
        return jsonify({'error': 'Nem található'}), 404
    book_data = dict(book)

    raw_instruct = body.get('instruct')
    instruct = (
        raw_instruct.strip()
        if isinstance(raw_instruct, str)
        else application._book_narrator_instruct(book_data)
    )
    if not instruct:
        return jsonify({'error': 'A narrátor hangleírása kötelező'}), 400

    raw_mode = body.get('single_narrator_mode', application._book_single_narrator_mode(book_data))
    if isinstance(raw_mode, str):
        single_narrator_mode = raw_mode.strip().lower() in {'1', 'true', 'yes', 'on'}
    else:
        single_narrator_mode = bool(raw_mode)
    narrator_changed = instruct != application._book_narrator_instruct(book_data)
    mode_changed = single_narrator_mode != application._book_single_narrator_mode(book_data)
    raw_ref_text = body.get('ref_text', book_data.get('narrator_ref_text') or '')
    ref_text = raw_ref_text.strip() if isinstance(raw_ref_text, str) else ''
    ref_text_changed = ref_text != (book_data.get('narrator_ref_text') or '')

    with get_conn() as conn:
        if ref_text_changed and book_data.get('narrator_ref_audio_path'):
            application.tts.invalidate_voice_prompt(
                book_data['narrator_ref_audio_path'],
                book_data.get('narrator_ref_text'),
            )
            application.tts.invalidate_voice_prompt(
                book_data['narrator_ref_audio_path'],
                ref_text or None,
            )
        conn.execute(
            'UPDATE books SET narrator_instruct=?, single_narrator_mode=?, '
            'narrator_ref_text=? WHERE id=?',
            (instruct, int(single_narrator_mode), ref_text, book_id)
        )

    if narrator_changed or mode_changed or ref_text_changed:
        application._clear_book_tts_segments(book_id)

    return jsonify({
        'ok': True,
        'instruct': instruct,
        'single_narrator_mode': single_narrator_mode,
        'ref_text': ref_text,
        'segments_cleared': narrator_changed or mode_changed or ref_text_changed,
    })


@bp.route('/api/books/<int:book_id>/characters/narrator/preview', methods=['POST'])
def preview_narrator(book_id):
    body = request.get_json(silent=True) or {}
    book = application._load_book(book_id)
    if not book:
        return jsonify({'error': 'Nem található'}), 404

    status = application.tts.status()
    if status['state'] != 'ready':
        return jsonify({'error': 'A beszédmotor még nem áll készen', 'status': status}), 503

    instruct = (body.get('instruct') or application._book_narrator_instruct(dict(book))).strip()
    narrator_ref, saved_ref_text = application._book_narrator_reference(book_id)
    requested_ref_text = body.get('ref_text', saved_ref_text)
    narrator_ref_text = requested_ref_text.strip() if narrator_ref and isinstance(requested_ref_text, str) and requested_ref_text.strip() else None
    try:
        result = application.tts.generate_preview(
            instruct=instruct,
            sample_text=str(body.get('text') or (
                'A délutáni fényben csendesen lapoztam a könyvet. Új történet kezdődik.'
                if (book['language'] or 'hu') == 'hu' else application.VOICE_PREVIEW_TEXT
            )).strip()[:1500],
            language=book['language'] or 'hu',
            ref_audio=narrator_ref,
            ref_text=narrator_ref_text,
        )
        return jsonify({'audio_url': f'/api/audio/{result["cache_key"]}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/characters/<int:char_id>/ref-audio', methods=['POST'])
def upload_ref_audio(char_id):
    if 'file' not in request.files:
        return jsonify({'error': 'Nincs kiválasztott fájl'}), 400
    f = request.files['file']
    name = (f.filename or '').lower()
    # sample=1: a mintavételező maga dekódol, így bármilyen hangformátum
    # elfogadható; a kimenet így is 24 kHz mono WAV lesz.
    sample = request.form.get('sample') == '1'
    if not name:
        return jsonify({'error': 'Nincs kiválasztott fájl'}), 400
    if not sample and not name.endswith('.wav'):
        return jsonify({'error': 'A referenciahang WAV-fájl legyen'}), 400
    ref_text = (request.form.get('ref_text') or '').strip()
    with get_conn() as conn:
        row = conn.execute(
            'SELECT book_id, ref_audio_path, ref_text FROM characters WHERE id=?',
            (char_id,),
        ).fetchone()
        if not row:
            return jsonify({'error': 'Nem található'}), 404
        if row['ref_audio_path']:
            application.tts.invalidate_voice_prompt(row['ref_audio_path'], row['ref_text'])
    if sample:
        path, report = application._sample_reference_upload(
            f, f'ref_{char_id}', clean=request.form.get('clean') == '1')
        if not path:
            return jsonify({'error': report.get('error') or
                            'Nem találtam megfelelő beszédszakaszt a felvételen',
                            'sample': report}), 422
    else:
        path = application._save_reference_upload(
            f, f'ref_{char_id}', clean=request.form.get('clean') == '1')
        report = None
    with get_conn() as conn:
        conn.execute(
            'UPDATE characters SET ref_audio_path=?, ref_audio_name=?, ref_text=? WHERE id=?',
            (path, os.path.basename(f.filename), ref_text, char_id),
        )
    application._delete_replaced_reference(row['ref_audio_path'], path, f'ref_{char_id}')
    application._clear_book_tts_segments(row['book_id'])
    result = {
        'ok': True,
        'ref_audio_name': os.path.basename(f.filename),
        'ref_text': ref_text,
    }
    if report is not None:
        # a jelentésből a mentett útvonal nem kell a kliensnek
        result['sample'] = {k: v for k, v in report.items() if k != 'stored_path'}
    return jsonify(result)


@bp.route('/api/characters/<int:char_id>/ref-audio', methods=['DELETE'])
def delete_ref_audio(char_id):
    with get_conn() as conn:
        row = conn.execute(
            'SELECT book_id, ref_audio_path, ref_text FROM characters WHERE id=?',
            (char_id,),
        ).fetchone()
        if not row:
            return jsonify({'error': 'Nem található'}), 404
        conn.execute(
            'UPDATE characters SET ref_audio_path=NULL, ref_audio_name=NULL, ref_text=NULL '
            'WHERE id=?', (char_id,)
        )
    if row['ref_audio_path']:
        application.tts.invalidate_voice_prompt(row['ref_audio_path'], row['ref_text'])
    application._delete_file_if_exists(row['ref_audio_path'])
    application._clear_book_tts_segments(row['book_id'])
    return jsonify({'ok': True, 'segments_cleared': True})


@bp.route('/api/books/<int:book_id>/narrator-ref-audio', methods=['POST'])
def upload_narrator_ref_audio(book_id):
    if 'file' not in request.files:
        return jsonify({'error': 'Nincs kiválasztott fájl'}), 400
    f = request.files['file']
    name = (f.filename or '').lower()
    # sample=1: a mintavételező maga dekódol, így bármilyen hangformátum
    # elfogadható; a kimenet így is 24 kHz mono WAV lesz.
    sample = request.form.get('sample') == '1'
    if not name:
        return jsonify({'error': 'Nincs kiválasztott fájl'}), 400
    if not sample and not name.endswith('.wav'):
        return jsonify({'error': 'A referenciahang WAV-fájl legyen'}), 400
    ref_text = (request.form.get('ref_text') or '').strip()
    with get_conn() as conn:
        prev = conn.execute(
            'SELECT narrator_ref_audio_path, narrator_ref_text FROM books WHERE id=?',
            (book_id,),
        ).fetchone()
        if not prev:
            return jsonify({'error': 'Nem található'}), 404
        if prev['narrator_ref_audio_path']:
            application.tts.invalidate_voice_prompt(
                prev['narrator_ref_audio_path'], prev['narrator_ref_text']
            )
    if sample:
        path, report = application._sample_reference_upload(
            f, f'narrator_ref_{book_id}', clean=request.form.get('clean') == '1')
        if not path:
            return jsonify({'error': report.get('error') or
                            'Nem találtam megfelelő beszédszakaszt a felvételen',
                            'sample': report}), 422
    else:
        path = application._save_reference_upload(
            f, f'narrator_ref_{book_id}', clean=request.form.get('clean') == '1')
        report = None
    with get_conn() as conn:
        conn.execute(
            'UPDATE books SET narrator_ref_audio_path=?, narrator_ref_audio_name=?, '
            'narrator_ref_text=? WHERE id=?',
            (path, os.path.basename(f.filename), ref_text, book_id),
        )
    application._delete_replaced_reference(
        prev['narrator_ref_audio_path'], path, f'narrator_ref_{book_id}'
    )
    application.tts.invalidate_voice_prompt(path, ref_text or None)
    application._clear_book_tts_segments(book_id)
    result = {
        'ok': True,
        'ref_audio_name': os.path.basename(f.filename),
        'ref_text': ref_text,
    }
    if report is not None:
        result['sample'] = {k: v for k, v in report.items() if k != 'stored_path'}
    return jsonify(result)


@bp.route('/api/books/<int:book_id>/narrator-ref-sample', methods=['POST'])
def sample_narrator_ref_audio(book_id):
    """A már aktív narrátorhangból kiválasztja a legjobb klónozási szakaszt.

    Nem új fájlt vár, hanem a könyv jelenlegi ``narrator_ref_audio_path``
    fájlját mintavételezi a ``core.reference_sampler``-ral: objektív szempontok
    szerint (beszédsűrűség, klip, jel-zaj arány, hangerő-stabilitás) megkeresi a
    legjobb 5–15 másodperces ablakot, kitisztítja és menti — ugyanazzal a
    tisztítólánccal és content-addressed névvel, mint a feltöltési út, hogy a
    hangklón-gyorsítótár ne adja vissza a régi hangot.
    """
    import hashlib
    import tempfile
    from core import reference_sampler

    with get_conn() as conn:
        row = conn.execute(
            'SELECT narrator_ref_audio_path, narrator_ref_audio_name, narrator_ref_text '
            'FROM books WHERE id=?',
            (book_id,),
        ).fetchone()
    if not row:
        return jsonify({'error': 'Nem található'}), 404
    source = row['narrator_ref_audio_path']
    if not source or not os.path.isfile(source):
        return jsonify({'error': 'Előbb tölts fel egy narrátorhangot.'}), 422
    if os.path.getsize(source) == 0:
        return jsonify({'error': 'A narrátorhang üres.'}), 422

    raw = None
    try:
        suffix = os.path.splitext(source)[1].lower() or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False,
                                         dir=application.UPLOAD_DIR,
                                         prefix='.narrator_ref.resample.') as handle:
            raw = handle.name
        with open(source, 'rb') as src, open(raw, 'wb') as dst:
            for chunk in iter(lambda: src.read(1 << 20), b''):
                dst.write(chunk)
        picked = raw + '.ref.wav'
        report = reference_sampler.sample_reference(raw, picked)
        if not report.get('sampled'):
            return jsonify({'error': report.get('error') or
                            'Nem találtam megfelelő beszédszakaszt a hanganyagban',
                            'sample': report}), 422
        digest = hashlib.sha256()
        with open(picked, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        path = os.path.join(application.UPLOAD_DIR,
                            f'narrator_ref_{digest.hexdigest()[:16]}.wav')
        os.replace(picked, path)
    except Exception as exc:
        return jsonify({'error': f'A mintavétel nem sikerült: {exc}'}), 500
    finally:
        for leftover in (raw, (raw + '.ref.wav') if raw else None):
            if leftover:
                try:
                    os.remove(leftover)
                except OSError:
                    pass

    ref_text = row['narrator_ref_text']
    application.tts.invalidate_voice_prompt(source, ref_text)
    application.tts.invalidate_voice_prompt(path, ref_text)
    with get_conn() as conn:
        conn.execute(
            'UPDATE books SET narrator_ref_audio_path=?, narrator_ref_audio_name=? '
            'WHERE id=?',
            (path, os.path.basename(source), book_id),
        )
    application._delete_replaced_reference(source, path, 'narrator_ref')
    application._clear_book_tts_segments(book_id)
    return jsonify({
        'ok': True,
        'ref_audio_name': os.path.basename(source),
        'ref_text': ref_text or '',
        'sample': {k: v for k, v in report.items() if k != 'stored_path'},
    })


@bp.route('/api/books/<int:book_id>/narrator-ref-transcribe', methods=['POST'])
def transcribe_narrator_ref_audio(book_id):
    """A aktív narrátorreferenciáról beszédfelismeréssel megírja az átiratot.

    Nem ment: a felismert szöveget a kliens a szövegdobozba tölti, a mentés a
    meglévő ``PUT /narrator`` úton történik. A modell a minőségellenőrzés
    ``Transcriber``-ja (Snapdragonon Parakeet TDT v3, ONNX, CPU); az első
    hívás betöltést igényelhet.
    """
    with get_conn() as conn:
        row = conn.execute(
            'SELECT narrator_ref_audio_path FROM books WHERE id=?', (book_id,)
        ).fetchone()
    if not row:
        return jsonify({'error': 'Nem található'}), 404
    path = row['narrator_ref_audio_path']
    if not path or not os.path.isfile(path):
        return jsonify({'error': 'Előbb tölts fel egy narrátorhangot.'}), 422

    try:
        from core.qa import Transcriber

        transcriber = Transcriber.for_language('hu')
        result = transcriber.transcribe(path, 'hu')
    except Exception as exc:
        return jsonify({'error': f'A beszédfelismerés nem sikerült: {exc}'}), 500
    text = str((result or {}).get('text') or '').strip()
    if not text:
        return jsonify({'error': 'A felismerő nem hallott ki érthető szöveget a felvételből.'}), 422
    return jsonify({'ok': True, 'text': text,
                    'model': getattr(transcriber, 'model_id', '')})


@bp.route('/api/books/<int:book_id>/narrator-ref-audio', methods=['DELETE'])
def delete_narrator_ref_audio(book_id):
    with get_conn() as conn:
        row = conn.execute(
            'SELECT narrator_ref_audio_path, narrator_ref_text FROM books WHERE id=?',
            (book_id,),
        ).fetchone()
        if not row:
            return jsonify({'error': 'Nem található'}), 404
        path = row['narrator_ref_audio_path']
        ref_text = row['narrator_ref_text']
        conn.execute(
            'UPDATE books SET narrator_ref_audio_path=NULL, narrator_ref_audio_name=NULL, '
            'narrator_ref_text=NULL WHERE id=?', (book_id,)
        )

    if path:
        application.tts.invalidate_voice_prompt(path, ref_text)
    application._delete_file_if_exists(path)
    application._clear_book_tts_segments(book_id)
    return jsonify({'ok': True, 'segments_cleared': True})
