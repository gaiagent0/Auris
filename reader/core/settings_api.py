"""Settings routes: engine and model setup, language models, Audiobookshelf, spaCy."""

import os
import sys
import re
from flask import jsonify, request
from core.database import get_conn
from core import security
from core import characters as char_module
from core import llm_characters
from core import settings as app_settings
from flask import Blueprint

import app as application

bp = Blueprint('settings_api', __name__)


@bp.route('/api/settings/abs-libraries', methods=['POST'])
def audiobookshelf_libraries():
    from core import publish
    body = request.get_json(silent=True) or {}
    url = str(body.get('url') or app_settings.get('abs_url', '')).strip()
    token = security.resolve_secret(body.get('token'), app_settings.get('abs_api_token', ''))
    try:
        return jsonify({'libraries': publish.list_libraries(url, token)})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@bp.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(security.mask_secrets(app_settings.load()))


@bp.route('/api/settings', methods=['POST'])
def save_settings():
    body = request.get_json(force=True) or {}
    previous = app_settings.load()
    allowed = {
        'tts_engine',
        'narrator_instruct', 'single_narrator_mode', 'default_speed', 'audio_format',
        'subtitle_format', 'theme', 'font_size', 'font_family', 'line_height',
        'normalize_text', 'tts_num_step', 'tts_batch_size', 'tts_coalesce_chars',
        'audio_mastering',
        'piper_voice', 'supertonic_voice', 'supertonic_steps', 'supertonic_batch',
        'supertonic_variant',
        'trim_segment_silence', 'export_room_tone',
        'narrator_credit', 'export_intro_template', 'export_outro_template',
        'abs_url', 'abs_api_token', 'abs_library_id', 'abs_folder_id', 'api_token',
        'qa_cer_warn', 'qa_cer_fail', 'qa_max_takes', 'qa_autostart', 'asr_model', 'asr_keep_loaded', 'asr_backend',
        'character_detection_mode', 'llm_provider',
        'llm_base_url', 'llm_api_key', 'llm_model',
        'openai_api_key', 'openai_model',
        'llm_timeout_sec', 'llm_max_output_tokens', 'llm_max_characters',
        'llm_batch_chars',
    }
    updates = security.drop_masked_secrets(
        {k: v for k, v in body.items() if k in allowed}
    )
    if 'tts_engine' in updates:
        from core.tts_router import ENGINE_NAMES
        engine = str(updates['tts_engine'] or 'auto').strip().lower()
        # 'auto' keeps the hardware-detected engine; an explicit name always wins.
        updates['tts_engine'] = engine if engine in (*ENGINE_NAMES, 'auto') else 'auto'
    if 'character_detection_mode' in updates:
        mode = str(updates['character_detection_mode'] or 'legacy').strip().lower()
        updates['character_detection_mode'] = (
            mode if mode in ('legacy', 'llm') else 'legacy'
        )
    if 'llm_provider' in updates:
        provider = str(updates['llm_provider'] or 'local').strip().lower()
        updates['llm_provider'] = (
            provider if provider in ('local', 'openai') else 'local'
        )
    if 'llm_base_url' in updates:
        updates['llm_base_url'] = str(updates['llm_base_url'] or '').strip().rstrip('/')
    if 'llm_model' in updates:
        updates['llm_model'] = str(updates['llm_model'] or '').strip()
    if 'openai_model' in updates:
        updates['openai_model'] = str(updates['openai_model'] or '').strip()
    for key, default, low, high in (
        ('llm_timeout_sec', 600, 15, 3600),
        ('llm_max_output_tokens', 8192, 512, 32768),
        ('llm_max_characters', 60, 1, 200),
        ('llm_batch_chars', 10000, 10000, 500000),
    ):
        if key in updates:
            try:
                updates[key] = max(low, min(int(updates[key]), high))
            except (TypeError, ValueError):
                updates[key] = default
    if 'normalize_text' in updates:
        updates['normalize_text'] = bool(updates['normalize_text'])
    if 'audio_mastering' in updates:
        updates['audio_mastering'] = bool(updates['audio_mastering'])
    from core.local_engines import PIPER_VOICES, SUPERTONIC_VARIANT_DIRS, SUPERTONIC_VOICES
    if 'piper_voice' in updates and updates['piper_voice'] not in PIPER_VOICES:
        updates['piper_voice'] = 'anna'
    if 'supertonic_voice' in updates and updates['supertonic_voice'] not in SUPERTONIC_VOICES:
        updates['supertonic_voice'] = 'F1'
    if ('supertonic_variant' in updates
            and updates['supertonic_variant'] not in SUPERTONIC_VARIANT_DIRS):
        updates['supertonic_variant'] = 'fp32'
    for key in ('narrator_credit', 'export_intro_template', 'export_outro_template',
                'abs_library_id', 'abs_folder_id', 'api_token'):
        if key in updates:
            updates[key] = str(updates[key] or '').strip()[:400]
    if 'abs_url' in updates:
        url = str(updates['abs_url'] or '').strip().rstrip('/')
        if url and not re.match(r'^https?://[^\s/]+', url):
            return jsonify({'error': 'Az Audiobookshelf címe http:// vagy https:// kezdetű legyen.'}), 400
        updates['abs_url'] = url
    for key in ('trim_segment_silence', 'export_room_tone', 'asr_keep_loaded', 'qa_autostart'):
        if key in updates:
            updates[key] = bool(updates[key])
    if 'asr_backend' in updates and updates['asr_backend'] not in ('auto', 'whisper', 'parakeet', 'hybrid'):
        return jsonify({'error': 'Ismeretlen beszédfelismerő.'}), 400
    if 'asr_model' in updates:
        model = str(updates['asr_model'] or '').strip()
        if model and not security.valid_hf_repo(model):
            return jsonify({'error': 'Érvénytelen Hugging Face repó-azonosító.'}), 400
        updates['asr_model'] = model
    for key, low, high, cast in (
        ('qa_cer_warn', 0.0, 1.0, float), ('qa_cer_fail', 0.01, 1.0, float),
        ('qa_max_takes', 0, 8, int),
        ('supertonic_steps', 4, 32, int), ('supertonic_batch', 1, 6, int),
    ):
        if key in updates:
            try:
                updates[key] = max(low, min(high, cast(updates[key])))
            except (TypeError, ValueError):
                updates.pop(key)
    if 'tts_num_step' in updates:
        try:
            step = int(updates['tts_num_step'])
        except (TypeError, ValueError):
            step = 16
        from core.tts_engine import ALLOWED_TTS_NUM_STEPS
        if step not in ALLOWED_TTS_NUM_STEPS:
            step = min(ALLOWED_TTS_NUM_STEPS, key=lambda s: abs(s - step))
        updates['tts_num_step'] = step
    if 'tts_batch_size' in updates:
        try:
            # 0 = the engine picks its default. Positive = fixed batch size.
            updates['tts_batch_size'] = max(0, min(int(updates['tts_batch_size']), 48))
        except (TypeError, ValueError):
            updates['tts_batch_size'] = 0
    if 'tts_coalesce_chars' in updates:
        # Kept in the settings schema for backward compatibility, but exact
        # spoken boundaries require this optimization to stay disabled.
        updates['tts_coalesce_chars'] = 0
    result = app_settings.save(updates)

    # Engine/model selection is applied on explicit Reload TTS. Keeping the
    # currently resident model alive makes Save Settings safe during playback.

    # Persisted segment rows short-circuit the engine cache entirely. Any
    # setting that changes synthesized audio therefore needs fresh segment
    # records, or playback would silently continue serving audio made with the
    # old settings. The engine-level cache keys still keep the distinct WAVs
    # separate; this clears only the database pointers used by playback.
    # Any setting that changes synthesized audio must clear the persisted
    # segment pointers, or playback would keep serving audio made with the
    # old settings (the engine-level cache keys still keep the WAVs apart).
    audio_affecting_keys = {
        'tts_engine',
        'tts_num_step',
        'normalize_text',
        'piper_voice', 'supertonic_voice', 'supertonic_steps', 'supertonic_variant',
    }
    if any(
        key in updates and updates[key] != previous.get(key)
        for key in audio_affecting_keys
    ):
        with get_conn() as conn:
            conn.execute('DELETE FROM tts_segments')

    if 'narrator_instruct' in updates and updates['narrator_instruct'] != previous.get('narrator_instruct'):
        with get_conn() as conn:
            conn.execute(
                'DELETE FROM tts_segments WHERE book_id IN (SELECT id FROM books WHERE narrator_instruct IS NULL)'
            )

    return jsonify({'ok': True, 'settings': result})


@bp.route('/api/settings/llm-test', methods=['POST'])
def llm_test():
    body = request.get_json(force=True) or {}
    provider = str(body.get('provider') or 'local').strip().lower()
    if provider == 'openai':
        base_url = 'https://api.openai.com/v1'
        api_key = security.resolve_secret(
            body.get('api_key'), app_settings.get('openai_api_key', '')
        )
        selected = str(
            body.get('model') or app_settings.get('openai_model', '')
        ).strip()
        if not api_key:
            return jsonify({
                'ok': False,
                'error': 'Az OpenAI-hoz API-kulcs szükséges.',
            }), 400
    else:
        provider = 'local'
        base_url = str(
            body.get('base_url') or app_settings.get('llm_base_url', '')
        ).strip()
        api_key = security.resolve_secret(
            body.get('api_key'), app_settings.get('llm_api_key', '')
        )
        selected = str(
            body.get('model') or app_settings.get('llm_model', '')
        ).strip()
    try:
        models = llm_characters.list_models(base_url, api_key=api_key)
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    return jsonify({
        'ok': True,
        'provider': provider,
        'models': models,
        'selected_available': bool(selected and selected in models),
    })


@bp.route('/api/settings/spacy-status')
def spacy_status_route():
    status = app_settings.spacy_status()
    # Only report load errors for an installed model; a missing optional
    # English model is already explained by the status itself.
    status['error'] = char_module.spacy_error() if status.get('model_installed') else ''
    return jsonify(status)


@bp.route('/api/settings/engine-install', methods=['POST'])
def engine_install():
    """Install an optional engine runtime into the Auris virtual environment."""
    import subprocess
    body = request.get_json(silent=True) or {}
    packages = {'piper': ['piper-tts==1.8.0']}
    engine = str(body.get('engine') or '')
    if engine not in packages:
        return jsonify({'ok': False, 'message': 'Ismeretlen motor.'}), 400
    from core.desktop_support import python_command
    result = subprocess.run(
        [*python_command(), '-m', 'pip', 'install', *packages[engine]],
        capture_output=True, text=True,
    )
    if result.returncode:
        return jsonify({'ok': False, 'message': (result.stderr or result.stdout)[-600:]}), 500
    return jsonify({'ok': True, 'message': 'Telepítve. Mentsd a beállítást, majd töltsd újra a beszédmotort.'})


@bp.route('/api/settings/spacy-install', methods=['POST'])
def spacy_install():
    body = request.get_json(silent=True) or {}
    language = 'hu' if str(body.get('language') or '').startswith('hu') else 'en'
    result = app_settings.install_spacy_model(language)
    if result['ok']:
        # Reset spaCy NLP so it reloads the new model
        char_module.reset_nlp_cache()
    return jsonify(result)


@bp.route('/api/settings/tts-reload', methods=['POST'])
def tts_reload():
    application.tts.reload()
    return jsonify({'ok': True})


