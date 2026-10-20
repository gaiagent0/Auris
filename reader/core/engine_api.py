"""Engine suggestion and switching routes.

Lets the UI ask which TTS engine this machine can actually run (so the
voice-cloning XTTS only shows up when its dependencies are present) and
switch the stored engine with an explicit, validated choice.
"""

import importlib.util

from flask import Blueprint, jsonify, request

from core import hardware
from core import settings as app_settings
from core.tts_router import ENGINE_NAMES

import app as application

bp = Blueprint('engine_api', __name__)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Actionable Hungarian message per engine when it cannot run here.
def _missing_deps(engine: str) -> list[str]:
    missing = []
    if engine == 'xtts_onnx':
        for module, package in (('onnxruntime', 'onnxruntime'),
                                ('librosa', 'librosa'),
                                ('pypinyin', 'pypinyin')):
            if not _module_available(module):
                missing.append(package)
    elif engine == 'supertonic_sherpa':
        if not _module_available('sherpa_onnx'):
            missing.append('sherpa-onnx')
    return missing


def _switch_error(engine: str) -> str | None:
    """Hungarian error for switching to `engine`, or None when it is allowed."""
    if engine not in ENGINE_NAMES:
        return 'Ismeretlen beszédmotor.'
    caps = hardware.detect()
    if not hardware.engine_available_on_this_machine(engine, caps):
        missing = _missing_deps(engine)
        if missing:
            return ('Ehhez a motorhoz hiányzó összetevők: '
                    + ', '.join(missing) + '. Telepítsd őket, majd próbáld újra.')
        return 'Ez a motor nem futtatható ezen a gépen.'
    return None


@bp.route('/api/engine/suggest')
def suggest():
    """Best engine for this machine plus a per-request availability check.

    With ?engine=<name> the answer also says whether that specific engine can
    run here (so the UI can hide the cloning engine when its deps are missing).
    """
    caps = hardware.detect()
    recommendation = hardware.recommend(caps)
    wanted = str(request.args.get('engine') or '').strip().lower()
    payload = {
        'recommended': recommendation.get('engine'),
        'reason': recommendation.get('reason', ''),
        'available': [e['engine'] for e in hardware.available_engines(caps)],
        'stored': str(app_settings.get('tts_engine', 'auto') or 'auto'),
    }
    if wanted:
        payload['engine'] = wanted
        payload['engine_available'] = _switch_error(wanted) is None
        if not payload['engine_available']:
            payload['engine_error'] = _switch_error(wanted)
    return jsonify(payload)


@bp.route('/api/engine/switch', methods=['POST'])
def switch():
    """Persist an explicit engine choice and reload the TTS router."""
    body = request.get_json(silent=True) or {}
    engine = str(body.get('engine') or '').strip().lower()
    error = _switch_error(engine)
    if error:
        return jsonify({'ok': False, 'error': error}), 400
    app_settings.save({'tts_engine': engine})
    application.tts.reload()
    return jsonify({'ok': True, 'engine': engine})
