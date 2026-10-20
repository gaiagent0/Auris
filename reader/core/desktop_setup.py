"""First-run model setup for the Snapdragon X Elite build.

Snapdragon X Elite (Windows ARM64) only: every engine runs through ONNX
Runtime on the CPU, so there is no GPU runtime to install here. The wizard
either verifies the bundled XTTSv2-ONNX cloning model or downloads
Supertonic 3 from Hugging Face.
"""
from __future__ import annotations

import fnmatch
import json
import os
import threading
from pathlib import Path


def setup_complete(root: Path) -> bool:
    try:
        return bool(json.loads((root / 'data' / 'settings.json').read_text(encoding='utf-8')).get('desktop_setup_complete'))
    except (OSError, ValueError):
        return False


def hardware() -> dict:
    """Describe this machine for the setup page. Read-only, no model load."""
    import platform

    machine = platform.machine().lower()
    arm64 = machine in {'arm64', 'aarch64'}
    return {
        'arm64': arm64,
        'snapdragon': arm64,
        'platform': f'{platform.system()} {platform.machine()}',
    }


class SetupController:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._state = {'state': 'idle', 'progress': 0, 'message': 'Válassz beszédmotort az első induláshoz.',
                       'file_bytes': 0, 'file_total': 0, 'restart_required': False}
        self._hardware = None

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _update(self, **values):
        with self._lock:
            self._state.update(values)

    def machine(self) -> dict:
        if self._hardware is None:
            self._hardware = hardware()
        return dict(self._hardware)

    def start(self, engine: str) -> None:
        if engine not in ('supertonic', 'xtts_onnx'):
            raise ValueError('Ismeretlen beszédmotor.')
        with self._lock:
            if self._state['state'] == 'running':
                raise ValueError('A kezdeti beállítás már folyamatban van.')
            self._state.update(state='running', progress=0, message='A letöltés előkészítése…',
                               file_bytes=0, file_total=0, restart_required=False)
        threading.Thread(target=self._run, args=(engine,), daemon=True).start()

    def _run(self, engine: str):
        try:
            self._download_model(engine)
            from core import settings
            values = {'tts_engine': engine, 'desktop_setup_complete': True,
                      'onboarding_complete': True}
            settings.save(values)
            self._update(state='done', progress=100, message='Az Auris használatra kész.',
                         restart_required=False)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Desktop setup failed')
            self._update(state='error', message=f'A beállítás nem fejeződött be: {exc}. Ellenőrizd az internetkapcsolatot és a szabad tárhelyet, majd próbáld újra.')

    def _download_model(self, engine: str):
        if engine == 'xtts_onnx':
            # The cloning model ships with the install; only verify it is here.
            self._update(message='A XTTSv2-ONNX klónmodell ellenőrzése…', progress=90)
            from core.xtts_onnx_engine import XttsOnnxEngine
            model = XttsOnnxEngine()
            if not model.model_present():
                raise RuntimeError(
                    'A XTTSv2-ONNX klónmodell hiányzik: a vendor/XTTSv2-Streaming-ONNX/'
                    'xtts_onnx könyvtár szükséges a hangklónozáshoz.')
            return

        from huggingface_hub import hf_hub_download, list_repo_files
        from tqdm.auto import tqdm
        from core.local_engines import SUPERTONIC_REPO, SUPERTONIC_REVISION

        repo, revision = SUPERTONIC_REPO, SUPERTONIC_REVISION
        dest = self.root / 'models' / 'supertonic-3'
        patterns = ['onnx/*', 'voice_styles/*', 'LICENSE*', 'README*']
        files = [name for name in list_repo_files(repo, revision=revision)
                 if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
        if not files:
            raise RuntimeError('A modelltároló nem tartalmaz letölthető fájlokat.')
        owner = self
        index = 0

        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs['disable'] = False
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                value = super().update(n)
                total = self.total or 0
                fraction = min(self.n, total) / total if total else 0
                owner._update(progress=10 + int(85 * (index + fraction) / len(files)),
                              file_bytes=self.n, file_total=total)
                return value

            def display(self, *args, **kwargs):
                return None

        for index, name in enumerate(files):
            self._update(message=f'Modell letöltése: {name} ({index + 1}/{len(files)})',
                         progress=10 + int(85 * index / len(files)), file_bytes=0, file_total=0)
            hf_hub_download(repo_id=repo, filename=name, revision=revision,
                            local_dir=str(dest), tqdm_class=DownloadProgress)
        # Load the selected model before reporting a successful setup.
        self._update(message='A letöltött modell ellenőrzése…', progress=96)
        from core.local_engines import SupertonicEngine
        model = SupertonicEngine()
        model._load_model()
        model._release()


def register(app, controller: SetupController, restart):
    from flask import Blueprint, jsonify, render_template, request
    bp = Blueprint('desktop', __name__)

    @bp.get('/desktop/setup')
    def setup_page():
        return render_template('desktop_setup.html', hardware=controller.machine(),
                               data_dir=str(controller.root), complete=setup_complete(controller.root))

    @bp.get('/api/desktop/status')
    def status():
        return jsonify(controller.status())

    @bp.post('/api/desktop/setup')
    def begin():
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify(error='Érvénytelen beállítás.'), 400
        try:
            controller.start(body.get('engine', 'supertonic'))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(ok=True), 202

    @bp.post('/api/desktop/restart')
    def restart_route():
        if controller.status()['state'] == 'running':
            return jsonify(error='Várd meg a folyamatban lévő beállítást.'), 409
        restart()
        return jsonify(ok=True)

    app.register_blueprint(bp)
