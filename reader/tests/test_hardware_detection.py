"""Tests for hardware detection, engine recommendation and the ARM64 install path."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import unittest

INSTALLER_SPEC = importlib.util.spec_from_file_location(
    'auris_installer', Path(__file__).parents[1] / 'setup.py')
installer = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(installer)

from core import hardware  # noqa: E402
from core import settings  # noqa: E402
from core import tts_router  # noqa: E402


def caps(**overrides):
    """A detect() result for a hypothetical machine."""
    base = {
        'platform': 'Windows ARM64', 'python': '3.12.10', 'arm64': True,
        'x64_emulated': False, 'accelerator': 'arm64', 'accelerator_detail': '',
        'torch': {'backend': 'cpu', 'detail': 'CPU', 'version': '2.10.0+cpu'},
        'torchaudio': False, 'onnxruntime': True, 'spacy': False,
        'piper_tts': False, 'ram_gb': 31.6, 'cpu_count': 12,
    }
    base.update(overrides)
    return base


class Arm64DetectionTests(unittest.TestCase):
    def test_arm64_on_windows_is_not_apple_silicon(self):
        with patch.object(installer.platform, 'machine', return_value='ARM64'), \
             patch.object(installer.platform, 'system', return_value='Windows'):
            self.assertTrue(installer.is_arm64())
            self.assertFalse(installer.is_apple_silicon())
            self.assertEqual(installer.arm64_platform_tag(), 'win_arm64')

    def test_aarch64_linux_uses_its_own_tag(self):
        with patch.object(installer.platform, 'machine', return_value='aarch64'), \
             patch.object(installer.platform, 'system', return_value='Linux'):
            self.assertTrue(installer.is_arm64())
            self.assertEqual(installer.arm64_platform_tag(), 'linux_aarch64')

    def test_x86_machines_are_not_arm(self):
        with patch.object(installer.platform, 'machine', return_value='AMD64'), \
             patch.object(installer.platform, 'system', return_value='Windows'):
            self.assertFalse(installer.is_arm64())
            self.assertIsNone(installer.arm64_platform_tag())

    def test_macos_arm64_keeps_the_mps_path(self):
        with patch.object(installer.platform, 'machine', return_value='arm64'), \
             patch.object(installer.platform, 'system', return_value='Darwin'):
            self.assertFalse(installer.is_arm64())
            self.assertTrue(installer.is_apple_silicon())

    def test_arm64_is_detected_before_any_gpu_probe(self):
        """An emulated x64 environment must not pick a nonexistent CUDA build."""
        def explode(*_args, **_kwargs):
            raise AssertionError('CUDA detection must not run on ARM64')

        with patch.object(installer.platform, 'machine', return_value='ARM64'), \
             patch.object(installer.platform, 'system', return_value='Windows'), \
             patch.object(installer, 'detect_cuda_version', side_effect=explode):
            self.assertEqual(installer.detect_hardware(), 'arm64_onnx')


class Arm64RequirementsTests(unittest.TestCase):
    def test_windows_arm64_drops_torchaudio_and_soxr(self):
        reqs = installer.arm64_requirements('win_arm64')
        self.assertNotIn('torchaudio>=2.4,<2.12', reqs)
        self.assertNotIn('soxr', reqs)
        self.assertIn('scipy', reqs, 'SciPy is the portable resample fallback')

    def test_aarch64_linux_keeps_torchaudio_but_pins_thinc(self):
        reqs = installer.arm64_requirements('linux_aarch64')
        self.assertIn('torchaudio>=2.4,<2.12', reqs)
        # thinc 9.1 dropped its aarch64 wheels, so spaCy needs the 9.0 series.
        self.assertIn('thinc<9.1', reqs)

    def test_omnivoice_is_absent_from_the_arm64_dependency_set(self):
        self.assertNotIn('omnivoice>=0.2.1', installer.ARM64_BASE_DEPS)


class Arm64InstallPathTests(unittest.TestCase):
    def test_install_torch_skips_torchaudio_on_windows_arm(self):
        calls = []
        with patch.object(installer, 'arm64_platform_tag', return_value='win_arm64'), \
             patch.object(installer, 'offline_wheels_available', return_value=False), \
             patch.object(installer, 'pip_install',
                          side_effect=lambda *a, **k: calls.append((a, k))):
            installer.install_torch('arm64_onnx')
        specs = ' '.join(' '.join(a) for a, _k in calls)
        kwargs = calls[0][1]
        self.assertIn('torch', specs)
        self.assertNotIn('torchaudio', specs)
        self.assertEqual(kwargs.get('index_url'), 'https://download.pytorch.org/whl/cpu')

    def test_aarch64_linux_installs_the_torchaudio_pair(self):
        installed = []
        with patch.object(installer, 'arm64_platform_tag', return_value='linux_aarch64'), \
             patch.object(installer, 'pip_install', side_effect=lambda *a, **k: installed.extend(a)):
            installer.install_torch('arm64_onnx')
        self.assertIn('torchaudio>=2.4,<2.12', installed)

    def test_runtime_check_imports_onnxruntime_not_torchaudio(self):
        with patch.object(installer, 'arm64_platform_tag', return_value='win_arm64'), \
             patch.object(installer, 'run') as run:
            installer.verify_torch('arm64_onnx')
        code = run.call_args.args[0][-1]
        self.assertIn('onnxruntime', code)
        self.assertNotIn('import torchaudio', code)

    def test_reader_deps_do_not_pull_requirements_txt_on_arm(self):
        installed = []
        with patch.object(installer, 'arm64_platform_tag', return_value='win_arm64'), \
             patch.object(installer, 'pip_install', side_effect=lambda *a, **k: installed.extend(a)):
            installer.install_reader_deps('arm64_onnx')
        flat = ' '.join(installed)
        self.assertNotIn('requirements.txt', flat)
        self.assertIn('onnxruntime', flat)

    def test_reader_deps_still_use_requirements_txt_on_cuda(self):
        installed = []
        with patch.object(installer, 'pip_install', side_effect=lambda *a, **k: installed.extend(a)):
            installer.install_reader_deps('cu128')
        self.assertIn('-r', installed[0])


class ResampleFallbackTests(unittest.TestCase):
    def test_resample_falls_back_to_scipy_when_soxr_is_missing(self):
        import numpy as np
        from core import local_engines

        audio = np.sin(np.linspace(0, 200 * np.pi, 4000)).astype(np.float32)
        real_import = __builtins__['__import__'] if isinstance(__builtins__, dict) \
            else __builtins__.__import__

        def no_soxr(name, *args, **kwargs):
            if name == 'soxr':
                raise ImportError('no soxr on win_arm64')
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=no_soxr):
            out = local_engines.resample(audio, 2000, 24000)
        self.assertEqual(out.dtype, np.float32)
        # 2000 Hz for 4000 samples is 2 s; at 24 kHz that is ~48000 samples.
        self.assertAlmostEqual(len(out) / 24000, 2.0, delta=0.05)


class EngineAvailabilityTests(unittest.TestCase):
    def test_arm64_recommends_the_measured_fastest_engine(self):
        result = hardware.recommend(caps())
        self.assertEqual(result['engine'], 'supertonic')

    def test_4b_engines_are_refused_without_a_gpu(self):
        names = [e['engine'] for e in hardware.available_engines(caps())]
        self.assertNotIn('higgs', names)
        self.assertNotIn('moss_tts', names)

    def test_omnivoice_needs_torchaudio_to_import(self):
        without = [e['engine'] for e in hardware.available_engines(caps(torchaudio=False))]
        self.assertNotIn('omnivoice', without)
        with_audio = [e['engine'] for e in hardware.available_engines(caps(torchaudio=True))]
        self.assertIn('omnivoice', with_audio)

    def test_cuda_machine_prefers_the_gpu_engines(self):
        gpu = caps(accelerator='cuda', arm64=False, ram_gb=48.0,
                   torch={'backend': 'cuda', 'detail': 'NVIDIA', 'version': '2.11.0+cu128'},
                   torchaudio=True)
        self.assertEqual(hardware.recommend(gpu)['engine'], 'higgs')

    def test_insufficient_ram_blocks_the_4b_engines_even_on_cuda(self):
        small = caps(accelerator='cuda', arm64=False, ram_gb=16.0,
                     torch={'backend': 'cuda', 'detail': 'NVIDIA', 'version': '2.11.0+cu128'},
                     torchaudio=True)
        names = [e['engine'] for e in hardware.available_engines(small)]
        self.assertNotIn('higgs', names)

    def test_piper_needs_its_separate_gpl_package(self):
        caps_no_piper = caps()
        self.assertNotIn('piper', [e['engine'] for e in hardware.available_engines(caps_no_piper)])
        caps_piper = caps(piper_tts=True)
        self.assertIn('piper', [e['engine'] for e in hardware.available_engines(caps_piper)])

    def test_no_onnxruntime_leaves_nothing_runnable(self):
        empty = caps(onnxruntime=False, torchaudio=False)
        self.assertIsNone(hardware.recommend(empty)['engine'])


class RouterSelectionTests(unittest.TestCase):
    """A választott motor olvasása a beállításfájlból történik.

    Ezért a teszteket `core.settings.load` rettekítésével futtatjuk: a
    DEFAULTS.patch önmagában nem elég, mert egy korábban mentett
    tts_engine felülírja, vagyis a teszt a gép állapotától függne.
    """

    @staticmethod
    def _stored(**overrides) -> dict:
        return {**settings.DEFAULTS, **overrides}

    def test_auto_follows_the_hardware_recommendation(self):
        with patch('core.settings.load', return_value=self._stored(tts_engine='auto')), \
             patch.object(tts_router, 'recommended_engine_name', return_value='supertonic'):
            self.assertEqual(tts_router.selected_engine_name(), 'supertonic')

    def test_an_explicit_choice_always_wins(self):
        for chosen in tts_router.ENGINE_NAMES:
            with patch('core.settings.load', return_value=self._stored(tts_engine=chosen)), \
                 patch.object(tts_router, 'recommended_engine_name', return_value='supertonic'):
                self.assertEqual(tts_router.selected_engine_name(), chosen)

    def test_an_unknown_stored_value_falls_back_to_the_recommendation(self):
        with patch('core.settings.load', return_value=self._stored(tts_engine='nonexistent')), \
             patch.object(tts_router, 'recommended_engine_name', return_value='piper'):
            self.assertEqual(tts_router.selected_engine_name(), 'piper')


if __name__ == '__main__':
    unittest.main()