"""Runtime selector that keeps every TTS engine lifecycle separate.

Snapdragon X Elite (Windows ARM64) only: every selectable engine runs on the
CPU through ONNX Runtime. CUDA/torch engines (OmniVoice, Higgs, MOSS) were
removed; an explicit user choice can only name an engine from ENGINE_NAMES.
"""

from __future__ import annotations

import threading
import time

ENGINE_NAMES = ("supertonic_sherpa", "supertonic", "xtts_onnx", "piper")


def selected_engine_name() -> str:
    try:
        from core.settings import get

        value = str(get("tts_engine", "auto") or "auto").lower()
    except Exception:
        value = "auto"

    if value == "auto":
        # No explicit choice yet: ask the hardware what it can actually run.
        # An engine the user picked always wins, even on unsupported hardware.
        return recommended_engine_name()

    return value if value in ENGINE_NAMES else recommended_engine_name()


def recommended_engine_name() -> str:
    """Best engine for this machine, falling back to the fastest measured one."""
    try:
        from core.hardware import recommend

        name = recommend().get("engine")
    except Exception:
        return "supertonic_sherpa"
    return name if name in ENGINE_NAMES else "supertonic_sherpa"


class TTSEngineRouter:
    def __init__(self):
        self._lock = threading.RLock()
        self._engine = self._create(selected_engine_name())

    @staticmethod
    def _create(name: str):
        if name == "supertonic_sherpa":
            from core.supertonic_sherpa_engine import SherpaSupertonicEngine

            return SherpaSupertonicEngine()
        if name == "xtts_onnx":
            from core.xtts_onnx_engine import XttsOnnxEngine

            return XttsOnnxEngine()
        from core.local_engines import ENGINE_CLASSES

        # "supertonic" and "piper" (and any future ONNX engine) live here.
        return ENGINE_CLASSES[name]()

    @property
    def engine_name(self) -> str:
        return self._engine.engine_name

    def _select_if_needed(self) -> None:
        wanted = selected_engine_name()
        with self._lock:
            if wanted == self.engine_name:
                return
            self._engine.unload()
            self._engine = self._create(wanted)

    def reload(self) -> None:
        wanted = selected_engine_name()
        with self._lock:
            if wanted != self.engine_name:
                self._engine.unload()
                self._engine = self._create(wanted)
                self._engine.load_async()
            else:
                self._engine.reload()

    def load_async(self) -> None:
        self._select_if_needed()
        self._engine.load_async()

    def unload(self) -> None:
        """Stop the active engine and release its memory before local LLM work."""
        with self._lock:
            cancel = getattr(self._engine, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            self._engine.unload()

    def wait_until_unloaded(self, timeout: float = 600) -> bool:
        """Wait for an in-flight load to notice cancellation and release memory."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            with self._lock:
                engine = self._engine
                if not getattr(engine, "_loading", False):
                    engine.unload()
                    return True
            time.sleep(0.1)
        return False

    def status(self) -> dict:
        self._select_if_needed()
        status = self._engine.status()
        status.setdefault("engine", self.engine_name)
        if "capabilities" not in status:
            from core.local_engines import ENGINE_INFO

            status["capabilities"] = dict(ENGINE_INFO.get(self.engine_name, {}))
        return status

    @property
    def capabilities(self) -> dict:
        from core.local_engines import ENGINE_INFO

        return dict(ENGINE_INFO.get(self.engine_name, {}))

    def cancel(self) -> bool:
        cancel = getattr(self._engine, "cancel", None)
        return bool(cancel()) if callable(cancel) else False

    def __getattr__(self, name):
        return getattr(self._engine, name)
