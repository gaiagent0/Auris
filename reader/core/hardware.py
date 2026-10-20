"""Hardware detection and automatic TTS engine recommendation.

Snapdragon X Elite (Windows ARM64) only. Picks the best local ONNX speech
engine for this machine; detection is read-only and cheap and never loads a
model. A recommendation only overrides the stored setting when the user has
not picked an engine themselves, so an explicit choice always wins.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

# Ordered best-first. Ranked by measured real-time factor, so the
# recommendation follows evidence rather than the order engines were added in.
#
# Measured on the Snapdragon X Elite, 2026-10-02, 268-character Hungarian
# paragraph, median of three runs:
#   supertonic_sherpa  RTF 0.14   (sherpa-onnx int8, fastest)
#   supertonic         RTF 0.24-0.26   (faster than real time)
#   xtts_onnx          RTF 0.58x — the only engine that CLONES a voice
#   piper              ONNX CPU, magyar hangok
#   omnivoice          needs torchaudio, which has no win_arm64 wheel — removed
# MOSS-TTS-Nano was removed: it ran, but its Hungarian was not intelligible.
# Its 0.1B size is the cause — measured 1.7x slower per character than English
# with degraded spectra, and no objective metric detects wrong pronunciation.
ENGINE_ORDER = (
    ("supertonic_sherpa", "Supertonic 3, sherpa-onnx int8 — mért RTF 0,14 (sherpa-onnx kell)"),
    ("supertonic", "Supertonic 3, ONNX CPU — mért RTF 0,25"),
    ("xtts_onnx", "XTTSv2-ONNX int8 — HANGKLÓNOZÁS, mért 0,58x (lassabb, de klónoz)"),
    ("piper", "Piper, ONNX CPU, magyar hangok"),
)


def is_arm64() -> bool:
    return platform.machine().lower() in {"arm64", "aarch64"}


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and is_arm64()


def is_x64_emulated() -> bool:
    """True when an x64 interpreter runs under emulation on ARM Windows/WSL2.

    A native ARM64 build reports platform.machine() == 'ARM64' and sizes its
    C types as 64-bit; an emulated x64 build reports 'AMD64'/'x86_64' while
    the host CPU is ARM. The pointer size is the reliable part, so fall back
    to the host CPU when the interpreter reports an x86 architecture.
    """
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        # An x64 interpreter on ARM hardware still sees an x86 architecture.
        if machine in {"amd64", "x86_64", "x64"}:
            try:
                import subprocess

                out = subprocess.run(
                    ["cmd", "/c", "echo %PROCESSOR_ARCHITECTURE%"],
                    capture_output=True, text=True, timeout=5,
                )
                host = out.stdout.upper()
                return "ARM" in host
            except Exception:
                return False
        return False
    try:
        import struct

        return struct.calcsize("P") < 8
    except Exception:
        return False


def total_ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / (1024 ** 3)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except Exception:
        return 0.0


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch_backend() -> dict | None:
    """Describe the installed torch runtime, or None when torch is absent.

    On this machine torch is always the CPU build — it only matters for the
    Whisper QA backend, never for synthesis.
    """
    if not _module_available("torch"):
        return None
    try:
        import torch

        return {"backend": "cpu", "detail": "CPU (PyTorch)", "version": torch.__version__}
    except Exception as exc:  # a broken torch must not break detection
        return {"backend": "error", "detail": str(exc)[:120], "version": "?"}


def _onnx_available() -> bool:
    if not _module_available("onnxruntime"):
        return False
    try:
        import onnxruntime as ort

        return "CPUExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _model_present(engine_name: str) -> bool:
    """Whether the engine's model files are already downloaded."""
    try:
        if engine_name == "supertonic":
            from core.local_engines import SupertonicEngine

            return SupertonicEngine().model_present()
        if engine_name == "piper":
            from core.local_engines import PiperEngine

            return PiperEngine().model_present()
        if engine_name == "supertonic_sherpa":
            from core.supertonic_sherpa_engine import SherpaSupertonicEngine

            return SherpaSupertonicEngine().model_present()
        if engine_name == "xtts_onnx":
            from core.xtts_onnx_engine import XttsOnnxEngine

            return XttsOnnxEngine().model_present()
    except Exception:
        return False
    return False


def detect() -> dict:
    """Return the machine's TTS-relevant capabilities. Read-only, no model load."""
    backend = _torch_backend()
    onnx = _onnx_available()
    ram = total_ram_gb()
    arm = is_arm64()
    emulated = is_x64_emulated()

    accelerator = "cpu"
    accelerator_detail = "Nincs gyorsító"
    if arm:
        accelerator = "arm64"
        accelerator_detail = ("ARM64 x64-emuláció alatt" if emulated
                             else "ARM64 natív (Snapdragon X, Adreno GPU/NPU)")

    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "arm64": arm,
        "x64_emulated": emulated,
        "accelerator": accelerator,
        "accelerator_detail": accelerator_detail,
        "torch": backend,
        "torchaudio": _module_available("torchaudio"),
        "onnxruntime": onnx,
        "spacy": _module_available("spacy"),
        "piper_tts": _module_available("piper"),
        "ram_gb": round(ram, 1),
        "cpu_count": os.cpu_count(),
    }


def _usable(engine_name: str, caps: dict) -> bool:
    """Whether an engine can actually run on this machine (ONNX CPU only)."""
    if engine_name in ("supertonic", "supertonic_sherpa", "piper"):
        if not caps["onnxruntime"]:
            return False
        if engine_name == "piper" and not caps["piper_tts"]:
            return False
        if engine_name == "supertonic_sherpa" and not _module_available("sherpa_onnx"):
            return False
        return True
    if engine_name == "xtts_onnx":
        # A klónozó XTTS ONNX-on fut, de a szöveg-előkészítéshez a pipeline
        # librosa-t és pypinyin-t is kér (a spacy helyett stub van).
        if not caps["onnxruntime"]:
            return False
        return all(_module_available(m) for m in ("librosa", "pypinyin"))
    return False


def available_engines(caps: dict | None = None) -> list[dict]:
    """Every engine this machine can run, best first, with the reason why."""
    caps = caps or detect()
    order = ENGINE_ORDER
    out = []
    for name, requirement in order:
        if _usable(name, caps):
            out.append({
                "engine": name,
                "requirement": requirement,
                "model_downloaded": _model_present(name),
            })
    return out


def recommend(caps: dict | None = None) -> dict:
    """Recommend the fastest engine this machine can actually run.

    Ranked by measured real-time factor, not by feature count: on a CPU-only
    machine a 4B model that needs twenty minutes for a page loses to a preset
    voice that beats real time. Model download state breaks ties only, so a
    first-run machine still gets the best engine rather than a stale default.
    """
    caps = caps or detect()
    usable = available_engines(caps)
    if not usable:
        return {
            "engine": None,
            "reason": "Nincs futtatható helyi motor: telepítsd az onnxruntime-ot "
                      "(pip install onnxruntime).",
            "hardware": caps,
        }
    chosen = usable[0]
    return {
        "engine": chosen["engine"],
        "reason": chosen["requirement"],
        "alternatives": [e["engine"] for e in usable[1:]],
        "needs_model_download": not chosen["model_downloaded"],
        "hardware": caps,
    }


def engine_available_on_this_machine(engine_name: str, caps: dict | None = None) -> bool:
    caps = caps or detect()
    return any(e["engine"] == engine_name for e in available_engines(caps))