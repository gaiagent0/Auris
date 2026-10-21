"""
Auris installer — Snapdragon X Elite (Windows ARM64) only.

Detects the ARM64 machine, installs the optional CPU PyTorch build (Whisper
QA only) and the ONNX engine/reader dependencies. No CUDA, no GPU runtime:
every TTS engine here runs through ONNX Runtime on the CPU.

Usage:
    python setup.py

Environment:
    AURIS_OFFLINE=1          Force local-wheel-only installs.
    AURIS_USE_LOCAL_WHEELS=1 Use a local wheel directory before package indexes.
    AURIS_WHEELS_DIR=...     Override the local wheel directory path.
"""

import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent

_WHEELS_OVERRIDE = os.environ.get("AURIS_WHEELS_DIR", "").strip()
WHEELS_DIR = Path(_WHEELS_OVERRIDE) if _WHEELS_OVERRIDE else (REPO_DIR / "wheels")
STRICT_OFFLINE = os.environ.get("AURIS_OFFLINE", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
USE_LOCAL_WHEELS = STRICT_OFFLINE or os.environ.get("AURIS_USE_LOCAL_WHEELS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

PIP = [sys.executable, "-m", "pip", "install", "--upgrade"]

W = "\033[0m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
B = "\033[34m"
BD = "\033[1m"


def banner():
    print(
        f"""
{BD}+------------------------------------------+{W}
{BD}|         Auris Setup Installer            |{W}
{BD}|   Snapdragon X Elite audiobook stack     |{W}
{BD}+------------------------------------------+{W}
"""
    )


def info(msg):
    print(f"  {G}*{W} {msg}")


def warn(msg):
    print(f"  {Y}!{W} {msg}")


def error(msg):
    print(f"  {R}x{W} {msg}")


def step(msg):
    print(f"\n{BD}{B}>{W} {BD}{msg}{W}")


def ok(msg):
    print(f"  {G}OK{W} {msg}")


def run(cmd, check=True, **kwargs):
    info(f"Running: {' '.join(str(part) for part in cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def is_arm64():
    """True on ARM64 Windows (Snapdragon X and friends) and on aarch64 Linux."""
    machine = platform.machine().lower()
    return machine in {"arm64", "aarch64"} and platform.system() != "Darwin"


def arm64_platform_tag():
    """Return 'win_arm64' or 'linux_aarch64' on ARM64, else None.

    The tag selects the dependency sets below. Windows on ARM has no
    torchaudio, thinc or soxr wheel, so the CPU engines there must be
    installed without them; aarch64 Linux has all of them.
    """
    if not is_arm64():
        return None
    return "win_arm64" if platform.system() == "Windows" else "linux_aarch64"


# Packages the CPU engines need, as (requirement, platforms_to_skip). torchaudio
# has no win_arm64 wheel; soxr has no ARM wheel at all; and thinc, spaCy's
# compiled backend, dropped its aarch64 wheels in the 9.1 series, so aarch64
# needs the 9.0 pin rather than a skip.
#
# spaCy and thinc are skipped on Windows ARM64: their compiled backend
# (blis) has no win_arm64 wheel, so pip would try to build it from source
# with Visual Studio and fail the whole install. The app runs without them:
# character detection falls back to its regex mode (core.characters).
ARM64_SOFT_DEPS = (
    ("torchaudio>=2.4,<2.12", ("win_arm64",)),
    ("soxr", ("win_arm64", "linux_aarch64")),
    ("scipy", ()),  # portable resample fallback, replaces soxr on every ARM tag
    ("thinc<9.1", ("win_arm64",)),
    ("spacy>=3.7", ("win_arm64",)),
)

# Platform-specific version pins, applied on top of the requirement list.
# The package name is the key, the value replaces the requirement.
#
# PyMuPDF 1.28.2 is the last release with a win_arm64 wheel
# (cp310-abi3, so it also serves Python 3.11+ on ARM64). Newer releases
# publish no win_arm64 build, so pip falls back to compiling the MuPDF
# sources with Visual Studio, which fails and aborts the install.
ARM64_PINNED_DEPS = {
    "win_arm64": {
        "pymupdf": "pymupdf==1.28.2",
    },
}

# Everything else the app needs on an ARM64 machine (Snapdragon X Elite),
# without any GPU stack.
ARM64_BASE_DEPS = (
    "onnxruntime>=1.20,<2",
    # Sherpa-onnx re-export of Supertonic 3 (the fastest measured engine on
    # Snapdragon, RTF 0.14). Pure-python + onnxruntime, no torchaudio need.
    # Installed as a soft dep so its absence only drops that one engine.
    "sherpa-onnx",
    "huggingface_hub",
    "sentencepiece>=0.2",
    "soundfile>=0.12",
    "librosa>=0.11",
    # XTTSv2-ONNX voice cloning needs pypinyin/hangul in its text pipeline.
    "pypinyin>=0.50",
    "hangul-romanize>=0.1",
    "num2words>=0.5.13",
    "pydub>=0.25",
    "flask>=3.0",
    "ebooklib>=0.18",
    "pymupdf>=1.24",
    "pillow>=10",
    "python-docx>=1.1",
    "pyphen>=0.15",
    "trafilatura>=2.2,<3",
    "onnx-asr==0.12.0",
)


def arm64_requirements(tag):
    """Filter the ARM64 dependency set for the given platform tag."""
    return [req for req, missing in ARM64_SOFT_DEPS if tag not in missing]


def apply_platform_pins(reqs, tag):
    """Replace a requirement with its platform-pinned version, if any."""
    pins = ARM64_PINNED_DEPS.get(tag, {})
    pinned = []
    for req in reqs:
        name = re.split(r"[<>=!~ ]", req, maxsplit=1)[0].strip()
        pinned.append(pins.get(name, req))
    return pinned


def detect_hardware():
    step("Detecting hardware")

    if is_arm64():
        tag = arm64_platform_tag()
        ok(f"ARM64 detected ({tag}) - installing the ONNX CPU engine set")
        return "arm64_onnx"

    raise RuntimeError(
        "This build targets Snapdragon X Elite (Windows ARM64) only; "
        f"detected {platform.system()} {platform.machine()}."
    )

def offline_wheels_available():
    return USE_LOCAL_WHEELS and WHEELS_DIR.exists() and any(WHEELS_DIR.glob("*.whl"))


def running_in_virtualenv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def ensure_pip():
    """Upgrade pip/setuptools/wheel so dependency resolution is reliable."""
    step("Upgrading pip")
    # Use ensurepip-safe invocation (not PIP, which already includes --upgrade).
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    ok("pip upgraded")


def pip_install(*args, no_index=False, index_url=None, extra_index_url=None):
    cmd = [*PIP]
    wheels_ready = offline_wheels_available()

    if wheels_ready:
        cmd.append(f"--find-links={WHEELS_DIR}")
        if STRICT_OFFLINE or no_index:
            cmd.append("--no-index")
    elif STRICT_OFFLINE or no_index:
        raise RuntimeError(
            f"Offline install requested, but no local wheels were found in: {WHEELS_DIR}"
        )

    if index_url:
        cmd.extend(["--index-url", index_url])
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])

    cmd.extend(str(arg) for arg in args)
    run(cmd)


# Tested range. torchaudio 2.9+ changed its I/O backends; an unbounded
# "latest" install could break the audio stack on a future release.
TORCH_SPEC = "torch>=2.4,<2.12"
TORCHAUDIO_SPEC = "torchaudio>=2.4,<2.12"


def install_torch(hw_tag):
    step("Installing PyTorch (CPU, optional)")

    # No CUDA build exists for ARM64 and torchaudio has no win_arm64 wheel.
    # torch is optional: only the Whisper QA backend imports it; every
    # synthesis engine runs on ONNX Runtime.
    tag = arm64_platform_tag()
    info(f"ARM64 ({tag}) detected. Installing CPU-only torch; torchaudio is skipped.")
    info("The synthesis engines (Supertonic 3, XTTSv2-ONNX, Piper) run on ONNX Runtime.")
    if tag == "linux_aarch64":
        # aarch64 Linux has both wheels. The CPU index also serves the aarch64
        # torch build, so the Whisper QA backend works in the container the
        # same way it does on Windows ARM64 (torchaudio alone does not pull
        # torch in, it is only the audio codec layer). Both come from the same
        # index so pip can match the torchaudio/torch pair.
        pip_install(TORCH_SPEC, TORCHAUDIO_SPEC,
                    index_url="https://download.pytorch.org/whl/cpu")
    elif offline_wheels_available():
        pip_install(TORCH_SPEC, no_index=True)
    else:
        pip_install(TORCH_SPEC, index_url="https://download.pytorch.org/whl/cpu")
    ok("PyTorch installed (CPU)")

def verify_torch(hw_tag):
    """Import native libraries in a fresh process before declaring setup ready."""
    step("Checking the ONNX runtime")
    code = (
        "import onnxruntime as ort; "
        "print('onnxruntime:', ort.__version__, ort.get_available_providers()); "
        "import importlib.util as u; "
        "print('torch:', 'installed (optional)' if u.find_spec('torch') else 'not installed (optional)');"
    )
    if arm64_platform_tag() == "linux_aarch64":
        code += " import torchaudio; print('torchaudio:', torchaudio.__version__);"
    run([sys.executable, "-c", code])
    ok("ARM64 native runtime verified")

def install_reader_deps(hw_tag=None):
    step("Installing the Snapdragon dependency set")
    tag = arm64_platform_tag()
    skipped = [req for req, missing in ARM64_SOFT_DEPS if tag in missing]
    requirements = apply_platform_pins([*ARM64_BASE_DEPS, *arm64_requirements(tag)], tag)
    pip_install(*requirements)
    if skipped:
        info("Not available for " + tag + ": " + ", ".join(skipped))
        info("The app runs without them: SciPy resamples, and character")
        info("detection falls back to its regex mode when spaCy is absent.")
    ok("Snapdragon dependencies installed")

def install_spacy_model():
    step("Installing spaCy language model (en_core_web_sm)")

    if STRICT_OFFLINE:
        warn("Strict offline mode enabled. Skipping spaCy model download.")
        warn("Install en_core_web_sm later from Settings or with: python -m spacy download en_core_web_sm")
        return

    try:
        import spacy

        try:
            spacy.load("en_core_web_sm")
            ok("en_core_web_sm already installed")
            return
        except OSError:
            pass
    except ImportError:
        warn("spaCy is not installed yet. Model download will be attempted after dependency install.")

    try:
        run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        ok("en_core_web_sm installed")
    except subprocess.CalledProcessError:
        warn("Could not download en_core_web_sm during setup.")
        warn("Install it later from Settings or with: python -m spacy download en_core_web_sm")


def install_hungarian_spacy_model():
    """HuSpaCy powers Hungarian character detection without a language model."""
    step("Installing Hungarian spaCy model (HuSpaCy hu_core_news_md)")
    if STRICT_OFFLINE:
        warn("Strict offline mode enabled. Install HuSpaCy later from Settings.")
        return
    import importlib.util

    if any(importlib.util.find_spec(name) for name in
           ("hu_core_news_lg", "hu_core_news_md", "hu_core_news_trf")):
        ok("HuSpaCy already installed")
        return
    try:
        sys.path.insert(0, str(APP_DIR))
        from core.settings import install_spacy_model as install_model

        result = install_model("hu")
    except Exception as exc:  # network or import failure must not stop setup
        result = {"ok": False, "message": str(exc)}
    if result.get("ok"):
        ok("HuSpaCy installed")
    else:
        warn("Could not install HuSpaCy during setup: " + str(result.get("message", ""))[-300:])
        warn("Install it later from Settings -> Character detection.")


def print_summary(hw_tag):
    device_label = {
        "arm64_onnx": "Snapdragon X Elite (ARM64, ONNX Runtime)",
    }.get(hw_tag, hw_tag)

    engine_hint = f"""
  {BD}Engines     : {BD}Supertonic 3 / Sherpa-Supertonic (fast), XTTSv2-ONNX (voice cloning), Piper{W}
                   Pick one under Settings -> Beszédmotorok.
                   Every engine runs locally on the CPU - no GPU required.
"""

    launch_hint = (
        r".venv\Scripts\python.exe app.py"
        if os.name == "nt"
        else "./.venv/bin/python app.py"
    )

    print(
        f"""
{BD}+------------------------------------------+{W}
{BD}|              Setup Complete              |{W}
{BD}+------------------------------------------+{W}

  Device      : {G}{device_label}{W}
{engine_hint}  To launch   : {BD}{launch_hint}{W}
  Windows     : {BD}run.bat{W}
  Linux / Mac : {BD}bash run.sh{W}
  Browser     : http://127.0.0.1:7860

  Model path is configured in Settings.
  The spaCy model can also be installed later from Settings.
"""
    )


def main():
    banner()
    os.chdir(APP_DIR)

    if running_in_virtualenv():
        ok(f"Using virtual environment: {sys.prefix}")
    else:
        warn("No virtual environment detected. setup.bat/setup.sh will create one automatically.")

    if offline_wheels_available():
        if STRICT_OFFLINE:
            info(f"Strict offline mode enabled with wheel cache: {WHEELS_DIR}")
        else:
            info(f"Using local wheels from {WHEELS_DIR}; missing packages will be downloaded if needed.")
    elif WHEELS_DIR.exists() and not USE_LOCAL_WHEELS:
        info(f"Ignoring local wheels at {WHEELS_DIR} unless AURIS_USE_LOCAL_WHEELS=1 is set.")
    elif STRICT_OFFLINE:
        raise RuntimeError(
            f"AURIS_OFFLINE=1 was set, but no wheel cache was found in: {WHEELS_DIR}"
        )

    if not STRICT_OFFLINE:
        ensure_pip()

    hw_tag = detect_hardware()

    # Snapdragon X Elite: wheel-only ONNX install, no GPU runtime anywhere.
    # The runtime check runs after install_reader_deps: it imports
    # onnxruntime, which only that step installs, so an earlier check fails
    # on every fresh virtual environment (fresh Windows install, Docker).
    install_torch(hw_tag)
    install_reader_deps(hw_tag)
    install_spacy_model()
    install_hungarian_spacy_model()
    verify_torch(hw_tag)
    print_summary(hw_tag)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        error(f"A step failed (exit code {exc.returncode}). Check the output above.")
        sys.exit(1)
    except RuntimeError as exc:
        error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        warn("Setup cancelled.")
        sys.exit(0)
