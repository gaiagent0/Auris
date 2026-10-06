"""NPU/HTP diagnosztika a Supertonic 3 NPU-gyorsításához.

Ez a szkript nem futtat TTS-t, hanem **mérni bizonyítja**, hogy ezen a gépen
az ONNX Runtime QNN execution provider tényleg az NPU-n fut-e, vagy csak
névleg van ott, és minden számítás CPU-n történik.

Használat (a reader venvből, ARM64 Python):
    .venv/Scripts/python.exe scripts/npu_probe.py
    .venv/Scripts/python.exe scripts/npu_probe.py --json-out npu_probe.json

A `--cpu-baseline` kapcsoló a vector_estimator CPU-időit méri (a referancia,
amivel minden NPU-mérést össze kell hasonlítani).

Három dolgot vizsgál:
  1. létezik-e a Hexagon NPU illesztőprogram,
  2. betölthető-e a QNN HTP stub (a libcdsprpc.dll a driver store-ban van,
     de nincs a betöltési útvonalon — ez a leggyakoribb hiba),
  3. az ORT QNN EP létrehoz-e valódi NPU-node-ot egy apró Conv modellen.

A 3. pont a döntő: ha a profilban 0 db `QNNExecutionProvider` node látszik,
akkor az EP nem az NPU-n fut, akármit ír a session provider-listája.
"""
from __future__ import annotations

import argparse
import base64
import collections
import ctypes
import json
import os
import subprocess
import sys
import time

# Apró, statikus Conv ONNX (1x4x8x8 -> 1x8x8x8), base64-ben, hogy a szkript
# ne függjön az `onnx` csomagtól. Csak a QNN EP "van-e NPU-node" kérdésére kell.
TINY_CONV_ONNX_B64 = (
    "CAg6vQoKPAoBeAoBdwoBYhIBeSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNA"
    "AUABQAFAAaABBxIFcHJvYmUqkAkICAgECAMIAxABQgF3SoAJbL8APn9GB7698iM/39XWPaEhCb/9Ir"
    "k+eemmP+Zzcj/+JzS/Vfmhv+qOH79uRSk9Ts0UwOwKYL4Dep+/4HU7v49UC78Y8qG+PcHSPhRxhT+Xng"
    "O+RuiuPzNKKr8h+bM+0klnP4WJwD34VT6/MvZrvwpb6r7VemE+KzuBvx8yVr7iCyO+23QKP5rPWz5p87U"
    "+UGEnv3C5BL6eskg/wCi/Pw8pob9ByME/pUWsPwYESD+6Zoc+frqgvmyguj+/6fo/+ZvmP1JVqD+R+rY+L6"
    "qav/rzkbu+Dig/B+mkv3BNyj4YF9w+2y8yPy2Rl79XZSm/cXTfvhK8lb+bo94/A+j9vrVuqD6fY4S+Pa/K"
    "P5cBqT9mIyI/TgYNwFUcVT0PBi8/0IGAPygvHr+rN+k/4gOpv+dZKb9wX28/fu1IPTMnAEAtC0E+Ahkiv"
    "wFQwb6tqou/Boujv6ZiIT9IxxQ/GrSlP9gtQb+sNNg/eySTvjaGyT8blt2+okg8v73Hfz6oBoQ/st8kPj"
    "jlFb8Wrau/BGWzv9OvAD/VXX0/1zwovsqEib+xf18/8+Ojv6GLNr8H+x4/UAIQwDzSxT5q5hS/Cc7fPWcJ"
    "m70T904+QbUxP4UkQr+94rU/SOE5P93+Vz9DGpU/Yp9JP4oVWD/T0Jo9h6C2v0NJCr7p/kS/Zxy2v+1Th"
    "D51jBG/otCDvw+Bhb/3bYk+2qO3PklGqT9a+mO8AVuFP2p9sz+hOJM/JGEXwIJFnT+s4q0+j/jYPoURvj"
    "6/+MM+Q4qjPnzDt77JaPO/rw7fvV/BTb/LQoo/NNmTviH1qj3Hf1m/J7gCvyv1PLzGIL6/ZvOZPmk82b2"
    "rxZe/pnwZwGJXAz/vXJi+oq4Hv4XScb5Jgug/GvxLvXFlsT1nWL6/AtzSP37gaj9SkYg/e0RDPeOpaj++7"
    "L0+9vkcP3nYG75cqLy/gLGDP8Ks97/1sXW+WG5RvnF8hb+j9Rw/OyNNvjWt375ZFAU/KwL0vhnKsT/r8bM"
    "+xtvyvq3d+L91ZKe/RR2LPzVGT73F9ZC+ElbSP9otpL+r7RW/A/fxvjMaFj9x3Sm/9Agdv4l1zb+ktjo/Jl"
    "9OP6fn876WQic+bnWlv36R8b6yYLA///wKPv/cE0B3hUm/hY0UP64ySL5w2RA/RU3su66qD78iHF6/8jlEQ"
    "BZnnr34EAHAsAomvwOULT+NAAC/GiOuP5ZOgD+p/hu+SsbxvlKdgL8CMTO/9I+8v6gpmj8VnMs/Ismgv2NB"
    "l7+ZXuK/J792vznORsAyNpK/UwGmP/37sL4Ixlo/J1r6voxd4T/N/0s+zpXDvupaI0AtIaa+DFGcv3/BTj5"
    "9ER+9U32IPzPwa7/uAU4/uU1aP47tKr9wKSc+KaxUv7ghFkB+QjS/YPnnvmFtiL/PNrG+r4vAu9SNRD/aSB"
    "y/hzs+voZPtb+i0FO/J18wQHVHhT9oC0i/1S+rv8y/eb8IsbG8uj4OPWuOPr95rqS/gBC2P1FD5z5hx7++"
    "A/Vhvk+PB78r6DvAHODsPZkHib/2V4C/KikICBABQgFiSiA96CO/IHg7P/TTlb+Jlre/WM0jP1IeQT+ufH"
    "W/S/kPP1obCgF4EhYKFAgBEhAKAggBCgIIBAoCCAgKAggIYhsKAXkSFgoUCAESEAoCCAEKAggICgIICA"
    "oCCAhCBAoAEBE="
)

DRIVERSTORE_DSPRPC = (
    r"C:\Windows\System32\DriverStore\FileRepository"
    r"\qcadsprpc8380.inf_arm64_81346390221b730a"
)


def _powershell(*args: str) -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", " ".join(args)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (out.stdout or "").strip()
    except Exception as exc:  # pragma: no cover - diagnosztikai szkript
        return f"<powershell hiba: {exc}>"


def probe_driver(report: dict) -> None:
    script = (
        "Get-CimInstance Win32_PnPSignedDriver | "
        "Where-Object { $_.DeviceName -match 'Hexagon' } | "
        "Select-Object -First 1 DeviceName,DriverVersion | ConvertTo-Json -Compress"
    )
    raw = _powershell(script)
    name = version = ""
    if raw and raw not in ("null", "<powershell hiba>"):
        try:
            data = json.loads(raw)
            name = data.get("DeviceName", "") or ""
            version = data.get("DriverVersion", "") or ""
        except Exception:
            name = raw
    found = "Hexagon" in name
    detail = f"{name} (driver {version})" if name else "nem talalhato"
    report["npu_driver"] = {"ok": found, "detail": detail}
    print(f"[{'OK ' if found else 'HIBA'}] NPU illesztoprogram: {detail}")


def _try_load(name: str) -> tuple[bool, str]:
    try:
        ctypes.WinDLL(name)
        return True, ""
    except OSError as exc:
        return False, str(exc)


def probe_rpc_libs(report: dict) -> None:
    ok, err = _try_load("libcdsprpc.dll")
    if not ok and os.path.isdir(DRIVERSTORE_DSPRPC):
        ok, err = _try_load(os.path.join(DRIVERSTORE_DSPRPC, "libcdsprpc.dll"))
        if ok:
            err = "driver store-bol betoltheto, de NINCS a betoltesi utvonalon"
    report["libcdsprpc"] = {"ok": ok, "detail": err}
    state = "OK " if ok else "HIBA"
    note = err if err else "betaltheto"
    print(f"[{state}] libcdsprpc.dll: {note}")
    if not ok:
        print("       -> a QnnHtpV81Stub.dll emiatt nem tolt be, es az EP CPU-ra es vissza.")


def probe_qnn_ep(report: dict) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        report["qnn_ep"] = {"ok": False, "detail": f"onnxruntime unavailable: {exc}"}
        print(f"[HIBA] onnxruntime nem elérhető: {exc}")
        return

    try:
        import onnxruntime_qnn as qnn_ep
    except Exception as exc:
        report["qnn_ep"] = {
            "ok": False,
            "detail": f"onnxruntime_qnn nincs telepitve ({exc}); "
            "a QNN EP nem is probalhato",
        }
        print(f"[HIBA] onnxruntime_qnn nincs telepitve: {exc}")
        return

    ort.register_execution_provider_library(
        qnn_ep.get_ep_name(), qnn_ep.get_library_path()
    )
    npu_devices = [
        d
        for d in ort.get_ep_devices()
        if d.ep_name == qnn_ep.get_ep_name() and "NPU" in str(d.device.type)
    ]

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.log_severity_level = 3
    so.add_provider_for_devices(
        npu_devices,
        {"backend_path": os.path.dirname(qnn_ep.__file__), "profiling_level": "basic"},
    )

    model = base64.b64decode(TINY_CONV_ONNX_B64)
    feed = {"x": np.ones((1, 4, 8, 8), dtype=np.float32)}
    try:
        sess = ort.InferenceSession(model, sess_options=so)
    except Exception as exc:
        report["qnn_ep"] = {"ok": False, "detail": f"session letepites: {exc}"}
        print(f"[HIBA] QNN EP session letepites: {exc}")
        return

    sess.run(None, feed)
    profile = sess.end_profiling()
    counts: collections.Counter = collections.Counter()
    try:
        with open(profile, encoding="utf-8") as fh:
            for event in json.load(fh):
                counts[event.get("args", {}).get("provider")] += 1
    finally:
        if os.path.exists(profile):
            os.remove(profile)

    qnn_nodes = counts.get("QNNExecutionProvider", 0)
    cpu_nodes = counts.get("CPUExecutionProvider", 0)
    listed = sess.get_providers()
    really_on_npu = qnn_nodes > 0
    report["qnn_ep"] = {
        "ok": really_on_npu,
        "listed_providers": listed,
        "qnn_nodes": qnn_nodes,
        "cpu_nodes": cpu_nodes,
    }
    if really_on_npu:
        print(f"[OK ] QNN EP: {qnn_nodes} node a HTP-n, {cpu_nodes} a CPU-n")
    else:
        print(f"[HIBA] QNN EP NEM fut az NPU-n: 0 QNN node, {cpu_nodes} CPU node.")
        print(f"       A session provider-listaja {listed} - ez nem bizonyit NPU-futasit.")
        print("       Minden korabbi 'QNN EP'-meres ebben az allapotban CPU-ido volt.")


def probe_cpu_baseline(report: dict, variants: int) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        print(f"[-- ] CPU baseline kihagyva: {exc}")
        return

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = os.path.join(root, "models")
    candidates = {
        "fp32": os.path.join(models, "supertonic-3", "onnx", "vector_estimator.onnx"),
        "int8": os.path.join(
            models, "supertonic-3-int8", "onnx", "vector_estimator.int8.onnx"
        ),
    }
    rng = np.random.default_rng(7)

    def feed_for(bsz: int) -> dict:
        def rnd(shape):
            return (rng.standard_normal(shape) * 0.1).astype(np.float32)

        return {
            "noisy_latent": rnd((bsz, 144, 64)),
            "text_emb": rnd((bsz, 256, 96)),
            "style_ttl": rnd((bsz, 50, 256)),
            "latent_mask": np.ones((bsz, 1, 64), dtype=np.float32),
            "text_mask": np.ones((bsz, 1, 96), dtype=np.float32),
            "current_step": np.zeros((bsz,), dtype=np.float32),
            "total_step": np.full((bsz,), 10.0, dtype=np.float32),
        }

    rows = []
    print("\nCPU referencia (vector_estimator, ONNX Runtime %s)" % ort.__version__)
    print("%-5s %-4s %-8s %9s %9s" % ("model", "bsz", "threads", "min ms", "median ms"))
    for label, path in candidates.items():
        if not os.path.exists(path):
            continue
        for bsz in (1, variants):
            for threads in (1, 12):
                so = ort.SessionOptions()
                so.intra_op_num_threads = threads
                sess = ort.InferenceSession(
                    path, sess_options=so, providers=["CPUExecutionProvider"]
                )
                feed = feed_for(bsz)
                for _ in range(3):
                    sess.run(None, feed)
                times = []
                for _ in range(8):
                    t0 = time.perf_counter()
                    sess.run(None, feed)
                    times.append(time.perf_counter() - t0)
                arr = np.array(times)
                rows.append(
                    {
                        "model": label,
                        "batch": bsz,
                        "threads": threads,
                        "min_ms": round(float(arr.min() * 1000), 2),
                        "median_ms": round(float(np.median(arr) * 1000), 2),
                    }
                )
                print(
                    "%-5s %-4d %-8d %9.2f %9.2f"
                    % (label, bsz, threads, arr.min() * 1000, np.median(arr) * 1000)
                )
    report["cpu_baseline"] = rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-baseline",
        type=int,
        default=4,
        metavar="BATCH",
        help="a vector_estimator CPU-merese ennyi elemu batchszel is (0 = kihagyja)",
    )
    parser.add_argument("--json-out", help="eredmeny JSON fajlba")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("Ez a szkript Windows ARM64-re készült.")
        return 2

    print("== NPU/HTP diagnosztika ==\n")
    report: dict = {}
    probe_driver(report)
    probe_rpc_libs(report)
    probe_qnn_ep(report)
    if args.cpu_baseline:
        probe_cpu_baseline(report, args.cpu_baseline)

    npu_ok = bool(report.get("qnn_ep", {}).get("ok"))
    print("\n== osszegzes ==")
    print("NPU-n fut-e a QNN EP:", "IGEN" if npu_ok else "NEM")
    if not npu_ok:
        print("A gyorsitas merese addig nem lehet hiteles, amíg ez nincs IGEN.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print("JSON: " + args.json_out)
    return 0 if npu_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())