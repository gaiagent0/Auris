"""Minimális torchvision shim Windows ARM64-hez.

A PyPI-n nincs torchvision-kerék ehhez a platformhoz, az onnx2torch viszont
importálja az NMS/RoIAlign konverterekhez. Az MDX/VR szétválasztó modellek
ezeket a műveleteket nem használják, ezért a tiszta torch implementáció elég.
"""
__version__ = "0.0.0+shim"

from . import ops  # noqa: F401