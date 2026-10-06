import os
_dll = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg-shared")
if os.path.isdir(_dll) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_dll)
