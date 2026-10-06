"""List the PE imports of a DLL and report which of them can be loaded."""
import ctypes
import struct
import sys


def imports(path):
    with open(path, "rb") as fh:
        data = fh.read()
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[e_lfanew:e_lfanew + 4] == b"PE\0\0"
    coff = e_lfanew + 4
    n_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    imp_rva, imp_size = struct.unpack_from("<II", data, dd + 8)
    sec = opt + opt_size
    sections = []
    for i in range(n_sections):
        o = sec + 40 * i
        va = struct.unpack_from("<I", data, o + 12)[0]
        vs = struct.unpack_from("<I", data, o + 8)[0]
        raw = struct.unpack_from("<I", data, o + 20)[0]
        sections.append((va, max(vs, 1), raw))

    def rva2off(rva):
        for va, vs, raw in sections:
            if va <= rva < va + vs:
                return raw + (rva - va)
        raise ValueError(rva)

    out, off = [], rva2off(imp_rva)
    while True:
        fields = struct.unpack_from("<IIIII", data, off)
        if not any(fields):
            break
        name_off = rva2off(fields[3])
        end = data.index(b"\0", name_off)
        out.append(data[name_off:end].decode())
        off += 20
    return out


for path in sys.argv[1:]:
    print("==", path)
    for dll in imports(path):
        try:
            ctypes.WinDLL(dll)
            print("   OK      ", dll)
        except OSError as e:
            print("   MISSING ", dll, "->", e)