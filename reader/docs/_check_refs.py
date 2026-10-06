"""A doc-okban említett útvonalak létezés-ellenőrzése.

Csak a frissített kutatási dokokat nézi: meresek.md, arm64-install.md,
handoff-2026-10-03.md.

Három szabály, hogy ne riasszon hamis positive-ot:
  1. a ``` kerített parancsblokkokon belüli útvonalak kimaradnak — azok
     utasítások („futtasd ezt"), nem állítások a jelenlegi állapotról;
  2. a törölt környezetekre utaló sorok kimaradnak (történeti említés);
  3. a {a,b} és a * mintákat felbontja, és elég, ha EGYRE van találat.

    python docs/_check_refs.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

DOCS = ["docs/meresek.md", "docs/arm64-install.md", "docs/handoff-2026-10-03.md"]

# "docs\meresi-adatok\..." vagy "D:\VoiceAI\..." elofordulas, lezaro karakter nelkul
PAT = re.compile(r"(docs[\\/]meresi-adatok[\\/][^\s`|,\)\"'>]+"
                 r"|D:[\\/]VoiceAI[\\/][^\s`|,\)\"'>]*)")

# A torolt utvonalak: ezek emlitese torteneti, nem hiba.
DELETED = re.compile(r"(clone-probe|npu-work|[\\/]venvs[\\/]|torchaudio_compat"
                     r"|C:[\\/]AI[\\/]venvs|exports[\\/]test-samples)")


def expand(path: str) -> list[str]:
    """A {a,b} es a * mintakat konkret utvonalakra bontja."""
    m = re.search(r"\{([^{}]*)\}", path)
    if m:
        out: list[str] = []
        for part in m.group(1).split(","):
            out.extend(expand(path[:m.start()] + part.strip() + path[m.end():]))
        return out
    if "*" in path or "?" in path:
        return glob.glob(path)
    # A PAT a vesszönél megszakad, ezért a {F1,M3} félig maradhat vissza.
    # Ilyenkor a nyitó { előtti rész egy FÁJLNÉV-előtag (pl.
    # "...\piper\hu_HU-{anna"), ami önmagában nem létező útvonal — a
    # szülőkönyvtárát ellenőrizzük, mert az a valódi állítás.
    if path.count("{") > path.count("}"):
        prefix = path[:path.index("{")].rstrip("\\/")
        parent = os.path.dirname(prefix)
        return [parent or prefix]
    return [path]


def exists_any(path: str) -> bool:
    return any(os.path.exists(p) for p in expand(path))


def main() -> int:
    seen: dict[str, list[str]] = {}
    skipped = 0
    for doc in DOCS:
        fenced = False
        with open(doc, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if line.lstrip().startswith("```"):
                    fenced = not fenced
                    continue
                if fenced:
                    skipped += 1
                    continue
                for raw in PAT.findall(line):
                    path = raw.strip().rstrip("\\/.,;:")
                    if DELETED.search(line) or "<" in path or "..." in path:
                        skipped += 1
                        continue
                    seen.setdefault(path, []).append(doc)

    missing = [(p, d) for p, d in seen.items() if not exists_any(p)]
    print(f"ellenorzott hivatkozas: {len(seen)}  "
          f"(kihagyva: {skipped} sor, parancsblokkban vagy torolt utvonal)")
    if missing:
        print("HIANYZO:")
        for p, d in sorted(missing):
            print(f"  {p}   ({d[0]})")
        return 1
    print("mindegyik atthalad a lemez-ellenorzesen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
