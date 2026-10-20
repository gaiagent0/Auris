"""Writable application paths; source installations keep their existing layout."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


def storage_root() -> Path:
    override = os.environ.get('AURIS_DATA_DIR', '').strip()
    return Path(override).expanduser().resolve() if override else APP_DIR


def user_path(*parts: str) -> Path:
    return storage_root().joinpath(*parts)
