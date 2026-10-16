"""Atomic same-directory config replacement; failed writes never publish runtime."""
import json
import os
from pathlib import Path
import tempfile


def atomic_save(path: Path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json")
    data.pop("config_path", None)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
