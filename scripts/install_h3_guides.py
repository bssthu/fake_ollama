"""Install the pinned ComfyUI H3 guide backport after stopping ComfyUI.

Default is a read-only check. --apply backs up and patches exactly three files.
Unrecognised versions are rejected before any file is changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PATCH = Path(__file__).resolve().parents[1] / "services/comfyui_h3_guides/patch.json"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(root: Path) -> list[tuple[Path, str]]:
    pending = []
    for entry in json.loads(PATCH.read_text(encoding="utf-8"))["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("patch path is outside ComfyUI")
        original = path.read_text(encoding="utf-8")
        current_hash = digest(original)
        if current_hash == entry["result_sha256"]:
            continue
        if current_hash != entry["source_sha256"]:
            raise ValueError(f"Unsupported or modified ComfyUI file: {path}; no files changed")
        lines = original.splitlines(keepends=True)
        for edit in reversed(entry["edits"]):
            lines[edit["start"]:edit["end"]] = edit["lines"]
        patched = "".join(lines)
        if digest(patched) != entry["result_sha256"]:
            raise ValueError(f"Patch result hash mismatch: {path}")
        compile(patched, str(path), "exec")
        pending.append((path, patched))
    return pending


def install(root: Path, *, apply: bool = False) -> dict:
    root = root.resolve()
    pending = prepare(root)
    result = {"root": str(root), "pending": [str(path.relative_to(root)) for path, _ in pending],
              "applied": False, "upstream": "https://github.com/Comfy-Org/ComfyUI/pull/15439"}
    if not apply or not pending:
        return result
    backup = root / ("h3-guide-backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    for path, _ in pending:
        saved = backup / path.relative_to(root)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    try:
        for path, patched in pending:
            # Re-check after the backup, before replacing any existing content.
            if path.read_bytes() != (backup / path.relative_to(root)).read_bytes():
                raise RuntimeError(f"File changed during installation: {path}")
        for path, patched in pending:
            path.write_text(patched, encoding="utf-8", newline="\n")
        if prepare(root):
            raise RuntimeError("Post-install verification failed")
    except Exception:
        for path, _ in pending:
            shutil.copy2(backup / path.relative_to(root), path)
        raise
    result.update(applied=True, backup=str(backup))
    (backup / "installation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write the verified patch after stopping ComfyUI")
    args = parser.parse_args()
    print(json.dumps(install(args.comfy_root, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
