"""Explicit, resumable download of the three Comfy-Org H3 BF16 components."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
from pathlib import Path
import shutil
import time

import httpx

REPO = "Comfy-Org/MiniMax-H3"
REVISION = "54472221d27c695fca9621b0c3ed52f06c4af707"
FILES = {
    "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors": (
        40225724176, "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4"
    ),
    "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors": (
        40225724176, "37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe"
    ),
    "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors": (
        51506295256, "600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: Path, name: str, workers: int = 6, direct_cdn: bool = False,
             *, repo: str = REPO, revision: str = REVISION,
             artifacts: dict | None = None) -> dict:
    size, expected = (FILES if artifacts is None else artifacts)[name]
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".partial")
    if dest.exists():
        if dest.stat().st_size != size or sha256(dest) != expected:
            raise ValueError(f"Existing file differs; refusing to overwrite: {dest}")
    else:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        if direct_cdn:
            response = httpx.head(url, follow_redirects=False, timeout=60)
            if not response.is_redirect:
                raise RuntimeError(f"Official download did not return a CDN redirect: {name}")
            url = response.headers["location"]
        state_path = part.with_suffix(part.suffix + ".json")
        block = 128 * 1024 * 1024
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["sha256"] != expected or state["bytes"] != size:
                raise ValueError(f"Partial download metadata differs: {state_path}")
            if not part.exists():
                raise ValueError(f"Partial download data missing: {part}")
        else:
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError(f"Oversized partial file: {part}")
            state = {"sha256": expected, "bytes": size, "prefix": offset, "done": []}
            state_path.write_text(json.dumps(state), encoding="utf-8")
        prefix = state["prefix"]
        ranges = [(start, min(start + block, size)) for start in range(prefix, size, block)]
        lock = threading.Lock()
        last = time.monotonic()
        with httpx.Client(follow_redirects=True, timeout=120, trust_env=not direct_cdn) as client:
            def fetch(bounds):
                nonlocal last
                start, end = bounds
                if start in state["done"]:
                    return
                for attempt in range(8):
                    try:
                        with client.stream("GET", url, headers={
                            "Range": f"bytes={start}-{end - 1}"
                        }) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end - 1}/{size}":
                                raise ValueError(f"Server did not honor range: {name}")
                            count = start
                            with part.open("r+b") as output:
                                output.seek(start)
                                for chunk in response.iter_bytes(4 * 1024 * 1024):
                                    if count + len(chunk) > end:
                                        raise ValueError("Response exceeds requested range")
                                    output.write(chunk)
                                    count += len(chunk)
                                output.flush()
                                os.fsync(output.fileno())
                            if count != end:
                                raise OSError("Incomplete range")
                        with lock:
                            state["done"].append(start)
                            tmp = state_path.with_suffix(".tmp")
                            tmp.write_text(json.dumps(state), encoding="utf-8")
                            tmp.replace(state_path)
                            if time.monotonic() - last >= 30:
                                done = prefix + sum(min(block, size - pos) for pos in state["done"])
                                print(f"{name}: {done / 2**30:.2f}/{size / 2**30:.2f} GiB", flush=True)
                                last = time.monotonic()
                        return
                    except (httpx.HTTPError, OSError) as exc:
                        print(f"Retry {attempt + 1}: {name}: {type(exc).__name__}", flush=True)
                        if attempt == 7:
                            raise
                        time.sleep(3)
            part.touch(exist_ok=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as chunks:
                list(chunks.map(fetch, ranges))
        print(f"Verifying SHA256: {name}", flush=True)
        if part.stat().st_size != size or sha256(part) != expected:
            raise ValueError(f"Checksum mismatch; retained partial file: {part}")
        part.rename(dest)
        state_path.unlink()
    print(f"Verified: {name}", flush=True)
    return {"file": name, "bytes": size, "sha256": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("J:/Projects/LLM_Models/MiniMax-H3"))
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 33), help="Range workers per file (1-32)")
    parser.add_argument("--direct-cdn", action="store_true", help="Resolve through the system proxy, then fetch the official CDN directly")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    needed = sum(size for name, (size, _) in FILES.items() if not (args.root / name).exists())
    if shutil.disk_usage(args.root).free < needed + 10 * 2**30:
        raise OSError("Insufficient space for weights plus 10 GiB headroom")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(lambda name: download(args.root, name, args.workers, args.direct_cdn), FILES))
    manifest = {"repo": REPO, "revision": REVISION, "verified": entries}
    (args.root / "bf16-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
