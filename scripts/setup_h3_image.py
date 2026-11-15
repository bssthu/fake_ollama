"""Download pinned ComfyUI H3-Image weights, with resumable ranges and SHA-256."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil

from setup_h3_bf16 import download

REPO = "Comfy-Org/MiniMax-H3-Image"
REVISION = "0bb149f51e3156ed4b90dd45d5620346bfe1df8f"
FILES = {
    "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors": (
        12549794530, "517cc19b85348386f4df7c806dce80d839d0899e7cdaa3e1dbb09de964381dbb"
    ),
    "diffusion_models/minimax_h3_image_pruned_bf16.safetensors": (
        40234657162, "a767f5325db019588b3491a93d80288d336c826a7aeb87bec67675e554367222"
    ),
    "vae/minimax_h3_image_vae_fp16.safetensors": (
        1168505312, "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("J:/Projects/LLM_Models/MiniMax-H3-Image"))
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 33))
    parser.add_argument("--direct-cdn", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    needed = sum(size for name, (size, _) in FILES.items() if not (args.root / name).exists())
    if shutil.disk_usage(args.root).free < needed + 10 * 2**30:
        raise OSError("Insufficient space for weights plus 10 GiB headroom")
    def fetch(name):
        return download(args.root, name, args.workers, args.direct_cdn,
                        repo=REPO, revision=REVISION, artifacts=FILES)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(fetch, FILES))
    manifest = {"repo": REPO, "revision": REVISION, "verified": entries}
    (args.root / "image-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
