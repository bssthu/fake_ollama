"""Build a portable H3 Ref2VA graph for the native ComfyUI /prompt API.

Reference names are files already uploaded into the selected ComfyUI input
folder. This does not submit a generation or change fake-ollama HTTP modes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "fake_ollama/workflows/minimax_h3_ref2va.json"


def build_workflow(*, prompt: str, images: list[str], videos: list[str],
                   width: int = 768, height: int = 768, frames: int = 124,
                   steps: int = 20, seed: int = 0, first_frame: str | None = None) -> dict:
    if not prompt.strip():
        raise ValueError("A nonempty prompt is required")
    if len(images) > 9 or len(videos) > 3:
        raise ValueError("H3 supports up to 9 reference images and 3 reference videos")
    if not images and not videos:
        raise ValueError("At least one image or video reference is required")
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("Width and height must be positive multiples of 32")
    if width * height > 768 * 1344:
        raise ValueError("This local 768p workflow is capped at 768 * 1344 pixels")
    if frames < 5 or frames > 362 or frames % 17 != 5:
        raise ValueError("Frame count must be 17k + 5, between 5 and 362, at 24 fps")
    if not 1 <= steps <= 100:
        raise ValueError("Steps must be between 1 and 100")
    refs = images + videos + ([first_frame] if first_frame else [])
    for name in refs:
        if not name or name.startswith(("/", "\\")) or ":" in name or ".." in name.replace("\\", "/").split("/"):
            raise ValueError("Use uploaded ComfyUI input filenames, not absolute paths or traversal")
    graph = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cond = graph["5"]["inputs"]
    cond.update(prompt=prompt, width=width, height=height, length=frames)
    graph["6"]["inputs"]["noise_seed"] = seed
    graph["10"]["inputs"]["steps"] = steps
    for i, name in enumerate(images):
        node = str(100 + i)
        graph[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        cond[f"ref_images.ref_image_{i}"] = [node, 0]
    for i, name in enumerate(videos):
        load, components = str(200 + 2*i), str(201 + 2*i)
        graph[load] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        graph[components] = {"class_type": "GetVideoComponents", "inputs": {"video": [load, 0]}}
        cond[f"ref_videos.ref_video_{i}"] = [components, 0]
    if first_frame:
        graph["300"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        graph["301"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
            "positive": ["5", 0], "latent": ["5", 1], "vae": ["3", 0],
            "image": ["300", 0], "frame_idx": 0}}
        graph["7"]["inputs"]["conditioning"] = ["301", 0]
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--image", action="append", default=[], help="Uploaded input filename; repeat in Picture order")
    parser.add_argument("--video", action="append", default=[], help="Uploaded 24-fps video filename; repeat in Video order")
    parser.add_argument("--first-frame", help="Optional input filename; requires MiniMaxH3AddGuide on the server")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        graph = build_workflow(prompt=args.prompt_file.read_text(encoding="utf-8-sig"),
            images=args.image, videos=args.video, first_frame=args.first_frame,
            width=args.width, height=args.height, frames=args.frames, steps=args.steps, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Workflow written: {args.output.resolve()} (not submitted)")


if __name__ == "__main__":
    main()
