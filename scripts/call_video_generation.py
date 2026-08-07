from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_api_interface(config: dict[str, Any]) -> dict[str, Any]:
    interfaces = config.get("api_interfaces") or []
    if not interfaces:
        raise SystemExit("config.json has no api_interfaces entry")
    return interfaces[0]


def token_from_config(config: dict[str, Any]) -> str:
    tokens = first_api_interface(config).get("access_tokens") or []
    if not tokens:
        raise SystemExit("api interface has no access_tokens entry")
    return str(tokens[0])


def base_url_from_config(config: dict[str, Any]) -> str:
    iface = first_api_interface(config)
    host = str(iface.get("host") or iface.get("bind") or "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(iface.get("port") or 21435)
    return f"http://{host}:{port}"


def image_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for group in args.image or []:
        if isinstance(group, Path):
            paths.append(group)
        else:
            paths.extend(group)
    return paths


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_mode": args.prompt_mode,
        "context_ir_mode": args.context_ir_mode,
        "size": args.size,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "prefetch_count": args.prefetch_count,
        "enable_tile": args.enable_tile,
        "enable_streaming": args.enable_streaming,
        "response_format": "b64_json",
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    image_paths = image_paths_from_args(args)
    if image_paths:
        payload["images"] = [encode_image(path) for path in image_paths]
        payload["filenames"] = [path.name for path in image_paths]
    return payload


def encode_image(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"reference image does not exist: {path}")
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def post_http(
    *,
    base_url: str,
    token: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/videos/generations"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": token,
        },
        method="POST",
    )
    # This script always targets the explicitly selected fake-ollama endpoint,
    # which is normally loopback.  Do not let Windows/system proxy settings
    # intercept a large local base64 media request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return int(response.status), json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw}
        return int(exc.code), data


def decode_video(data: dict[str, Any]) -> tuple[bytes, str, str]:
    items = data.get("data") or []
    if not items:
        raise SystemExit(f"response has no data items: {json.dumps(data, ensure_ascii=False)[:1000]}")
    item = items[0]
    mime_type = str(item.get("mime_type") or "video/mp4")
    filename = str(item.get("filename") or "joyai-video.mp4")
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"]), filename, mime_type
    url = str(item.get("url") or "")
    if url.startswith("data:") and "," in url:
        header, encoded = url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or mime_type
        return base64.b64decode(encoded), filename, mime_type
    raise SystemExit(f"unsupported response item: {json.dumps(item, ensure_ascii=False)[:1000]}")


def write_output(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call the running fake-ollama /v1/videos/generations API and "
            "save the first mp4 result. Pass --image for image-to-video."
        )
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for the video.")
    parser.add_argument("--model", default="joyai-echo")
    parser.add_argument(
        "--prompt-mode",
        choices=("raw", "auto", "enhance"),
        default="auto",
        help="H3 prompt planning mode (default: auto).",
    )
    parser.add_argument(
        "--context-ir-mode",
        choices=("auto", "t2va", "i2va", "fl2va", "l2va"),
        default="auto",
        help="H3 base workflow mode (default: auto by reference-image count).",
    )
    parser.add_argument("--size", default="256x256")
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--prefetch-count", type=int, default=1)
    parser.add_argument(
        "--enable-tile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tiled VAE decode (default: enabled).",
    )
    parser.add_argument(
        "--enable-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model-internal streaming/offload (default: disabled).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--image",
        "--images",
        dest="image",
        action="append",
        nargs="+",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Reference image path(s) for image-to-video. Accepts one or more "
            "paths and can be repeated for multi-ref workflows."
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".tmp" / "joyai-video.mp4")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--base-url", default=None, help="Default: first api_interface in config.json.")
    parser.add_argument("--api-key", default=None, help="Default: first access token in config.json.")
    parser.add_argument("--timeout", type=float, default=7200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    token = args.api_key or token_from_config(config)
    base_url = args.base_url or base_url_from_config(config)
    payload = build_payload(args)

    try:
        status, response = post_http(
            base_url=base_url,
            token=token,
            payload=payload,
            timeout=args.timeout,
        )
    except OSError as exc:
        raise SystemExit(f"fake-ollama API call failed: {exc}") from exc
    if status >= 400:
        raise SystemExit(json.dumps(response, ensure_ascii=False, indent=2))
    video, filename, mime_type = decode_video(response)
    output = write_output(args.output, video)
    print(json.dumps({
        "status": status,
        "base_url": base_url,
        "reference_images": len(payload.get("images") or []),
        "mime_type": mime_type,
        "filename": filename,
        "output": str(output),
        "bytes": len(video),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
