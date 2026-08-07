"""Example client for fake_ollama's Mage-VL integration."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _config_defaults(path: Path) -> tuple[str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    interfaces = config.get("api_interfaces") or []
    if not interfaces:
        raise SystemExit("config.json has no api_interfaces entry")
    interface = interfaces[0]
    host = str(interface.get("host") or "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    base_url = f"http://{host}:{int(interface.get('port') or 21435)}"
    tokens = interface.get("access_tokens") or []
    return base_url, (str(tokens[0]) if tokens else "")


def _video_data_url(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"video does not exist: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _post(url: str, token: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["x-api-key"] = token
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Mage-VL request failed ({exc.code}): {detail}") from exc


def _answer(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError):
        raise SystemExit(
            "Mage-VL returned an unexpected response:\n"
            + json.dumps(response, ensure_ascii=False, indent=2)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--prompt",
        default="请按时间顺序描述视频中的场景、主体、动作变化、关键事件和可见文字。",
    )
    parser.add_argument("--model", default="mage-vl-local")
    parser.add_argument("--segment-seconds", type=float, default=8)
    parser.add_argument("--frames-per-segment", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--include-summary", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_url, default_token = _config_defaults(args.config)
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": _video_data_url(args.video)},
                    },
                ],
            }
        ],
        "stream": False,
        "segment_seconds": args.segment_seconds,
        "frames_per_segment": args.frames_per_segment,
        "max_segments": args.max_segments,
        "include_summary": args.include_summary,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    response = _post(
        args.base_url or default_url,
        default_token if args.api_key is None else args.api_key,
        payload,
        args.timeout,
    )
    answer = _answer(response)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(answer, encoding="utf-8")
    print(answer)


if __name__ == "__main__":
    main()
