"""Run a real video through Playground discovery and fake_ollama's Mage proxy."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fake_ollama.config import load_settings
from fake_ollama.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--segment-seconds", type=float, default=8)
    parser.add_argument("--frames-per-segment", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"video does not exist: {args.video}")
    settings = load_settings(args.config)
    api_interface = settings.api_interfaces[0]
    if not api_interface.access_tokens:
        raise SystemExit("the first api interface has no access token")
    token = api_interface.access_tokens[0]
    headers = {"x-api-key": token}
    playground_port = int(settings.playground_port or 21431)
    api_port = int(api_interface.port)

    mime = mimetypes.guess_type(args.video.name)[0] or "video/mp4"
    data_url = "data:" + mime + ";base64," + base64.b64encode(
        args.video.read_bytes()
    ).decode("ascii")
    payload: dict[str, Any] = {
        "model": "mage-vl-local",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请按时间顺序描述视频中能确认的场景、主体和动作变化。",
                    },
                    {"type": "video_url", "video_url": {"url": data_url}},
                ],
            }
        ],
        "stream": args.stream,
        "segment_seconds": args.segment_seconds,
        "frames_per_segment": args.frames_per_segment,
        "max_segments": args.max_segments,
        "include_summary": False,
        "max_tokens": 96,
        "temperature": 0,
    }

    app = create_app(settings)
    with TestClient(app, base_url=f"http://testserver:{api_port}") as client:
        discovery = client.get(
            f"http://testserver:{playground_port}/playground/api/models",
            headers=headers,
        )
        discovery.raise_for_status()
        mage_entry = next(
            item for item in discovery.json()["models"] if item["id"] == "mage-vl-local"
        )
        operation = next(
            item for item in mage_entry["operations"] if item["id"] == "video_analysis"
        )
        request_url = f"http://testserver:{api_port}/v1/chat/completions"
        if args.stream:
            answer_parts: list[str] = []
            proxied_model = ""
            with client.stream(
                "POST", request_url, headers=headers, json=payload, timeout=1200
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    proxied_model = str(chunk.get("model") or proxied_model)
                    choices = chunk.get("choices") or []
                    if choices:
                        answer_parts.append(str(choices[0].get("delta", {}).get("content") or ""))
            answer = "".join(answer_parts)
        else:
            response = client.post(
                request_url, headers=headers, json=payload, timeout=1200
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            proxied_model = result["model"]

    summary = {
        "discovered_capabilities": mage_entry["capabilities"],
        "discovered_operation": operation,
        "stream": args.stream,
        "proxied_model": proxied_model,
        "answer": answer,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
