"""Example client for JoyAI VL recognition through fake_ollama."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shlex
import shutil
import subprocess
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


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


def guess_mime_type(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type:
        return mime_type
    suffix = path.suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mkv":
        return "video/x-matroska"
    return "application/octet-stream"


def encode_file_data_uri(path: Path, *, mime_type: str | None = None) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(f"input file does not exist: {path}")
    mime_type = mime_type or guess_mime_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", mime_type


def media_kind_from_mime(mime_type: str) -> str:
    if mime_type == "image/gif":
        return "video"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("image/"):
        return "image"
    return "video"


def image_content_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def parse_command(command: str) -> list[str]:
    return shlex.split(command, posix=(os.name != "nt"))


def windows_path_to_wsl(path: Path) -> str:
    resolved = str(path.resolve())
    drive, rest = os.path.splitdrive(resolved)
    if not drive:
        return resolved.replace("\\", "/")
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive_letter}{rest}"


def command_uses_wsl(command: list[str]) -> bool:
    if not command:
        return False
    return Path(command[0]).name.lower() in {"wsl", "wsl.exe"}


def joyai_wsl_distro_from_config(config: dict[str, Any], model: str) -> str | None:
    targets = config.get("generic_openai_targets") or []
    for target in targets:
        models = target.get("models") or []
        names = {str(item.get("name") or "") for item in models if isinstance(item, dict)}
        aliases = {str(item.get("alias") or "") for item in models if isinstance(item, dict)}
        if model not in names and model not in aliases:
            continue
        start_command = str(target.get("start_command") or "")
        parts = parse_command(start_command)
        for idx, part in enumerate(parts[:-1]):
            if part == "-d":
                return parts[idx + 1]
    return None


def default_ffmpeg_command(config: dict[str, Any], model: str) -> list[str]:
    local = shutil.which("ffmpeg")
    if local:
        return [local]
    if os.name == "nt" and shutil.which("wsl"):
        distro = joyai_wsl_distro_from_config(config, model) or "Ubuntu-24.04"
        return ["wsl", "-d", distro, "--exec", "ffmpeg"]
    return ["ffmpeg"]


def default_ffprobe_command(ffmpeg_command: list[str]) -> list[str]:
    if command_uses_wsl(ffmpeg_command):
        command = list(ffmpeg_command)
        command[-1] = "ffprobe"
        return command
    executable = Path(ffmpeg_command[0]).name.lower()
    if executable in {"ffmpeg", "ffmpeg.exe"}:
        ffprobe = shutil.which("ffprobe")
        return [ffprobe] if ffprobe else ["ffprobe"]
    sibling = Path(ffmpeg_command[0]).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if sibling.exists():
        return [str(sibling)]
    return ["ffprobe"]


def path_arg_for_command(path: Path, command: list[str]) -> str:
    if command_uses_wsl(command) and os.name == "nt":
        return windows_path_to_wsl(path)
    return str(path)


def probe_duration_seconds(path: Path, ffprobe_command: list[str]) -> float | None:
    path_arg = path_arg_for_command(path, ffprobe_command)
    command = [
        *ffprobe_command,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path_arg,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def extract_frames(
    path: Path,
    *,
    config: dict[str, Any],
    model: str,
    ffmpeg_command_arg: str | None,
    max_frames: int,
    frame_width: int,
) -> list[Path]:
    if max_frames <= 0:
        raise SystemExit("--max-frames must be greater than 0 for video/GIF input")

    ffmpeg_command = (
        parse_command(ffmpeg_command_arg)
        if ffmpeg_command_arg
        else default_ffmpeg_command(config, model)
    )
    ffprobe_command = default_ffprobe_command(ffmpeg_command)
    duration = probe_duration_seconds(path, ffprobe_command)
    fps = 1.0 if duration is None else max(1.0 / max(duration, 1.0), max_frames / duration)
    fps = min(fps, 8.0)

    frame_dir = ROOT / ".tmp" / "joyai-vl-frames" / f"{path.stem}-{uuid.uuid4().hex[:8]}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frame_dir / "frame-%03d.jpg"
    input_arg = path_arg_for_command(path, ffmpeg_command)
    output_arg = path_arg_for_command(output_pattern, ffmpeg_command)
    scale_expr = f"scale=min({frame_width}\\,iw):-2"
    command = [
        *ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_arg,
        "-vf",
        f"fps={fps:.6f},{scale_expr}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        output_arg,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "ffmpeg was not found; install ffmpeg or pass --ffmpeg-command "
            "for example: --ffmpeg-command \"wsl -d Ubuntu-24.04 -- ffmpeg\""
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise SystemExit(f"ffmpeg failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("ffmpeg timed out while extracting frames") from exc

    frames = sorted(frame_dir.glob("frame-*.jpg"))
    if not frames:
        raise SystemExit("ffmpeg did not extract any frames")
    return frames


def extract_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def normalize_joyai_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("</response>"):
        return stripped[len("</response>"):].strip()
    return stripped


def build_payload(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    mime_type = guess_mime_type(args.input)
    kind = args.media_kind if args.media_kind != "auto" else media_kind_from_mime(mime_type)
    content: list[dict[str, Any]] = [{"type": "text", "text": args.prompt}]

    frame_count = 0
    if kind == "image":
        media_url, _ = encode_file_data_uri(args.input, mime_type=mime_type)
        content.append(image_content_part(media_url))
    elif kind == "video":
        frames = extract_frames(
            args.input,
            config=config,
            model=args.model,
            ffmpeg_command_arg=args.ffmpeg_command,
            max_frames=args.max_frames,
            frame_width=args.frame_width,
        )
        try:
            frame_count = len(frames)
            content.append({
                "type": "text",
                "text": (
                    f"\n\nThe following {frame_count} images are sampled frames "
                    f"from {args.input.name}, in chronological order."
                ),
            })
            for frame in frames:
                frame_url, _ = encode_file_data_uri(frame, mime_type="image/jpeg")
                content.append(image_content_part(frame_url))
        finally:
            if frames and not args.keep_frames:
                shutil.rmtree(frames[0].parent, ignore_errors=True)
    else:
        raise SystemExit(f"unsupported media kind: {args.media_kind}")

    messages: list[dict[str, Any]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": content})

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "user": args.session_id,
    }
    if frame_count:
        payload["metadata"] = {"sampled_frame_count": frame_count}
    return payload


def post_http(
    *,
    base_url: str,
    token: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": token,
        },
        method="POST",
    )
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


def write_output(path: Path, response: dict[str, Any], text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call the running fake-ollama /v1/chat/completions API with a "
            "video/GIF/image file for JoyAI-VL-Interaction recognition."
        )
    )
    parser.add_argument("input", type=Path, help="Input video, GIF, or image file.")
    parser.add_argument(
        "--prompt",
        default=(
            "Describe this media in detail, including scene, subjects, actions, "
            "timeline changes, and any visible text."
        ),
        help="Recognition prompt sent with the media file.",
    )
    parser.add_argument("--system", default=None, help="Optional system message.")
    parser.add_argument("--model", default="joyai-vl-interaction")
    parser.add_argument(
        "--media-kind",
        choices=["auto", "video", "image"],
        default="auto",
        help="Default: extract frames for video files/GIFs, send still images directly.",
    )
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--frame-width", type=int, default=512)
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted JPEG frames under .tmp/joyai-vl-frames for debugging.",
    )
    parser.add_argument(
        "--ffmpeg-command",
        default=None,
        help=(
            "Command used to extract video/GIF frames. Default: local ffmpeg, "
            "or WSL ffmpeg using the JoyAI target distro from config.json."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--session-id",
        default=None,
        help="JoyAI adapter session id. Default: unique session per script run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--base-url", default=None, help="Default: first api_interface in config.json.")
    parser.add_argument("--api-key", default=None, help="Default: first access token in config.json.")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames > 32:
        raise SystemExit(
            "--max-frames cannot exceed 32 for the current JoyAI vLLM service "
            "(start_joyai_vl_interaction.sh sets --limit-mm-per-prompt "
            "'{\"image\":32,\"video\":1}')."
        )
    if not args.session_id:
        args.session_id = f"joyai-vl-{uuid.uuid4().hex[:12]}"
    config = load_config(args.config)
    token = args.api_key or token_from_config(config)
    base_url = args.base_url or base_url_from_config(config)
    payload = build_payload(args, config)

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
        raise SystemExit(json.dumps({
            "status": status,
            "base_url": base_url,
            "response": response,
        }, ensure_ascii=False, indent=2))

    text = normalize_joyai_text(extract_message_text(response))
    if args.output:
        write_output(args.output, response, text)

    print(json.dumps({
        "status": status,
        "base_url": base_url,
        "model": args.model,
        "session_id": args.session_id,
        "input": str(args.input),
        "output": str(args.output) if args.output else None,
        "text": text,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
