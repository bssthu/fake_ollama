"""H3 reference media and timeline guides, independent of prompt planning."""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any

MAX_REFERENCE_VIDEOS = 3
MAX_KEYFRAMES = 16


def decode_media(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must contain base64 data")
    value = value.strip()
    if value.startswith("data:"):
        value = value.partition(",")[2]
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"invalid base64 {label}") from exc
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


async def video_media_from_request(request, payload, image_inputs, *, frames, fps):
    """Parse JSON or multipart references; guide images have a separate role.

    A guide's image_index addresses the original image upload list, and removes
    that image from ordinary first/last/reference inputs. Multipart guides may
    instead use an image_field naming an uploaded file.
    """
    videos = []
    files = {}
    if (request.headers.get("content-type") or "").startswith("multipart/"):
        form = await request.form()
        for key, value in form.multi_items():
            if not hasattr(value, "read"):
                continue
            if key == "video" or key.startswith("video["):
                videos.append((await value.read(), value.filename or "input.mp4"))
            elif not (key == "image" or key.startswith("image[")):
                files[key] = value
    else:
        raw = []
        for field in ("video", "videos"):
            values = payload.get(field, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ValueError("videos must be a list of base64 strings")
            raw.extend(values)
        for field in ("image", "images"):
            values = payload.get(field, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError("images must contain base64 strings")
        names = payload.get("video_filenames", [])
        if not isinstance(names, list):
            raise ValueError("video_filenames must be a list")
        for i, value in enumerate(raw):
            name = names[i] if i < len(names) else f"input-{i + 1}.mp4"
            videos.append((decode_media(value, "video"), str(name)))
    if len(videos) > MAX_REFERENCE_VIDEOS:
        raise ValueError("Ref2VA accepts at most 3 reference videos")
    if any(not data for data, _ in videos):
        raise ValueError("reference videos must not be empty")

    raw_guides = payload.get("keyframes", [])
    if isinstance(raw_guides, str):
        try:
            raw_guides = json.loads(raw_guides)
        except json.JSONDecodeError as exc:
            raise ValueError("keyframes must be a JSON array") from exc
    if not isinstance(raw_guides, list) or len(raw_guides) > MAX_KEYFRAMES:
        raise ValueError(f"keyframes must be an array of at most {MAX_KEYFRAMES} entries")
    guides, used_images, positions = [], set(), set()
    for i, guide in enumerate(raw_guides):
        if not isinstance(guide, dict):
            raise ValueError("each keyframe must be an object")
        if set(guide) - {"frame_index", "time_seconds", "image", "image_index", "image_field", "filename"}:
            raise ValueError("unknown keyframe field")
        if ("frame_index" in guide) == ("time_seconds" in guide):
            raise ValueError("each keyframe needs exactly one of frame_index or time_seconds")
        if "time_seconds" in guide:
            seconds = guide["time_seconds"]
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("keyframe time_seconds must be a finite nonnegative number")
            if seconds * fps >= frames:
                raise ValueError("keyframe time_seconds is outside the video")
            index = math.floor(seconds * fps + 0.5)
        else:
            index = guide["frame_index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("keyframe frame_index must be an integer")
            if index < 0:
                index += frames
        if not 0 <= index < frames:
            raise ValueError(f"keyframe frame_index is outside the video's {frames} frames")
        if index in positions:
            raise ValueError("duplicate keyframe position")
        positions.add(index)
        sources = [name for name in ("image", "image_index", "image_field") if name in guide]
        if len(sources) != 1:
            raise ValueError("each keyframe needs exactly one image, image_index or image_field")
        if "image_index" in guide:
            source = guide["image_index"]
            if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < len(image_inputs):
                raise ValueError("keyframe image_index is outside the image upload list")
            data, name = image_inputs[source]
            used_images.add(source)
        elif "image_field" in guide:
            field = guide["image_field"]
            upload = files.get(field) if isinstance(field, str) else None
            if upload is None:
                raise ValueError("keyframe image_field must name a separate multipart file")
            await upload.seek(0)
            data, name = await upload.read(), upload.filename or f"guide-{i}.png"
        else:
            data = decode_media(guide["image"], "keyframe image")
            name = str(guide.get("filename") or f"guide-{i}.png")
        if not data:
            raise ValueError("keyframe images must not be empty")
        guides.append((index, data, name))
    refs = [image for i, image in enumerate(image_inputs) if i not in used_images]
    if any(not data for data, _ in refs):
        raise ValueError("reference images must not be empty")
    return refs, videos, guides


def add_reference_inputs(graph: dict, node: str, slot: str, names: list[str], *, video: bool) -> None:
    """Autogrow inputs are separate references, never an ImageBatch."""
    inputs = graph[node]["inputs"]
    for key in list(inputs):
        if key.startswith(slot + "."):
            del inputs[key]
    next_id = max((int(key) for key in graph if key.isdigit()), default=0) + 1
    for i, name in enumerate(names):
        load = str(next_id)
        next_id += 1
        graph[load] = {"class_type": "LoadVideo" if video else "LoadImage",
                       "inputs": {"file" if video else "image": name}}
        if video:
            components = str(next_id)
            next_id += 1
            graph[components] = {"class_type": "GetVideoComponents", "inputs": {"video": [load, 0]}}
            inputs[f"{slot}.ref_video_{i}"] = [components, 0]
        else:
            inputs[f"{slot}.ref_image_{i}"] = [load, 0]


def add_keyframes(graph: dict, guides: list[tuple[int, str]]) -> None:
    if not guides:
        return
    conditioners = [(key, value) for key, value in graph.items()
                    if value.get("class_type") in {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"}]
    if len(conditioners) != 1:
        raise ValueError("keyframes require a workflow with one native H3 conditioning node")
    source, conditioning = conditioners[0]
    positive = [source, 0]
    consumers = [node["inputs"] for node in graph.values()
                 if node.get("class_type") == "BasicGuider" and node.get("inputs", {}).get("conditioning") == positive]
    if not consumers:
        raise ValueError("keyframes require BasicGuider connected to H3 conditioning")
    next_id = max((int(key) for key in graph if key.isdigit()), default=0) + 1
    for index, name in guides:
        load, guide = str(next_id), str(next_id + 1)
        next_id += 2
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph[guide] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
            "positive": positive, "latent": [source, 1],
            "vae": conditioning["inputs"]["vae"], "image": [load, 0], "frame_idx": index,
        }}
        positive = [guide, 0]
    for inputs in consumers:
        inputs["conditioning"] = positive
