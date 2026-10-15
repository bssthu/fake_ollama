"""Exercise the public endpoint through the real workflow client (mock transport)."""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.config import Settings
from fake_ollama.server import create_app


def encoded(data=b"image"):
    return base64.b64encode(data).decode()


@pytest.fixture
def gateway():
    settings = Settings(
        comfyui_targets=[dict(name="h3", model="h3", preset="minimax_h3", base_url="http://comfy.test",
                             auto_start=False, max_reference_images=9, min_width=32, min_height=32,
                             width_modulo=32, height_modulo=32, default_width=256, default_height=256,
                             min_num_frames=5, default_num_frames=22, max_num_frames=362,
                             num_frames_modulo=17, num_frames_offset=5,
                             min_frame_rate=24, max_frame_rate=24)],
        api_interfaces=[dict(name="api", host="127.0.0.1", port=21435, access_tokens=["tk"],
                             exposed_models=[dict(model="h3", target="h3", alias="h3")])],
        model_profiles=[dict(model="h3", target="h3", capabilities=["video_generation"])],
        playground_enabled=True, playground_port=21431,
    )
    observed = {"prompts": [], "uploads": [], "guide_available": True, "paths": []}

    def transport(request):
        path = request.url.path
        observed["paths"].append(path)
        if path == "/system_stats":
            return httpx.Response(200, json={"devices": []})
        if path == "/object_info/MiniMaxH3AddGuide":
            return httpx.Response(200, json={"MiniMaxH3AddGuide": {}} if observed["guide_available"] else {})
        if path == "/upload/image":
            observed["uploads"].append(request.content)
            suffix = "mp4" if b"video/mp4" in request.content else "png"
            return httpx.Response(200, json={"name": f"u{len(observed['uploads'])}.{suffix}", "subfolder": "session"})
        if path == "/prompt":
            observed["prompts"].append(json.loads(request.content)["prompt"])
            return httpx.Response(200, json={"prompt_id": "p"})
        if path == "/history/p":
            return httpx.Response(200, json={"p": {"status": {"completed": True}, "outputs": {
                "9": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}}})
        if path == "/view":
            return httpx.Response(200, content=b"mp4", headers={"Content-Type": "video/mp4"})
        return httpx.Response(404)

    app = create_app(settings)
    client = ComfyUIClient("http://comfy.test", workflow_config=settings.comfyui_targets[0].workflow_config(),
                           client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    app.state.comfyui_clients = {"h3": client}
    with TestClient(app, base_url="http://testserver:21431", headers={"x-api-key": "tk"}) as http:
        yield http, observed


def generate(http, **kwargs):
    return http.post("/v1/videos/generations", json={
        "model": "h3", "prompt": "Use Picture 1 and Video 1.", "prompt_mode": "raw", "video_mode": "ref2va",
        "images": [encoded()], **kwargs,
    })


def test_ref2va_routes_image_video_and_chained_timeline_guides(gateway):
    http, seen = gateway
    response = generate(http, videos=[encoded(b"video")], video_filenames=["motion.mp4"], seed=7, n=2,
                        keyframes=[{"time_seconds": 0.25, "image": encoded(b"middle")},
                                   {"frame_index": -1, "image": encoded(b"last")}])
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2
    assert len(seen["uploads"]) == 4  # uploaded once, reused for independent seeds
    graph, second = seen["prompts"]
    assert graph["1"]["inputs"]["unet_name"].startswith("minimax_h3_ref2va_")
    assert graph["5"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    refs = graph["5"]["inputs"]
    assert graph[refs["ref_images.ref_image_0"][0]]["inputs"]["image"] == "session/u1.png"
    components = graph[refs["ref_videos.ref_video_0"][0]]
    assert components["class_type"] == "GetVideoComponents"
    assert graph[components["inputs"]["video"][0]]["inputs"]["file"] == "session/u2.mp4"
    guides = [(node, value["inputs"]) for node, value in graph.items() if value["class_type"] == "MiniMaxH3AddGuide"]
    assert [value["frame_idx"] for _, value in guides] == [6, 21]
    assert guides[0][1]["positive"] == ["5", 0]
    assert guides[1][1]["positive"] == [guides[0][0], 0]
    assert graph["7"]["inputs"]["conditioning"] == [guides[1][0], 0]
    assert graph["11"]["inputs"]["latent_image"] == ["5", 1]
    assert second["6"]["inputs"]["noise_seed"] == 8


def test_ref2va_accepts_video_only_and_all_nine_images(gateway):
    http, seen = gateway
    assert generate(http, images=[], videos=[encoded(b"video")]).status_code == 200
    assert generate(http, images=[encoded()] * 9).status_code == 200
    refs = seen["prompts"][-1]["5"]["inputs"]
    assert len([name for name in refs if name.startswith("ref_images.")]) == 9


def test_multipart_separates_reference_media_from_keyframes(gateway):
    http, seen = gateway
    response = http.post("/v1/videos/generations", data={
        "model": "h3", "prompt": "move", "video_mode": "ref2va", "prompt_mode": "raw",
        "keyframes": json.dumps([{"image_index": 1, "frame_index": 4}, {"image_field": "guide", "frame_index": -1}]),
    }, files=[("image[]", ("ref.png", b"ref", "image/png")),
              ("image[]", ("middle.png", b"mid", "image/png")),
              ("video[]", ("ref.mp4", b"vid", "video/mp4")),
              ("guide", ("last.png", b"last", "image/png"))])
    assert response.status_code == 200, response.text
    refs = seen["prompts"][0]["5"]["inputs"]
    assert "ref_images.ref_image_1" not in refs
    assert len(seen["uploads"]) == 4


@pytest.mark.parametrize("mode,count,node,slots", [
    ("auto", 0, "MiniMaxH3ImageToVideo", []),
    ("i2va", 1, "MiniMaxH3ImageToVideo", ["first_frame"]),
    ("fl2va", 2, "MiniMaxH3ImageToVideo", ["first_frame", "last_frame"]),
    ("l2va", 1, "MiniMaxH3ImageToVideo", ["last_frame"]),
])
def test_base_modes_keep_their_weights_and_slots(gateway, mode, count, node, slots):
    http, seen = gateway
    response = generate(http, video_mode=mode, images=[encoded()] * count,
                        keyframes=[{"frame_index": 10, "image": encoded()}])
    assert response.status_code == 200, response.text
    graph = seen["prompts"][0]
    assert graph["1"]["inputs"]["unet_name"].startswith("minimax_h3_fl2va_")
    assert graph["5"]["class_type"] == node
    assert set(graph["5"]["inputs"]) & {"first_frame", "last_frame"} == set(slots)


@pytest.mark.parametrize("override", [
    {"images": []}, {"images": [encoded()] * 10}, {"videos": [encoded()] * 4},
    {"video_mode": "auto", "images": [encoded()] * 3},
    {"video_mode": "auto", "videos": [encoded()]},
    {"prompt_mode": "auto"}, {"prompt_mode": "enhance"},
    {"context_ir_mode": "i2va"}, {"videos": ["not-base64"]}, {"videos": {}},
    {"images": [{}], "videos": [encoded()]}, {"fps": "NaN"}, {"fps": "Infinity"},
    {"keyframes": [{"frame_index": 22, "image": encoded()}]},
    {"keyframes": [{"frame_index": -23, "image": encoded()}]},
    {"keyframes": [{"time_seconds": -1, "image": encoded()}]},
    {"keyframes": [{"frame_index": 1.5, "image": encoded()}]},
    {"keyframes": [{"frame_index": True, "image": encoded()}]},
    {"keyframes": [{"frame_index": 1, "time_seconds": 1, "image": encoded()}]},
    {"keyframes": [{"frame_index": 1, "image_index": 9}]},
    {"keyframes": [{"frame_index": 1, "image": encoded(), "image_index": 0}]},
    {"keyframes": [{"frame_index": -1, "image": encoded()}, {"frame_index": 21, "image": encoded()}]},
    {"keyframes": [{"frame_index": i, "image": encoded()} for i in range(17)]},
    {"video_mode": "i2va", "keyframes": [{"frame_index": 0, "image": encoded()}]},
])
def test_invalid_inputs_fail_before_upload_or_submission(gateway, override):
    http, seen = gateway
    response = generate(http, **override)
    assert response.status_code == 400, response.text
    assert seen["prompts"] == []
    assert seen["uploads"] == []


def test_missing_guide_node_has_actionable_error_and_no_uploads(gateway):
    http, seen = gateway
    seen["guide_available"] = False
    response = generate(http, keyframes=[{"frame_index": 4, "image": encoded()}])
    assert response.status_code == 502, response.text
    assert "MiniMaxH3AddGuide" in response.text
    assert seen["uploads"] == []


def test_discovery_exposes_reference_mode_videos_and_keyframes(gateway):
    http, _ = gateway
    response = http.get("/playground/api/models")
    op = response.json()["models"][0]["operations"][0]
    assert "ref2va" in op["workflow_modes"]
    assert op["accepts_videos"] and op["multiple_videos"]
    assert op["limits"]["max_reference_images"] == 9
    assert op["limits"]["max_videos"] == 3
    params = {item["name"]: item for item in op["parameters"]}
    assert params["keyframes"]["default"] == "[]"
    assert "ref2va" in [choice["value"] for choice in params["video_mode"]["choices"]]
