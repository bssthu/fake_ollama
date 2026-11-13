"""Download publication must follow verified content, including sparse resumes."""
import hashlib
import json

import httpx
import pytest
import respx

from scripts import setup_h3_bf16 as setup


@pytest.fixture
def artifact(monkeypatch):
    name = "diffusion_models/example.safetensors"
    data = b"bf16-weight-test-content"
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(setup, "FILES", {name: (len(data), checksum)})
    url = f"https://huggingface.co/{setup.REPO}/resolve/{setup.REVISION}/{name}"
    return name, data, checksum, url


def test_resume_uses_recorded_prefix_not_sparse_file_size(tmp_path, artifact):
    name, data, checksum, url = artifact
    dest = tmp_path / name
    dest.parent.mkdir()
    part = dest.with_suffix(dest.suffix + ".partial")
    part.write_bytes(data[:5] + b"\0" * (len(data) - 5))
    state = part.with_suffix(part.suffix + ".json")
    state.write_text(json.dumps({"sha256": checksum, "bytes": len(data), "prefix": 5, "done": []}))
    with respx.mock as router:
        route = router.get(url, headers={"Range": f"bytes=5-{len(data)-1}"}).respond(
            206, content=data[5:], headers={"content-range": f"bytes 5-{len(data)-1}/{len(data)}"}
        )
        setup.download(tmp_path, name)
        assert route.call_count == 1
    assert dest.read_bytes() == data
    assert not part.exists()
    assert not state.exists()


def test_bad_checksum_is_not_published(tmp_path, artifact):
    name, data, _, url = artifact
    with respx.mock as router:
        router.get(url).respond(206, content=b"x" * len(data), headers={
            "content-range": f"bytes 0-{len(data)-1}/{len(data)}"
        })
        with pytest.raises(ValueError, match="Checksum mismatch"):
            setup.download(tmp_path, name)
    assert not (tmp_path / name).exists()
    assert (tmp_path / (name + ".partial")).exists()


def test_server_ignoring_range_cannot_corrupt_resume(tmp_path, artifact):
    name, data, _, url = artifact
    part = tmp_path / (name + ".partial")
    part.parent.mkdir()
    part.write_bytes(data[:5])
    with respx.mock as router:
        router.get(url).respond(200, content=data)
        with pytest.raises(ValueError, match="honor range"):
            setup.download(tmp_path, name)
    assert part.read_bytes() == data[:5]
    assert not (tmp_path / name).exists()


def test_existing_different_file_is_never_overwritten(tmp_path, artifact):
    name, data, _, _ = artifact
    dest = tmp_path / name
    dest.parent.mkdir()
    original = b"x" * len(data)
    dest.write_bytes(original)
    with respx.mock:
        with pytest.raises(ValueError, match="refusing to overwrite"):
            setup.download(tmp_path, name)
    assert dest.read_bytes() == original


def test_direct_cdn_uses_official_redirect_and_verifies_content(tmp_path, artifact):
    name, data, _, url = artifact
    with respx.mock as router:
        redirect = router.head(url).respond(302, headers={"location": "https://cdn.example/weight"})
        content = router.get("https://cdn.example/weight").respond(206, content=data, headers={
            "content-range": f"bytes 0-{len(data)-1}/{len(data)}"
        })
        setup.download(tmp_path, name, direct_cdn=True)
        assert redirect.called and content.called
    assert (tmp_path / name).read_bytes() == data
