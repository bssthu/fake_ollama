"""Offline lifecycle contracts; no real HTTP listener, model, or shutdown."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mage_vl_adapter import api, control, server
from mage_vl_adapter.settings import AdapterSettings


TOKEN = "t" * 43
STOP_SCRIPT = Path(__file__).parents[1] / "scripts" / "stop.ps1"


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    app = api.create_app(AdapterSettings(temp_root=tmp_path), engine=SimpleNamespace())
    app.state.shutdown_token = TOKEN
    app.state.uvicorn_server = SimpleNamespace(should_exit=False)
    callbacks = []
    loop = SimpleNamespace(call_later=lambda delay, callback: callbacks.append((delay, callback)))
    monkeypatch.setattr(api, "asyncio", SimpleNamespace(get_running_loop=lambda: loop))
    return app, callbacks


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_local_authenticated_shutdown_schedules_graceful_exit(local_app, host):
    app, callbacks = local_app
    # Starlette's installed TestClient cannot parse an IPv6 URL authority;
    # supply its real Host header separately while retaining the socket port.
    with TestClient(app, base_url="http://localhost:8071", client=("::1" if host == "[::1]" else "127.0.0.1", 50000)) as client:
        response = client.post("/shutdown", headers={"Authorization": f"Bearer {TOKEN}", "Host": f"{host}:8071"})
    assert response.status_code == 200
    assert TOKEN not in response.text
    assert not app.state.uvicorn_server.should_exit
    assert len(callbacks) == 1 and callbacks[0][0] == 0.2
    callbacks[0][1]()
    assert app.state.uvicorn_server.should_exit


@pytest.mark.parametrize("peer,headers,initialized", [
    ("127.0.0.1", {}, True),
    ("127.0.0.1", {"Authorization": "Bearer wrong"}, True),
    ("127.0.0.1", {"Authorization": f"Bearer {TOKEN}"}, False),
    ("testclient", {"Authorization": f"Bearer {TOKEN}"}, True),
    ("192.0.2.1", {"Authorization": f"Bearer {TOKEN}", "X-Forwarded-For": "127.0.0.1"}, True),
    ("127.0.0.1", {"Authorization": f"Bearer {TOKEN}", "Origin": "null"}, True),
    ("127.0.0.1", {"Authorization": f"Bearer {TOKEN}", "Sec-Fetch-Site": "cross-site"}, True),
    ("127.0.0.1", {"Authorization": f"Bearer {TOKEN}", "Host": "unrelated.example:8071"}, True),
    ("127.0.0.1", {"Authorization": f"Bearer {TOKEN}", "Host": "localhost:9999"}, True),
])
def test_shutdown_requires_credentials_and_local_authority(local_app, peer, headers, initialized):
    app, callbacks = local_app
    if not initialized:
        app.state.shutdown_token = None
    with TestClient(app, base_url="http://127.0.0.1:8071", client=(peer, 50000)) as client:
        response = client.post("/shutdown", headers=headers)
    assert response.status_code == 403
    assert not callbacks
    assert TOKEN not in response.text


def test_shutdown_without_runtime_owner_fails(local_app):
    app, callbacks = local_app
    app.state.uvicorn_server = None
    with TestClient(app, base_url="http://localhost:8071", client=("::1", 50000)) as client:
        response = client.post("/shutdown", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 503
    assert not callbacks


def test_token_rotation_cleanup_and_exclusive_port(tmp_path, monkeypatch):
    directory = tmp_path / "control"
    monkeypatch.setattr(control, "control_directory", lambda: directory)
    path = directory / "shutdown-8071.token"
    with control.shutdown_credentials(8071) as first:
        assert len(first) == 43 and path.read_text() == first
        with pytest.raises(RuntimeError, match="already owned"):
            with control.shutdown_credentials(8071):
                pytest.fail("duplicate control owner")
        assert path.read_text() == first
        with control.shutdown_credentials(8072) as other:
            assert other != first
    assert not path.exists()
    assert not list(directory.glob("*.token"))
    path.write_text("stale-token")
    with pytest.raises(RuntimeError, match="runtime failed"):
        with control.shutdown_credentials(8071) as second:
            assert second != first and path.read_text() == second
            raise RuntimeError("runtime failed")
    assert not path.exists()
    assert not list(directory.glob(".shutdown-*"))


def test_permission_failure_does_not_publish_token(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "control_directory", lambda: tmp_path)
    def deny(_):
        raise PermissionError("cannot restrict directory")
    monkeypatch.setattr(control, "_private_directory", deny)
    with pytest.raises(PermissionError):
        with control.shutdown_credentials(8071):
            pytest.fail("insecure directory accepted")
    assert not list(tmp_path.iterdir())


def test_token_directory_and_file_are_private(tmp_path, monkeypatch):
    directory = tmp_path / "control"
    monkeypatch.setattr(control, "control_directory", lambda: directory)
    with control.shutdown_credentials(8071):
        path = directory / "shutdown-8071.token"
        if os.name == "nt":
            result = _powershell("""$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$directory = Get-Acl -LiteralPath $env:MAGE_TEST_DIRECTORY
$file = Get-Acl -LiteralPath (Join-Path $env:MAGE_TEST_DIRECTORY 'shutdown-8071.token')
if (-not $directory.AreAccessRulesProtected) { throw 'inherited directory ACL' }
foreach ($acl in @($directory, $file)) {
    $rules = $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])
    if ($rules.Count -ne 1 -or $rules[0].IdentityReference.Value -ne $sid -or
        $rules[0].AccessControlType -ne 'Allow') { throw 'non-user access rule' }
}
""", {"MAGE_TEST_DIRECTORY": str(directory)})
            assert result.returncode == 0, result.stderr
        else:
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("bind_fails,run_fails", [(False, False), (True, False), (False, True)])
def test_entrypoint_owns_token_only_after_binding(monkeypatch, bind_fails, run_fails):
    app = SimpleNamespace(state=SimpleNamespace())
    events = []
    socket = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 8071), close=lambda: events.append("close"))
    def config(app_arg, **kwargs):
        assert app_arg is app and kwargs["proxy_headers"] is False
        def bind():
            events.append("bind")
            if bind_fails:
                raise OSError("port in use")
            return socket
        return SimpleNamespace(bind_socket=bind)
    @contextmanager
    def credentials(port):
        assert port == 8071
        events.append("publish")
        try:
            yield TOKEN
        finally:
            events.append("remove")
    def run(*, sockets):
        assert sockets == [socket] and app.state.shutdown_token == TOKEN
        events.append("run")
        if run_fails:
            raise RuntimeError("startup failed")
    monkeypatch.setattr(server, "create_app", lambda: app)
    monkeypatch.setattr(server, "shutdown_credentials", credentials)
    monkeypatch.setattr("uvicorn.Config", config)
    monkeypatch.setattr("uvicorn.Server", lambda _: SimpleNamespace(run=run))
    if bind_fails or run_fails:
        with pytest.raises((OSError, RuntimeError)):
            server.main([])
    else:
        assert server.main([]) == 0
    assert events == (["bind"] if bind_fails else ["bind", "publish", "run", "remove", "close"])
    if not bind_fails:
        assert app.state.shutdown_token is None


def _powershell(script, env):
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "$ErrorActionPreference = 'Stop'\n" + script],
        env={**os.environ, **env}, capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


# Load real script functions without executing its entrypoint or opening HTTP.
LOAD_STOP_FUNCTIONS = """$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:MAGE_TEST_STOP_SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'stop.ps1 parse error' }
foreach ($statement in $ast.EndBlock.Statements) {
    if ($statement -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
        Invoke-Expression $statement.Extent.Text
    }
}
"""


@pytest.mark.skipif(os.name != "nt", reason="Windows lifecycle script")
@pytest.mark.parametrize("has_token,post_status", [(True, 200), (False, 200), (False, 403), (True, 403)])
def test_stop_script_discovers_token_and_preserves_upgrade_flow(tmp_path, has_token, post_status):
    if has_token:
        directory = tmp_path / "fake-ollama" / "mage-vl" / "control"
        directory.mkdir(parents=True)
        (directory / "shutdown-18071.token").write_text(TOKEN)
    script = LOAD_STOP_FUNCTIONS + """$script:requests = @()
function Invoke-MageControlRequest {
    param([uri]$Uri, [string]$Method = 'GET', [string]$Token = '', [int]$TimeoutSeconds = 2)
    if ($Uri.Port -ne 18071) { throw 'wrong port' }
    $script:requests += $Method
    if ($Method -eq 'POST') {
        if ($Token -cne $env:MAGE_TEST_EXPECT_TOKEN) { throw 'wrong token' }
        return [int]$env:MAGE_TEST_POST_STATUS
    }
    if ($env:MAGE_TEST_POST_STATUS -eq '403') { return 503 }
    throw [System.Net.WebException]::new('closed', [System.Net.WebExceptionStatus]::ConnectFailure)
}
$message = ''
try { $message = Stop-MageService -BaseUrl 'http://localhost:18071' -TimeoutSeconds 5 }
catch { $message = $_.Exception.Message }
@{ message = $message; requests = $script:requests } | ConvertTo-Json -Compress
"""
    result = _powershell(script, {
        "MAGE_TEST_STOP_SCRIPT": str(STOP_SCRIPT), "LOCALAPPDATA": str(tmp_path),
        "MAGE_TEST_EXPECT_TOKEN": TOKEN if has_token else "",
        "MAGE_TEST_POST_STATUS": str(post_status),
    })
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["requests"] == ["POST", "GET"]
    assert ("stopped gracefully" if post_status == 200 else "HTTP 403") in output["message"]
    assert TOKEN not in result.stdout + result.stderr


@pytest.mark.skipif(os.name != "nt", reason="Windows lifecycle script")
def test_stop_script_does_not_confuse_timeouts_with_exit(tmp_path):
    result = _powershell(LOAD_STOP_FUNCTIONS + """
function Invoke-MageControlRequest {
    throw [System.Net.WebException]::new('slow', [System.Net.WebExceptionStatus]::Timeout)
}
if (-not (Test-MageRunning -Uri 'http://localhost:8071/health')) { throw 'timeout mistaken for exit' }
$rejected = $false
try { Stop-MageService -BaseUrl 'http://192.0.2.1:8071' -TimeoutSeconds 0 }
catch { if ($_.Exception.Message -match 'local HTTP') { $rejected = $true } else { throw } }
if (-not $rejected) { throw 'non-local endpoint accepted' }
""", {"MAGE_TEST_STOP_SCRIPT": str(STOP_SCRIPT), "LOCALAPPDATA": str(tmp_path)})
    assert result.returncode == 0, result.stderr
