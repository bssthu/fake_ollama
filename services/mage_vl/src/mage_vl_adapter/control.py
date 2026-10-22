"""Per-start credentials for local lifecycle control, separate from model access."""

from __future__ import annotations

import ipaddress
import os
import secrets
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from fastapi import HTTPException, Request


def control_directory() -> Path:
    # Independent of model/install/temp overrides so stop.ps1 needs no new args.
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return root / "fake-ollama" / "mage-vl" / "control"


def _private_directory(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise RuntimeError("Mage-VL control directory must not be a link or reparse point")
    if os.name == "nt":
        # chmod does not restrict Windows read access on all supported Pythons.
        # Protect the directory before creating a secret; files inherit this ACL.
        script = """$ErrorActionPreference = 'Stop'
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = New-Object System.Security.AccessControl.DirectorySecurity
$acl.SetOwner($sid)
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $sid, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')
$acl.AddAccessRule($rule)
[System.IO.Directory]::SetAccessControl($env:MAGE_VL_PRIVATE_CONTROL_DIRECTORY, $acl)
"""
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**os.environ, "MAGE_VL_PRIVATE_CONTROL_DIRECTORY": str(directory)},
            check=True, capture_output=True, timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        if info.st_uid != os.getuid():
            raise RuntimeError("Mage-VL control directory belongs to another user")
        directory.chmod(0o700)


@contextmanager
def _control_lock(path: Path) -> Iterator[None]:
    # Separate from the readable token: a second process using the same control
    # port must not replace it, including Windows SO_REUSEADDR duplicate binds.
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Mage-VL control port is already owned by another instance") from None
        yield


@contextmanager
def shutdown_credentials(port: int) -> Iterator[str]:
    """Publish only after binding; rotate stale tokens and remove our own on exit."""
    directory = control_directory()
    _private_directory(directory)
    path = directory / f"shutdown-{port}.token"
    with _control_lock(directory / f"shutdown-{port}.lock"):
        with _published_token(path) as token:
            yield token


@contextmanager
def _published_token(path: Path) -> Iterator[str]:
    token = secrets.token_urlsafe(32)
    fd, temporary = tempfile.mkstemp(prefix=".shutdown-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(token)
        os.replace(temporary, path)
        try:
            yield token
        finally:
            # An older instance must not remove a successor's credential.
            try:
                if secrets.compare_digest(path.read_bytes(), token.encode("ascii")):
                    path.unlink()
            except FileNotFoundError:
                pass
    finally:
        Path(temporary).unlink(missing_ok=True)


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_shutdown_access(request: Request) -> None:
    peer = request.client
    if not peer or not _loopback(peer.host):
        raise HTTPException(403, "shutdown is loopback-only")
    host = request.headers.get("host", "")
    scheme = request.scope.get("scheme", "http")
    try:
        authority = urlsplit(scheme + "://" + host)
        server = request.scope.get("server")
        valid_host = (
            len(request.headers.getlist("host")) == 1
            and authority.hostname is not None
            and (authority.hostname == "localhost" or _loopback(authority.hostname))
            and authority.username is None and authority.password is None
            and not (authority.path or authority.query or authority.fragment)
            and not any(char.isspace() for char in host)
            and server is not None
            and (authority.port or (443 if scheme == "https" else 80)) == server[1]
        )
    except ValueError:
        valid_host = False
    origin = request.headers.get("origin")
    if (not valid_host or (origin is not None and origin != f"{scheme}://{host}")
            or request.headers.get("sec-fetch-site", "").lower() not in {"", "same-origin", "none"}):
        raise HTTPException(403, "shutdown requires a local same-origin request")
    expected = getattr(request.app.state, "shutdown_token", None)
    authorization = request.headers.get("authorization", "").split(" ", 1)
    if (not expected or len(request.headers.getlist("authorization")) != 1
            or len(authorization) != 2 or authorization[0].lower() != "bearer"
            or not secrets.compare_digest(authorization[1].encode("utf-8"), expected.encode("ascii"))):
        raise HTTPException(403, "shutdown credential is missing or invalid")
