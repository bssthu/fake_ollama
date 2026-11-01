"""Startup-only content persistence policy, inherited by owned workers."""

import os
import shlex
import sys
from urllib.parse import urlsplit


PRIVACY_MODE_ENV = "FAKE_OLLAMA_PRIVACY_MODE"


def privacy_mode_enabled() -> bool:
    return os.environ.get(PRIVACY_MODE_ENV) == "1"


def configure_privacy_mode(enabled: bool) -> None:
    if enabled:
        os.environ[PRIVACY_MODE_ENV] = "1"
    else:
        os.environ.pop(PRIVACY_MODE_ENV, None)


def privacy_status() -> dict:
    enabled = privacy_mode_enabled()
    return {"enabled": enabled, "content_storage": "memory_only" if enabled else "standard",
            "exports_allowed": not enabled,
            "scope": "application_and_managed_workers"}


def worker_environment() -> dict:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    if privacy_mode_enabled():
        env[PRIVACY_MODE_ENV] = "1"
        env.update(CUDA_CACHE_DISABLE="1", TORCH_COMPILE_DISABLE="1",
                   TOKENIZERS_PARALLELISM="false")
    return env


def backend_privacy_reason(source) -> str | None:
    """Fail closed for external/custom runtimes with unverified persistence."""
    if type(source).__name__ != 'LlamaCppTarget':
        return '此后端的内容保存行为尚不可由本服务控制，隐私模式下不可用'
    if (not source.auto_start or source.start_command or not source.binary_path
            or urlsplit(source.base_url).hostname not in ('127.0.0.1', 'localhost', '::1')):
        return '隐私模式仅支持由本服务直接启动的本地 llama.cpp 标准运行时'
    allowed = {'--jinja', '--no-mmap', '--mlock', '--no-warmup', '--no-webui', '--log-disable',
               '--reasoning-format', '--reasoning-budget', '--chat-template', '--threads', '-t',
               '--threads-batch', '-tb', '--prio', '--poll', '--temp', '--top-p', '--top-k',
               '--min-p', '--repeat-penalty', '--seed', '--cache-reuse', '--no-context-shift'}
    try:
        flags = shlex.split(source.extra_args or '', posix=False)
    except ValueError:
        return '无法确认启动参数的隐私约束'
    for flag in flags:
        if flag.startswith('-') and not flag.lstrip('-').isdigit() and flag.split('=', 1)[0] not in allowed:
            return '自定义启动参数尚未通过无落盘校验，隐私模式下不可用'
    return None


def require_private_backend(source):
    if privacy_mode_enabled() and (reason := backend_privacy_reason(source)):
        from fastapi import HTTPException
        raise HTTPException(409, reason)


_write_guard_installed = False


def install_worker_write_guard():
    """Deny Python file writes in strict workers, including third-party temp IO.

    Native libraries additionally require their cache/output paths to be disabled;
    this guard does not claim to control OS paging, dumps or external programs.
    """
    global _write_guard_installed
    if not privacy_mode_enabled() or _write_guard_installed:
        return
    def guard(event, args):
        if event == 'open':
            path, mode, flags = args
            writable = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writable and not isinstance(path, int) and str(path).lower() not in ('nul', os.devnull.lower()):
                raise PermissionError('隐私模式禁止模型进程写入文件')
    sys.addaudithook(guard)
    _write_guard_installed = True
