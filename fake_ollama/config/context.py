"""Per-request forwarding context used for runtime cycle detection."""

from __future__ import annotations

import sys
import uuid
from contextvars import ContextVar
from typing import Dict, Tuple


FORWARDED_BY_HEADER = "x-fake-ollama-forwarded-by"
INSTANCE_ID: str = uuid.uuid4().hex
_inbound_forwarded_chain: ContextVar[Tuple[str, ...]] = ContextVar(
    "fake_ollama_forwarded_chain", default=()
)


def set_inbound_forwarded_chain(chain: Tuple[str, ...]):
    """Record the forwarded-by chain parsed from the inbound request."""

    return _inbound_forwarded_chain.set(tuple(chain))


def reset_inbound_forwarded_chain(token) -> None:
    _inbound_forwarded_chain.reset(token)


def current_inbound_forwarded_chain() -> Tuple[str, ...]:
    return _inbound_forwarded_chain.get()


def outbound_forwarded_chain() -> str:
    """Build the outbound chain while honoring package-level test overrides."""

    chain = list(_inbound_forwarded_chain.get())
    package = sys.modules.get(__package__)
    instance_id = getattr(package, "INSTANCE_ID", INSTANCE_ID)
    if instance_id not in chain:
        chain.append(instance_id)
    return ",".join(chain)


def outbound_cycle_headers() -> Dict[str, str]:
    return {FORWARDED_BY_HEADER: outbound_forwarded_chain()}


def parse_forwarded_chain(raw: str) -> Tuple[str, ...]:
    if not raw:
        return ()
    return tuple(token.strip() for token in raw.split(",") if token.strip())
