"""Fake Ollama server backed by an Anthropic-compatible upstream."""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
__version__ = "0.1.0"
