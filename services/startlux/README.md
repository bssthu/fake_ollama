# Optional official StartLux integration

Setup, local-model requirements, isolated Python environments, validation and licensing:
[docs/decision-startlux.md](../../docs/decision-startlux.md).

The runnable companion is packaged at `fake_ollama/startlux_companion.py`, so it
also works from an installed fake-ollama wheel. This folder intentionally contains
no model weights, copied upstream implementation, installer or auto-downloader.

Upstream: [StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision).
