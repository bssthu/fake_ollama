"""Public configuration API.

The implementation lives in focused submodules while this package preserves
the original ``fake_ollama.config`` import surface.
"""

from . import models as _models
from .context import *  # noqa: F401,F403
from .loader import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403


# Preserve the few intentionally imported underscore helpers from the former
# single-module layout without maintaining a second export list.
for _name in dir(_models):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_models, _name))

del _name
