from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_target = Path(__file__).resolve().parents[3] / "scripts" / "data" / "dl3dv_conditions" / "common.py"
_spec = importlib.util.spec_from_file_location("_vgm_dl3dv_conditions_common", _target)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_target}")
_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_vgm_dl3dv_conditions_common", _module)
_spec.loader.exec_module(_module)
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})
