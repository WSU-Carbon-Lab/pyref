from __future__ import annotations

import importlib.util
from pathlib import Path

_image_py = Path(__file__).resolve().parent.parent / "image.py"
_spec = importlib.util.spec_from_file_location("pyref._image_impl", _image_py)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    apply_mask = _mod.apply_mask
    locate_beam = _mod.locate_beam
    reduction = _mod.reduction
else:
    apply_mask = None
    locate_beam = None
    reduction = None

__all__ = ["apply_mask", "locate_beam", "reduction"]
