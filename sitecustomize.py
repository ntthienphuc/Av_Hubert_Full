try:
    import numpy as _np

    for _name, _value in {
        "float": float,
        "int": int,
        "bool": bool,
        "object": object,
    }.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _value)
except Exception:
    pass

try:
    import torch as _torch

    _orig_load = _torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _torch.load = _load_compat
except Exception:
    pass
