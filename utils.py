# utils.py
import numpy as np
import json
import os
import random

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)

def calculate_entropy_from_params(params):
    """Shannon-style entropy for numeric parameter dict (normalized)."""
    vals = []
    for v in params.values():
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    arr = np.array(vals, dtype=float)
    s = np.sum(arr)
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def append_json_log(filepath, record):
    safe_mkdir(os.path.dirname(filepath) or ".")
    existing = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(record)
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)
