"""Shared helpers: result serialisation + comparison. Used by the sandbox runner,
the engine, and (via source inspection) by the standalone proof script, so all
three serialise results identically."""
import math, datetime
import numpy as np
import pandas as pd

MAX_ROWS = 200
ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "datetime", "statistics", "collections", "itertools", "functools"}


class CannotDetermine(Exception):
    """Raised by analysis code when the data cannot support a reliable answer."""


def _scalar(v):
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return None
        return str(f) if math.isinf(f) else f
    if isinstance(v, (pd.Timestamp, datetime.date, datetime.datetime)):
        return str(v)
    return str(v)


def to_payload(r):
    if isinstance(r, pd.DataFrame):
        h = r.head(MAX_ROWS)
        return {"kind": "frame", "columns": [str(c) for c in h.columns],
                "index": [_scalar(i) for i in h.index],
                "rows": [[_scalar(x) for x in row] for row in h.itertuples(index=False, name=None)],
                "total_rows": int(len(r))}
    if isinstance(r, dict):
        r = pd.Series(r)
    if isinstance(r, (list, tuple, np.ndarray)):
        r = pd.Series(list(r))
    if isinstance(r, pd.Series):
        h = r.head(MAX_ROWS)
        return {"kind": "series", "name": None if r.name is None else str(r.name),
                "index": [_scalar(i) for i in h.index], "values": [_scalar(x) for x in h.values],
                "total_rows": int(len(r))}
    return {"kind": "scalar", "value": _scalar(r)}


def payloads_match(a, b, rel=1e-9, abs_=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(payloads_match(a[k], b[k], rel, abs_) for k in a if k != "name")
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(payloads_match(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    return a == b
