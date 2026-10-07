"""Sandbox runner. Executed as a separate, isolated Python process:
    python -I runner.py spec.json
spec = {"tables": {name: pickle_path}, "code": "..."}
Prints one marker line to stdout containing a JSON status object."""
import sys, os, io, json, pickle, builtins, traceback, contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 ** 3, 3 * 1024 ** 3))
except Exception:
    pass

import numpy as np
import pandas as pd
import common

MARK = "__PROOFAI__"
_real_import = builtins.__import__


def _safe_import(name, g=None, l=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] not in common.ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is not allowed in ProofAI sandbox")
    return _real_import(name, g, l, fromlist, level)


SAFE = ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance",
        "len", "list", "map", "max", "min", "pow", "print", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None", "Exception", "ValueError",
        "KeyError", "TypeError", "ZeroDivisionError", "IndexError", "AttributeError", "ArithmeticError", "bytes",
        "frozenset", "hasattr", "iter", "next", "callable", "chr", "ord", "format", "hash", "id", "type"]


def main():
    spec = json.load(open(sys.argv[1]))
    ns_builtins = {k: getattr(builtins, k) for k in SAFE if hasattr(builtins, k)}
    ns_builtins["__import__"] = _safe_import
    ns = {"__builtins__": ns_builtins, "pd": pd, "np": np, "CannotDetermine": common.CannotDetermine}
    for name, path in spec["tables"].items():
        with open(path, "rb") as f:
            ns[name] = pickle.load(f)
    buf = io.StringIO()
    out = {"status": "ok"}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(spec["code"], "<proofai>", "exec"), ns)
        if "result" not in ns:
            raise NameError("the code never assigned a variable named `result`")
        out["payload"] = common.to_payload(ns["result"])
    except common.CannotDetermine as e:
        out = {"status": "cannot_determine", "reason": str(e)}
    except BaseException as e:  # noqa
        tb = traceback.extract_tb(e.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<proofai>"), None)
        out = {"status": "error", "error": f"{type(e).__name__}: {e}", "line": line}
    out["output"] = buf.getvalue()[:4000]
    sys.stdout.write("\n" + MARK + json.dumps(out, default=str) + "\n")


main()
