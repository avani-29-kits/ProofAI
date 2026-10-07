"""ProofAI engine: ingestion, profiling, guardrails, sandboxed execution, independent
verification and response building. Framework-independent (no FastAPI import)."""
from __future__ import annotations
import ast, inspect, json, keyword, os, pickle, re, subprocess, sys, tempfile, time, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import common, planner

DATA_DIR = Path(os.environ.get("PROOFAI_DATA", Path(__file__).parent / "data"))
RUNNER = Path(__file__).parent / "runner.py"
MARK = "__PROOFAI__"
MAX_ATTEMPTS = 3
RESERVED = {"pd", "np", "result", "json", "math", "re", "datetime", "CannotDetermine"}
P = "@@P@@"  # placeholder for the file path inside read expressions


# ───────────────────────── ingestion ─────────────────────────
def ident(s: str) -> str:
    s = re.sub(r"\W+", "_", s).strip("_").lower() or "table"
    if s[0].isdigit():
        s = "t_" + s
    return s + "_df" if (s in RESERVED or keyword.iskeyword(s)) else s


def _is_num(s): return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
def _is_text(s): return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)


_ISO = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ T].*)?$")
_SLASH = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$")
_NAMED = re.compile(r"^(\d{1,2}[ -][A-Za-z]{3,9},?[ -]\d{2,4}|[A-Za-z]{3,9}\.? \d{1,2},? \d{4})$")


def detect_dates(nn: pd.Series):
    """Return date info for a text column, or None if it isn't a date column."""
    if nn.empty:
        return None
    vals = nn.astype(str).str.strip()
    sample = vals.head(500)
    iso = sample.str.match(_ISO).sum(); named = sample.str.match(_NAMED).sum()
    sl = [m for m in (_SLASH.match(v) for v in sample) if m]
    if (iso + named + len(sl)) / len(sample) < 0.8:
        return None
    a_gt12 = any(int(m.group(1)) > 12 for m in sl); b_gt12 = any(int(m.group(2)) > 12 for m in sl)
    if not sl: fmt = "unambiguous"
    elif a_gt12 and b_gt12: fmt = "mixed"
    elif a_gt12: fmt = "dayfirst"
    elif b_gt12: fmt = "monthfirst"
    else: fmt = "ambiguous"
    try:
        parsed = pd.to_datetime(vals, errors="coerce", dayfirst=(fmt == "dayfirst"), format="mixed")
    except Exception:
        parsed = pd.to_datetime(vals, errors="coerce")
    ok = parsed.dropna()
    if ok.empty:
        return None
    return {"format": fmt, "min": str(ok.min().date()), "max": str(ok.max().date()),
            "years": {int(k): int(v) for k, v in ok.dt.year.value_counts().sort_index().items()},
            "parsed_pct": round(100 * len(ok) / len(vals), 1)}


_CUR = {"$": "USD", "us$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR"}
_UNITS = {"lbs": "lb", "lb": "lb", "pounds": "lb", "kg": "kg", "kgs": "kg", "g": "g", "km": "km", "mi": "mi", "miles": "mi", "m": "m",
          "cm": "cm", "ft": "ft", "l": "l", "ml": "ml", "gal": "gal", "%": "%"}
_NT = re.compile(r"^\s*(?P<pre>US\$|[$€£₹]|USD|EUR|GBP|INR|Rs\.?)?\s*-?[\d,]*\.?\d+\s*(?P<post>[A-Za-z%]{1,6})?\s*$", re.I)


def numeric_text_info(nn: pd.Series):
    """Text column that holds numbers with embedded currency/units ('$12.50', '9 kg', '1,200')."""
    vals = nn.astype(str).head(500); units = []; hit = 0
    for v in vals:
        m = _NT.match(v)
        if not m: continue
        tok = (m.group("pre") or m.group("post") or "").strip().lower()
        if tok and tok not in _CUR and tok not in _UNITS: continue
        hit += 1; units.append(_CUR.get(tok) or _UNITS.get(tok))
    if not len(vals) or hit / len(vals) < 0.8: return None
    kinds = sorted({u for u in units if u})
    return {"units": kinds, "mixed": len(kinds) > 1, "with_unit_pct": round(100 * sum(1 for u in units if u) / hit)}


def profile_df(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        try: nn = s.dropna(); uniq = int(nn.nunique())
        except TypeError: nn = s.dropna().astype(str); uniq = int(nn.nunique())
        info = {"name": str(c), "dtype": str(s.dtype), "nulls": int(s.isna().sum()), "unique": uniq,
                "sample": [str(x)[:40] for x in nn.drop_duplicates().head(4)]}
        if _is_num(s) and len(nn):
            info["min"], info["max"] = float(nn.min()), float(nn.max())
            if re.search(r"year|yr", str(c).lower()) and uniq <= 60 and (nn % 1 == 0).all() and 1900 <= nn.min() <= nn.max() <= 2200:
                info["date"] = {"format": "year_column", "min": str(int(nn.min())), "max": str(int(nn.max())),
                                "years": {int(k): int(v) for k, v in nn.astype(int).value_counts().sort_index().items()}}
        elif pd.api.types.is_datetime64_any_dtype(s) and len(nn):
            info["date"] = {"format": "unambiguous", "min": str(nn.min().date()), "max": str(nn.max().date()),
                            "years": {int(k): int(v) for k, v in nn.dt.year.value_counts().sort_index().items()}, "parsed_pct": 100.0}
        elif _is_text(s) and len(nn):
            d = detect_dates(nn)
            if d: info["date"] = d
            else:
                nt = numeric_text_info(nn)
                if nt: info["numeric_text"] = nt
        cols.append(info)
    try: dups = int(df.duplicated().sum())
    except TypeError: dups = int(df.astype(str).duplicated().sum())
    return {"rows": int(len(df)), "duplicate_rows": dups, "columns": cols}


class Workspace:
    """One uploaded-data workspace per session id (files + pickles + meta.json on disk)."""

    def __init__(self, session: str = "default"):
        self.id = re.sub(r"[^A-Za-z0-9_-]", "", session)[:40] or "default"
        self.dir = DATA_DIR / "sessions" / self.id
        self.files = self.dir / "files"; self.pk = self.dir / "pkl"
        for d in (self.files, self.pk): d.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {"tables": {}}

    @property
    def tables(self) -> dict: return self.meta["tables"]
    def _save(self): self.meta_path.write_text(json.dumps(self.meta))

    def ingest(self, filename: str, content: bytes) -> list[dict]:
        fname = re.sub(r"[^\w.\- ]", "_", Path(filename).name) or "upload"
        path = self.files / fname; path.write_bytes(content)
        ext = path.suffix.lower(); stem = path.stem; p = repr(str(path))
        plans = []  # (table_name_hint, display, expr_template)
        if ext in (".csv", ".tsv", ".txt"):
            enc = "utf-8"
            try: content.decode("utf-8")
            except UnicodeDecodeError: enc = "latin-1"
            sep = "\t" if ext == ".tsv" else ","
            head = content[:2000].decode(enc, "ignore").split("\n")[0]
            if ext != ".tsv" and head.count(";") > head.count(","): sep = ";"
            plans.append((stem, fname, f"pd.read_csv({P}, sep={sep!r}, encoding={enc!r})"))
        elif ext in (".xlsx", ".xlsm", ".xls"):
            sheets = pd.ExcelFile(path).sheet_names
            for sh in sheets:
                plans.append((stem if len(sheets) == 1 else f"{stem}_{sh}", fname if len(sheets) == 1 else f"{fname} [{sh}]",
                              f"pd.read_excel({P}, sheet_name={sh!r})"))
        elif ext == ".json":
            raw = json.loads(content.decode("utf-8-sig"))
            load = f"json.load(open({P}, encoding='utf-8-sig'))"
            if isinstance(raw, list):
                plans.append((stem, fname, f"pd.json_normalize({load})"))
            elif isinstance(raw, dict):
                lists = [k for k, v in raw.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
                if lists:
                    for k in lists:
                        plans.append((k if len(lists) > 1 else stem, f"{fname} [{k}]", f"pd.json_normalize({load}[{k!r}])"))
                else:
                    plans.append((stem, fname, f"pd.json_normalize({load})"))
            else:
                raise ValueError("JSON must be a list of records or an object containing lists of records")
        else:
            raise ValueError(f"Unsupported file type '{ext}'. Use CSV, XLSX or JSON.")
        added = []
        for hint, display, tpl in plans:
            df = eval(tpl.replace(P, p), {"pd": pd, "json": json})
            name = ident(hint)
            old = self.tables.get(name)
            if old and old["file"] != fname:
                i = 2
                while f"{name}_{i}" in self.tables: i += 1
                name = f"{name}_{i}"
            with open(self.pk / f"{name}.pkl", "wb") as f: pickle.dump(df, f)
            self.tables[name] = {"name": name, "file": fname, "display": display, "tpl": tpl, "rows": int(len(df)),
                                 "cols": int(df.shape[1]), "profile": profile_df(df)}
            added.append(self.tables[name])
        self._contradictions(); self._save()
        return added


    def _contradictions(self):
        out, names = [], list(self.tables)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = {c["name"] for c in self.tables[a]["profile"]["columns"]} & {c["name"] for c in self.tables[b]["profile"]["columns"]}
                if len(common) < 2: continue
                try:
                    A = pickle.load(open(self.pk / f"{a}.pkl", "rb")).drop_duplicates(); B = pickle.load(open(self.pk / f"{b}.pkl", "rb")).drop_duplicates()
                except Exception: continue
                if len(A) > 300000 or len(B) > 300000: continue
                for k in sorted(common):
                    try:
                        if A[k].isna().any() or B[k].isna().any() or A[k].nunique() != len(A) or B[k].nunique() != len(B) or len(A) < 2: continue
                    except TypeError: continue
                    j = A.merge(B, on=k, suffixes=("_a", "_b"))
                    if len(j) == 0: continue
                    for c in sorted(common - {k}):
                        x, y = j[c + "_a"], j[c + "_b"]; both = x.notna() & y.notna()
                        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y): diff = (x - y).abs() > 1e-6 * np.maximum(1, x.abs())
                        else: diff = x.astype(str).str.strip().str.lower() != y.astype(str).str.strip().str.lower()
                        bad = diff & both
                        if bad.any():
                            ex = j[bad].iloc[0]
                            out.append({"a": a, "b": b, "key": k, "column": c, "mismatches": int(bad.sum()), "compared": int(both.sum()),
                                        "example": [str(ex[k]), str(ex[c + "_a"]), str(ex[c + "_b"])]})
                    break
        self.meta["contradictions"] = out

    def delete(self, name: str) -> bool:
        if name not in self.tables: return False
        fname = self.tables.pop(name)["file"]
        (self.pk / f"{name}.pkl").unlink(missing_ok=True)
        if not any(t["file"] == fname for t in self.tables.values()): (self.files / fname).unlink(missing_ok=True)
        self._contradictions(); self._save(); return True

    def public_list(self):
        return [{"name": t["name"], "file": t["display"], "rows": t["rows"], "cols": t["cols"],
                 "duplicate_rows": t["profile"]["duplicate_rows"],
                 "nulls": sum(c["nulls"] for c in t["profile"]["columns"])} for t in self.tables.values()]

    def pickles(self, names): return {n: str(self.pk / f"{n}.pkl") for n in names}


# ───────────────────────── guardrails ─────────────────────────
BAD_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars", "breakpoint",
             "exit", "quit", "getattr", "setattr", "delattr", "help", "memoryview", "dir"}
BAD_ATTRS = {"to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_sql", "to_feather", "to_hdf", "to_clipboard",
             "to_stata", "to_html", "to_latex", "eval", "system", "popen", "load", "save", "savez", "loadtxt", "genfromtxt",
             "fromfile", "tofile", "memmap", "savetxt"}


def guard(code: str) -> list[str]:
    try: tree = ast.parse(code)
    except SyntaxError as e: return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [f"import '{a.name}' not allowed" for a in n.names if a.name.split('.')[0] not in common.ALLOWED_IMPORTS]
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split('.')[0] not in common.ALLOWED_IMPORTS: out.append(f"import from '{n.module}' not allowed")
        elif isinstance(n, ast.Name) and (n.id in BAD_NAMES or n.id.startswith("__")):
            out.append(f"use of '{n.id}' is not allowed")
        elif isinstance(n, ast.Attribute) and (n.attr in BAD_ATTRS or n.attr.startswith("__") or n.attr.startswith("read_")):
            out.append(f"use of attribute '{n.attr}' is not allowed (no file/network/system access)")
    return sorted(set(out))


def referenced_tables(code: str, tables) -> list[str]:
    try: tree = ast.parse(code)
    except SyntaxError: return []
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in tables})


def string_literals(code: str) -> set[str]:
    try: return {n.value for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    except SyntaxError: return set()


# ───────────────────────── execution + proof script ─────────────────────────
def _parse_marker(stdout: str):
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    return None


def _env():
    e = {"PATH": os.environ.get("PATH", "")}
    if os.environ.get("SYSTEMROOT"): e["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return e


def sandbox_run(pickles: dict, code: str, timeout: int = 30) -> dict:
    with tempfile.TemporaryDirectory() as td:
        spec = Path(td) / "spec.json"; spec.write_text(json.dumps({"tables": pickles, "code": code}))
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-I", str(RUNNER), str(spec)], capture_output=True, text=True,
                               timeout=timeout, cwd=td, env=_env())
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"TimeoutError: execution exceeded {timeout}s", "secs": timeout}
        res = _parse_marker(p.stdout) or {"status": "error", "error": "Sandbox crashed: " + (p.stderr.strip()[-300:] or "no output")}
        res["secs"] = time.perf_counter() - t0
        return res


def build_script(question: str, ws: Workspace, used: list[str], code: str, emit: bool = False) -> str:
    q = re.sub(r"\s+", " ", question)[:200]
    lines = ['"""ProofAI proof script — run with:  python proof.py', "Keep the original data files in the same folder as this script.",
             f'Question: {q}', '"""', "import json, math, datetime", "import pandas as pd", "import numpy as np", "",
             "class CannotDetermine(Exception):", "    pass", ""]
    for n in used:
        t = ws.tables[n]
        lines.append(f"{n} = " + t["tpl"].replace(P, repr(t["file"])))
    lines += ["", "# ───── generated analysis ─────", code.strip(), ""]
    if emit:
        lines += ["# ───── result emitter (verification only) ─────", "import sys as _sys", "MAX_ROWS = %d" % common.MAX_ROWS,
                  inspect.getsource(common._scalar), inspect.getsource(common.to_payload),
                  '_sys.stdout.write("\\n%s" % ("' + MARK + '" + json.dumps({"status": "ok", "payload": to_payload(result)}, default=str)))']
    else:
        lines += ['pd.set_option("display.width", 200)', "print(result)"]
    return "\n".join(lines) + "\n"


def reproduce(ws: Workspace, question: str, used: list[str], code: str, timeout: int = 60) -> dict:
    """Independent re-run: the standalone script re-reads the ORIGINAL files in a fresh interpreter."""
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "verify.py"; sp.write_text(build_script(question, ws, used, code, emit=True), encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(sp)], capture_output=True, text=True, timeout=timeout, cwd=ws.files, env=_env())
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "TimeoutError"}
        return _parse_marker(p.stdout) or {"status": "error", "error": (p.stderr.strip().splitlines() or ["no output"])[-1]}


# ───────────────────────── input checks ─────────────────────────
_UNIT_COL = re.compile(r"^(unit|units|uom|currency|ccy|unit_of_measure|currency_code)$|_(unit|units|uom|currency|ccy)$")
_TIME_Q = re.compile(r"\b(q[1-4]|quarter\w*|month\w*|week\w*|daily|day|year\w*|annual\w*|jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*|20\d\d|19\d\d)\b", re.I)


def input_checks(code: str, ws: Workspace, used: list[str]):
    """Returns (hard_issues, warnings, ok_lines). Hard issues make the engine retry, then refuse."""
    lits = string_literals(code); hard, warn, ok = [], [], []
    dup_handled = bool(re.search(r"drop_duplicates|\.duplicated\(", code))
    unit_ok = True; dup_total = 0
    for n in used:
        prof = ws.tables[n]["profile"]
        for c in prof["columns"]:
            if _UNIT_COL.search(c["name"].lower()) and c["unique"] > 1 and c["name"] not in lits:
                unit_ok = False
                hard.append(f"Table '{n}' column '{c['name']}' holds mixed units/currencies (e.g. {', '.join(c['sample'][:3])}) and the data contains no verified conversion for them.\n"
                            "Convert/filter explicitly using data present in the tables; if no conversion data exists, raise CannotDetermine.")
            nt = c.get("numeric_text")
            if nt and nt["mixed"] and c["name"] in lits:
                unit_ok = False
                hard.append(f"Column '{n}.{c['name']}' mixes units/currencies inside its values ({', '.join(nt['units'])}) and the data contains no verified conversion rates.\n"
                            "Raise CannotDetermine; never invent an exchange rate or unit factor.")
            d = c.get("date")
            if d and d["format"] in ("ambiguous", "mixed") and c["name"] in lits:
                hard.append(f"Date column '{n}.{c['name']}' is {d['format']} (DD/MM vs MM/DD cannot be established from the data). "
                            "Raise CannotDetermine unless the question fixes the convention.")
        dup_total += prof["duplicate_rows"]
        if prof["duplicate_rows"] and not dup_handled:
            warn.append(f"⚠ {prof['duplicate_rows']:,} exact duplicate rows in {n} were not removed")
        for c in prof["columns"]:
            if c["name"] in lits and c["nulls"]:
                warn.append(f"⚠ {c['nulls']:,} missing values in {n}.{c['name']}")
    for ct in ws.meta.get("contradictions", []):
        if (ct["a"] in used or ct["b"] in used) and ct["column"] in lits:
            ex = ct["example"]
            hard.append(f"Tables '{ct['a']}' and '{ct['b']}' contradict each other: '{ct['column']}' differs for {ct['mismatches']:,} of {ct['compared']:,} matching {ct['key']} values (e.g. {ct['key']}={ex[0]}: {ex[1]} vs {ex[2]}).\n"
                        "Raise CannotDetermine unless the question says which source to trust.")
    ok.append("Input validated: " + ", ".join(used))
    if unit_ok: ok.append("Units consistent")
    if dup_total == 0: ok.append("Duplicate check passed")
    elif dup_handled: ok.append(f"{dup_total:,} duplicate rows removed in code")
    return hard, warn, ok


def time_ambiguity(question: str, ws: Workspace):
    """Refuse up-front when the question is date-dependent and every date column is ambiguous/mixed."""
    if not _TIME_Q.search(question): return None
    date_cols = [(t["name"], c["name"], c["date"]) for t in ws.tables.values() for c in t["profile"]["columns"] if c.get("date")]
    if not date_cols or any(d["format"] not in ("ambiguous", "mixed") for _, _, d in date_cols): return None
    t, c, d = date_cols[0]
    why = ("mixes DD/MM and MM/DD formats" if d["format"] == "mixed" else "uses slash dates where every value could be read as DD/MM or MM/DD")
    return (f"Column '{c}' in {t} {why}, so the period in the question cannot be established reliably.", t, c, d)


def year_coverage(question: str, ws: Workspace):
    ys = sorted({int(y) for y in re.findall(r"\b(19\d\d|20\d\d|21\d\d)\b", question)})
    cands = [(t["name"], c["name"], c["date"]) for t in ws.tables.values() for c in t["profile"]["columns"] if c.get("date")]
    if not ys or not cands: return None
    covered = {int(y) for _, _, d in cands for y in d["years"]}
    missing = [y for y in ys if y not in covered]
    if not missing: return None
    tname, col, d = max(cands, key=lambda x: sum(x[2]["years"].values()))
    return missing, tname, col, d, covered



_BADQ = re.compile(r"\bq\s?([05-9])\b|\bquarter\s+([05-9])\b|\bmonth\s+(1[3-9]|[2-9]\d|0)\b", re.I)
_ENT = re.compile(r"\b([A-Z][A-Za-z\-]+(?:\s[A-Z][A-Za-z\-]+)*)\s+(region|category|segment|product|customer|city|state|country|branch|store)\b"
                  r"|\b(region|category|segment|product|customer|city|state|country|branch|store)\s+([A-Z][\w\-]+)")
_SKIP = {"top", "which", "what", "compare", "show", "list", "find", "the", "total", "average", "highest", "lowest", "best", "worst",
         "test", "how", "each", "every", "any", "all", "our", "my", "your", "and", "for", "per", "by", "in", "of"}


def invalid_period(question: str):
    m = _BADQ.search(question)
    if not m: return None
    tok = m.group(0).strip()
    return refusal(f"'{tok}' is not a valid period (quarters are Q1–Q4, months 1–12), so the question cannot be answered as asked.",
                   ev=["invalid period in question"], chk=["Period validated against the calendar: invalid", "No value was estimated"], vt="Invalid question: answer withheld")


_REL = re.compile(r"\b(last|this|next|previous|current|past|prior)\s+(day|week|month|quarter|year|fy|period)s?\b|\b(recent|recently|ytd|mtd|yesterday|today|tomorrow|lately|nowadays)\b", re.I)


def relative_time(question: str, ws: Workspace):
    m = _REL.search(question)
    if not m: return None
    ends = [(t["name"], c["date"]["max"]) for t in ws.tables.values() for c in t["profile"]["columns"] if c.get("date")]
    latest = f" (the latest record in the data is {max(e[1] for e in ends)})" if ends else ""
    return refusal(f"'{m.group(0)}' is relative to today's date, which the data does not define{latest}. Name the exact period (for example 'Q2 2024') and ask again.",
                   ev=["relative time phrase detected"] + ([f"data ends {max(e[1] for e in ends)}"] if ends else []),
                   chk=["Time reference is ambiguous", "No value was estimated"], vt="Ambiguous period: answer withheld")


def entity_check(question: str, ws: Workspace):
    for m in _ENT.finditer(question):
        ent = (m.group(1) or m.group(4)).strip(); noun = (m.group(2) or m.group(3)).lower()
        if m.start() == 0 or ent.lower() in _SKIP or any(w.lower() in _SKIP for w in ent.split()): continue
        for t in ws.tables.values():
            for c in t["profile"]["columns"]:
                if noun not in re.split(r"[^a-z]+", c["name"].lower()) or c["unique"] == 0: continue
                df = pickle.load(open(ws.pk / f"{t['name']}.pkl", "rb"))
                vals = df[c["name"]].dropna().astype(str)
                if (vals.str.lower() == ent.lower()).any(): return None
                code = (f'# Does "{ent}" exist as a {noun} in {t["name"]}.{c["name"]}?\n'
                        f'print(sorted({t["name"]}["{c["name"]}"].dropna().astype(str).unique())[:10])\n'
                        f'result = int(({t["name"]}["{c["name"]}"].astype(str).str.lower() == "{ent.lower()}").sum())')
                run = sandbox_run(ws.pickles([t["name"]]), code)
                if run["status"] != "ok" or run["payload"].get("value") != 0: return None
                known = ", ".join(sorted(vals.unique())[:5])
                return refusal(f"No {noun} named '{ent}' exists in the data (known {noun}s: {known}).",
                               ev=[t["display"], f"{c['name']}: {c['unique']} distinct values", f"'{ent}' rows: 0"],
                               chk=[f"Looked up '{ent}' in {t['name']}.{c['name']}: 0 rows", "No value was estimated"],
                               code=code, tm=f"{run['secs']:.2f}s", vt="Verified: the entity does not exist in the data")
    return None

# ───────────────────────── response building ─────────────────────────
def fmt_num(v, unit=""):
    if v is None: return "—"
    if isinstance(v, bool) or isinstance(v, str): return str(v)
    f = float(v)
    if unit in ("₹", "INR", "Rs"):
        if abs(f) >= 1e7: return f"₹{f / 1e7:.2f} crore"
        if abs(f) >= 1e5: return f"₹{f / 1e5:.2f} lakh"
        return f"₹{f:,.2f}"
    sym = {"$": "$", "USD": "$", "€": "€", "EUR": "€", "£": "£", "GBP": "£"}.get(unit)
    if sym: return f"{sym}{f:,.2f}"
    s = f"{int(f):,}" if f == int(f) and abs(f) < 1e15 else f"{f:,.2f}"
    return f"{s} {unit}" if unit else s


def _chart(payload, unit):
    labels, vals = [], []
    if payload["kind"] == "series":
        labels, vals = [str(i) for i in payload["index"]], payload["values"]
    elif payload["kind"] == "frame" and payload["rows"]:
        cols = payload["columns"]; r0 = payload["rows"][0]
        ni = next((i for i, v in enumerate(r0) if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if ni is None: return None, ""
        li = next((i for i, v in enumerate(r0) if isinstance(v, str)), None)
        labels = [str(r[li]) if li is not None else str(payload["index"][k]) for k, r in enumerate(payload["rows"])]
        vals = [r[ni] for r in payload["rows"]]
    pairs = [(l, v) for l, v in zip(labels, vals) if isinstance(v, (int, float)) and not isinstance(v, bool)][:8]
    if not pairs: return None, ""
    u = "value"
    if unit in ("₹", "INR", "Rs") and max(abs(v) for _, v in pairs) >= 1e5:
        pairs = [(l, round(v / 1e5, 2)) for l, v in pairs]; u = "₹ lakh"
    elif unit: u = unit
    return {"l": [p[0] for p in pairs], "v": [round(p[1], 2) if isinstance(p[1], float) else p[1] for p in pairs]}, u


def headline(payload, plan):
    unit, desc, kind = plan.get("unit", ""), plan.get("description", "Result"), plan.get("answer_kind", "scalar")
    if payload["kind"] == "scalar":
        v = payload["value"]
        return (fmt_num(v, unit) if not isinstance(v, str) else v), desc
    n = payload["total_rows"]
    if n == 0: return "None found", desc
    if kind == "ranking":
        if payload["kind"] == "series":
            return str(payload["index"][0]), f"{fmt_num(payload['values'][0], unit)} · {desc}"
        r0 = payload["rows"][0]
        li = next((i for i, v in enumerate(r0) if isinstance(v, str)), None)
        ni = next((i for i, v in enumerate(r0) if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if li is not None and ni is not None: return str(r0[li]), f"{fmt_num(r0[ni], unit)} · {desc}"
    return f"{n:,} result{'s' if n != 1 else ''}", desc


def refusal(reason, ev=None, chk=None, code=None, ch=None, t="Evidence", u="", tm="—", vt="Not enough evidence to verify", note=None, **extra):
    return {"no": 1, "a": "Cannot determine", "s": reason, "ev": ev or [], "t": t, "u": u, "ch": ch,
            "code": code or f"# No numerical answer was produced.\n# Reason: {reason}", "tm": tm, "vt": vt,
            "chk": chk or ["No value was estimated", "ProofAI declines to guess"],
            "note": note or "ProofAI will not invent a value when the data does not contain enough evidence.", **extra}


def _ev_tables(ws, used):
    out = []
    for n in used:
        out += [ws.tables[n]["display"], f"{ws.tables[n]['rows']:,} rows"]
    return out


def _year_refusal(ws, missing, tname, col, d, covered):
    info = ws.tables[tname]["profile"]["columns"]; cinfo = next(c for c in info if c["name"] == col)
    if d["format"] == "year_column": ycode = f'years = {tname}["{col}"]'
    else:
        ycode = f'years = pd.to_datetime({tname}["{col}"], errors="coerce", {"format=\"mixed\"" if d["format"] in ("mixed", "ambiguous") else "dayfirst=" + str(d["format"] == "dayfirst")}).dt.year'
    code = f'{ycode}\n\nprint(sorted(years.dropna().astype(int).unique()))   # years present in the data\nresult = int(years.isin({missing}).sum())    # rows matching the requested year(s)'
    run = sandbox_run(ws.pickles([tname]), code)
    if run["status"] != "ok" or run["payload"].get("value") != 0: return None
    ys = sorted(int(y) for y in d["years"]); span = f"{ys[0]}–{ys[-1]}" if ys[0] != ys[-1] else str(ys[0])
    allyears = sorted(set(ys) | set(missing)); mlabel = ", ".join(map(str, missing))
    return refusal(f"The available dataset contains no {mlabel} records.",
                   ev=[ws.tables[tname]["display"], f"dates: {span}", f"{mlabel} rows: 0"],
                   chk=[f"Date range checked: {d['min']} → {d['max']}", f"{mlabel} rows found: 0", "No value was estimated"],
                   code=code, ch={"l": [str(y) for y in allyears], "v": [d["years"].get(y, d["years"].get(str(y), 0)) for y in allyears]},
                   t="Coverage by year", u="rows", tm=f"{run['secs']:.2f}s", vt="Verified: the data has no rows for that period")


# ───────────────────────── the pipeline ─────────────────────────
def analyze(question: str, ws: Workspace, store=None) -> dict:
    t0 = time.perf_counter(); question = question.strip()
    if not ws.tables:
        return refusal("No datasets have been uploaded yet.", ev=["no datasets"], chk=["Upload CSV, XLSX or JSON files first"])
    yc = year_coverage(question, ws)
    if yc:
        r = _year_refusal(ws, *yc)
        if r: return _finish(r, question, ws, store, t0)
    for pre in (invalid_period(question), relative_time(question, ws), entity_check(question, ws)):
        if pre: return _finish(pre, question, ws, store, t0)
    ta = time_ambiguity(question, ws)
    if ta:
        reason, t, c, d = ta
        return _finish(refusal(reason, ev=[ws.tables[t]["display"], f"{c}: {d['format']} dates", f"sample: {', '.join(next(x['sample'] for x in ws.tables[t]['profile']['columns'] if x['name'] == c)[:2])}"],
                               chk=["Date format could not be established", "No value was estimated"], vt="Ambiguous dates: answer withheld"), question, ws, store, t0)
    history, last_issue = [], None
    for attempt in range(MAX_ATTEMPTS):
        plan = planner.plan(question, ws.tables, history)
        if not plan["answerable"]:
            return _finish(refusal(plan["reason"] or "The data cannot support a reliable answer.", ev=plan.get("evidence") or [], planner=plan["planner"]), question, ws, store, t0)
        code = plan["code"]; used = referenced_tables(code, ws.tables)
        problems = guard(code) + ([] if used else ["The code must use at least one of the loaded tables: " + ", ".join(ws.tables)])
        run = None
        if not problems:
            run = sandbox_run(ws.pickles(used), code)
            if run["status"] == "cannot_determine":
                return _finish(refusal(run["reason"], ev=_ev_tables(ws, used), code=code, tm=f"{run['secs']:.2f}s",
                                       chk=["Executed code detected insufficient or contradictory evidence", "No value was estimated"],
                                       vt="Verified: the data is insufficient", planner=plan["planner"]), question, ws, store, t0)
            if run["status"] == "error": problems.append(f"Execution failed{' on line %s' % run['line'] if run.get('line') else ''}: {run['error']}")
            elif run["payload"]["kind"] == "scalar" and run["payload"]["value"] is None: problems.append("`result` is empty/NaN")
        hard, warn, ok = ([], [], [])
        if not problems:
            hard, warn, ok = input_checks(code, ws, used); problems += hard
        if problems:
            last_issue = problems[0]
            history.append((plan.get("_raw") or json.dumps({k: plan[k] for k in ("answerable", "description", "unit", "answer_kind", "code")}),
                            "The analysis was rejected:\n- " + "\n- ".join(problems) + "\nFix the code and return the full JSON object again."))
            if attempt == MAX_ATTEMPTS - 1 and hard:
                return _finish(refusal(hard[0].split("\n")[0], ev=_ev_tables(ws, used), code=code, chk=["Input validation failed", "No value was estimated"], vt="Verification failed: unresolved data issue"), question, ws, store, t0)
            continue
        rep = reproduce(ws, question, used, code)
        if rep["status"] != "ok" or not common.payloads_match(run["payload"], rep["payload"]):
            return _finish({**refusal("The generated code did not reproduce the same result when re-run independently, so the answer is invalid.",
                                      ev=_ev_tables(ws, used), code=code, chk=["Independent re-run failed or differed", "Answer marked invalid"], vt="Verification failed", tm=f"{run['secs']:.2f}s"),
                            "a": "Answer invalid"}, question, ws, store, t0)
        a, s = headline(run["payload"], plan); ch, u = _chart(run["payload"], plan.get("unit", ""))
        ev = _ev_tables(ws, used) + [str(x)[:60] for x in (plan.get("evidence") or [])[:3]]
        notes = list(plan.get("assumptions") or []) + warn
        r = {"no": 0, "a": a, "s": s, "ev": ev, "t": plan.get("description", "Result"), "u": u, "ch": ch, "code": code.strip(),
             "tm": f"{run['secs']:.2f}s", "vt": "Result reproduced successfully", "chk": ok + ["Calculation reproduced in a fresh process"],
             "note": ("; ".join(notes) + ". " if notes else "") + "The generated code was executed against the uploaded dataset and an independent re-run of the standalone proof script produced the same result.",
             "output": run.get("output", ""), "result": run["payload"], "planner": plan["planner"], "attempts": attempt + 1}
        return _finish(r, question, ws, store, t0, used=used)
    return _finish(refusal(f"No verifiable analysis could be produced after {MAX_ATTEMPTS} attempts. Last issue: {last_issue}", chk=["Could not produce code that runs and verifies", "No value was estimated"]), question, ws, store, t0)


def _finish(r, question, ws, store, t0, used=None):
    r["elapsed"] = round(time.perf_counter() - t0, 2)
    if used is None: used = referenced_tables(r.get("code", ""), ws.tables)
    r["script"] = build_script(question, ws, used, r["code"]) if used and not r["code"].startswith("# No ") else None
    if store:
        r["id"] = store.save(ws.id, question, r, used)
        r["script_url"] = f"/api/proof/{r['id']}.py" if r["script"] else None
    return r


# ───────────────────────── tamper / re-run ─────────────────────────
def summarize(p) -> str:
    def n(v): return f"{v:,.2f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v))
    if p["kind"] == "scalar": return n(p["value"])
    if p["kind"] == "series":
        return "no rows" if not p["values"] else f"{p['index'][0]} = {n(p['values'][0])}" + (f"  (+{p['total_rows'] - 1} more)" if p["total_rows"] > 1 else "")
    return "no rows" if not p["rows"] else ", ".join(f"{c} = {n(v)}" for c, v in zip(p["columns"], p["rows"][0])) + (f"  (+{p['total_rows'] - 1} more)" if p["total_rows"] > 1 else "")


def rerun(ws: Workspace, question: str, original_payload: dict, code: str) -> dict:
    """Execute user-edited code, independently reproduce it, and compare with the originally verified result."""
    problems = guard(code); used = referenced_tables(code, ws.tables)
    if not problems and not used: problems.append("The code must use at least one loaded table: " + ", ".join(ws.tables))
    if problems: return {"status": "rejected", "error": "; ".join(problems)}
    run = sandbox_run(ws.pickles(used), code)
    if run["status"] == "cannot_determine": return {"status": "cannot_determine", "error": run["reason"]}
    if run["status"] != "ok": return {"status": "error", "error": run["error"] + (f" (line {run['line']})" if run.get("line") else "")}
    rep = reproduce(ws, question, used, code)
    return {"status": "ok", "same_as_original": common.payloads_match(run["payload"], original_payload),
            "reproduced": rep["status"] == "ok" and common.payloads_match(run["payload"], rep["payload"]),
            "original": summarize(original_payload), "edited": summarize(run["payload"]), "secs": round(run["secs"], 2)}
