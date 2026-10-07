"""Planner: turns a question + table profiles into {answerable, code, ...}.
Uses the Anthropic Messages API when ANTHROPIC_API_KEY is set (with a repair loop fed by the
engine's validation errors). Otherwise / on API failure it falls back to an offline heuristic
planner that handles common aggregate questions, so the demo always works."""
from __future__ import annotations
import json, os, re, urllib.request
from pathlib import Path
import pandas as pd


def _load_env():
    p = Path(__file__).parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()
MODEL = os.environ.get("PROOFAI_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are the planning core of ProofAI, a proof-carrying data analyst. You receive profiles of pandas tables and a question. Return ONLY one JSON object (no markdown fences):
{"answerable": bool,
 "reason": "if not answerable: why, max 25 words, citing concrete evidence from the profiles",
 "description": "max 8 words naming the answer, e.g. 'Q2 revenue'",
 "unit": "currency symbol or short unit, '' if unknown (do not assume a currency the data does not state)",
 "answer_kind": "scalar" | "ranking" | "list",
 "assumptions": ["short strings, e.g. period assumptions"],
 "evidence": ["up to 3 short strings: filters / columns used"],
 "code": "python"}
CODE RULES
1. Every table is already loaded as a pandas DataFrame named exactly like its table name. `pd` and `np` are available. No file, network or system access; imports limited to pandas, numpy, math, re, datetime, statistics.
2. Assign the final answer to a variable `result`: a scalar, or a pandas Series (index = label) / DataFrame. For which/top/highest/lowest questions return a Series sorted best-to-worst, at most 10 rows, and set answer_kind "ranking".
3. Never hardcode numbers taken from the profiles; compute everything from the data.
4. Clean explicitly in the code: drop exact duplicates (drop_duplicates), coerce numerics (pd.to_numeric(errors="coerce")), parse dates explicitly, handle missing values, join tables on real keys.
5. If the answer cannot be computed reliably — missing period/entity/column, ambiguous date format (DD/MM vs MM/DD), mixed units or currencies with no conversion data in the tables, tables that contradict each other, an invalid or trick question — set answerable=false, or check in code and `raise CannotDetermine("reason")`. NEVER guess, invent an exchange rate, or fill in a value.
6. If the period is unspecified and the data spans several periods, aggregate over all of it and say so in assumptions."""


def prompt_profile(metas: dict) -> str:
    out = []
    for n, m in metas.items():
        p = m["profile"]
        cols = []
        for c in p["columns"]:
            d = {"name": c["name"], "dtype": c["dtype"], "nulls": c["nulls"], "unique": c["unique"], "sample": c["sample"][:3]}
            if "min" in c: d["range"] = [c["min"], c["max"]]
            if "date" in c: d["date"] = {"format": c["date"]["format"], "min": c["date"]["min"], "max": c["date"]["max"], "years": list(c["date"]["years"])}
            cols.append(d)
        out.append({"table": n, "file": m["display"], "rows": p["rows"], "duplicate_rows": p["duplicate_rows"], "columns": cols})
    return json.dumps(out, ensure_ascii=False)


def llm_enabled() -> bool: return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_llm(messages, max_tokens=2500) -> str:
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "system": SYSTEM, "messages": messages}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)
    return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


def _normalize(raw: str) -> dict:
    i, j = raw.find("{"), raw.rfind("}")
    d = json.loads(raw[i:j + 1])
    return {"answerable": bool(d.get("answerable", True)), "reason": str(d.get("reason", "")), "description": str(d.get("description", "Result"))[:80],
            "unit": str(d.get("unit", "") or ""), "answer_kind": d.get("answer_kind") if d.get("answer_kind") in ("scalar", "ranking", "list") else "scalar",
            "assumptions": [str(x) for x in d.get("assumptions", [])][:4], "evidence": [str(x) for x in d.get("evidence", [])][:3],
            "code": str(d.get("code", "")), "_raw": raw}


def plan(question: str, metas: dict, history: list) -> dict:
    if llm_enabled():
        try:
            msgs = [{"role": "user", "content": f"Table profiles:\n{prompt_profile(metas)}\n\nQuestion: {question}"}]
            for prev, fb in history:
                msgs += [{"role": "assistant", "content": prev}, {"role": "user", "content": fb}]
            p = _normalize(call_llm(msgs))
            if p["answerable"] and not p["code"].strip(): raise ValueError("empty code")
            p["planner"] = "llm"; return p
        except Exception as e:  # network / parse failure → offline planner
            p = heuristic(question, metas); p["planner"] = f"offline (LLM unavailable: {type(e).__name__})"; return p
    p = heuristic(question, metas); p["planner"] = "offline"; return p


# ───────────────────────── offline heuristic planner ─────────────────────────
MONEY = ("revenue", "sales", "amount", "price", "profit", "cost", "income", "spend", "total", "value")


def _num(c): return c["dtype"].lower().startswith(("int", "float", "uint")) and not re.search(r"(^|_)id$|^id$", c["name"].lower())
def _numish(c): return _num(c) or "numeric_text" in c
def _txt(c): return not _num(c) and "numeric_text" not in c and "date" not in c and not c["dtype"].lower().startswith(("datetime", "bool"))
def _idlike(n): return bool(re.search(r"(^|_)id$|^id$|code$", n.lower()))
def _tokens(s): return set(re.findall(r"[a-z]+", s.lower()))


def _no(reason): return {"answerable": False, "reason": reason, "description": "", "unit": "", "answer_kind": "scalar", "assumptions": [], "evidence": [], "code": ""}


def heuristic(question: str, metas: dict) -> dict:
    q = question.lower(); qt = _tokens(q); qt |= {t.rstrip("s") for t in qt}
    if re.search(r"missing|null|blank|incomplete", q):
        parts = ", ".join(f'"{n}": {n}.isna().sum()' for n in metas)
        code = (f"nulls = pd.concat({{{parts}}})\nnulls.index = [f\"{{t}}.{{c}}\" for t, c in nulls.index]\nresult = nulls[nulls > 0].sort_values(ascending=False)")
        return {"answerable": True, "reason": "", "description": "Missing values by column", "unit": "", "answer_kind": "list", "assumptions": [], "evidence": ["isna() on every column"], "code": code}
    cands = [(t, c) for t, m in metas.items() for c in m["profile"]["columns"] if _numish(c)]
    def mscore(tc):
        n = tc[1]["name"].lower(); parts = set(n.split("_"))
        return 3 * bool(parts & qt) + 2 * any(k in n and k in q for k in MONEY)
    def default_score(tc): return 1 if any(k in tc[1]["name"].lower() for k in MONEY[:6]) else 0
    ranked = sorted(cands, key=mscore, reverse=True)
    if ranked and mscore(ranked[0]) == 0 and re.search(r"compare|rank|performance|best|top\b|worst", q):
        ranked = sorted(cands, key=default_score, reverse=True)   # generic comparison: fall back to the main money column
        if not ranked or default_score(ranked[0]) == 0: ranked = []
    elif ranked and mscore(ranked[0]) == 0:
        ranked = []
    nomeasure = False
    if not ranked and re.search(r"how many|count|number of", q):   # a count needs no measure column: count rows of the main table
        mt0 = max(metas, key=lambda t: metas[t]["profile"]["rows"])
        ranked = [(mt0, metas[mt0]["profile"]["columns"][0])]; nomeasure = True
    if not ranked:
        cols = ", ".join(sorted({c["name"] for _, c in cands})[:8]) or "none"
        return _no(f"No column in the data matches this question (numeric columns available: {cols}), so it cannot be answered reliably.")
    mt, mc = ranked[0][0], ranked[0][1]["name"]
    agg = "mean" if re.search(r"average|mean|avg", q) else "count" if re.search(r"how many|count|number of", q) else "sum"
    asc = bool(re.search(r"lowest|least|smallest|minimum|fewest|worst|bottom", q))
    m = re.search(r"(?:top|bottom)\s+(\d+)", q); topn = int(m.group(1)) if m else 10
    dims = []
    for t, mm in metas.items():
        for c in mm["profile"]["columns"]:
            if _txt(c) and (_tokens(c["name"]) & qt) and (not _idlike(c["name"]) or "id" in qt):
                dims.append((t, c["name"], _idlike(c["name"])))
    dims.sort(key=lambda x: x[2])
    wants_dim = bool(re.search(r"\bwhich\b|\bby\b|\bper\b|\btop\b|compare|each|highest|lowest|best|worst", q))
    dim = dims[0] if (dims and wants_dim) else None
    L = [f"data = {mt}.drop_duplicates()"]; ev = ["duplicates removed"]; assumptions = []; filt = ""
    if dim and dim[0] != mt:
        common = [c["name"] for c in metas[mt]["profile"]["columns"] if c["name"] in {x["name"] for x in metas[dim[0]]["profile"]["columns"]}]
        if not common: return _no(f"No shared key column links {mt} and {dim[0]}, so they cannot be combined reliably.")
        key = sorted(common, key=lambda n: not _idlike(n))[0]; d = dim[0]
        L += [f"lookup = {d}.drop_duplicates()[[\"{key}\", \"{dim[1]}\"]].drop_duplicates()",
              f"if lookup.duplicated(subset=\"{key}\").any():",
              f"    raise CannotDetermine(\"{d} lists conflicting {dim[1]} values for the same {key}\")",
              f"data = data.merge(lookup, on=\"{key}\", how=\"left\")"]
        ev.append(f"joined on {key}")
    ntinfo = next((c.get("numeric_text") for c in metas[mt]["profile"]["columns"] if c["name"] == mc), None)
    if ntinfo and not nomeasure:
        L.append(f"data[\"{mc}\"] = pd.to_numeric(data[\"{mc}\"].astype(str).str.replace(r\"[^0-9.\\-]\", \"\", regex=True), errors=\"coerce\")"); ev.append("units/symbols stripped")
    elif not nomeasure: L.append(f"data[\"{mc}\"] = pd.to_numeric(data[\"{mc}\"], errors=\"coerce\")")
    if agg != "count": L.append(f"data = data.dropna(subset=[\"{mc}\"])"); ev.append("missing values excluded")
    # time filters
    qm = re.search(r"\bq([1-4])\b", q); ym = re.search(r"\b(20\d\d)\b", q)
    if qm or ym:
        src = next(((t, c) for t in ([mt] + list(metas)) for c in metas[t]["profile"]["columns"] if c.get("date") and t == mt), None)
        qcol = next((c["name"] for c in metas[mt]["profile"]["columns"] if "quarter" in c["name"].lower()), None)
        if qm and qcol:
            L.append(f"data = data[data[\"{qcol}\"].astype(str).str.upper().str.contains(\"Q{qm.group(1)}\")]"); filt = f"Q{qm.group(1)} "; ev.append(f"{qcol} == Q{qm.group(1)}")
        elif src:
            col = src[1]; dd = col["date"]
            arg = "format=\"mixed\"" if dd["format"] == "mixed" else f"dayfirst={dd['format'] == 'dayfirst'}"
            L.append(f"_d = pd.to_datetime(data[\"{col['name']}\"], errors=\"coerce\", {arg})")
            conds = []
            if qm: conds.append(f"(_d.dt.quarter == {qm.group(1)})"); filt += f"Q{qm.group(1)} "; assumptions.append(f"Q{qm.group(1)} read as a calendar quarter ({['Jan–Mar','Apr–Jun','Jul–Sep','Oct–Dec'][int(qm.group(1))-1]}), not a fiscal quarter")
            if ym: conds.append(f"(_d.dt.year == {ym.group(1)})"); filt += f"{ym.group(1)} "
            L.append(f"data = data[{' & '.join(conds)}]"); ev.append(f"filter on {col['name']}")
            if qm and not ym and len(dd["years"]) > 1:
                assumptions.append(f"Q{qm.group(1)} aggregated across all years in the data ({min(dd['years'])}–{max(dd['years'])})")
        else:
            return _no("The question refers to a time period but no usable date or quarter column exists.")
        L += ["if data.empty:", "    raise CannotDetermine(\"No rows match the requested period\")"]
    fn = {"sum": "sum", "mean": "mean", "count": "size"}[agg]
    monetary = any(k in mc.lower() for k in MONEY[:7]) and not any(_UNIT(c["name"]) for c in metas[mt]["profile"]["columns"])
    unit = "₹" if monetary and agg != "count" else ""
    if ntinfo and len(ntinfo["units"]) == 1 and agg != "count":
        u0 = ntinfo["units"][0]; unit = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(u0, u0); monetary = False
    elif ntinfo: unit = ""
    if unit == "₹" and monetary: assumptions.append("Currency assumed to be INR (no currency column in the data)")
    word = {"sum": "", "mean": "average ", "count": "count of "}[agg]
    desc = (f"{filt}row count" if agg == "count" else f"{filt}{word}{mc.replace('_', ' ')}").strip()
    if re.search(r"unusual|outlier|anomal|strange|suspicious", q):
        idc = next((c["name"] for c in metas[mt]["profile"]["columns"] if _idlike(c["name"])), None)
        L += [f"z = (data[\"{mc}\"] - data[\"{mc}\"].mean()) / data[\"{mc}\"].std()",
              f"result = data.loc[z.abs() > 3, [{', '.join(repr(x) for x in [idc, mc] if x)}]].sort_values(\"{mc}\", ascending=False)"]
        return {"answerable": True, "reason": "", "description": f"Unusual {mc.replace('_', ' ')} values (|z| > 3)", "unit": unit, "answer_kind": "list", "assumptions": assumptions, "evidence": ev[:3], "code": "\n".join(L)}
    if dim:
        gb = f"data.groupby(\"{dim[1]}\")" + (f"[\"{mc}\"].{fn}()" if agg != "count" else ".size()")
        L.append(f"result = {gb}.sort_values(ascending={asc}).head({topn})")
        ev.append(f"grouped by {dim[1]}"); kind = "ranking"
    else:
        L.append(f"result = {'int(len(data))' if agg == 'count' else f'data[\"{mc}\"].{agg}()'}"); kind = "scalar"
    return {"answerable": True, "reason": "", "description": desc, "unit": unit, "answer_kind": kind, "assumptions": assumptions, "evidence": ev[-3:], "code": "\n".join(L)}


def _UNIT(n): return bool(re.search(r"^(unit|units|uom|currency|ccy|unit_of_measure|currency_code)$|_(unit|units|uom|currency|ccy)$", n.lower()))
