"""ProofAI benchmark: answerable questions (checked against independent pandas ground truth)
and trick questions (must be refused). Run:  python evaluate.py   [--llm to use ANTHROPIC_API_KEY]"""
import os, sys, tempfile, re, time
os.environ["PROOFAI_DATA"] = tempfile.mkdtemp()
if "--llm" not in sys.argv: os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, engine
from pathlib import Path
D = Path(__file__).parent / "sample_data"
sales = pd.read_csv(D / "sales.csv").drop_duplicates(); sales["revenue"] = pd.to_numeric(sales["revenue"], errors="coerce")
cust = pd.read_csv(D / "customers.csv"); prod = pd.read_csv(D / "products.csv")
sales["dt"] = pd.to_datetime(sales["order_date"]); m = sales.merge(cust, on="customer_id", how="left").merge(prod[["product_id", "product_name"]], on="product_id", how="left")
STD = ["sales.csv", "customers.csv", "products.csv"]
tp = pd.read_csv(D / "text_prices_usd_only.csv"); tp["p"] = tp["price"].str.replace(r"[^0-9.]", "", regex=True).astype(float)

def num(r):  # headline value as float from the structured result
    p = r.get("result") or {}
    return p.get("value") if p.get("kind") == "scalar" else (p.get("values") or [None])[0]

# (name, files, question, kind, check)   kind: ok → check(r) must be True ; no → must be refused (substring optional)
Q = [
 ("Q2 top region", STD, "Which region generated the highest Q2 revenue?", "ok", lambda r: r["a"] == m[m.dt.dt.quarter == 2].groupby("region")["revenue"].sum().idxmax()),
 ("Q2 top region value", STD, "Which region generated the highest Q2 revenue?", "ok", lambda r: abs(num(r) - m[m.dt.dt.quarter == 2].groupby("region")["revenue"].sum().max()) < 0.01),
 ("Total revenue", STD, "What is the total revenue?", "ok", lambda r: abs(num(r) - sales["revenue"].sum()) < 0.01),
 ("Top product", STD, "Top 5 products by revenue", "ok", lambda r: r["a"] == m.groupby("product_name")["revenue"].sum().idxmax()),
 ("Lowest region", STD, "Which region has the lowest revenue?", "ok", lambda r: r["a"] == m.groupby("region")["revenue"].sum().idxmin()),
 ("Avg quantity", STD, "What is the average quantity per order?", "ok", lambda r: abs(num(r) - sales["quantity"].mean()) < 1e-6),
 ("Orders in 2024", STD, "How many orders were placed in 2024?", "ok", lambda r: num(r) == (sales.dt.dt.year == 2024).sum()),
 ("2024 revenue by region", STD, "Compare regions by revenue in 2024", "ok", lambda r: r["a"] == m[m.dt.dt.year == 2024].groupby("region")["revenue"].sum().idxmax()),
 ("Missing data report", STD, "Test missing data", "ok", lambda r: "revenue" in str(r["result"])),
 ("Outlier detection", STD, "Find unusual values in revenue", "ok", lambda r: "4500000" in str(r["result"])),
 ("Year not in data (2021)", STD, "What was the revenue in 2021?", "no", "2021"),
 ("Year not in data (2019)", STD, "Total revenue for 2019", "no", "2019"),
 ("Future year (2026)", STD, "What will be the revenue in 2026?", "no", "2026"),
 ("Partly missing years", STD, "Compare revenue 2021 vs 2024", "no", "2021"),
 ("Invalid quarter Q5", STD, "What was the revenue in Q5?", "no", "Q5"),
 ("Nonexistent entity", STD, "What was the revenue in the Atlantis region?", "no", "Atlantis"),
 ("Column does not exist", ["sales.csv"], "What is the average employee salary?", "no", None),
 ("Not a data question", STD, "What is the meaning of life?", "no", None),
 ("Ambiguous dates", ["trap_ambiguous_dates.csv"], "What was the total revenue in Q2?", "no", "DD/MM"),
 ("Mixed currencies", ["trap_mixed_currency.csv"], "What is the total revenue?", "no", "currenc"),
 ("Contradictory tables", ["sales.csv", "trap_conflicting_customers.csv"], "Which region has the highest revenue?", "no", "conflicting"),
 ("$ and € inside values", ["trap_usd_eur_values.csv"], "What is the total price?", "no", "EUR"),
 ("Single-currency text values", ["text_prices_usd_only.csv"], "What is the total price?", "ok", lambda r: abs(num(r) - tp["p"].sum()) < 0.01 and r["a"].startswith("$")),
 ("Tables disagree (amount)", ["trap_orders.csv", "trap_payments.csv"], "What is the total amount?", "no", "contradict"),
 ("Relative time: last quarter", STD, "What was the revenue last quarter?", "no", "relative"),
 ("Relative time: recent", STD, "Which product sold best recently?", "no", "relative"),
 ("No data uploaded", [], "What is the total revenue?", "no", None),
]

rows, ok_n = [], 0
for i, (name, files, q, kind, chk) in enumerate(Q):
    ws = engine.Workspace(f"eval{i}")
    for f in files: ws.ingest(f, (D / f).read_bytes())
    t = time.time(); r = engine.analyze(q, ws)
    if kind == "ok": passed = (not r["no"]) and bool(chk(r)); verified = "✓ verified" if not r["no"] else "✗ refused"
    else: passed = bool(r["no"]) and (chk is None or chk.lower() in r["s"].lower()); verified = "✓ refused" if r["no"] else "✗ ANSWERED"
    ok_n += passed
    rows.append((passed, name, kind, verified, (r["a"] + " — " + r["s"])[:58], f"{time.time() - t:.1f}s"))
if "--json" in sys.argv:
    import json; print(json.dumps({"rows": [{"pass": bool(p), "name": n, "kind": k, "detail": d, "secs": t} for p, n, k, v, d, t in rows], "passed": int(ok_n), "total": len(rows)})); sys.exit(0)
w = max(len(r[1]) for r in rows)
print(f"\n{'':2}{'case':<{w}}  {'expect':<8}{'got':<12}detail")
for p, n, k, v, d, t in rows: print(f"{'✅' if p else '❌'} {n:<{w}}  {'answer' if k == 'ok' else 'refuse':<8}{v:<12}{d}")
ans = [r for r in rows if r[2] == "ok"]; ref = [r for r in rows if r[2] == "no"]
print(f"\nanswerable correct: {sum(r[0] for r in ans)}/{len(ans)}   trick questions refused: {sum(r[0] for r in ref)}/{len(ref)}   overall {ok_n}/{len(rows)} = {100 * ok_n / len(rows):.0f}%")
sys.exit(0 if ok_n == len(rows) else 1)
