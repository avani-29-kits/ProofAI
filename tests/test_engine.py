import os, sys, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PROOFAI_DATA"] = tempfile.mkdtemp(); os.environ.pop("ANTHROPIC_API_KEY", None)
import engine, store, planner
from pathlib import Path
S = Path(__file__).parent.parent / "sample_data"

def ws_with(*files, sid="t"):
    ws = engine.Workspace(sid)
    for f in files: ws.ingest(f, (S / f).read_bytes())
    return ws

def run(ws, q, st=None): return engine.analyze(q, ws, st)

def test_verified_answer():
    ws = ws_with("sales.csv", "customers.csv", "products.csv"); st = store.Store()
    r = run(ws, "Which region generated the highest Q2 revenue?", st)
    print(r["a"], r["s"], r["ev"], r["chk"], r["tm"]); assert not r["no"] and r["a"] == "South"
    assert "lakh" in r["s"] or "crore" in r["s"]; assert r["ch"]["l"][0] == "South"
    assert "Calculation reproduced" in " ".join(r["chk"]); assert r["script_url"]
    got = st.get(r["id"]); assert "print(result)" in got["script"]
    # the downloadable script must run standalone and print the same ranking
    import subprocess, sys as _s
    p = subprocess.run([_s.executable, "-"], input=got["script"].replace("'" + str(ws.files) + "/", "'"), capture_output=True, text=True, cwd=ws.files)
    print(p.stdout[-300:], p.stderr[-300:]); assert "South" in p.stdout

def test_missing_year_refusal():
    r = run(ws_with("sales.csv", sid="y"), "What was the revenue in 2021?")
    print(r["a"], r["s"], r["ch"]); assert r["no"] and "2021" in r["s"] and r["ch"]["v"][0] == 0

def test_ambiguous_dates():
    r = run(ws_with("trap_ambiguous_dates.csv", sid="a"), "What was the total revenue in Q2?")
    print(r["s"]); assert r["no"] and "DD/MM" in r["s"]

def test_mixed_currency():
    r = run(ws_with("trap_mixed_currency.csv", sid="m"), "What is the total revenue?")
    print(r["s"]); assert r["no"] and "currency" in r["s"].lower()

def test_contradictory_tables():
    r = run(ws_with("sales.csv", "trap_conflicting_customers.csv", sid="c"), "Which region has the highest revenue?")
    print(r["s"]); assert r["no"] and "conflicting" in r["s"]

def test_scalar_and_outliers_and_missing():
    ws = ws_with("sales.csv", "customers.csv", sid="s")
    r = run(ws, "What is the total revenue?"); print(r["a"], r["note"][:80]); assert not r["no"]
    r = run(ws, "Find unusual values in revenue"); print(r["a"], r["s"]); assert not r["no"] and "4500000" in str(r["result"])
    r = run(ws, "Test missing data"); print(r["a"]); assert not r["no"]

def test_sandbox_blocks_and_tamper():
    assert engine.guard("import os\nresult=1") and engine.guard("result = open('x').read()") and engine.guard("import pandas as pd\nresult = pd.read_csv('a')")
    assert not engine.guard("import pandas as pd\nresult = 1+1")
    ws = ws_with("sales.csv", sid="g"); pk = ws.pickles(["sales"])
    r = engine.sandbox_run(pk, "result = sales['revenue'].sum()"); assert r["status"] == "ok"
    r = engine.sandbox_run(pk, "x = 1/0\nresult = x"); assert r["status"] == "error" and "ZeroDivision" in r["error"]
    r = engine.sandbox_run(pk, "while True: pass", timeout=3); assert r["status"] == "error"

def test_verification_catches_nondeterminism():
    ws = ws_with("sales.csv", sid="n"); code = "import numpy as np\nresult = float(np.random.rand())"
    run1 = engine.sandbox_run(ws.pickles(["sales"]), code); rep = engine.reproduce(ws, "q", ["sales"], code)
    assert not engine.common.payloads_match(run1["payload"], rep["payload"])

def test_llm_repair_loop(monkeypatch=None):
    calls = []
    bad = '{"answerable":true,"description":"Total revenue","unit":"₹","answer_kind":"scalar","code":"result = sales[\'revnue\'].sum()"}'
    good = '{"answerable":true,"description":"Total revenue","unit":"₹","answer_kind":"scalar","assumptions":[],"evidence":["revenue column"],"code":"d = sales.drop_duplicates()\\nresult = pd.to_numeric(d[\'revenue\'], errors=\'coerce\').sum()"}'
    def fake(msgs, max_tokens=0): calls.append(len(msgs)); return bad if len(calls) == 1 else good
    planner.call_llm = fake; os.environ["ANTHROPIC_API_KEY"] = "x"
    try:
        r = run(ws_with("sales.csv", sid="l"), "What is the total revenue?")
    finally: os.environ.pop("ANTHROPIC_API_KEY"); 
    print(r["planner"], r["attempts"], r["a"]); assert r["planner"] == "llm" and r["attempts"] == 2 and not r["no"]

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): print("──", k); f()
    print("\nALL TESTS PASSED")
