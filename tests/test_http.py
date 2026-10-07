"""End-to-end over real HTTP: sample data -> answer -> proof -> re-verify -> refusal.   python tests/test_http.py"""
import json, os, sys, tempfile, threading, urllib.request
os.environ["PROOFAI_DATA"] = tempfile.mkdtemp(); os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import server
from http.server import ThreadingHTTPServer
srv = ThreadingHTTPServer(("127.0.0.1", 0), server.H); threading.Thread(target=srv.serve_forever, daemon=True).start()
B = f"http://127.0.0.1:{srv.server_port}"
def call(path, body=None):
    req = urllib.request.Request(B + path, data=None if body is None else json.dumps(body).encode(), headers={"X-Session": "t", "Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req) as r: return r.read().decode()
j = lambda *a: json.loads(call(*a))
assert len(j("/api/datasets/sample", {})["datasets"]) == 3
r = j("/api/analyze", {"question": "Which region generated the highest Q2 revenue?"}); assert not r["no"] and r["a"] == "South", r
assert "result =" in call(r["script_url"]); v = j("/api/verify/" + r["id"], {}); assert v["reproduced"], v
r2 = j("/api/analyze", {"question": "What was the revenue in 2021?"}); assert r2["no"] and "2021" in r2["s"]
assert "<html" in call("/").lower(); print("HTTP end-to-end: all checks passed")
