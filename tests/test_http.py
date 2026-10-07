"""Full HTTP test against the real server (no mocks)."""
import os, sys, tempfile, threading, json, time, urllib.request, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PROOFAI_DATA"] = tempfile.mkdtemp(); os.environ.pop("ANTHROPIC_API_KEY", None)
import server
from http.server import ThreadingHTTPServer
from pathlib import Path
srv = ThreadingHTTPServer(("127.0.0.1", 0), server.H); PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
B = f"http://127.0.0.1:{PORT}"; SID = {"X-Session": "e2e"}

def call(method, path, data=None, headers=None, raw=False):
    req = urllib.request.Request(B + path, data=data, method=method, headers={**SID, **(headers or {})})
    try: r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e: r = e
    b = r.read(); return (b if raw else json.loads(b)), r.status

def post_json(path, obj): return call("POST", path, json.dumps(obj).encode(), {"Content-Type": "application/json"})

def upload(*names):
    bd = uuid.uuid4().hex; body = b""
    for n in names:
        body += (f'--{bd}\r\nContent-Disposition: form-data; name="files"; filename="{n}"\r\nContent-Type: text/csv\r\n\r\n').encode() + (Path(__file__).parent.parent / "sample_data" / n).read_bytes() + b"\r\n"
    body += f"--{bd}--\r\n".encode()
    return call("POST", "/api/datasets", body, {"Content-Type": f"multipart/form-data; boundary={bd}"})

page, st = call("GET", "/", raw=True); assert st == 200 and b"ProofAI" in page; print("✓ serves the web app")
h, _ = call("GET", "/api/health"); assert h["ok"]; print("✓ health", h)
r, st = upload("sales.csv", "customers.csv", "products.csv"); assert [d["rows"] for d in r["datasets"]] == [3040, 300, 40], r; print("✓ multipart upload", [(d["file"], d["rows"]) for d in r["datasets"]])
r, _ = post_json("/api/analyze", {"question": "Which region generated the highest Q2 revenue?"}); assert r["a"] == "South" and not r["no"]; print("✓ verified answer:", r["a"], "|", r["s"], "|", r["tm"])
aid = r["id"]; s, st = call("GET", r["script_url"], raw=True); assert st == 200 and b"print(result)" in s; print("✓ proof.py download", len(s), "bytes")
v, _ = post_json(f"/api/verify/{aid}", {}); assert v["reproduced"]; print("✓ re-verify endpoint:", v)
r2, _ = post_json("/api/analyze", {"question": "What was the revenue in 2021?"}); assert r2["no"] and "2021" in r2["s"]; print("✓ smart refusal:", r2["a"], "|", r2["s"])
r3, st = post_json("/api/analyze", {"question": ""}); assert st == 400; print("✓ empty question rejected")
hist, _ = call("GET", "/api/history"); assert len(hist["history"]) == 2; print("✓ history", [h["status"] for h in hist["history"]])
bad, _ = upload("trap_mixed_currency.csv"); r4, _ = post_json("/api/analyze", {"question": "What is the total revenue?"}); print("✓ mixed-currency question →", r4["a"], "|", r4["s"][:70])
d, st = call("DELETE", "/api/datasets/trap_mixed_currency"); assert st == 200; print("✓ delete dataset")
call("GET", "/api/datasets"); d, st = call("DELETE", "/api/datasets/nope"); assert st == 404
code = r["code"]
same, _ = post_json(f"/api/rerun/{aid}", {"code": code}); assert same["status"] == "ok" and same["same_as_original"] and same["reproduced"]; print("✓ rerun unchanged code → same:", same["original"])
tam, _ = post_json(f"/api/rerun/{aid}", {"code": code.replace("quarter == 2", "quarter == 3")}); assert tam["status"] == "ok" and not tam["same_as_original"]; print("✓ tampered code → differs:", tam["original"], "→", tam["edited"])
bad2, _ = post_json(f"/api/rerun/{aid}", {"code": "import os\nresult = 1"}); assert bad2["status"] == "rejected"; print("✓ unsafe edit rejected:", bad2["error"])
err, _ = post_json(f"/api/rerun/{aid}", {"code": "result = nosuch.sum()"}); assert err["status"] in ("error", "rejected"); print("✓ broken edit reported:", err["status"], err["error"][:50])
print("\nHTTP E2E PASSED")
