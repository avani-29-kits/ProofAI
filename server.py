"""ProofAI server — standard library only (needs just pandas/numpy/openpyxl).
    python server.py            → http://localhost:8000   (PORT env var respected)
Same API as main.py (FastAPI), but with zero web-framework dependencies."""
from __future__ import annotations
import json, os, re
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
import common, engine, planner
from store import Store

HERE = Path(__file__).parent
store = Store()
MAX_BODY = 60 * 1024 * 1024


def multipart(ctype: str, body: bytes):
    msg = BytesParser(policy=default).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    return [(p.get_filename(), p.get_payload(decode=True)) for p in msg.iter_parts() if p.get_filename()]


class H(BaseHTTPRequestHandler):
    server_version = "ProofAI/1.0"

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else (json.dumps(body, default=str).encode() if ctype == "application/json" else body.encode())
        self.send_response(code); self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
        self.send_header("Content-Length", str(len(data))); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*"); self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(data)

    def _err(self, code, msg): self._send(code, {"error": msg})
    def ws(self): return engine.Workspace(self.headers.get("X-Session", "default"))
    def log_message(self, *a): pass
    def do_OPTIONS(self): self._send(204, b"", "text/plain")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY: raise ValueError("Upload too large (60 MB max)")
        return self.rfile.read(n)

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p in ("/", "/index.html"): return self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html")
            if p == "/api/health":
                return self._send(200, {"ok": True, "planner": "llm" if planner.llm_enabled() else "offline", "model": planner.MODEL if planner.llm_enabled() else None})
            if p == "/api/datasets": return self._send(200, {"datasets": self.ws().public_list()})
            if p == "/api/history": return self._send(200, {"history": store.history(self.ws().id)})
            m = re.fullmatch(r"/api/proof/([A-Za-z0-9]+)\.py", p)
            if m:
                row = store.get(m.group(1))
                if not row or not row["script"]: return self._err(404, "No proof script for this analysis")
                return self._send(200, row["script"], "text/plain", {"Content-Disposition": f'attachment; filename="proof_{m.group(1)}.py"'})
            return self._err(404, "Not found")
        except Exception as e: return self._err(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        p = self.path.split("?")[0]
        try:
            if p == "/api/datasets":
                ws, errors, added = self.ws(), [], []
                for fn, data in multipart(self.headers.get("Content-Type", ""), self._body()):
                    try: added += [a["name"] for a in ws.ingest(fn, data)]
                    except Exception as e: errors.append({"file": fn, "error": str(e)})
                return self._send(200, {"datasets": ws.public_list(), "added": added, "errors": errors})
            if p == "/api/datasets/sample":
                ws = self.ws()
                for name in ("sales.csv", "customers.csv", "products.csv"):
                    f = HERE / "sample_data" / name
                    if not f.exists():
                        import make_sample_data  # noqa: F401  (generates the files on first use)
                    ws.ingest(name, f.read_bytes())
                return self._send(200, {"datasets": ws.public_list()})
            if p == "/api/analyze":
                q = (json.loads(self._body() or b"{}").get("question") or "").strip()
                if not q: return self._err(400, "Empty question")
                r = engine.analyze(q, self.ws(), store); r.pop("script", None)
                return self._send(200, r)
            m = re.fullmatch(r"/api/rerun/([A-Za-z0-9]+)", p)
            if m:
                row = store.get(m.group(1))
                if not row or row["status"] != "verified": return self._err(404, "Only verified analyses can be edited and re-run")
                code = (json.loads(self._body() or b"{}").get("code") or "").strip()
                if not code: return self._err(400, "Empty code")
                ws = engine.Workspace(row["session"])
                return self._send(200, engine.rerun(ws, row["question"], json.loads(row["response"])["result"], code))
            m = re.fullmatch(r"/api/verify/([A-Za-z0-9]+)", p)
            if m:
                row = store.get(m.group(1))
                if not row or row["status"] != "verified": return self._err(404, "Only verified analyses can be re-verified")
                ws = engine.Workspace(row["session"]); used = json.loads(row["tables"])
                if any(u not in ws.tables for u in used): return self._err(409, "Datasets have since been removed")
                rep = engine.reproduce(ws, row["question"], used, row["code"])
                same = rep["status"] == "ok" and common.payloads_match(json.loads(row["response"])["result"], rep["payload"])
                return self._send(200, {"reproduced": same, "status": rep["status"], "detail": None if same else rep.get("error", "result differs")})
            return self._err(404, "Not found")
        except Exception as e: return self._err(500, f"{type(e).__name__}: {e}")

    def do_DELETE(self):
        m = re.fullmatch(r"/api/datasets/([\w\-]+)", self.path.split("?")[0])
        if m and self.ws().delete(unquote(m.group(1))): return self._send(200, {"ok": True})
        return self._err(404, "No such dataset")


def serve(port=None):
    port = int(port or os.environ.get("PORT", 8000))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"ProofAI running → http://localhost:{port}   planner: {'Claude' if planner.llm_enabled() else 'offline heuristic'}")
    srv.serve_forever()


if __name__ == "__main__":
    serve()
