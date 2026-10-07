"""ProofAI API.   uvicorn main:app --reload --port 8000"""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import common, engine, planner
from store import Store

app = FastAPI(title="ProofAI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
store = Store()
HERE = Path(__file__).parent


class Ask(BaseModel):
    question: str


def W(session: str) -> engine.Workspace: return engine.Workspace(session)


@app.get("/api/health")
def health():
    return {"ok": True, "planner": "llm" if planner.llm_enabled() else "offline", "model": planner.MODEL if planner.llm_enabled() else None}


@app.get("/api/datasets")
def list_datasets(x_session: str = Header("default")):
    return {"datasets": W(x_session).public_list()}


@app.post("/api/datasets")
async def upload(files: list[UploadFile] = File(...), x_session: str = Header("default")):
    ws, added, errors = W(x_session), [], []
    for f in files:
        try: added += ws.ingest(f.filename or "upload", await f.read())
        except Exception as e: errors.append({"file": f.filename, "error": str(e)})
    return {"datasets": ws.public_list(), "added": [a["name"] for a in added], "errors": errors}


@app.post("/api/datasets/{which}")
def load_set(which: str, x_session: str = Header("default")):
    if which not in engine.SETS: raise HTTPException(404, "Unknown dataset set")
    ws = W(x_session); engine.load_set(ws, which, replace=which == "traps")
    return {"datasets": ws.public_list()}


@app.get("/api/benchmark")
def benchmark(): return engine.benchmark()


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str, x_session: str = Header("default")):
    if not W(x_session).delete(name): raise HTTPException(404, "No such dataset")
    return {"ok": True}


@app.post("/api/analyze")
def analyze(body: Ask, x_session: str = Header("default")):
    if not body.question.strip(): raise HTTPException(400, "Empty question")
    r = engine.analyze(body.question, W(x_session), store)
    r.pop("script", None)
    return r


@app.get("/api/history")
def history(x_session: str = Header("default")):
    return {"history": store.history(W(x_session).id)}


@app.get("/api/proof/{aid}.py", response_class=PlainTextResponse)
def proof(aid: str):
    row = store.get(aid)
    if not row or not row["script"]: raise HTTPException(404, "No proof script for this analysis")
    return PlainTextResponse(row["script"], headers={"Content-Disposition": f'attachment; filename="proof_{aid}.py"'})


@app.post("/api/verify/{aid}")
def verify(aid: str):
    """Re-run the stored proof script in a fresh interpreter against the original files and compare."""
    row = store.get(aid)
    if not row or row["status"] != "verified": raise HTTPException(404, "Only verified analyses can be re-verified")
    ws = W(row["session"]); used = json.loads(row["tables"])
    if any(u not in ws.tables for u in used): raise HTTPException(409, "Datasets have since been removed")
    rep = engine.reproduce(ws, row["question"], used, row["code"])
    same = rep["status"] == "ok" and common.payloads_match(json.loads(row["response"])["result"], rep["payload"])
    return {"reproduced": same, "status": rep["status"], "detail": None if same else rep.get("error", "result differs")}


class Rerun(BaseModel):
    code: str


@app.post("/api/rerun/{aid}")
def rerun(aid: str, body: Rerun):
    row = store.get(aid)
    if not row or row["status"] != "verified": raise HTTPException(404, "Only verified analyses can be edited and re-run")
    if not body.code.strip(): raise HTTPException(400, "Empty code")
    return engine.rerun(W(row["session"]), row["question"], json.loads(row["response"])["result"], body.code)


@app.get("/light")
def light():
    return FileResponse(HERE / "static" / "light.html")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
