"""SQLite persistence for analysis history (question, answer, code, script, full response)."""
import json, sqlite3, time, uuid
from pathlib import Path
import engine


class Store:
    def __init__(self):
        engine.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = engine.DATA_DIR / "proofai.db"
        with self._c() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY, session TEXT, ts REAL, question TEXT,
                         status TEXT, answer TEXT, tables TEXT, code TEXT, script TEXT, response TEXT)""")

    def _c(self): return sqlite3.connect(self.path)

    def save(self, session, question, r, used) -> str:
        aid = uuid.uuid4().hex[:10]
        with self._c() as c:
            c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?,?)", (aid, session, time.time(), question,
                      "refused" if r["no"] else "verified", r["a"], json.dumps(used), r["code"], r.get("script"),
                      json.dumps({k: v for k, v in r.items() if k != "script"}, default=str)))
        return aid

    def get(self, aid):
        with self._c() as c:
            row = c.execute("SELECT id,session,question,status,answer,tables,code,script,response FROM analyses WHERE id=?", (aid,)).fetchone()
        return dict(zip(("id", "session", "question", "status", "answer", "tables", "code", "script", "response"), row)) if row else None

    def history(self, session, limit=30):
        with self._c() as c:
            rows = c.execute("SELECT id,ts,question,status,answer FROM analyses WHERE session=? ORDER BY ts DESC LIMIT ?", (session, limit)).fetchall()
        return [dict(zip(("id", "ts", "question", "status", "answer"), r)) for r in rows]
