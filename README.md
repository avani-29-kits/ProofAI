# ProofAI backend — Proof-Carrying Data Analyst 

Every numeric answer ships with executable code, is re-run in a **fresh interpreter from the original files**,
and is only shown as *Verified* if both runs match. Unreliable questions return **Cannot determine**.

## Run (1 command)
- **Windows:** double-click `run.bat` · **Mac/Linux:** `./run.sh`
- Open **http://localhost:8000** — the UI auto-loads the sample sales/customers/products files and talks to the live API.
- Manual: `pip install -r requirements.txt && python make_sample_data.py && python server.py`
- Optional: copy `.env.example` → `.env` and add `ANTHROPIC_API_KEY` for the Claude planner. Without it the offline
  planner answers common aggregate questions (sum/avg/count, top-N, group-by, quarter/year filters, joins, outliers, missing data).
- `server.py` needs only pandas/numpy/openpyxl. `main.py` is an equivalent FastAPI app (`pip install -r requirements-fastapi.txt`, `uvicorn main:app`).

## Deploy a public link (judges can open it)
- **Render:** push this folder to GitHub → New → Blueprint → it reads `render.yaml` (Docker). Add `ANTHROPIC_API_KEY` in the dashboard.
- **Railway / Fly / any Docker host:** `docker build -t proofai . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... proofai`

## Judge / verifier mode (batch)
`python batch.py <data_folder> <questions.txt> [out_folder]` answers every question, writes `results.csv`
(ANSWER / CANNOT_DETERMINE + reason + assumptions) and one `proof_<n>.py` per answer, copies the data beside them,
and runs each script to confirm it executes. Anyone can then `python proof_1.py` to re-check a number.

## Traps handled (HNX26PSI08)
Mixed units/currencies (separate column **or** embedded like `$12`/`€10`/`5 kg`) · ambiguous DD/MM vs MM/DD dates · duplicate rows ·
missing values · tables that contradict each other (same key, different value) · conflicting lookup keys · years/entities/columns that
don't exist · invalid periods (Q5) · relative time ("last quarter") · non-data questions · calendar-vs-fiscal quarter assumption stated.

## Benchmark
`python evaluate.py` runs 27 cases (11 answerable questions checked against independent pandas ground truth, 16 trick questions that must be refused: missing/future years, partial year coverage, invalid quarter, nonexistent entity, nonexistent column, non-data question, ambiguous dates, mixed currencies, contradictory tables, $/€ inside values, tables disagreeing on a column, relative time, no data). `python evaluate.py --llm` runs the same set through Claude. Add your own cases to the `Q` list.

## Tests
`python tests/test_engine.py` (engine) · `python tests/test_http.py` (real HTTP end-to-end: upload → answer → proof → re-verify → refusal)

## Pipeline
`upload → profile (dtypes, nulls, duplicates, date formats DD/MM vs MM/DD, year coverage)`
`→ deterministic pre-checks (year not in data, ambiguous dates → refuse, with executed evidence)`
`→ planner (Claude or offline) writes pandas code`
`→ AST guard (no imports beyond pandas/numpy/math/re/datetime, no file/network/eval)`
`→ sandbox run #1 (isolated subprocess, CPU/RAM limits, timeout)`
`→ input checks (mixed units, ambiguous dates, duplicates, missing values)`
`→ run #2: standalone proof.py re-reads the ORIGINAL files in a fresh process → results must match`
`→ answer + evidence + chart + downloadable proof.py`

Refusal triggers: no data for the asked year · ambiguous/mixed date formats · mixed currencies/units without conversion data ·
contradicting tables (e.g. one customer in two regions) · code that raises `CannotDetermine` · code that can't be made to run/verify.

## API (header `X-Session` isolates users)
| | |
|-|-|
| `GET /api/health` | planner mode |
| `GET/POST /api/datasets` | list / upload CSV, XLSX (each sheet a table), JSON |
| `POST /api/datasets/sample`, `DELETE /api/datasets/{name}` | |
| `POST /api/analyze {"question"}` | → `{no, a, s, ev, t, u, ch, code, tm, vt, chk, note, output, result, id, script_url}` |
| `GET /api/proof/{id}.py` | standalone script; put it beside the data files and `python proof_<id>.py` |
| `POST /api/verify/{id}` | re-verify a stored answer now |
| `GET /api/history` | SQLite log of questions/answers/code |

## Security note
The sandbox is process-level (AST allow-list + `-I` isolated interpreter + rlimits + timeout). For production run the
runner inside a container/gVisor with no network.
