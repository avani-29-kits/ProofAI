# ProofAI: a proof-carrying data analyst (HNX26PSI08)

**Team:** _add names_ · **Problem statement:** PSI08

## What it does
Ask a question about your own CSV / XLSX / JSON files. ProofAI writes pandas code, runs it, and then **re-runs it in a fresh Python process that re-reads the original files**. The answer is shown as **Verified** only if both runs match. If the data cannot support a reliable answer, ProofAI says **Cannot determine** and explains why. It never guesses or invents a value.

Every answer ships with: the code, the evidence used, the checks that passed, a chart/table, and a downloadable `proof.py` that anyone can run to re-check the number.

## Quick start (about 2 minutes)
Requires Python 3.10+.

| OS | Command |
|----|---------|
| Windows | double-click `run.bat` (or run it in Command Prompt) |
| Mac/Linux | `./run.sh` |

Then open **http://localhost:8000**. The sample sales data loads automatically.

Manual setup:
```
pip install -r requirements.txt
python make_sample_data.py     # generates ./sample_data (messy demo data + trap files)
python server.py               # http://localhost:8000  (PORT env var respected)
```
**Optional Claude planner:** copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY=...`. Without a key the built-in offline planner is used, and the whole system works with no network and no key.
Docker: `docker build -t proofai . && docker run -p 8000:8000 proofai`. Render: push the folder and use `render.yaml`.

## Reproduce the demonstrated results
1. `python evaluate.py` runs the **27-case benchmark**: 11 answerable questions checked against independent pandas ground truth and 16 trick questions that must be refused. Expected: `overall 27/27 = 100%`. (`--llm` runs the same set through Claude.)
2. `python tests/test_http.py` runs a real HTTP end-to-end test: load data, ask, download proof, re-verify, refusal.
3. In the UI, click **Benchmark → Run benchmark** to see the same 27 cases live.
4. Judge/verifier mode: `python batch.py sample_data questions.txt out/` answers a file of questions, writes `results.csv` and one runnable `proof_<n>.py` per answer.

## Data pipeline
```
upload (CSV/XLSX/JSON)
 → profile each column: dtype, nulls, duplicates, date format (DD/MM vs MM/DD), year coverage, embedded currency/units
 → deterministic pre-checks (year not in data, invalid period like Q5, relative time like "last quarter",
   nonexistent entity, ambiguous dates)  → refuse early, with executed evidence
 → planner writes pandas code (Claude, or the offline heuristic planner)
 → AST guard (only pandas/numpy/math/re/datetime/statistics; no file, network, eval)
 → sandbox run #1 (isolated subprocess, CPU/RAM limits, timeout)
 → input checks (mixed units/currencies, ambiguous dates, duplicates, missing values, contradicting tables)
 → run #2: standalone proof script re-reads the ORIGINAL files in a fresh process; results must match
 → answer + evidence + chart + proof.py   (or "Cannot determine" + reason)
```
Files: `engine.py` (ingest, profiling, guards, execution, verification), `planner.py` (code generation), `runner.py` (sandbox), `common.py` (result serialisation shared by sandbox and proof), `server.py` / `main.py` (API), `store.py` (SQLite history), `static/index.html` (UI).

## Core reasoning
The central idea is **proof-carrying answers**: a number is trusted only because independently re-executed code reproduces it. Refusal is a first-class outcome, with explicit triggers: no data for the asked year, ambiguous or mixed date formats, mixed currencies/units without conversion data, tables that contradict each other, invalid or relative periods, non-existent entities or columns, non-data questions, and code that cannot be made to run and verify.

## Sample input and output
**Input** (`sample_data/sales.csv`, `customers.csv`, `products.csv`): *"Which region generated the highest Q2 revenue?"*

**Output**
- Answer: **South** (₹54.01 lakh, Q2 revenue), status **Verified**, 0.55 s
- Evidence: `sales.csv` 3,040 rows, `customers.csv` 300 rows, missing values excluded, filtered on `order_date`, grouped by `region`
- Checks: input validated · units consistent · 40 duplicate rows removed in code · calculation reproduced in a fresh process
- Assumptions stated: Q2 read as calendar Apr–Jun, aggregated across all years; currency assumed INR (no currency column)
- Code produced (excerpt):
```python
data = sales.drop_duplicates()
data = data.merge(lookup, on="customer_id", how="left")      # lookup = customers (region), checked for conflicts
data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
data = data.dropna(subset=["revenue"])
_d = pd.to_datetime(data["order_date"], errors="coerce", dayfirst=False)
data = data[(_d.dt.quarter == 2)]
result = data.groupby("region")["revenue"].sum().sort_values(ascending=False).head(10)
```
The full runnable script is in `examples/proof_q2_top_region.py` (put it beside the sample CSVs and run it).

**Refusal example**, *"What was the revenue in 2021?"* → **Cannot determine**: "The available dataset contains no 2021 records." Evidence: dates span 2023-01-01 → 2025-12-30, 2021 rows found: 0, no value estimated.

## Evidence and explanation
Each result exposes intermediate outputs: the profile-driven evidence chips, the list of passed checks, timings, printed output, the generated code (editable and re-runnable via **Edit and re-run**, which shows whether the edit changes the answer), the stored history in SQLite, and `proof.py` for independent re-checking. `POST /api/verify/{id}` re-verifies any stored answer on demand.

## Scope note
**Minimum viable solution (implemented):** upload and profiling; offline planner for common aggregates (sum/avg/count, top-N, group-by, quarter/year filters, joins, outliers, missing-data report); AST guard + sandbox; independent re-run verification; refusal logic for the trap cases above; proof script download; web UI.
**Stretch goals (implemented):** Claude planner with a repair loop; judge/batch mode; edit-and-re-run tamper test; live benchmark in the UI; FastAPI alternative server (`main.py`); Docker/Render deployment; alternate light UI at `/light`.
**Not done / limitations:** the offline planner only covers common question shapes (harder questions need the Claude planner); "Verified" means the calculation reproduces from the files, not that the source data is correct; currency is assumed INR when no currency is stated and this is surfaced as an assumption; the sandbox is process-level (use a container/gVisor with no network in production).

## Technologies, libraries and resources declared
- Python 3.10+ standard library (`http.server`, `subprocess`, `sqlite3`, `ast`), **pandas**, **numpy**, **openpyxl**; optional **FastAPI + uvicorn** (`requirements-fastapi.txt`).
- Optional: Anthropic Messages API (model set by `PROOFAI_MODEL`, default `claude-sonnet-5-5`) for the planner. Not required.
- Frontend: plain HTML/CSS/JS, Google Fonts (Bricolage Grotesque, Instrument Sans, JetBrains Mono) with system fallbacks.
- Data: all sample and trap data is synthetic, generated by `make_sample_data.py`. No external datasets or pre-trained models are bundled.
- AI usage: AI tools (Claude) assisted development; the team reviewed the code and is responsible for it.

## API (header `X-Session` isolates users)
`GET /api/health` · `GET/POST /api/datasets` · `POST /api/datasets/sample|traps` · `DELETE /api/datasets/{name}` · `POST /api/analyze` · `GET /api/proof/{id}.py` · `POST /api/verify/{id}` · `POST /api/rerun/{id}` · `GET /api/history` · `GET /api/benchmark`
