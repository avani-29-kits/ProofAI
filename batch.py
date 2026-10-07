"""Judge/verifier mode: answer a whole file of questions over a folder of data files and emit
re-runnable proof scripts.
    python batch.py <data_folder> <questions.txt|csv> [out_folder]
Writes out_folder/results.csv + out_folder/proof_<n>.py (+ copies of the data so each script runs as-is),
then executes every proof script to confirm it runs and prints the same answer."""
import csv, os, shutil, subprocess, sys, tempfile
from pathlib import Path
os.environ.setdefault("PROOFAI_DATA", tempfile.mkdtemp())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine


def main():
    if len(sys.argv) < 3: sys.exit(__doc__)
    data, qfile = Path(sys.argv[1]), Path(sys.argv[2]); out = Path(sys.argv[3] if len(sys.argv) > 3 else "proofai_output"); out.mkdir(parents=True, exist_ok=True)
    qs = [l.strip().strip('"') for l in qfile.read_text(encoding="utf-8-sig").splitlines() if l.strip() and not l.startswith("#")]
    if qs and qs[0].lower() in ("question", "questions"): qs = qs[1:]
    ws = engine.Workspace("batch")
    for f in sorted(data.iterdir()):
        if f.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls", ".json"):
            ws.ingest(f.name, f.read_bytes()); shutil.copy(f, out / f.name)
    rows = []
    for i, q in enumerate(qs, 1):
        r = engine.analyze(q, ws); proof, runs = "", ""
        if not r["no"] and r.get("script"):
            proof = f"proof_{i}.py"; (out / proof).write_text(r["script"], encoding="utf-8")
            p = subprocess.run([sys.executable, proof], capture_output=True, text=True, cwd=out, timeout=120)
            runs = "yes" if p.returncode == 0 else "NO: " + (p.stderr.strip().splitlines() or [""])[-1]
        rows.append({"id": i, "question": q, "status": "CANNOT_DETERMINE" if r["no"] else "ANSWER", "answer": r["a"], "detail": r["s"],
                     "assumptions_and_warnings": r.get("note", "") if not r["no"] else "", "proof_script": proof, "script_runs": runs})
        print(f"[{i}] {rows[-1]['status']:<17} {r['a'][:30]:<30} {r['s'][:60]}")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}/results.csv and {sum(1 for r in rows if r['proof_script'])} proof scripts")


main()
