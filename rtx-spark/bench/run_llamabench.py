"""Run llama-bench (DGX Spark discussion #16578 methodology) on each local model, sequentially."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
P = json.load(open(os.path.join(RESULTS, "model_paths.json")))
EXE = r"C:\llama.cpp\llama-bench.exe"
BASE = ["-fa", "1", "-ub", "2048", "-p", "2048", "-n", "32", "-d", "0,16384", "-r", "2", "-ngl", "99", "-o", "md"]
RUNS = {
    "q27": ["-m", P["d27"]],
    "qfn": ["-m", P["qfn"], "-lzm", "on"],
    "nem": ["-m", P["nem"]],
}

for tag in (sys.argv[1:] or RUNS.keys()):
    out = os.path.join(RESULTS, f"llamabench_{tag}{os.environ.get('LB_SUFFIX', '')}.md")
    t0 = time.time()
    res = subprocess.run([EXE] + RUNS[tag] + BASE, capture_output=True, text=True, timeout=3600,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    open(out, "w", encoding="utf-8").write(res.stdout + "\n<!-- stderr tail -->\n" + "\n".join(res.stderr.splitlines()[-15:]))
    print(f"=== {tag} ({time.time() - t0:.0f} s, rc={res.returncode}) ===", flush=True)
    print("\n".join(l for l in res.stdout.splitlines() if l.startswith("|")), flush=True)

