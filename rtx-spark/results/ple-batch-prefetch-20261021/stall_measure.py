"""Cold vs warm lazy-PLE prefill on novel prompts with the deployed Flash-Next recipe.
Gate (written before measurement): proceed to a one-ubatch-ahead prefetch prototype only if the
median cold penalty, 1 - warm_ms/cold_ms, is >= 5% at 2K or 8K. Otherwise stop the PLE candidate."""
import glob, json, os, random, statistics, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\fabraham\.copilot\repos\copilot-worktrees\llama.cpp\fabraham-microsoft-jubilant-fiesta\rtx-spark"
P = json.load(open(os.path.join(REPO, "results", "model_paths.json")))
DRAFT = os.path.join(os.path.dirname(HERE), "benchmark-models", "mtp-Qwen3.8-Flash-Next-Q8_0-output-Q4_K.gguf")
EXE = r"C:\llama.cpp\llama-server.exe"
PORT = 9947
URL = f"http://127.0.0.1:{PORT}"
ARGS = ["-m", P["qfn"], "--mmproj", P["qfn_mmproj"], "-lzm", "on", "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "262144", "-np", "1", "-fa", "on", "-ngl", "999", "-b", "2048", "-ub", "512",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-md", DRAFT, "--spec-draft-ngl", "999",
        "--spec-draft-vocab-dynamic", "16384"]

def words(n, seed):
    r = random.Random(seed)
    return " ".join("".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(3, 9))) for _ in range(n))

LIB = os.path.dirname(os.__file__)
CODE_FILES = sorted(glob.glob(os.path.join(LIB, "*", "*.py")) + glob.glob(os.path.join(LIB, "*", "*", "*.py")))
def code(n_chars, part):
    files = CODE_FILES[part::9]
    out, total = [], 0
    for f in files:
        s = open(f, encoding="utf-8", errors="replace").read()
        out.append(s); total += len(s)
        if total >= n_chars: break
    return "".join(out)[:n_chars]

CASES = []
for L, nw, nc in ((512, 160, 1600), (2048, 650, 6500), (8192, 2600, 26000)):
    for i in range(3):
        CASES.append((f"words{L}#{i}", words(nw, 9100 + L + i) + "\nSummarize:"))
        CASES.append((f"code{L}#{i}", code(nc, {512: 0, 2048: 3, 8192: 6}[L] + i) + "\n# Summarize:\n"))

def call(prompt):
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0.0, "top_k": 1, "cache_prompt": False, "stream": False}
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        t = json.loads(r.read())["timings"]
    return t["prompt_n"], t["prompt_ms"]

def healthy():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r: return r.status == 200
    except Exception: return False

log = open(os.path.join(HERE, "server.log"), "x")
proc = subprocess.Popen([EXE] + ARGS, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
rows = []
try:
    t0 = time.time()
    while time.time() - t0 < 900 and proc.poll() is None and not healthy(): time.sleep(2)
    if not healthy(): raise RuntimeError("server failed")
    call("Hello world")
    for name, prompt in CASES:
        runs = [call(prompt) for _ in range(3)]
        n = runs[0][0]; assert all(r[0] == n for r in runs)
        ms = [r[1] for r in runs]
        row = {"case": name, "prompt_n": n, "ms": ms, "cold_tps": n / ms[0] * 1000,
               "penalty": 1 - statistics.median(ms[1:]) / ms[0], "order_check": ms[2] / ms[1] - 1}
        rows.append(row); print(json.dumps(row), flush=True)
finally:
    proc.terminate(); proc.wait(60)
summary = {}
for L in (512, 2048, 8192):
    sel = [r for r in rows if r["case"].endswith(tuple(f"{L}#{i}" for i in range(3)))]
    summary[L] = {"median_penalty": statistics.median(r["penalty"] for r in sel),
                  "max_penalty": max(r["penalty"] for r in sel),
                  "median_order_check": statistics.median(r["order_check"] for r in sel)}
proceed = any(summary[L]["median_penalty"] >= 0.05 for L in (2048, 8192))
json.dump({"args": [os.path.basename(a) if os.sep in a else a for a in ARGS], "exe": EXE, "rows": rows,
           "summary": summary, "proceed_to_prototype": proceed}, open(os.path.join(HERE, "results.json"), "x"), indent=2)
print(json.dumps(summary, indent=1), "proceed", proceed)
