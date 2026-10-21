import glob, json, os, random, statistics, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\fabraham\.copilot\repos\copilot-worktrees\llama.cpp\fabraham-microsoft-jubilant-fiesta\rtx-spark"
P = json.load(open(os.path.join(REPO, "results", "model_paths.json")))
DRAFT = os.path.join(os.path.dirname(HERE), "benchmark-models", "mtp-Qwen3.8-Flash-Next-Q8_0-output-Q4_K.gguf")
RT = {"A": r"C:\llama-builds\ple-ctrl", "B": r"C:\llama-builds\ple-cand"}
PORT = 9948
URL = f"http://127.0.0.1:{PORT}"
ARGS = ["-m", P["qfn"], "--mmproj", P["qfn_mmproj"], "-lzm", "on", "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "262144", "-np", "1", "-fa", "on", "-ngl", "999", "-b", "2048", "-ub", "512",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-md", DRAFT, "--spec-draft-ngl", "999",
        "--spec-draft-vocab-dynamic", "16384"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"
def words(n, seed):
    r = random.Random(seed)
    return " ".join("".join(r.choice(LETTERS) for _ in range(r.randint(3, 9))) for _ in range(n))
LIB = os.path.dirname(os.__file__)
FILES = sorted(glob.glob(os.path.join(LIB, "*", "*.py")) + glob.glob(os.path.join(LIB, "*", "*", "*.py")))[::-1]
def code(n_chars, part):  # parts 0..15 of a stride-16 split taken from the end of the sorted list
    out, total = [], 0
    for fn in FILES[part::16]:
        s = open(fn, encoding="utf-8", errors="replace").read(); out.append(s); total += len(s)
        if total >= n_chars: break
    return "".join(out)[:n_chars]
def cases(phase):
    c = []
    for j, (L, nw, nc) in enumerate(((2048, 650, 6500), (8192, 2600, 26000))):
        for i in range(4): c.append((f"words{L}#{i}", words(nw, 77000 + 100 * phase + 10 * j + i) + "\nSummarize:"))
        for i in range(2): c.append((f"code{L}#{i}", code(nc, 4 * phase + 2 * j + i) + "\n# Summarize:\n"))
    return c
def post(body):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read())
def call(prompt):
    t = post({"prompt": prompt, "n_predict": 1, "temperature": 0.0, "top_k": 1, "cache_prompt": False})["timings"]
    return t["prompt_n"], t["prompt_ms"]
def healthy():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r: return r.status == 200
    except Exception: return False
FIXED = "Write a short Python function that merges two sorted lists, then explain its complexity."
rows, gens = [], {}
for phase, arm in enumerate("ABBA"):
    log = open(os.path.join(HERE, f"server-{phase}{arm}.log"), "x")
    proc = subprocess.Popen([os.path.join(RT[arm], "llama-server.exe")] + ARGS, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        t0 = time.time()
        while time.time() - t0 < 900 and proc.poll() is None and not healthy(): time.sleep(2)
        if not healthy(): raise RuntimeError("server failed")
        call("Hello world")
        g = post({"prompt": FIXED, "n_predict": 64, "temperature": 0.0, "top_k": 1, "cache_prompt": False})
        gens.setdefault(arm, []).append(g["content"])
        for name, prompt in cases(phase):
            (n, cold), (n2, warm) = call(prompt), call(prompt)
            assert n == n2
            row = {"phase": phase, "arm": arm, "case": name, "prompt_n": n, "cold_tps": n / cold * 1000, "warm_tps": n / warm * 1000}
            rows.append(row); print(json.dumps(row), flush=True)
    finally:
        proc.terminate(); proc.wait(60)
summary = {}
for L in (2048, 8192):
    s = {}
    for m in ("cold_tps", "warm_tps"):
        med = {a: statistics.median(r[m] for r in rows if r["arm"] == a and f"{L}#" in r["case"]) for a in "AB"}
        s[m] = {"A": med["A"], "B": med["B"], "gain": med["B"] / med["A"] - 1}
    summary[L] = s
same = len(set(gens["A"] + gens["B"])) == 1
cold = [summary[L]["cold_tps"]["gain"] for L in (2048, 8192)]
warm = [summary[L]["warm_tps"]["gain"] for L in (2048, 8192)]
verdict = "ACCEPT" if min(cold) >= 0.03 and min(warm) >= -0.01 and same else ("REJECT" if min(cold) < 0.01 or not same else "INCONCLUSIVE")
json.dump({"runtimes": RT, "args": [os.path.basename(a) if os.sep in a else a for a in ARGS], "rows": rows,
           "generations": gens, "same_generation": same, "summary": summary, "verdict": verdict},
          open(os.path.join(HERE, "results.json"), "x"), indent=2)
print(json.dumps(summary, indent=1), "same", same, "verdict", verdict)
