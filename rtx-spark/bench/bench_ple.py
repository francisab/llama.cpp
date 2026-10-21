"""Does lazy PLE (per-layer token embedding read from SSD) hurt Qwen3.8-Flash-Next on realistic text?
Launch no-spec server, run prompts of different n-gram diversity, record speed and process disk reads."""
import glob, json, os, random, subprocess, sys, time
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_matrix as M
from bench_server import call, text_of

def stdlib_code(n_chars, seed):
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    rnd = random.Random(seed); rnd.shuffle(files)
    out, total = [], 0
    for f in files:
        s = open(f, encoding="utf-8", errors="replace").read()
        out.append(s); total += len(s)
        if total >= n_chars:
            break
    return "".join(out)[:n_chars]

def random_words(n_words, seed):
    rnd = random.Random(seed)
    return " ".join("".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9))) for _ in range(n_words))

CASES = [
    ("lowdiv-8K", text_of(6000, 6000) + "\nSummarize:"),
    ("stdlib-code-8K#1", stdlib_code(26000, 1) + "\n# Summarize the module above:\n"),
    ("stdlib-code-8K#2", stdlib_code(26000, 2) + "\n# Summarize the module above:\n"),
    ("randwords-8K", random_words(2600, 3) + "\nSummarize:"),
    ("stdlib-code-8K#1-again", stdlib_code(26000, 1) + "\n# Summarize the module above:\n"),
]

args = M.qfn + ["-b", "2048", "-ub", "2048", "--spec-type", "none"]
log = os.path.join(M.LOGS, "server_PLE.log")
proc = subprocess.Popen([M.EXE] + args, stdout=open(log, "w"), stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
t0 = time.time()
while time.time() - t0 < 600 and proc.poll() is None and not M.healthy():
    time.sleep(2)
print(f"ready in {time.time() - t0:.0f} s", flush=True)
ps = psutil.Process(proc.pid)
call("Hello", 8)
rows = []
for name, prompt in CASES:
    io0 = ps.io_counters()
    r = call(prompt, 128)
    io1 = ps.io_counters()
    r.update(case=name, disk_read_mb=round((io1.read_bytes - io0.read_bytes) / 1e6, 1),
             read_ops=io1.read_count - io0.read_count)
    rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(rows, open(os.path.join(M.RESULTS, "ple_diversity_qfn.json"), "w"), indent=1)
print("stop:", M.stop(proc), flush=True)
