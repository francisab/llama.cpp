"""Launch llama-server configurations one at a time, benchmark each, then stop it with Ctrl+C."""
import json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_server
from bench_server import call, text_of

EXE = r"C:\llama.cpp\llama-server.exe"
PORT = 9940
bench_server.URL = f"http://127.0.0.1:{PORT}/completion"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
LOGS = os.path.join(RESULTS, "logs")
P = json.load(open(os.path.join(RESULTS, "model_paths.json")))
OUT = os.path.join(RESULTS, os.environ.get("MATRIX_OUT", "matrix_results.json"))

CODE = ("Write a complete, well-commented C++ implementation of a thread-safe LRU cache with "
        "templated key and value types, including unit tests using only the standard library.\n\n")
PROSE = "Explain in detail how a CPU cache hierarchy works, from L1 to DRAM.\n\n"

SUITES = {
    "decode": [("code256", CODE, 256), ("prose256", PROSE, 256), ("prose512", PROSE, 512),
               ("novel2K", text_of(1500, 1500) + "\nSummarize:", 128)],
    "prefill": [("pp8K", text_of(6000, 6000) + "\nSummarize:", 32), ("pp25K", text_of(24000, 24000) + "\nSummarize:", 32)],
}
SUITES["full"] = SUITES["decode"] + [("pp2K", text_of(1500, 1500) + "\nSummarize:", 64)] + SUITES["prefill"]
SUITES["full+override"] = SUITES["full"] + [("prose256-req-nmax0", PROSE, 256, {"speculative.n_max": 0}),
                                            ("prose256-req-types-none", PROSE, 256, {"speculative.types": "none"})]

common = ["--host", "127.0.0.1", "--port", str(PORT), "-c", "262144", "-np", "1", "-fa", "on", "-ngl", "999"]
qfn = ["-m", P["qfn"], "--mmproj", P["qfn_mmproj"], "-lzm", "on"] + common
nem = ["-m", P["nem"], "--mmproj", P["nem_mmproj"], "--fit", "off"] + common
qmtp = lambda n: ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n), "-md", P["qfn_mtp"], "--spec-draft-ngl", "999"]

CONFIGS = [
    ("Q1 qfn mtp3 ub2048", qfn + ["-b", "2048", "-ub", "2048"] + qmtp(3), "full"),
    ("Q2 qfn mtp2 ub2048", qfn + ["-b", "2048", "-ub", "2048"] + qmtp(2), "decode"),
    ("Q3 qfn mtp4 ub2048", qfn + ["-b", "2048", "-ub", "2048"] + qmtp(4), "decode"),
    ("Q4 qfn mtp3 ub4096", qfn + ["-b", "4096", "-ub", "4096"] + qmtp(3), "prefill"),
    ("N1 nem nospec ub4096", nem + ["-b", "4096", "-ub", "4096", "--spec-type", "none"], "full"),
    ("N2 nem mtp3 ub4096", nem + ["-b", "4096", "-ub", "4096", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], "full"),
    ("N3 nem dflash3 ub4096", nem + ["-b", "4096", "-ub", "4096", "--spec-type", "draft-dflash", "--spec-draft-n-max", "3",
                                     "-md", P["nem_dfl"], "--spec-draft-ngl", "999"], "decode"),
    ("D1 q27 nospec ub2048", ["-m", P["d27"], "--mmproj", P["d27_mmproj"]] + common + ["-b", "2048", "-ub", "2048", "--spec-type", "none"], "full"),
    ("D2 q27 dflash7 ub2048", ["-m", P["d27"], "--mmproj", P["d27_mmproj"]] + common + ["-b", "2048", "-ub", "2048", "--spec-type", "draft-dflash",
                              "--spec-draft-n-max", "7", "-md", P["d27_dfl"], "--spec-draft-ngl", "999"], "decode"),
    ("D3 q27 dflash4 ub2048", ["-m", P["d27"], "--mmproj", P["d27_mmproj"]] + common + ["-b", "2048", "-ub", "2048", "--spec-type", "draft-dflash",
                              "--spec-draft-n-max", "4", "-md", P["d27_dfl"], "--spec-draft-ngl", "999"], "decode"),
    ("U1 qfn mtp3 ub512", qfn + ["-b", "2048", "-ub", "512"] + qmtp(3), "full+override"),
    ("N4 nem mtp3 pmin0.5", nem + ["-b", "4096", "-ub", "4096", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                                   "--spec-draft-p-min", "0.5"], "decode"),
    ("N5 nem mtp3 pmin0.75", nem + ["-b", "4096", "-ub", "4096", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                                    "--spec-draft-p-min", "0.75"], "decode"),
    ("Q5 qfn mtp3 pmin0.5", qfn + ["-b", "2048", "-ub", "512"] + qmtp(3) + ["--spec-draft-p-min", "0.5"], "decode"),
    ("D4 q27 dflash7 pmin0.5", ["-m", P["d27"], "--mmproj", P["d27_mmproj"]] + common + ["-b", "2048", "-ub", "2048", "--spec-type", "draft-dflash",
                               "--spec-draft-n-max", "7", "-md", P["d27_dfl"], "--spec-draft-ngl", "999", "--spec-draft-p-min", "0.5"], "decode"),
]


def gpu_mem_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ctrl_c(pid):
    helper = ("import ctypes,sys,time;k=ctypes.windll.kernel32;k.FreeConsole();"
              "ok=k.AttachConsole(%d);k.SetConsoleCtrlHandler(None,True);"
              "k.GenerateConsoleCtrlEvent(0,0) if ok else None;time.sleep(1);k.FreeConsole();sys.exit(0 if ok else 2)" % pid)
    return subprocess.run([sys.executable, "-c", helper], timeout=30,
                          creationflags=subprocess.CREATE_NO_WINDOW).returncode


def stop(proc):
    rc = ctrl_c(proc.pid)
    try:
        proc.wait(timeout=45)
        how = "ctrl+c" if rc == 0 else "exited"
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=30); how = "killed"
    for _ in range(60):
        if gpu_mem_mib() < 8000:
            break
        time.sleep(1)
    return how


def tail(path, n=25):
    try:
        return open(path, encoding="utf-8", errors="replace").read().splitlines()[-n:]
    except Exception:
        return []


def main():
    only = set(sys.argv[1:])
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, args, suite in CONFIGS:
        tag = name.split()[0]
        if only and tag not in only:
            continue
        log = os.path.join(LOGS, f"server_{tag}.log")
        t0 = time.time()
        proc = subprocess.Popen([EXE] + args, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        while time.time() - t0 < 600 and proc.poll() is None and not healthy():
            time.sleep(2)
        entry = {"args": args, "suite": suite, "load_s": round(time.time() - t0, 1)}
        if proc.poll() is not None or not healthy():
            entry["error"] = "failed to start"; entry["log_tail"] = tail(log)
            if proc.poll() is None:
                stop(proc)
            results[name] = entry; json.dump(results, open(OUT, "w"), indent=1)
            print(f"[{name}] FAILED to start; see {log}", flush=True)
            continue
        entry["vram_mib"] = gpu_mem_mib()
        print(f"[{name}] ready in {entry['load_s']} s, VRAM {entry['vram_mib']} MiB", flush=True)
        try:
            call("Hello", 8)
        except Exception as e:
            entry["error"] = f"warmup failed: {e}"; entry["log_tail"] = tail(log)
            entry["stop"] = stop(proc)
            results[name] = entry; json.dump(results, open(OUT, "w"), indent=1)
            print(f"[{name}] warmup FAILED ({e}); stopped via {entry['stop']}", flush=True)
            continue
        rows = []
        for case, prompt, n, *extra in SUITES[suite]:
            try:
                r = call(prompt, n, extra[0] if extra else None)
            except Exception as e:
                r = {"error": str(e)}
            r["case"] = case
            rows.append(r)
            print(f"[{name}] {json.dumps(r)}", flush=True)
        entry["rows"] = rows
        entry["stop"] = stop(proc)
        print(f"[{name}] stopped via {entry['stop']}", flush=True)
        results[name] = entry
        json.dump(results, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

