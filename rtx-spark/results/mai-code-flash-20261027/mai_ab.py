"""Server A/B for MAI-Code-1.1-Flash: launches each runtime with the user's command, runs fixed requests, records speed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import heldout_suite  # noqa: E402

MODEL_DIR = Path(__import__("os").environ["MAI_MODEL_DIR"])  # folder with MAI-Code-1.1-Flash-Q2-XS.gguf and dflash-v3-lc1.gguf
PORT = 49990
ARGS = ["--host", "127.0.0.1", "--port", str(PORT), "--model", str(MODEL_DIR / "MAI-Code-1.1-Flash-Q2-XS.gguf"),
        "-lv", "4", "--n-gpu-layers", "999", "--model-draft", str(MODEL_DIR / "dflash-v3-lc1.gguf"),
        "--spec-type", "draft-dflash", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.2", "--spec-draft-ngl", "all",
        "--flash-attn", "on", "--batch-size", "2048", "--ubatch-size", "2048", "--fit", "off", "--ctx-checkpoints", "0",
        "--no-host", "-np", "1", "-c", "262145"]
CODE = [
    ("py_lru", "Implement an LRU cache class in Python with get and put in O(1), using a dict and a doubly linked list. Include type hints and docstrings."),
    ("rust_parse", "Write a Rust function that parses a CSV line with quoted fields (commas and escaped quotes inside quotes) into a Vec<String>, plus unit tests."),
    ("ts_api", "Write a TypeScript Express handler for POST /orders that validates the JSON body with zod, stores it in a Map, and returns 201 with the created order."),
]


def gpu_mib():
    out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def chat(content, max_tokens):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0,
            "top_k": 1, "seed": 1234, "ignore_eos": True, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    t = r["timings"]
    m = r["choices"][0]["message"]; text = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    return {"wall_s": round(time.perf_counter() - t0, 3), "prompt_n": t["prompt_n"], "prompt_tps": t["prompt_per_second"],
            "predicted_n": t["predicted_n"], "decode_tps": t["predicted_per_second"], "draft_n": t.get("draft_n", 0),
            "draft_accepted": t.get("draft_n_accepted", 0), "chars": len(text), "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}


def main():
    label, exe, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix(".log"), "w")
    idle = gpu_mib()
    t0 = time.monotonic()
    proc = subprocess.Popen([exe] + ARGS, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                break
            except OSError:
                if time.monotonic() - t0 > 900:
                    raise TimeoutError("not ready")
                time.sleep(2)
        load_s = time.monotonic() - t0
        ready = gpu_mib()
        doc = heldout_suite.long_context(64000)[:32000] + "\n\nSummarize the build options described above as a bullet list."
        cases = [(n, p, 256) for n, p in CODE] + [("long8k", doc, 64)]
        rows = []
        for name, prompt, n in cases:
            chat(prompt, n)  # warmup
            for rep in range(int(__import__("os").environ.get("MAI_REPS", "3"))):
                rows.append({"case": name, "rep": rep, **chat(prompt, n)})
        res = {"label": label, "exe_version": subprocess.run([exe, "--version"], capture_output=True, text=True).stderr.strip().splitlines()[-2:],
               "load_s": round(load_s, 1), "ready_gpu_mib": ready - idle if ready and idle else None, "rows": rows}
        out.write_text(json.dumps(res, indent=1))
        print(label, "ok", flush=True)
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        log.close()
        time.sleep(5)


if __name__ == "__main__":
    main()
