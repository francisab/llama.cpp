"""Soak the installed llama-server with mixed long-document and short chat requests; records errors, latency drift and GPU memory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heldout_suite  # noqa: E402
import run_matrix as matrix  # noqa: E402

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 1234}


def post(path, body, timeout=1800):
    request = urllib.request.Request(f"http://127.0.0.1:{matrix.PORT}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def wait_ready(proc, timeout=600):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{matrix.PORT}/health", timeout=5) as response:
                if response.status == 200:
                    return time.monotonic() - start
        except OSError:
            time.sleep(2)
    raise TimeoutError("server not ready")


def workload():
    doc = heldout_suite.long_context(64000)
    short = [
        "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
        "Give three bullet points on why unit tests matter, in plain English.",
        "Return a JSON object with keys name, age and city for a fictional person. Output only JSON.",
    ]
    long_sizes = [16000, 32000, 64000]
    while True:
        for size, prompt in zip(long_sizes, short):
            yield f"long{size}", doc[:size] + "\n\nSummarize the main points above in five bullet points.", 192
            yield "short", prompt, 192


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", choices=["U1"])
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configs = {name.split()[0]: list(a) for name, a, _ in matrix.CONFIGS}
    command = [matrix.EXE] + configs[args.tag] + args.extra_args.split()
    command[command.index("-md") + 1] = str(args.draft_model.resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=False)
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    rows, errors = [], []
    try:
        ready_s = wait_ready(proc)
        ready_mib = gpu_used_mib()
        end = time.monotonic() + args.minutes * 60
        for name, prompt, count in workload():
            if time.monotonic() > end:
                break
            start = time.perf_counter()
            try:
                # chat template: a raw ~17K-token document makes this instruct model stop at once
                result = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}],
                                                       "max_tokens": count, **SAMPLING, "cache_prompt": True,
                                                       "chat_template_kwargs": {"enable_thinking": False}})
                t = result["timings"]
                content = result["choices"][0]["message"].get("content") or ""
                rows.append({"case": name, "wall_s": round(time.perf_counter() - start, 3),
                             "prompt_n": t["prompt_n"], "prompt_per_second": round(t["prompt_per_second"], 2),
                             "predicted_n": t["predicted_n"], "predicted_per_second": round(t["predicted_per_second"], 2),
                             "draft_n": t.get("draft_n", 0), "draft_n_accepted": t.get("draft_n_accepted", 0),
                             "gpu_mib": gpu_used_mib(),
                             "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                             "empty": not content.strip()})
            except Exception as error:  # record and continue; the soak counts failures
                errors.append({"case": name, "error": f"{type(error).__name__}: {error}"[:300]})
                if proc.poll() is not None:
                    break
    finally:
        alive = proc.poll() is None
        if alive:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        log.close()
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    summary = {}
    for case, items in by_case.items():
        half = max(1, len(items) // 2)
        first = sum(r["predicted_per_second"] for r in items[:half]) / half
        last = sum(r["predicted_per_second"] for r in items[-half:]) / half
        summary[case] = {"requests": len(items), "decode_tps_first_half": round(first, 2),
                         "decode_tps_last_half": round(last, 2), "empty_outputs": sum(r["empty"] for r in items)}
    mem = [r["gpu_mib"] for r in rows if r["gpu_mib"] is not None]
    result = {"config": args.tag, "exe": Path(matrix.EXE).name, "minutes": args.minutes, "ready_s": round(ready_s, 1),
              "ready_gpu_mib": ready_mib, "gpu_mib_min_max": [min(mem), max(mem)] if mem else None,
              "server_alive_at_end": alive, "requests": len(rows), "errors": errors,
              "summary": summary, "rows": rows}
    (args.out / "soak.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("requests", "errors", "ready_gpu_mib", "gpu_mib_min_max", "summary")}, indent=1))


if __name__ == "__main__":
    main()
