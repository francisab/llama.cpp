"""Check existing shared-prefix reuse, including full request latency around cache restore."""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "rtx-spark" / "bench"))
import run_matrix as matrix
import run_rebaseline as runner

PORT = 9949
URL = f"http://127.0.0.1:{PORT}"
COLORS = ["amber", "blue", "green", "violet"]


def fixture(family, index):
    offset = 0 if family == "A" else 1000
    lines = [f"Record R{i:03d} has color {COLORS[(i + offset // 1000) % 4]} and limit {100 + offset + i} units."
             for i in range(96)]
    system = ("Answer inventory lookups using only the inventory below. "
              "Return exactly one compact JSON object with keys record, color, limit. "
              "The limit is an integer. Do not add prose or markdown.\nInventory:\n" + "\n".join(lines))
    return {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": f"Lookup record R{index:03d}."}],
        "expected": {"record": f"R{index:03d}", "color": COLORS[(index + offset // 1000) % 4],
                     "limit": 100 + offset + index}
    }


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def common_prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def complete(case, cache):
    body = {"messages": case["messages"], "max_tokens": 64, "temperature": 0,
            "top_k": 1, "seed": 1234, "cache_prompt": cache, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    content, usage, timings, first, done = "", None, None, None, False
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data: "):
                raise ValueError(f"Unexpected SSE line: {line}")
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content") or ""
                if text and first is None:
                    first = time.perf_counter() - start
                content += text
            if event.get("usage") is not None:
                usage = event["usage"]
            if event.get("timings") is not None:
                timings = event["timings"]
    if not done or first is None or usage is None or timings is None:
        raise ValueError("Incomplete SSE content/usage/timings")
    answer = json.loads(content)
    if answer != case["expected"]:
        raise ValueError(f"Wrong lookup: {answer}, expected {case['expected']}")
    return {"content": content, "answer": answer, "usage": usage, "timings": timings,
            "first_content_ms": first * 1000, "wall_ms": (time.perf_counter() - start) * 1000}


def run(out, draft):
    if out.exists():
        raise FileExistsError(out)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"Port {PORT} is already occupied")
    out.mkdir(parents=True)
    sequence = [(f"{family}{visit}", fixture(family, index))
                for family, visit, index in (("A", 1, 5), ("A", 2, 17), ("B", 1, 5),
                                            ("B", 2, 17), ("A", 3, 29), ("B", 3, 29))]
    result = {"completed": False, "rows": [], "prefix_tokens": {}, "template_hashes": {}}
    args = list(matrix.qfn)
    args[args.index("--port") + 1] = str(PORT)
    args += ["-b", "2048", "-ub", "512", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
             "-md", str(draft.resolve(strict=True)), "--spec-draft-ngl", "999",
             "--spec-draft-vocab-dynamic", "16384", "--reasoning", "off", "--log-verbosity", "4"]
    try:
        for arm, extra in (("defaults", []), ("no_ram_cache", ["--cache-ram", "0"])):
            with (out / f"{arm}.log").open("x") as log:
                proc = subprocess.Popen([matrix.EXE, *args, *extra], stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    deadline = time.monotonic() + 600
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"{arm} server exited: {proc.returncode}")
                        try:
                            with urllib.request.urlopen(URL + "/health", timeout=3) as response:
                                if response.status == 200:
                                    break
                        except urllib.error.URLError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f"{arm} server did not become ready")
                        time.sleep(1)
                    print(f"{arm} ready, owned pid={proc.pid}", flush=True)
                    tokenized = {}
                    for name, case in sequence:
                        prompt = post("/apply-template", {"messages": case["messages"],
                                      "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
                        tokenized[name] = post("/tokenize", {"content": prompt, "add_special": True})["tokens"]
                        result["template_hashes"][name] = hashlib.sha256(prompt.encode()).hexdigest()
                    for family in ("A", "B"):
                        prefix = common_prefix(tokenized[family + "1"], tokenized[family + "2"])
                        if prefix < 1000:
                            raise ValueError(f"Fixture prefix too small: {prefix}")
                        result["prefix_tokens"][family] = prefix
                    requests = [(name, case, True) for name, case in sequence]
                    if arm == "defaults":
                        requests += [(name + "_uncached", case, False) for name, case in sequence
                                     if name[-1] != "1"]
                    for name, case, cache in requests:
                        row = {"arm": arm, "case": name, "cache_prompt": cache, **complete(case, cache)}
                        result["rows"].append(row)
                        n = row["usage"]["prompt_tokens"]
                        if n != len(tokenized[name.split("_")[0]]):
                            raise ValueError(f"Template/usage token mismatch: {n}")
                        cached = row["usage"]["prompt_tokens_details"]["cached_tokens"]
                        print(f"{arm} {name}: cached={cached}/{n}, first_content_ms={row['first_content_ms']:.1f}",
                              flush=True)
                        runner.save(out / "results.json", result)
                finally:
                    if proc.poll() is None:
                        runner.stop_process(proc.pid)
                    proc.wait(timeout=30)

        for row in result["rows"]:
            cached = row["usage"]["prompt_tokens_details"]["cached_tokens"]
            if not row["cache_prompt"]:
                assert cached == 0
                continue
            name = row["case"]
            if name[-1] != "1":
                ratio = cached / result["prefix_tokens"][name[0]]
                if row["arm"] == "no_ram_cache" and name[-1] == "3":
                    assert ratio < 0.1, row
                else:
                    assert ratio >= 0.95, row
        result["completed"] = True
        result["decision"] = "Existing upstream prefix/checkpoint retention passes this diagnostic; no new pinning implementation."
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        runner.save(out / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    run(options.out, options.draft)
