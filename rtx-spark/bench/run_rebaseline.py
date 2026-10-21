"""Run repeated native and server benchmarks without replacing earlier results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.request

import heldout_suite
import en_us_quality
import run_matrix as matrix


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def stop_process(pid):
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -ErrorAction Stop"],
        check=True, timeout=30, capture_output=True,
    )


def gpu_memory_mib():
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=20,
    )
    return int(output.strip())


def server_request(path, body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{matrix.PORT}{path}",
        data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def spec_metrics():
    values = {}
    for line in server_request("/metrics").splitlines():
        if line.startswith("llamacpp:spec_decode_"):
            name, value = line.rsplit(" ", 1)
            values[name.removeprefix("llamacpp:")] = float(value)
    return values


def metric_delta(before, after):
    return {name: after[name] - before.get(name, 0.0) for name in sorted(after)}


def completion(prompt, count, allow_eos=False):
    body = {
        "prompt": prompt, "n_predict": count, "temperature": 0.0, "top_k": 1,
        "seed": 1234, "cache_prompt": False, "stream": False,
        "ignore_eos": not allow_eos, "return_tokens": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{matrix.PORT}/completion",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    result["wall_s"] = time.perf_counter() - start
    result["request"] = body
    result["content_sha256"] = hashlib.sha256(result["content"].encode()).hexdigest()
    if not allow_eos and (result["timings"]["predicted_n"] != count or len(result["tokens"]) != count):
        raise ValueError(
            f"Expected {count} generated tokens, got "
            f"timings={result['timings']['predicted_n']}, tokens={len(result['tokens'])}"
        )
    return result


BENCH_PROFILES = {
    "base": (["-ub", "2048", "-p", "2048", "-n", "32,128", "-d", "0,16384"], {}),
    # Target-only small batches; not actual speculative verification.
    "verify": (["-ub", "512", "-p", "1,2,3,4,5,6,8,16", "-n", "0", "-d", "0,4096"], {}),
    "ubatch": (["-ub", "512,2048", "-p", "2048", "-n", "0"], {}),
    "lazy": (["-ub", "512", "-p", "512", "-n", "128"], {}),
    "decode": (["-ub", "512", "-p", "0", "-n", "128"], {}),
    "nographs": (["-ub", "512", "-p", "0", "-n", "128"], {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
    "nofusion": (["-ub", "512", "-p", "0", "-n", "128"], {"GGML_CUDA_DISABLE_FUSION": "1"}),
}


def run_plain(tag, out, repetitions, bench="base"):
    keys = {"qfn": "qfn", "nem": "nem", "q27": "d27"}
    bench_args, bench_env = BENCH_PROFILES[bench]
    command = [
        str(Path(matrix.EXE).with_name("llama-bench.exe")), "-m", matrix.P[keys[tag]], "-fa", "on",
        *bench_args, "-r", str(repetitions), "-ngl", "99", "-o", "json", "--progress",
    ]
    if bench == "lazy":
        command += ["-lzm", "on,off"]
    elif tag == "qfn":
        command += ["-lzm", "on"]
    name = f"plain_{tag}" if bench == "base" else f"plain_{tag}_{bench}"
    output = out / f"{name}.json"
    if output.exists() or (out / f"{name}.meta.json").exists():
        raise FileExistsError(output)
    metadata = {"command": command, "env": bench_env, "repetitions": repetitions, "completed": False}
    start = time.perf_counter()
    try:
        with output.open("x", encoding="utf-8") as stdout, (out / f"{name}.log").open("x") as stderr:
            print(f"[plain {tag} {bench}] starting", flush=True)
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=3600,
                           env={**os.environ, **bench_env}, creationflags=subprocess.CREATE_NO_WINDOW)
        rows = json.loads(output.read_text(encoding="utf-8"))
        for row in rows:
            samples = row["samples_ts"]
            if len(samples) != repetitions:
                raise ValueError(f"Expected {repetitions} samples: {row}")
            print(f"[plain {tag} {bench}] pp{row['n_prompt']} tg{row['n_gen']} d{row['n_depth']} "
                  f"ub{row['n_ubatch']} lzm={row.get('lazy_mode')}: "
                  f"median={statistics.median(samples):.2f} tok/s, "
                  f"range={min(samples):.2f}-{max(samples):.2f}", flush=True)
        metadata["completed"] = True
    except Exception as error:
        metadata["error"] = str(error)
        raise
    finally:
        metadata["wall_s"] = time.perf_counter() - start
        save(out / f"{name}.meta.json", metadata)


def run_server(tag, out, repetitions, suite, allow_eos=False, draft_model=None, capture_metrics=False, case_names=None,
               exe=None, extra_args=()):
    configs = {name.split()[0]: (name, args) for name, args, _ in matrix.CONFIGS}
    configs["Q0"] = ("Q0 qfn nospec ub512",
                     matrix.qfn + ["-b", "2048", "-ub", "512", "--spec-type", "none"])
    for depth in [2, 4]:
        configs[f"U{depth}"] = (f"U{depth} qfn mtp{depth} ub512",
                                matrix.qfn + ["-b", "2048", "-ub", "512"] + matrix.qmtp(depth))
    name, args = configs[tag]
    if draft_model is not None:
        if "-md" not in args:
            raise ValueError(f"{tag} does not use an external draft model")
        args = list(args)
        args[args.index("-md") + 1] = str(draft_model.resolve(strict=True))
    cases = []
    if suite in ("heldout", "en-us"):
        source_cases = heldout_suite.cases(en_us=suite == "en-us")
    elif suite == "en-us-quality":
        if not allow_eos:
            raise ValueError("en-us-quality requires --allow-eos")
        source_cases = en_us_quality.cases()
    elif suite == "novel":
        source_cases = matrix.SUITES["decode"][-1:]
    else:
        source_cases = matrix.SUITES[suite]
    for case in source_cases:
        if len(case) != 3:
            raise ValueError(f"Request overrides are not supported by this benchmark: {case[0]}")
        cases.append((case[0], case[1], case[2]))
    if case_names is not None:
        if len(set(case_names)) != len(case_names) or set(case_names) - {case[0] for case in cases}:
            raise ValueError(f"Unknown or duplicate case selection: {case_names}")
        cases = [case for case in cases if case[0] in case_names]
        if not cases:
            raise ValueError("Case selection is empty")
    exe = str(exe.resolve(strict=True)) if exe is not None else matrix.EXE
    command = [exe] + list(args) + list(extra_args) + (["--metrics"] if capture_metrics else [])
    output = out / f"server_{tag}_{suite}.json"
    if output.exists():
        raise FileExistsError(output)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", matrix.PORT)) == 0:
            raise RuntimeError(f"Port {matrix.PORT} is already occupied")
    result = {"name": name, "command": command, "repetitions": repetitions, "suite": suite, "allow_eos": allow_eos,
              "capture_metrics": capture_metrics, "case_names": [case[0] for case in cases],
              "prompts": {}, "warmups": [], "rows": [], "completed": False}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with (out / f"server_{tag}_{suite}.log").open("x") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, startupinfo=startup,
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
        start = time.perf_counter()
        result["pid"] = proc.pid
        try:
            while not matrix.healthy():
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited with {proc.returncode}; see server log")
                if time.perf_counter() - start > 600:
                    raise TimeoutError(f"{name} was not ready within 600 seconds")
                time.sleep(2)
            result["load_s"] = time.perf_counter() - start
            result["gpu_memory_ready_mib"] = gpu_memory_mib()
            print(f"[{name}] ready in {result['load_s']:.1f}s, pid={proc.pid}", flush=True)
            resolved = []
            for case, prompt, count in cases:
                if isinstance(prompt, dict):
                    prompt = json.loads(server_request("/apply-template", prompt))["prompt"]
                result["prompts"][case] = {"sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                           "characters": len(prompt)}
                resolved.append((case, prompt, count))
            cases = resolved
            last_metrics = spec_metrics() if capture_metrics else None

            def measured(prompt, count):
                nonlocal last_metrics
                response = completion(prompt, count, allow_eos)
                if capture_metrics:
                    current = spec_metrics()
                    response["spec_metrics"] = metric_delta(last_metrics, current)
                    last_metrics = current
                    timing = response["timings"]
                    for metric, field in [("spec_decode_num_draft_tokens_total", "draft_n"),
                                          ("spec_decode_num_accepted_tokens_total", "draft_n_accepted")]:
                        if response["spec_metrics"].get(metric, 0.0) != timing.get(field, 0):
                            raise ValueError(f"{metric} does not match response {field}")
                return response

            for case, prompt, count in cases:
                result["warmups"].append({"case": case, "response": measured(prompt, count)})
                save(output, result)
            for repeat in range(repetitions):
                offset = repeat % len(cases)
                for case, prompt, count in cases[offset:] + cases[:offset]:
                    response = measured(prompt, count)
                    result["rows"].append({"case": case, "repeat": repeat + 1, "response": response})
                    save(output, result)
                    timing = response["timings"]
                    print(f"[{name}] {case} #{repeat + 1}: "
                          f"pp={timing['prompt_per_second']:.1f}, tg={timing['predicted_per_second']:.2f}, "
                          f"accepted={timing.get('draft_n_accepted', 0)}/{timing.get('draft_n', 0)}",
                          flush=True)
            result["gpu_memory_done_mib"] = gpu_memory_mib()
            result["completed"] = True
        except Exception as error:
            result["error"] = str(error)
            raise
        finally:
            try:
                if proc.poll() is None:
                    matrix.ctrl_c(proc.pid)
                    try:
                        proc.wait(timeout=45)
                        result["stop"] = "ctrl+c"
                    except subprocess.TimeoutExpired:
                        stop_process(proc.pid)
                        proc.wait(timeout=30)
                        result["stop"] = "forced"
                        result["completed"] = False
                        raise RuntimeError(f"{name} did not stop with Ctrl+C")
            finally:
                result["exit_code"] = proc.returncode
                result["wall_s"] = time.perf_counter() - start
                save(output, result)
            print(f"[{name}] stopped, exit={proc.returncode}", flush=True)
            deadline = time.monotonic() + 60
            while gpu_memory_mib() >= 8000:
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU memory was not released after server exit")
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plain", "server"])
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--suite", choices=["decode", "prefill", "novel", "heldout", "en-us", "en-us-quality"],
                        default="decode")
    parser.add_argument("--allow-eos", action="store_true", help="Measure natural completion lengths instead")
    parser.add_argument("--draft-model", type=Path, help="Replace the selected external draft GGUF")
    parser.add_argument("--spec-metrics", action="store_true", help="Record per-request speculative counters")
    parser.add_argument("--cases", nargs="+", help="Run only these named server cases, preserving suite order")
    parser.add_argument("--bench", choices=list(BENCH_PROFILES), default="base", help="llama-bench profile")
    parser.add_argument("--exe", type=Path, help="llama-server executable to use instead of the installed one")
    parser.add_argument("--extra-args", default="", help="Extra server arguments, space separated")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.mode == "plain" and (args.allow_eos or args.draft_model is not None or args.spec_metrics
                                 or args.exe is not None or args.extra_args):
        parser.error("--allow-eos, --draft-model, --spec-metrics, --exe and --extra-args require server mode")
    if args.mode == "server" and args.bench != "base":
        parser.error("--bench requires plain mode")
    if args.mode != "server" and args.cases is not None:
        parser.error("--cases requires server mode")
    if args.suite == "en-us-quality" and (args.mode != "server" or not args.allow_eos):
        parser.error("--suite en-us-quality requires server mode and --allow-eos")
    if args.bench == "lazy" and args.tags != ["qfn"]:
        parser.error("--bench lazy applies only to qfn")
    args.out.mkdir(parents=True, exist_ok=True)
    label = args.bench if args.mode == "plain" and args.bench != "base" else args.suite
    telemetry_path = args.out / f"telemetry_{args.mode}_{label}_{'-'.join(args.tags)}.csv"
    with telemetry_path.open("x") as telemetry_file:
        telemetry = subprocess.Popen(
            ["nvidia-smi", "-i", "0",
             "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
             "--format=csv", "-l", "5"],
            stdout=telemetry_file, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            for tag in args.tags:
                if args.mode == "plain":
                    run_plain(tag, args.out, args.repetitions, args.bench)
                else:
                    run_server(tag, args.out, args.repetitions, args.suite, args.allow_eos, args.draft_model,
                               args.spec_metrics, args.cases, args.exe, args.extra_args.split())
                if telemetry.poll() is not None:
                    raise RuntimeError(f"GPU telemetry exited with {telemetry.returncode}")
        finally:
            if telemetry.poll() is None:
                stop_process(telemetry.pid)
                telemetry.wait(timeout=30)


if __name__ == "__main__":
    main()
