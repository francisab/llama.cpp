"""Verify the rebuilt compact-prefill runtime against all twelve CPU-reference fixtures."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "rtx-spark" / "bench"))
from precision import sha, runtime_manifest, save
import run_rebaseline as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    results = []
    for arm in ("control", "candidate"):
        if runner.gpu_memory_mib() >= 10000:
            raise RuntimeError("GPU is occupied")
        runtime = args.runtime_root / arm
        before = runtime_manifest(runtime)
        executable = runtime / "test-backend-ops.exe"
        tester_hash = sha(executable)
        env = {key: value for key, value in os.environ.items() if not key.startswith("MMQ_MOE_")}
        if arm == "candidate":
            env["MMQ_MOE_COMPACT"] = "1"
        command = [str(executable), "test", "-b", "CUDA0", "-o", "MUL_MAT_ID", "-p", "prefill_route"]
        log = args.work / f"{arm}.log"
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                code = process.wait(timeout=900)
            finally:
                if process.poll() is None:
                    runner.stop_process(process.pid)
                    process.wait(timeout=30)
        text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(encoding="utf-8"))
        cases = re.findall(r"(MUL_MAT_ID\([^\n]*prefill_route=\d+\)):\s+(OK|FAIL)", text)
        passed = (code == 0 and len(cases) == len(dict(cases)) == 12
                  and all(status == "OK" for _, status in cases) and "12/12 tests passed" in text)
        if before != runtime_manifest(runtime) or tester_hash != sha(executable):
            raise ValueError("Runtime changed during correctness tests")
        result = {"arm": arm, "passed": passed, "exit_code": code, "cases": dict(cases),
                  "tester_sha256": tester_hash, "runtime_sha256": before, "log_sha256": sha(log),
                  "command": [executable.name] + command[1:],
                  "environment": {key: value for key, value in env.items() if key.startswith("MMQ_MOE_")}}
        results.append(result)
        runner.save(args.work / f"{arm}.json", result)
        if not passed:
            raise ValueError(f"{arm}: correctness failed; inspect {log}")
        print(f"{arm}: 12/12 CPU-reference cases passed", flush=True)
    if (results[0]["tester_sha256"] != results[1]["tester_sha256"]
            or results[0]["cases"] != results[1]["cases"]):
        raise ValueError("Control/candidate correctness fixtures differ")
    installed = runtime_manifest(Path(runner.matrix.EXE).parent)
    if ({key: value for key, value in results[0]["runtime_sha256"].items() if key != "test-backend-ops.exe"}
            != {key: value for key, value in installed.items() if key != "test-backend-ops.exe"}):
        raise ValueError("Control differs from the installed runtime")
    differences = [name for name in results[0]["runtime_sha256"]
                   if results[0]["runtime_sha256"][name] != results[1]["runtime_sha256"].get(name)]
    if differences != ["ggml-cuda.dll"]:
        raise ValueError(f"Unexpected candidate runtime differences: {differences}")
    save(HERE / "compact-correctness.json",
         {"passed": True, "script_sha256": sha(Path(__file__)), "results": results,
          "installed_runtime_sha256": installed})


if __name__ == "__main__":
    main()
