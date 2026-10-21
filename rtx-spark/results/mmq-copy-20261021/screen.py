"""Run the pinned MMQ copy correctness and fail-fast prefill screen."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fixtures(protocol):
    rows = []
    spec = protocol["correctness"]
    k = spec["inner"]
    for name in spec["types"]:
        kind = GGMLQuantizationType[name]
        block, size = GGML_QUANT_SIZES[kind]
        if k % block:
            raise ValueError(f"Unaligned {name} matrix")
        stride = k // block * size
        for m in spec["rows"]:
            for n in spec["columns"]:
                values = [
                    29, 0, m, n, 1, 1, 0, 2,
                    int(kind), k, m, 1, 1, size, stride, stride * m, stride * m,
                    0, k, n, 1, 1, 4, k * 4, k * n * 4, k * n * 4,
                ]
                rows.append(" ".join(map(str, values)) + f" {name.lower()}_m{m}_n{n}_k{k}")
    if len(rows) != spec["cases_per_arm"]:
        raise ValueError("Protocol fixture count does not match")
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--candidate-cuda", type=Path, required=True)
    parser.add_argument("--test-exe", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if digest(args.installed / "ggml-cuda.dll") != protocol["control_cuda_sha256"]:
        raise ValueError("Installed control has changed")
    if digest(args.candidate_cuda) == protocol["control_cuda_sha256"]:
        raise ValueError("Candidate and control CUDA libraries are identical")
    args.work.mkdir(parents=True, exist_ok=True)
    test_file = HERE / "test-ops.txt"
    test_file.write_text(fixtures(protocol), encoding="utf-8")
    runtimes = {}
    manifest = {"protocol_sha256": digest(HERE / "protocol.json"),
                "test_file_sha256": digest(test_file), "arms": {}}
    for arm in ("base", "cand"):
        runtime = args.work / arm
        runtime.mkdir()
        for path in args.installed.iterdir():
            if path.is_file() and path.suffix.lower() in (".dll", ".exe"):
                shutil.copy2(path, runtime / path.name)
        shutil.copy2(args.test_exe, runtime / "test-backend-ops.exe")
        if arm == "cand":
            shutil.copy2(args.candidate_cuda, runtime / "ggml-cuda.dll")
        runtimes[arm] = runtime
        manifest["arms"][arm] = {p.name: digest(p) for p in sorted(runtime.iterdir())}
    changed = [name for name, value in manifest["arms"]["base"].items()
               if manifest["arms"]["cand"].get(name) != value]
    if changed != ["ggml-cuda.dll"] or set(manifest["arms"]["base"]) != set(manifest["arms"]["cand"]):
        raise ValueError(f"Unexpected runtime differences: {changed}")
    manifest["changed_files"] = changed
    save(HERE / "manifest.json", manifest)
    for arm, runtime in runtimes.items():
        command = [str(runtime / "test-backend-ops.exe"), "test", "-b", "CUDA0",
                   "--test-file", str(test_file)]
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, timeout=600,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        (args.work / f"{arm}-correctness.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        count = protocol["correctness"]["cases_per_arm"]
        summary = re.search(r"(\d+)/(\d+) tests passed", result.stdout)
        passed = (result.returncode == 0 and summary is not None
                  and tuple(map(int, summary.groups())) == (count, count)
                  and "FAIL" not in result.stdout and "not supported" not in result.stdout.lower())
        evidence = {"command": [Path(command[0]).name, *command[1:-1], test_file.name],
                    "exit_code": result.returncode, "expected_cases": count,
                    "summary": summary.group(0) if summary else None, "passed": passed,
                    "elapsed_seconds": time.perf_counter() - start,
                    "log_sha256": digest(args.work / f"{arm}-correctness.log")}
        save(HERE / f"{arm}-correctness.json", evidence)
        print(arm, evidence, flush=True)
        if not passed:
            raise RuntimeError(f"{arm} correctness failed; inspect the private log")
    command = [
        sys.executable, str(ROOT / "rtx-spark" / "bench" / "ab_llamabench.py"), "q27",
        "--base", str(runtimes["base"] / "llama-bench.exe"),
        "--cand", str(runtimes["cand"] / "llama-bench.exe"),
        "--screen", "prefill", "--out", str(HERE / "speed"),
    ]
    result = subprocess.run(command, timeout=650, check=False)
    if result.returncode not in (0, 2, 3):
        raise RuntimeError(f"Speed screen failed with exit code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
