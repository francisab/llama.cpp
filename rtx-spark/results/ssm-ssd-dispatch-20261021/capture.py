"""Observe the existing long-sequence SSM dispatch in an isolated diagnostic build."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rtx-spark" / "bench"))

TRACE = re.compile(
    r"SPARK_SSM name=(\S+) n_t=(\d+) K=(\d+) d_state=(\d+) head_dim=(\d+) "
    r"n_head=(\d+) n_seq=(\d+) scalar_a=(\d+) ssd=(\d+)"
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hashes(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in (".exe", ".dll"):
            with path.open("rb") as stream:
                result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def parse_trace(text):
    fields = ("n_t", "K", "d_state", "head_dim", "n_head", "n_seq", "scalar_a", "ssd")
    return [{"name": match[0], **dict(zip(fields, map(int, match[1:])))} for match in TRACE.findall(text)]


def summarize(tag, result, traces):
    responses = [row["response"] for group in ("warmups", "rows") for row in result[group]]
    prompt_counts = {response["timings"]["prompt_n"] for response in responses}
    selected = [row for row in traces if 128 < row["n_t"] <= max(prompt_counts, default=0)]
    expected_k, expected_ssd = (1, 1) if tag == "N1" else (4, 0)
    matched = (result["stop"] == "ctrl+c" and result["exit_code"] == 0
               and result["gpu_memory_released_mib"] < 8000
               and len(responses) == 2 and len(prompt_counts) == 1 and len(selected) >= 40
               and all(row["K"] == expected_k and row["ssd"] == expected_ssd
                       and row["scalar_a"] == 1 and row["d_state"] == 128
                       and row["head_dim"] == 64 and row["n_head"] == 128
                       and row["n_seq"] == 1 for row in selected))
    return {
        "completed": result["completed"],
        "matched_hypothesis": matched,
        "prompt_token_counts": sorted(prompt_counts),
        "request_count": len(responses),
        "stop": result["stop"],
        "exit_code": result["exit_code"],
        "gpu_memory_ready_mib": result["gpu_memory_ready_mib"],
        "gpu_memory_done_mib": result["gpu_memory_done_mib"],
        "gpu_memory_released_mib": result["gpu_memory_released_mib"],
        "long_dispatch_records": traces,
        "request_dispatch_groups": [
            {"K": k, "ssd": ssd, "records": count}
            for (k, ssd), count in sorted(Counter((row["K"], row["ssd"]) for row in selected).items())
        ],
        "profiling_perturbed": True,
        "timings_are_performance_evidence": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed arms with matching commands and runtime hashes")
    args = parser.parse_args()
    import run_rebaseline as runner
    import run_matrix as matrix

    before = hashes(args.exe.parent)
    installed = hashes(args.installed)
    differences = [name for name in installed if before.get(name) != installed[name]]
    if differences != ["ggml-cuda.dll"]:
        raise ValueError(f"Expected only an instrumented CUDA DLL: {differences}")
    manifest = {
        "diagnostic_binary_hashes": before,
        "installed_binary_hashes": installed,
        "changed_files": differences,
        "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
    }
    if args.resume:
        if json.loads((HERE / "manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cannot resume a different runtime or protocol")
    else:
        save(HERE / "manifest.json", manifest)
    matrix.PORT = 9949
    configs = []
    for name, values, suite in matrix.CONFIGS:
        if name.split()[0] not in ("N1", "N4"):
            continue
        values = list(values)
        values[values.index("--port") + 1] = str(matrix.PORT)
        values[values.index("-c") + 1] = "8192"
        configs.append((name, values, suite))
    matrix.CONFIGS = configs
    matrix.SUITES["ssm-dispatch"] = [("prefix", matrix.text_of(1500, 1500), 1)]
    summaries = {}
    for tag in ("N1", "N4"):
        work = args.work / tag
        output = work / f"server_{tag}_ssm-dispatch.json"
        if not (args.resume and output.exists()):
            work.mkdir(parents=True)
            runner.run_server(tag, work, 1, "ssm-dispatch", exe=args.exe, extra_args=("--no-warmup",))
        result = json.loads(output.read_text(encoding="utf-8"))
        values = next(values for name, values, _ in configs if name.split()[0] == tag)
        if result["command"] != [str(args.exe.resolve(strict=True)), *values, "--no-warmup"]:
            raise ValueError(f"{tag} command differs from the protocol")
        result["gpu_memory_released_mib"] = runner.gpu_memory_mib()
        log = (work / f"server_{tag}_ssm-dispatch.log").read_text(encoding="utf-8", errors="replace")
        summary = summarize(tag, result, parse_trace(log))
        save(HERE / f"{tag}.json", summary)
        if not summary["completed"] or not summary["matched_hypothesis"]:
            raise RuntimeError(f"{tag} does not establish the expected dispatch; inspect preserved evidence")
        summaries[tag] = {key: value for key, value in summary.items() if key != "long_dispatch_records"}
        print(tag, json.dumps(summaries[tag]), flush=True)
    if hashes(args.exe.parent) != before or hashes(args.installed) != installed:
        raise RuntimeError("Runtime changed during the diagnostic")
    save(HERE / "summary.json", {
        "observed_rollback_blocks_ssd": True,
        "arms": summaries,
        "runtime_hashes_unchanged": True,
        "speedup_measured": False,
        "deployment_changed": False,
        "prompt_accounting": "Response prompt_n aggregates batches; the long SSM dispatch need not equal that total.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
