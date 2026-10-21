"""Bounded, routing-aware NVFP4 screen for Nemotron's actual expert projections."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
COLUMNS = (1, 4, 512)
SHAPES = {1: ("q4_k", 2688, 1024), 0: ("iq4_nl", 1024, 2688)}
CONTROL_CUDA = "898fb56c5a64f75e945e0823efa9a773cbdec66e5de9f1f253a771911b172c9f"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PERF = re.compile(r"(?P<op>NEMOTRON_FFN|PERF_GRAPH)\((?P<vars>[^)\n]+)\):\s+"
                  r"(?P<runs>\d+) runs\s+-\s+(?P<us>[\d.]+) us/run")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(directory):
    return {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.suffix.lower() in (".exe", ".dll")}


def inventory(first):
    sys.path.insert(0, str(ROOT / "gguf-py"))
    from gguf import GGUFReader

    if not first.name.endswith("-00001-of-00002.gguf"):
        raise ValueError("Expected the first of two original Nemotron shards")
    shards = [first, first.with_name(first.name.replace("-00001-of-", "-00002-of-"))]
    counts = Counter()
    names = set()
    metadata = None
    tensor_count = None
    for index, path in enumerate(shards):
        reader = GGUFReader(path, mode="r")
        if index == 0:
            arch = reader.fields["general.architecture"].contents()
            metadata = {key: int(reader.fields[f"{arch}.{key}"].contents())
                        for key in ("block_count", "nextn_predict_layers", "expert_count", "expert_used_count")}
            if arch != "nemotron_h_moe" or metadata != {
                "block_count": 89, "nextn_predict_layers": 1, "expert_count": 512, "expert_used_count": 22
            }:
                raise ValueError("Unexpected Nemotron architecture or expert metadata")
            tensor_count = int(reader.fields["split.tensors.count"].contents())
        else:
            # Later GGUF shards may contain only split metadata.
            for key, value in {"general.architecture": "nemotron_h_moe",
                               **{f"nemotron_h_moe.{k}": v for k, v in metadata.items()}}.items():
                if key in reader.fields and reader.fields[key].contents() != value:
                    raise ValueError("Shard metadata differs")
        if int(reader.fields["split.no"].contents()) != index or int(reader.fields["split.count"].contents()) != 2:
            raise ValueError("Unexpected shard metadata")
        if int(reader.fields["split.tensors.count"].contents()) != tensor_count:
            raise ValueError("Shard tensor counts differ")
        for tensor in reader.tensors:
            if tensor.name in names:
                raise ValueError("Duplicate tensor across shards")
            names.add(tensor.name)
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(up_exps|down_exps|up_shexp|down_shexp|latent_down|latent_up)\.weight",
                                tensor.name)
            if match and int(match[1]) < 88:
                counts[(match[2], tensor.tensor_type.name.lower(), tuple(map(int, tensor.shape)))] += 1
    if len(names) != tensor_count:
        raise ValueError("Incomplete shard-wide tensor inventory")
    expected = {
        ("up_exps", "q4_k", (1024, 2688, 512)): 40,
        ("down_exps", "iq4_nl", (2688, 1024, 512)): 40,
        ("up_shexp", "q4_k", (4096, 5376)): 40,
        ("down_shexp", "q4_k", (5376, 4096)): 20,
        ("down_shexp", "q6_k", (5376, 4096)): 20,
        ("latent_down", "q4_k", (4096, 1024)): 40,
        ("latent_up", "q4_k", (1024, 4096)): 40,
    }
    if counts != expected:
        raise ValueError(f"Unexpected complete target FFN inventory: {counts}")
    return {"model_shards": [p.name for p in shards], "metadata": metadata, "source_tensors": len(names), "target_layers": 88,
            "target_ffn_tensors": 240, "selected_expert_tensors": 80,
            "tensors": [{"role": role, "type": kind, "shape": shape, "count": count,
                         "selected": role in ("up_exps", "down_exps")}
                        for (role, kind, shape), count in sorted(counts.items())]}


def parse_perf(text):
    rows = []
    for match in PERF.finditer(ANSI.sub("", text)):
        fields = dict(part.split("=", 1) for part in match["vars"].split(","))
        row = {"op": match["op"], "runs": int(match["runs"]), "time_us": float(match["us"])}
        if match["op"] == "NEMOTRON_FFN":
            row.update({key: int(fields[key]) for key in ("n_mats", "n_used", "b", "m", "n", "k", "ids_offset")})
            row.update(type=fields["type_a"].lower(), route=int(fields["nem_route"]))
            if fields["type_b"] != "f32" or float(fields["amax"]) != 1 or int(fields["m_v"]) != 0:
                raise ValueError("Unexpected activation type, magnitude or expert view")
        else:
            row.update({key: int(fields[key]) for key in ("whole_graph", "by_flops")})
        rows.append(row)
    return rows


def validate_cases(rows, arm):
    expected = {(kind if arm == "original" else "nvfp4", b, m, n, k, route)
                for b, (kind, m, k) in SHAPES.items() for n in COLUMNS for route in range(3)}
    keys = [(r["type"], r["b"], r["m"], r["n"], r["k"], r["route"]) for r in rows]
    if len(keys) != 18 or set(keys) != expected:
        raise ValueError("Performance cases do not match the frozen expert shapes and routes")
    if any(r["n_mats"] != 512 or r["n_used"] != 22 or r["ids_offset"] != 1 for r in rows):
        raise ValueError("Wrong expert count, top-k or IDs offset")
    if any(r["runs"] <= 0 or not math.isfinite(r["time_us"]) or r["time_us"] <= 0 for r in rows):
        raise ValueError("Performance timings must be positive and finite")


def compare(base, candidate):
    validate_cases(base, "original")
    validate_cases(candidate, "nvfp4")
    target = {(r["b"], r["n"], r["route"]): r["time_us"] for r in candidate}
    rows = [{**r, "nvfp4_us": target[(r["b"], r["n"], r["route"])],
             "throughput_ratio": r["time_us"] / target[(r["b"], r["n"], r["route"])]} for r in base]
    gains = {str(n): math.exp(sum(math.log(r["throughput_ratio"]) for r in rows if r["n"] == n) / 6)
             for n in COLUMNS}
    regression = any(r["throughput_ratio"] < 0.97 for r in rows)
    decision = "reject" if regression else "advance" if gains["512"] >= 1.05 else "park"
    return {"decision": decision, "comparisons": rows, "geomean_ratios_by_columns": gains,
            "regression_limit_exceeded": regression, "weights_converted": False,
            "model_quality_checked": False, "deployed": False}


def separated_prefill_ranges(first, second):
    other = {(r["b"], r["n"], r["route"]): r for r in second["comparisons"]}
    return all(max(r["nvfp4_us"], other[(r["b"], r["n"], r["route"])]["nvfp4_us"])
               < min(r["time_us"], other[(r["b"], r["n"], r["route"])]["time_us"])
               for r in first["comparisons"] if r["n"] == 512)


def run(exe, work, label, arguments, expected, timeout):
    before = hashes(exe.parent)
    start = time.perf_counter()
    command = [str(exe), *arguments]
    save(work / f"{label}-command.json", command)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    text = ANSI.sub("", result.stdout + result.stderr)
    (work / f"{label}.log").write_text(text, encoding="utf-8")
    cases = parse_perf(text) if arguments[0] == "perf" else []
    summary = None
    if arguments[0] == "test":
        summary = re.search(r"(\d+)/(\d+) tests passed", text)
        valid = summary is not None and tuple(map(int, summary.groups())) == (expected, expected)
    elif label == "whole-graph-regression":
        valid = (len(cases) == expected and
                 {(r["whole_graph"], r["by_flops"]) for r in cases} == {(0, 0), (0, 1), (1, 0), (1, 1)})
    else:
        validate_cases(cases, label.split("-")[0])
        valid = len(cases) == expected
    passed = (result.returncode == 0 and valid and "FAIL" not in text and "not supported" not in text.lower()
              and hashes(exe.parent) == before)
    evidence = {"label": label, "exit_code": result.returncode, "expected_cases": expected,
                "cases": cases, "passed": passed, "elapsed_seconds": time.perf_counter() - start,
                "correctness_summary": summary.group(0) if summary is not None else None,
                "log_sha256": sha256(work / f"{label}.log"),
                "command": [exe.name, *arguments]}
    save(HERE / f"{label}.json", evidence)
    print(label, "passed=", passed, "result=", evidence["correctness_summary"] or f"{len(cases)} performance cases", flush=True)
    if not passed:
        print(text[-4000:], flush=True)
        raise RuntimeError(f"{label} failed")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True)
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    binaries = hashes(args.test_exe.parent)
    if binaries.get("ggml-cuda.dll") != CONTROL_CUDA:
        raise ValueError("The CUDA backend is not the frozen original")
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=30)
    if int(gpu.stdout.strip()) >= 8000:
        raise RuntimeError("GPU is not idle")
    save(HERE / "inventory.json", inventory(args.model))
    save(HERE / "manifest.json", {
        "binary_hashes": binaries,
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "fixture_sha256": sha256(HERE / "fixture.patch"),
        "runner_sha256": sha256(Path(__file__)),
        "cuda_environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("CUDA_", "GGML_CUDA_"))},
    })
    timeout = protocol["performance"]["timeout_per_launch_seconds"]
    run(args.test_exe, args.work, "whole-graph-regression",
        ["perf", "-b", "CUDA0", "-o", "PERF_GRAPH"], 4, timeout)
    for arm, types in (("original", "(q4_K|iq4_nl)"), ("nvfp4", "nvfp4")):
        run(args.test_exe, args.work, f"{arm}-correctness",
            ["test", "-b", "CUDA0", "-o", "NEMOTRON_FFN", "-p", f"type_a={types},"], 18, timeout)
    summaries = []
    for pair, order in ((1, ("original", "nvfp4")), (2, ("nvfp4", "original"))):
        results = {}
        for arm in order:
            types = "(q4_K|iq4_nl)" if arm == "original" else "nvfp4"
            results[arm] = run(args.test_exe, args.work, f"{arm}-r{pair}",
                               ["perf", "-b", "CUDA0", "-o", "NEMOTRON_FFN", "-p", f"type_a={types},"],
                               18, timeout)
        summary = compare(results["original"], results["nvfp4"])
        summaries.append(summary)
        save(HERE / f"comparison-r{pair}.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=2), flush=True)
        if summary["decision"] != "advance":
            return 3 if summary["decision"] == "reject" else 2
    separated = separated_prefill_ranges(*summaries)
    save(HERE / "decision.json", {"decision": "advance" if separated else "park",
                                 "prefill_ranges_separated": separated, "deployed": False})
    return 0 if separated else 2


if __name__ == "__main__":
    raise SystemExit(main())
