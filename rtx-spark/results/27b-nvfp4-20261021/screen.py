"""Screen the actual 27B FFN type/shape mix before any NVFP4 conversion."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType, GGML_QUANT_SIZES

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PERF = re.compile(r"(MUL_MAT\([^\n]*\)):\s+(\d+) runs\s+-\s+([\d.]+) us/run")
LABEL = re.compile(r"q27_(\w+)_m(\d+)_n(\d+)_k(\d+)")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hashes(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in (".exe", ".dll"):
            with path.open("rb") as stream:
                result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def fixture(kind, m, n, k):
    block, size = GGML_QUANT_SIZES[kind]
    if k % block:
        raise ValueError(f"Unaligned {kind.name} matrix")
    stride = k // block * size
    values = [
        29, 0, m, n, 1, 1, 0, 2,
        int(kind), k, m, 1, 1, size, stride, stride * m, stride * m,
        0, k, n, 1, 1, 4, k * 4, k * n * 4, k * n * 4,
    ]
    return " ".join(map(str, values)) + f" q27_{kind.name.lower()}_m{m}_n{n}_k{k}"


def run(exe, work, label, arguments, expected, timeout):
    command = [str(exe), *arguments]
    before = hashes(exe.parent)
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    text = ANSI.sub("", result.stdout + result.stderr)
    (work / f"{label}.log").write_text(text, encoding="utf-8")
    cases = []
    if arguments[0] == "test":
        summary = re.search(r"(\d+)/(\d+) tests passed", text)
        valid = summary is not None and tuple(map(int, summary.groups())) == (expected, expected)
    else:
        for description, repetitions, elapsed in PERF.findall(text):
            match = LABEL.search(description)
            if match is None:
                raise ValueError(f"Missing fixture label: {description}")
            kind, m, n, k = match.groups()
            cases.append({"type": kind, "m": int(m), "n": int(n), "k": int(k),
                          "runs": int(repetitions), "time_us": float(elapsed)})
        valid = (len(cases) == expected
                 and len({(c["type"], c["m"], c["n"], c["k"]) for c in cases}) == expected
                 and all(c["runs"] > 0 and math.isfinite(c["time_us"]) and c["time_us"] > 0 for c in cases))
    passed = result.returncode == 0 and valid and "FAIL" not in text and hashes(exe.parent) == before
    evidence = {"label": label, "exit_code": result.returncode, "expected_cases": expected,
                "cases": cases, "passed": passed, "elapsed_seconds": time.perf_counter() - start,
                "command": [exe.name, *[Path(a).name if "\\" in a else a for a in arguments]]}
    save(HERE / f"{label}.json", evidence)
    print(label, "passed=", passed, "cases=", len(cases) if cases else expected, flush=True)
    if not passed:
        print(text[-4000:], flush=True)
        raise RuntimeError(f"{label} failed")
    return cases


def compare(base, candidate, counts):
    expected_base = {(kind, m, n, k) for kind, m, k in counts for n in (1, 8, 512)}
    expected_candidate = {("nvfp4", m, n, k) for _, m, n, k in expected_base}
    for rows, expected in ((base, expected_base), (candidate, expected_candidate)):
        keys = [(c["type"], c["m"], c["n"], c["k"]) for c in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Performance cases do not match the complete FFN inventory")
        if any(not math.isfinite(c["time_us"]) or c["time_us"] <= 0 for c in rows):
            raise ValueError("Performance timings must be positive and finite")
    if not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("Expected positive tensor counts")
    target = {(c["m"], c["n"], c["k"]): c["time_us"] for c in candidate}
    comparisons = []
    for c in base:
        key = (c["type"], c["m"], c["k"])
        ratio = c["time_us"] / target[(c["m"], c["n"], c["k"])]
        comparisons.append({**c, "nvfp4_us": target[(c["m"], c["n"], c["k"])],
                            "throughput_ratio": ratio, "tensor_count": counts[key]})
    weighted = {}
    for n in (1, 8, 512):
        rows = [c for c in comparisons if c["n"] == n]
        weighted[str(n)] = math.exp(sum(c["tensor_count"] * math.log(c["throughput_ratio"]) for c in rows)
                                   / sum(c["tensor_count"] for c in rows))
    regression = any(c["throughput_ratio"] < 0.97 for c in comparisons)
    decision = "reject" if regression else "advance" if weighted["512"] >= 1.05 else "park"
    return {"decision": decision, "comparisons": comparisons,
            "weighted_geomean_ratios_by_columns": weighted,
            "regression_limit_exceeded": regression, "model_quality_checked": False, "deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True)
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    reader = GGUFReader(args.model)
    layers = int(reader.fields["qwen35.block_count"].contents()) - int(reader.fields["qwen35.nextn_predict_layers"].contents())
    if layers != 64:
        raise ValueError(f"Unexpected target layer count: {layers}")
    counts = Counter()
    for tensor in reader.tensors:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(up|gate|down)\.weight", tensor.name)
        if match and int(match[1]) < layers:
            k, m = map(int, tensor.shape)
            counts[(tensor.tensor_type.name.lower(), m, k)] += 1
    if sum(counts.values()) != 192 or {(m, k) for _, m, k in counts} != {(17408, 5120), (5120, 17408)}:
        raise ValueError("Unexpected target FFN inventory")
    source = [{"type": kind, "m": m, "k": k, "count": count} for (kind, m, k), count in sorted(counts.items())]
    save(HERE / "inventory.json", {"model": args.model.name, "target_layers": layers, "tensors": source})
    original = [fixture(GGMLQuantizationType[kind.upper()], m, n, k)
                for kind, m, k in sorted(counts) for n in (1, 8, 512)]
    candidate = [fixture(GGMLQuantizationType.NVFP4, m, n, k)
                 for m, k in sorted({(m, k) for _, m, k in counts}) for n in (1, 8, 512)]
    for name, lines in (("original", original), ("nvfp4", candidate)):
        (HERE / f"{name}-ops.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pattern = protocol["correctness"]["parameter_filter"]
    support = subprocess.run([str(args.test_exe), "support", "-b", "CPU", "-o", "MUL_MAT.*", "-p", pattern],
                             capture_output=True, text=True, check=True, timeout=60)
    expected = len(re.findall(r": SUPPORTED", ANSI.sub("", support.stdout)))
    if expected < 4:
        raise ValueError("Too few NVFP4 reference fixtures")
    save(HERE / "manifest.json", {"binary_hashes": hashes(args.test_exe.parent),
                                 "selected_correctness_cases": expected,
                                 "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()})
    run(args.test_exe, args.work, "correctness",
        ["test", "-b", "CUDA0", "-o", "MUL_MAT.*", "-p", pattern], expected, 600)
    for pair, order in ((1, ("original", "nvfp4")), (2, ("nvfp4", "original"))):
        results = {}
        for arm in order:
            results[arm] = run(
                args.test_exe, args.work, f"{arm}-r{pair}",
                ["perf", "-b", "CUDA0", "--test-file", str(HERE / f"{arm}-ops.txt")],
                len(original) if arm == "original" else len(candidate), 600)
        summary = compare(results["original"], results["nvfp4"], counts)
        save(HERE / f"comparison-r{pair}.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=2), flush=True)
        if summary["decision"] != "advance":
            return 3 if summary["decision"] == "reject" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
