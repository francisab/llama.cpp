"""Bounded NVFP4 requantization and precision checks; never installs a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType

CASES = {
    "27b": ("d27", 64, 65, 192, r"blk\.(\d+)\.ffn_(gate|up|down)\.weight"),
    "nemotron": ("nem", 88, 89, 80, r"blk\.(\d+)\.ffn_(up|down)_exps\.weight"),
    "flash-next": ("qfn", 48, 48, 144, r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight"),
}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shards(first):
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", first.name)
    if not match:
        return [first]
    count = int(match[2])
    return [first.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


def tensors(readers):
    result = {}
    for reader in readers:
        for tensor in reader.tensors:
            require(tensor.name not in result, f"Duplicate tensor {tensor.name}")
            result[tensor.name] = tensor
    if len(readers) > 1:
        for index, reader in enumerate(readers):
            require(reader.fields["split.no"].contents() == index, "Incorrect split index")
            require(reader.fields["split.count"].contents() == len(readers), "Incorrect split count")
            require(reader.fields["split.tensors.count"].contents() == len(result), "Incomplete split tensors")
    return result


def parse_metrics(text):
    chunks = [int(value) for value in re.findall(r"^\s*(\d+)\s+[\d.]+\s+\u00b1", text, re.M)]
    require(chunks == list(range(1, 17)), "Expected all 16 precision chunks exactly once")
    patterns = {
        "ppl": r"Mean PPL\(Q\)\s*:\s*([-\d.eE+]+)",
        "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([-\d.eE+]+)",
        "mean_kl_nats": r"Mean\s+KLD:\s*([-\d.eE+]+)",
        "same_top_percent": r"Same top p:\s*([-\d.eE+]+)",
    }
    result = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        require(len(matches) == 1, f"Missing or duplicate {name}")
        value = float(matches[0])
        require(math.isfinite(value), f"Nonfinite {name}")
        result[name] = value
    require(result["ppl"] > 0 and result["ppl_ratio"] > 0, "Invalid perplexity")
    require(0 <= result["same_top_percent"] <= 100, "Invalid top-token agreement")
    return result


def validate_calibration(calibration, reference_text, protocol):
    matches = re.findall(r"Final estimate: PPL = ([\d.eE+-]+)", reference_text)
    require(len(matches) == 1, "Missing or duplicate original perplexity estimate")
    original_ppl = float(matches[0])
    require(math.isfinite(original_ppl) and original_ppl > 0, "Invalid original perplexity")
    require(abs(calibration["mean_kl_nats"]) <= protocol["maximum_calibration_absolute_mean_kl_nats"],
            "Same-model KL exceeds its numerical floor")
    require(abs(calibration["ppl"] / original_ppl - 1) <= protocol["maximum_calibration_actual_ppl_relative_difference"],
            "Same-model actual PPL differs from the original reference pass")
    return original_ppl


def precision_passes(metrics, calibration, protocol):
    ratio = metrics["ppl"] / calibration["ppl"]
    return (ratio <= protocol["maximum_mean_perplexity_ratio"]
            and metrics["mean_kl_nats"] <= protocol["maximum_mean_kl_nats"]), ratio


def run(command, work, label, timeout):
    start = time.monotonic()
    log = work / f"{label}.log"
    record = {"command": list(map(str, command)), "completed": False}
    print(f"{label}: starting", flush=True)
    try:
        with log.open("xb") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        record.update(exit_code=process.returncode, completed=True)
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
        raise
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        record["log_sha256"] = sha(log)
        save(work / f"{label}-run.json", record)
    require(process.returncode == 0, f"{label} failed with exit {process.returncode}; inspect its retained log")
    print(f"{label}: completed in {record['elapsed_seconds']}s", flush=True)
    return record


def runtime_manifest(runtime):
    files = sorted([*runtime.glob("*.dll"), *runtime.glob("*.exe")])
    require(files, "Empty runtime")
    return {path.name: sha(path) for path in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args()
    public = Path(__file__).parent
    protocol = json.loads((public / "protocol.json").read_text())["nvfp4_precision_preflight"]
    require(sha(args.dataset) == protocol["dataset_sha256"], "Unexpected corpus")
    args.work.mkdir(parents=True, exist_ok=False)
    baseline_manifest = runtime_manifest(args.runtime)
    save(args.work / "runtime-before.json", baseline_manifest)
    try:
        result = evaluate(args, protocol)
    finally:
        final_manifest = runtime_manifest(args.runtime)
        save(args.work / "runtime-after.json", final_manifest)
        require(final_manifest == baseline_manifest, "Runtime changed during precision evaluation")
    result["runtime_sha256"] = baseline_manifest
    result["protocol_sha256"] = sha(public / "protocol.json")
    result["script_sha256"] = sha(Path(__file__))
    save(public / f"nvfp4-{args.case}-precision.json", result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate(args, protocol):
    key, layers, total_layers, expected, pattern = CASES[args.case]
    paths = json.loads((ROOT / "rtx-spark" / "results" / "model_paths.json").read_text())
    source_files = shards(Path(paths[key]))
    sources = [GGUFReader(str(path)) for path in source_files]
    architecture = sources[0].fields["general.architecture"].contents()
    require(sources[0].fields[f"{architecture}.block_count"].contents() == total_layers, "Unexpected target/MTP layout")
    before = tensors(sources)
    selected = {}
    for name, tensor in before.items():
        match = re.fullmatch(pattern, name)
        if match and int(match[1]) < layers:
            selected[name] = tensor
    require(len(selected) == expected, f"Expected {expected} selected tensors, got {len(selected)}")
    desired = {
        name: GGMLQuantizationType.NVFP4 if name in selected else tensor.tensor_type
        for name, tensor in before.items()
    }
    mapping = args.work / "tensor-types.txt"
    with mapping.open("x", encoding="ascii", newline="\n") as stream:
        for name, kind in sorted(desired.items()):
            stream.write("^" + re.escape(name) + "$=" + kind.name + "\n")
    model_dir = args.work / "model"
    model_dir.mkdir()
    output = model_dir / f"{args.case}-ffn-nvfp4"
    ftype = str(sources[0].fields["general.file_type"].contents())
    command = [
        str(args.runtime / "llama-quantize.exe"), "--allow-requantize", "--pure",
        "--keep-split", "--max-buffer-size", "512", "--tensor-type-file", str(mapping),
        str(source_files[0]), str(output), ftype, "8",
    ]
    run([command[0], "--dry-run", *command[1:]], args.work, "dry-run", 120)
    run(command, args.work, "conversion", protocol["timeout_per_process_seconds"])
    candidate_files = sorted(model_dir.glob("*.gguf"))
    require(len(candidate_files) == len(source_files), "Unexpected output shard count")
    candidates = [GGUFReader(str(path)) for path in candidate_files]
    after = tensors(candidates)
    require(before.keys() == after.keys(), "Tensor names changed")
    for source, candidate in zip(sources, candidates):
        require(source.fields.keys() == candidate.fields.keys(), "Metadata fields changed")
        for name in source.fields:
            if name.startswith("GGUF."):
                continue
            a, b = source.fields[name], candidate.fields[name]
            require(a.types == b.types and a.contents() == b.contents(), f"Metadata changed: {name}")
    unchanged, changed = [], []
    for name in sorted(before):
        a, b = before[name], after[name]
        require(list(a.shape) == list(b.shape) and b.tensor_type == desired[name], f"Wrong type/shape: {name}")
        digest_a = hashlib.sha256(a.data).hexdigest()
        digest_b = hashlib.sha256(b.data).hexdigest()
        if name in selected:
            require(a.tensor_type != b.tensor_type, f"Unconverted selection: {name}")
            changed.append([name, digest_a, digest_b])
        else:
            require(a.tensor_type == b.tensor_type and a.n_bytes == b.n_bytes and digest_a == digest_b,
                    f"Unselected payload changed: {name}")
            unchanged.append([name, digest_a])
    audit = {
        "source_filenames": [path.name for path in source_files],
        "candidate_filenames": [path.name for path in candidate_files],
        "source_bytes": sum(path.stat().st_size for path in source_files),
        "candidate_bytes": sum(path.stat().st_size for path in candidate_files),
        "changed_tensors": len(changed), "unchanged_payloads_verified": len(unchanged),
        "metadata_unchanged": True, "tensor_map_sha256": sha(mapping),
        "unchanged_payload_manifest_sha256": hashlib.sha256(json.dumps(unchanged).encode()).hexdigest(),
        "changed_payload_manifest_sha256": hashlib.sha256(json.dumps(changed).encode()).hexdigest(),
    }
    save(args.work / "conversion-audit.json", audit)
    save(args.work / "payload-manifest.json", {"changed": changed, "unchanged": unchanged})
    print("Full tensor/metadata audit passed", flush=True)
    reference = args.work / "baseline.kld"
    common = [
        str(args.runtime / "llama-perplexity.exe"), "-f", str(args.dataset),
        "-c", "1024", "-b", "1024", "-ub", "512", "-ngl", "99", "-fa", "on",
        "-lzm", "on", "--chunks", "16", "-t", "20", "-tb", "20",
        "--kl-divergence-base", str(reference),
    ]
    run([*common, "-m", str(source_files[0])], args.work, "reference", protocol["timeout_per_process_seconds"])
    require(reference.is_file() and reference.stat().st_size > 0, "No reference logits")
    run([*common, "-m", str(source_files[0]), "--kl-divergence"], args.work,
        "calibration", protocol["timeout_per_process_seconds"])
    calibration = parse_metrics((args.work / "calibration.log").read_text(encoding="utf-8"))
    original_ppl = validate_calibration(
        calibration, (args.work / "reference.log").read_text(encoding="utf-8"), protocol)
    run([*common, "-m", str(candidate_files[0]), "--kl-divergence"], args.work,
        "candidate", protocol["timeout_per_process_seconds"])
    metrics = parse_metrics((args.work / "candidate.log").read_text(encoding="utf-8"))
    passed, ppl_ratio = precision_passes(metrics, calibration, protocol)
    return {
        "case": args.case, "candidate": "NVFP4 from existing quantized GGUF, not BF16",
        "status": "development_precision_pass" if passed else "development_precision_fail",
        "decision": "Continue held-out quality validation" if passed else "Do not enable this requantized candidate",
        "speed_measured": False, "deployment_changed": False,
        "conversion": audit, "calibration": calibration, "metrics": metrics,
        "actual_ppl_ratio": ppl_ratio, "reference_printed_actual_ppl": original_ppl,
    }


if __name__ == "__main__":
    main()
