"""Score a prebuilt BF16-derived candidate and the deployed GGUF against BF16 logits (bf16-reference-protocol.json)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precision import (CASES, GGUFReader, audit_prebuilt, parse_metrics, re, require, run, runtime_manifest, save,  # noqa: E402
                       sha, shards, tensors, validate_calibration)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def non_inferior(candidate, control):
    return candidate["mean_kl_nats"] <= control["mean_kl_nats"] and candidate["ppl"] <= control["ppl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["27b"])
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path, help="BF16 GGUF")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    args = parser.parse_args()
    parent = json.loads((HERE / "protocol.json").read_text())["nvfp4_precision_preflight"]
    declared = HERE / "bf16-reference-protocol.json"
    require(json.loads(declared.read_text())["parent_protocol_sha256"] == sha(HERE / "protocol.json"), "Parent protocol changed")
    require(sha(args.dataset) == parent["dataset_sha256"], "Unexpected corpus")
    args.work.mkdir(parents=True, exist_ok=False)
    baseline = runtime_manifest(args.runtime)
    try:
        key, layers, total_layers, expected, pattern = CASES[args.case]
        source_files = shards(Path(json.loads((ROOT / "rtx-spark" / "results" / "model_paths.json").read_text())[key]))
        sources = [GGUFReader(str(path)) for path in source_files]
        before = tensors(sources)
        selected = [n for n in before if (m := re.fullmatch(pattern, n)) and int(m[1]) < layers]
        require(len(selected) == expected, f"Expected {expected} selected tensors, got {len(selected)}")
        candidate_files, audit = audit_prebuilt(args.candidate_dir, sources, before, selected)
        print("Full tensor/metadata audit passed", flush=True)
        base = args.work / "bf16.kld"
        common = [
            str(args.runtime / "llama-perplexity.exe"), "-f", str(args.dataset),
            "-c", "1024", "-b", "1024", "-ub", "512", "-ngl", "99", "-fa", "on",
            "-lzm", "on", "--chunks", "16", "-t", "20", "-tb", "20", "--kl-divergence-base", str(base),
        ]
        timeout = parent["timeout_per_process_seconds"]
        run([*common, "-m", str(args.reference_model)], args.work, "reference", timeout)
        require(base.is_file() and base.stat().st_size > 0, "No reference logits")
        scores = {}
        for label, model in (("calibration", args.reference_model), ("control", source_files[0]),
                             ("candidate", candidate_files[0])):
            run([*common, "-m", str(model), "--kl-divergence"], args.work, label, timeout)
            scores[label] = parse_metrics((args.work / f"{label}.log").read_text(encoding="utf-8"))
        bf16_ppl = validate_calibration(scores["calibration"], (args.work / "reference.log").read_text(encoding="utf-8"), parent)
        passed = non_inferior(scores["candidate"], scores["control"])
        result = {
            "case": args.case, "candidate": "Scaled per-tensor NVFP4 FFN from original BF16 (bf16-reference-protocol.json)",
            "reference": "BF16 GGUF from the pinned checkpoint", "reference_actual_ppl": bf16_ppl,
            "status": "bf16_reference_non_inferior" if passed else "bf16_reference_inferior",
            "decision": "Continue full-model speed and quality checks" if passed else "Do not enable this candidate",
            "speed_measured": False, "deployment_changed": False, "conversion": audit, **scores,
            "reference_model_bytes": args.reference_model.stat().st_size,
        }
    finally:
        after = runtime_manifest(args.runtime)
        require(after == baseline, "Runtime changed during precision evaluation")
    result.update(runtime_sha256=baseline, protocol_sha256=sha(declared), parent_protocol_sha256=sha(HERE / "protocol.json"),
                  script_sha256=sha(Path(__file__)))
    save(HERE / f"nvfp4-bf16ref-{args.case}-precision.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
