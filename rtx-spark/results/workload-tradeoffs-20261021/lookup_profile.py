"""Full structured-edit validation of the unchanged rollback-capacity candidate."""
import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "rtx-spark" / "bench"))
sys.path.insert(0, str(HERE))
import en_us_quality
from config_profiles import compare
from precision import runtime_manifest, sha
import statistics


def fixtures():
    result = {}
    for name, field, replacement, factor in (
        ("worker_timeouts", "timeout_seconds", "timeout_ms", 1000),
        ("buffer_sizes", "buffer_kib", "buffer_bytes", 1024),
    ):
        rows = [
            {"id": f"spruce-{index:02d}", "queue": f"jobs-{index % 3}", field: (index + 2) * 8,
             "enabled": index % 4 != 0, "attempts": index % 3 + 1,
             "labels": ["stable", f"zone-{index % 5}", "Case_Exact"]}
            for index in range(12)
        ]
        source = {"schema_version": 7, "workers": rows}
        expected = {"schema_version": 7, "workers": [
            {(replacement if key == field else key): value * factor if key == field else value
             for key, value in row.items()} for row in rows
        ]}
        prompt = (
            f"In every worker, rename {field} to {replacement} and multiply its integer value by {factor}. "
            "Preserve all other keys, values, string spelling and array order. Return the complete modified "
            "JSON object, not a patch, with no explanation or Markdown fences.\n\n" + json.dumps(source, indent=2)
        )
        result[name] = {"prompt": {"messages": [{"role": "user", "content": prompt}]}, "expected": expected}
    return result


def score(response, expected):
    text = en_us_quality.final_text(response)
    actual = json.loads(text, object_pairs_hook=en_us_quality.unique_object)
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def analyze(data, suite):
    if not data["completed"] or data["exit_code"] != 0 or data["stop"] != "ctrl+c" or not data["allow_eos"]:
        raise ValueError("Incomplete natural-EOS edit run")
    scores = []
    for phase in ("warmups", "rows"):
        count = 1 if phase == "warmups" else 3
        if len(data[phase]) != len(suite) * count:
            raise ValueError("Incomplete response count")
        for name in suite:
            rows = [row for row in data[phase] if row["case"] == name]
            if len(rows) != count or (phase == "rows" and sorted(row["repeat"] for row in rows) != [1, 2, 3]):
                raise ValueError("Incomplete per-case repetitions")
            for row in rows:
                scores.append({"phase": phase, "case": name, "passed": score(row["response"], suite[name]["expected"])})
    totals = [sum(row["response"]["wall_s"] for row in data["rows"] if row["repeat"] == repeat)
              for repeat in range(1, 4)]
    return {
        "quality_passed": all(item["passed"] for item in scores), "quality": scores,
        "total_request_seconds": totals, "median_total_seconds": statistics.median(totals),
        "load_seconds": data["load_s"], "ready_memory_mib": data["gpu_memory_ready_mib"],
        "rows": [{"case": row["case"], "repeat": row["repeat"], "wall_s": row["response"]["wall_s"],
                  "timings": row["response"]["timings"], "content_sha256": row["response"]["content_sha256"]}
                 for row in data["rows"]],
    }


def main():
    import run_rebaseline as runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    before = runtime_manifest(Path(runner.matrix.EXE).parent)
    candidate = runtime_manifest(args.runtime)
    differences = [name for name in before if candidate.get(name) != before[name]]
    if differences != ["llama-common.dll"]:
        raise ValueError(f"Unexpected runtime changes: {differences}")
    prior = json.loads((HERE.parent / "lookup-rollback-capacity-20261021" / "manifest.json").read_text())
    if candidate["llama-common.dll"] != prior["preflight"]["candidate_common_sha256"]:
        raise ValueError("Different rollback-capacity candidate")
    draft_hash = sha(args.draft)
    suite = fixtures()
    runner.matrix.SUITES["decode"] = [(name, entry["prompt"], 4096) for name, entry in suite.items()]
    protocol = {
        "parent_protocol_sha256": sha(HERE / "protocol.json"), "script_sha256": sha(Path(__file__)),
        "cases": suite, "repetitions": 3, "order": ["control", "candidate", "candidate", "control"],
        "maximum_new_tokens": 4096, "natural_eos_required": True, "maximum_added_ready_memory_mib": 2048,
        "score": "Every key, value, type and array order in the complete edited JSON must match.",
        "scope": "Two fresh twelve-worker full JSON edits; not the historical 512-token prefix cases.",
    }
    runner.save(args.work / "protocol.json", protocol)
    runs = []
    for index, arm in enumerate(protocol["order"]):
        out = args.work / f"{index}-{arm}"
        out.mkdir()
        extra = ["--spec-draft-vocab-dynamic", "16384", "--reasoning", "off"]
        exe = None
        if arm == "candidate":
            exe = args.runtime / "llama-server.exe"
            extra += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "12",
                      "--spec-ngram-simple-size-m", "12"]
        runner.run_server("U1", out, 3, "decode", allow_eos=True, draft_model=args.draft,
                          capture_metrics=True, exe=exe, extra_args=extra)
        if arm == "candidate":
            values = re.findall(r"lookup_snapshot target_n_rs_seq=(\d+)",
                                (out / "server_U1_decode.log").read_text(encoding="utf-8"))
            if values != ["12"]:
                raise ValueError("Expected exactly twelve actual target rollback states")
        runs.append(analyze(json.loads((out / "server_U1_decode.json").read_text()), suite))
        if not runs[-1]["quality_passed"]:
            break
    if runtime_manifest(Path(runner.matrix.EXE).parent) != before or runtime_manifest(args.runtime) != candidate:
        raise ValueError("Runtime changed during validation")
    if sha(args.draft) != draft_hash:
        raise ValueError("Draft model changed during validation")
    gate = json.loads((HERE / "protocol.json").read_text())["configuration_profiles"]["minimum_request_latency_reduction_percent"]
    comparison = compare(runs, gate) if len(runs) == 4 else None
    added = [runs[b]["ready_memory_mib"] - runs[a]["ready_memory_mib"]
             for a, b in ((0, 1), (3, 2))] if len(runs) == 4 else []
    qualified = (len(runs) == 4 and all(run["quality_passed"] for run in runs)
                 and comparison["passed"] and max(added) <= protocol["maximum_added_ready_memory_mib"])
    report = {
        "candidate": "lookup-rollback-capacity", "qualified": qualified, "runs": runs,
        "comparison": comparison, "additional_ready_memory_mib": added, "deployment_changed": False,
        "runtime_sha256": before, "candidate_runtime_sha256": candidate, "draft_sha256": draft_hash,
        "protocol": {key: value for key, value in protocol.items() if key != "cases"},
    }
    with (HERE / "lookup-full-edits.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"candidate": report["candidate"], "qualified": qualified, "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
