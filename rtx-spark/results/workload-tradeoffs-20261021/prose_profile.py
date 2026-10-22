"""Fresh complete source-grounded prose at the actual 27B launcher's two draft depths."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "rtx-spark" / "bench"))
import en_us_quality
from config_profiles import compare
from precision import runtime_manifest, sha, save

FACTS = {
    "incident": (
        "Spruce-Gateway release 4.8.2 was deployed on May 14, 2032. Errors occurred from 10:05 UTC to "
        "10:17 UTC, a duration of 12 minutes. There were 1200 requests and 72 failures, a 6% failure rate. "
        "The deployment changed the timeout from 2000 ms to 500 ms. Restoring 2000 ms ended the errors. "
        "No database corruption or data loss was observed. Version 4.8.1 is the rollback option, but no "
        "software rollback was performed. The next rollout starts at 5%, then 25%, then 100%; each "
        "expansion requires an error rate below 1% for 30 minutes.",
        ["Spruce-Gateway", "4.8.2", "May 14, 2032", "10:05", "10:17", "12", "1200", "72", "6%",
         "2000", "500", "4.8.1", "5%", "25%", "100%", "1%", "30"],
    ),
    "cache_policy": (
        "Birch-Cache uses policy version 2.6.0. Its capacity is 240 entries and its fixed TTL is 90 seconds. "
        "A hit does not extend the TTL. At capacity it evicts the least recently used entry, not the "
        "oldest inserted entry. Expired entries are removed before capacity eviction. Keys are case-sensitive: "
        "Order-A and order-a are different keys. Failed fetches are never cached. A successful write invalidates "
        "the matching key immediately. A deployment starts with an empty cache; no cache is shared across "
        "processes. The measurement window contained 800 reads, with 600 hits and 200 misses, a 75% hit rate. "
        "The measurements do not establish a latency or monetary saving.",
        ["Birch-Cache", "2.6.0", "240", "90", "Order-A", "order-a", "800", "600", "200", "75%"],
    ),
}


def cases():
    return [
        (name, {"messages": [{"role": "user", "content":
            "Write a complete source-grounded explanation in US English, about 250 to 350 words in prose "
            "paragraphs. Explain the sequence or policy and its operational implications. Include every "
            "fact and caveat below. Do not invent events, causes or benefits. Preserve identifiers and "
            "numerical literals exactly as written, but use your own sentences. Return only the explanation.\n\n"
            + facts}]}, 1536)
        for name, (facts, _) in FACTS.items()
    ]


def check_literals(case, response):
    text = en_us_quality.final_text(response)
    missing = [value for value in FACTS[case][1]
               if not re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text)]
    words = len(re.findall(r"\b[\w'-]+\b", text))
    return {"passed": not missing and 120 <= words <= 650, "missing": missing, "words": words,
            "content_sha256": response["content_sha256"]}


def main():
    import run_rebaseline as runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    prior = HERE.parent / "27b-profiles-20261021"
    spec = importlib.util.spec_from_file_location("launcher_profile", prior / "validate.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    catalog_hash = sha(args.catalog)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8-sig"))
    model = next(model for model in catalog["models"] if model["id"] == "qwen/qwen3.8-27b")
    recipe_id = json.loads((prior / "protocol.json").read_text())["control_recipe"]
    recipe = next(recipe for recipe in model["recipes"] if recipe["id"] == recipe_id)
    source = json.loads((prior / "launcher-source.json").read_text())
    if recipe["serve"] != source["serve"]:
        raise ValueError("Actual 27B launcher command changed")
    runner.matrix.CONFIGS = [(f"P{depth} q27 launcher dflash{depth}",
                             launcher.resolve_command(recipe, runner.matrix.P, runner.matrix.PORT, depth), "decode")
                            for depth in (7, 4)]
    runner.matrix.SUITES["decode"] = cases()
    baseline = runtime_manifest(Path(runner.matrix.EXE).parent)
    protocol = {
        "script_sha256": sha(Path(__file__)), "launcher_resolver_sha256": sha(prior / "validate.py"),
        "parent_protocol_sha256": sha(HERE / "protocol.json"), "catalog_sha256": catalog_hash,
        "recipe": recipe_id, "cases": cases(), "order": [7, 4, 4, 7], "repetitions": 3,
        "primary_score": "Sum of complete natural-EOS request times for both fresh prose tasks",
        "quality": "Literal/completion checks are preliminary. Independently review every unique full response against all supplied facts and caveats before promotion.",
    }
    save(args.work / "protocol.json", protocol)
    runs = []
    for index, depth in enumerate(protocol["order"]):
        out = args.work / f"{index}-depth{depth}"
        out.mkdir()
        runner.run_server(f"P{depth}", out, 3, "decode", allow_eos=True,
                          capture_metrics=True, extra_args=["--reasoning", "off"])
        data = json.loads((out / f"server_P{depth}_decode.json").read_text())
        if not data["completed"] or data["exit_code"] != 0 or data["stop"] != "ctrl+c" or not data["allow_eos"]:
            raise ValueError("Incomplete prose run")
        checks = []
        for phase in ("warmups", "rows"):
            expected = 1 if phase == "warmups" else 3
            if len(data[phase]) != expected * len(FACTS):
                raise ValueError("Incomplete prose response set")
            for case in FACTS:
                rows = [row for row in data[phase] if row["case"] == case]
                if len(rows) != expected or (phase == "rows" and sorted(row["repeat"] for row in rows) != [1, 2, 3]):
                    raise ValueError("Missing or duplicate prose responses")
                checks += [{"phase": phase, "case": case, **check_literals(case, row["response"])} for row in rows]
        totals = [sum(row["response"]["wall_s"] for row in data["rows"] if row["repeat"] == repeat)
                  for repeat in range(1, 4)]
        runs.append({"depth": depth, "checks": checks, "total_request_seconds": totals,
                     "median_total_seconds": statistics.median(totals), "ready_memory_mib": data["gpu_memory_ready_mib"],
                     "rows": [{"case": row["case"], "repeat": row["repeat"], "timings": row["response"]["timings"],
                               "wall_s": row["response"]["wall_s"], "content_sha256": row["response"]["content_sha256"]}
                              for row in data["rows"]]})
        if not all(item["passed"] for item in checks):
            break
    if runtime_manifest(Path(runner.matrix.EXE).parent) != baseline or sha(args.catalog) != catalog_hash:
        raise ValueError("Runtime or launcher changed during validation")
    gate = json.loads((HERE / "protocol.json").read_text())["configuration_profiles"]["minimum_request_latency_reduction_percent"]
    comparison = compare(runs, gate) if len(runs) == 4 else None
    result = {
        "candidate": "27b-dflash4", "qualified": False, "independent_prose_review": "pending",
        "runs": runs, "comparison": comparison, "deployment_changed": False,
        "protocol": {key: value for key, value in protocol.items() if key != "cases"}, "runtime_sha256": baseline,
    }
    save(HERE / "27b-complete-prose.json", result)
    print(json.dumps({"candidate": result["candidate"], "comparison": comparison,
                      "independent_prose_review": "pending"}), flush=True)


if __name__ == "__main__":
    main()
