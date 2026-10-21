"""Summarize generated token-ID coverage and speculative acceptance from benchmark results."""
import argparse
import json
from pathlib import Path
import re
import statistics

from heldout_suite import EN_US_CASES

POSITION = re.compile(r'^spec_decode_num_accepted_tokens_per_pos_total\{position="(\d+)"\}$')


def summarize(rows, cutoffs):
    tokens = [token for row in rows for token in row["response"]["tokens"]]
    timings = [row["response"]["timings"] for row in rows]
    result = {
        "responses": len(rows), "tokens": len(tokens), "distinct": len(set(tokens)),
        "max_id": max(tokens, default=None),
        # This measures a low-ID prefix, not a learned vocabulary shortlist.
        "id_coverage": {str(k): sum(token < k for token in tokens) / len(tokens) if tokens else None for k in cutoffs},
        "median_tg": statistics.median(t["predicted_per_second"] for t in timings) if timings else None,
    }
    if any("draft_n" in t for t in timings):
        result["draft_n"] = sum(t.get("draft_n", 0) for t in timings)
        result["draft_n_accepted"] = sum(t.get("draft_n_accepted", 0) for t in timings)
        result["acceptance"] = result["draft_n_accepted"] / result["draft_n"] if result["draft_n"] else None
    metrics = [row["response"]["spec_metrics"] for row in rows if "spec_metrics" in row["response"]]
    if metrics:
        totals = {}
        for values in metrics:
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
        steps = totals.get("spec_decode_num_drafts_total", 0.0)
        positions = sorted((int(match[1]), value) for name, value in totals.items() if (match := POSITION.match(name)))
        result["verify_steps"] = steps
        result["mean_accepted_length"] = 1 + totals.get("spec_decode_num_accepted_tokens_total", 0.0) / steps if steps else None
        result["accepted_per_position"] = [value / steps for _, value in positions] if steps else None
    return result


def analyze(path, cutoffs, en_us=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data["completed"] or data.get("exit_code") != 0 or not data["rows"]:
        raise ValueError(f"Incomplete benchmark: {path}")
    rows = data["rows"]
    if en_us:
        missing = set(EN_US_CASES) - {row["case"] for row in rows}
        if missing:
            raise ValueError(f"Missing en-us cases in {path}: {sorted(missing)}")
        rows = [row for row in rows if row["case"] in EN_US_CASES]
    cases = {}
    for row in rows:
        cases.setdefault(row["case"], []).append(row)
    return {"path": str(path), "name": data["name"], "suite": data["suite"], "completed": data["completed"],
            "scope": "en-us" if en_us else "all", "overall": summarize(rows, cutoffs),
            "cases": {case: summarize(rows, cutoffs) for case, rows in cases.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoffs", default="4096,8192,16384,32768,65536,131072")
    parser.add_argument("--scope", choices=["all", "en-us"], default="all",
                        help="Select English/code held-out cases; fail if any selected case is missing")
    args = parser.parse_args()
    cutoffs = [int(value) for value in args.cutoffs.split(",")]
    if not cutoffs or any(value < 1 for value in cutoffs):
        parser.error("--cutoffs must contain positive integers")
    summary = {"cutoffs": cutoffs, "scope": args.scope,
               "files": [analyze(path, cutoffs, en_us=args.scope == "en-us") for path in args.inputs]}
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    for item in summary["files"]:
        for scope, stats in [("overall", item["overall"])] + list(item["cases"].items()):
            coverage = " ".join(f"{value:.3f}" if value is not None else "N/A" for value in stats["id_coverage"].values())
            accepted = f" acc={stats['acceptance']:.3f}" if stats.get("acceptance") is not None else ""
            depth = stats.get("accepted_per_position")
            depth = " pos=" + ",".join(f"{value:.3f}" for value in depth) if depth else ""
            print(f"{item['name']} {scope}: tok={stats['tokens']} distinct={stats['distinct']} "
                  f"cov={coverage}{accepted}{depth}")


if __name__ == "__main__":
    main()
