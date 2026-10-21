"""Fresh complete-task screen of existing speculative/sampling options."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rtx-spark" / "bench"))
import en_us_quality

HERE = Path(__file__).parent
ROWS = [
    [("Birch", 600, 1), ("Maple", 1800, 1), ("Birch", 700, 1), ("Maple", 9000, 0), ("Ash", 900, 1)],
    [("Elm", 1000, 1), ("Ash", 1000, 1), ("Elm", 300, 0), ("Oak", 999, 1)],
    [("Pine", 400, 1), ("Pine", 600, 1), ("Elm", 5000, 0), ("Ash", 800, 1), ("Ash", 1200, 1)],
]
EXPECTED_JSON = {
    "ticket": "SPK-08312-bC", "region": "us-west-7", "enabled": False,
    "limits": {"workers": 12, "timeout_ms": 7500}, "tags": ["steady", "v2.04", "Case_Sensitive"],
    "amount_text": "000079.300", "checksum": "dA91f03B7c",
}


def cases():
    sql = (
        "Write only one SQLite SELECT query, without explanation. The table orders has columns "
        "customer TEXT, amount_cents INTEGER, paid INTEGER. Select each customer and their total "
        "amount_cents from rows with paid=1. Keep only customers whose paid total is at least 1000. "
        "Return exactly two columns named customer and total_cents, sorted by total_cents descending, "
        "then customer ascending. The table contents are not known in advance."
    )
    document = (
        "Return only a JSON object describing record TARGET. Include exactly ticket, region, enabled, "
        "limits, tags, amount_text, checksum. limits contains exactly workers and timeout_ms. "
        "Preserve string case, punctuation, tag order, and textual zeros. Use a boolean for enabled "
        "and integers for both limits. Do not copy any DECOY fields.\n"
        "DECOY: ticket=SPK-08321-bC; region=us-east-7; enabled=true; workers=21; timeout_ms=5700; "
        "tags=steady,v2.40,case_sensitive; amount_text=000079.030; checksum=dA19f03B7c\n"
        "TARGET: ticket=SPK-08312-bC; region=us-west-7; enabled=false; workers=12; timeout_ms=7500; "
        "tags=steady,v2.04,Case_Sensitive; amount_text=000079.300; checksum=dA91f03B7c"
    )
    return [(name, {"messages": [{"role": "user", "content": text}]}, 768)
            for name, text in (("sql", sql), ("structured_copy", document))]


def score_sql(text):
    fence = re.fullmatch(r"```(?:sql|sqlite)?\s*\n(.*?)\n```", text, re.S)
    query = fence[1] if fence else text
    if len(query) > 4096 or not re.match(r"(?:SELECT|WITH)\b", query, re.I):
        raise ValueError("Expected a bounded SELECT query")
    for rows in ROWS:
        db = sqlite3.connect(":memory:")
        try:
            db.execute("CREATE TABLE orders(customer TEXT, amount_cents INTEGER, paid INTEGER)")
            db.executemany("INSERT INTO orders VALUES(?,?,?)", rows)
            totals = {}
            for customer, amount, paid in rows:
                if paid == 1:
                    totals[customer] = totals.get(customer, 0) + amount
            expected = sorted(((name, total) for name, total in totals.items() if total >= 1000),
                              key=lambda row: (-row[1], row[0]))
            reads = []

            def authorize(action, a, b, _database, _trigger):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and a == "orders":
                    reads.append(b)
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and b.lower() in {"sum", "count", "min", "max", "coalesce"}:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            db.set_authorizer(authorize)
            db.set_progress_handler(lambda: 1, 10000)
            cursor = db.execute(query)
            actual = cursor.fetchmany(100)
            if ([column[0] for column in cursor.description] != ["customer", "total_cents"]
                    or actual != expected or not reads):
                raise ValueError("Query fails held-out table contents, ordering or output schema")
        finally:
            db.close()
    return {"passed": True, "datasets": len(ROWS)}


def score(case, response):
    try:
        text = en_us_quality.final_text(response)
        if case == "sql":
            return score_sql(text)
        if case != "structured_copy":
            raise ValueError(f"Unknown task: {case}")
        result = json.loads(text, object_pairs_hook=en_us_quality.unique_object)
        if result != EXPECTED_JSON or type(result["enabled"]) is not bool:
            raise ValueError("Incorrect or incomplete structured record")
        if any(type(result["limits"][key]) is not int for key in ("workers", "timeout_ms")):
            raise ValueError("Wrong limit value types")
        return {"passed": True, "fields": len(EXPECTED_JSON)}
    except (ValueError, TypeError, KeyError, sqlite3.Error) as error:
        return {"passed": False, "error": str(error)}


def audit(data):
    if not data["completed"] or data["exit_code"] != 0 or data["stop"] != "ctrl+c" or not data["allow_eos"]:
        raise ValueError("Incomplete natural-EOS server run")
    names = [item[0] for item in cases()]
    scored = []
    for phase in ("warmups", "rows"):
        expected = 1 if phase == "warmups" else 3
        if len(data[phase]) != expected * len(names):
            raise ValueError("Incomplete task counts")
        for name in names:
            rows = [row for row in data[phase] if row["case"] == name]
            if len(rows) != expected:
                raise ValueError("Wrong per-task counts")
            if phase == "rows" and sorted(row["repeat"] for row in rows) != [1, 2, 3]:
                raise ValueError("Missing or duplicate repetitions")
            for row in rows:
                if not math.isfinite(row["response"]["wall_s"]) or row["response"]["wall_s"] <= 0:
                    raise ValueError("Invalid request duration")
                scored.append({"phase": phase, "case": name, **score(name, row["response"])})
    return {"passed": all(item["passed"] for item in scored), "scores": scored}


def summarize(data):
    totals = [sum(row["response"]["wall_s"] for row in data["rows"] if row["repeat"] == repeat)
              for repeat in range(1, 4)]
    details = {}
    for name, _, _ in cases():
        responses = [row["response"] for row in data["rows"] if row["case"] == name]
        details[name] = {
            "wall_seconds": [item["wall_s"] for item in responses],
            "generated_tokens": [item["timings"]["predicted_n"] for item in responses],
            "content_sha256": [item["content_sha256"] for item in responses],
            "draft_tokens": [item["timings"].get("draft_n", 0) for item in responses],
            "accepted_tokens": [item["timings"].get("draft_n_accepted", 0) for item in responses],
        }
    return {"total_request_seconds": totals, "median_total_seconds": statistics.median(totals),
            "load_seconds": data["load_s"], "ready_memory_mib": data["gpu_memory_ready_mib"], "cases": details}


def compare(runs, gate):
    pairs = []
    for a, b in ((0, 1), (3, 2)):
        base, candidate = runs[a], runs[b]
        pairs.append({
            "latency_reduction_percent": 100 * (1 - candidate["median_total_seconds"] / base["median_total_seconds"]),
            "separated": max(candidate["total_request_seconds"]) < min(base["total_request_seconds"]),
        })
    drift = 100 * abs(runs[0]["median_total_seconds"] - runs[3]["median_total_seconds"]) / min(
        runs[0]["median_total_seconds"], runs[3]["median_total_seconds"])
    passed = all(pair["separated"] and pair["latency_reduction_percent"] >= gate
                 and pair["latency_reduction_percent"] > drift for pair in pairs)
    return {"passed": passed, "pairs": pairs, "control_reload_drift_percent": drift}


def main():
    import run_rebaseline as runner
    from precision import runtime_manifest, sha

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=["backend-sampling", "mtp4", "q5-head"])
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--candidate-draft", type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    if (args.candidate == "q5-head") != (args.candidate_draft is not None):
        parser.error("Only q5-head requires --candidate-draft")
    args.draft = args.draft.resolve(strict=True)
    args.work.mkdir(parents=True, exist_ok=False)
    runtime = Path(runner.matrix.EXE).parent
    binaries = runtime_manifest(runtime)
    draft_hash = sha(args.draft)
    candidate_draft_hash = sha(args.candidate_draft) if args.candidate_draft else None
    runner.matrix.SUITES["decode"] = cases()
    runner.matrix.PORT = 9940
    policy = json.loads((HERE / "protocol.json").read_text())["configuration_profiles"]
    snapshot = {
        "candidate": args.candidate, "order": ["control", "candidate", "candidate", "control"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
        "cases": cases(), "repetitions": 3, "natural_eos": True,
        "draft_filename": args.draft.name, "scope": "Greedy SQL generation and structured-record copying; no grammar.",
    }
    runner.save(args.work / "protocol.json", snapshot)
    summaries, quality = [], []
    for index, arm in enumerate(snapshot["order"]):
        out = args.work / f"{index}-{arm}"
        out.mkdir()
        candidate = arm == "candidate"
        tag = "U4" if candidate and args.candidate == "mtp4" else "U1"
        extra = ["--spec-draft-vocab-dynamic", "16384", "--reasoning", "off"]
        if candidate and args.candidate == "backend-sampling":
            extra.append("--backend-sampling")
        draft = args.candidate_draft if candidate and args.candidate == "q5-head" else args.draft
        runner.run_server(tag, out, 3, "decode", allow_eos=True, draft_model=draft,
                          capture_metrics=True, extra_args=extra)
        data = json.loads((out / f"server_{tag}_decode.json").read_text())
        quality.append(audit(data))
        summaries.append(summarize(data))
        if not quality[-1]["passed"]:
            break
    timing = compare(summaries, policy["minimum_request_latency_reduction_percent"]) if len(summaries) == 4 else None
    if runtime_manifest(runtime) != binaries or sha(args.draft) != draft_hash:
        raise ValueError("Runtime or control draft changed during evaluation")
    if args.candidate_draft and sha(args.candidate_draft) != candidate_draft_hash:
        raise ValueError("Candidate draft changed during evaluation")
    passed = all(item["passed"] for item in quality) and timing is not None and timing["passed"]
    result = {
        "candidate": args.candidate, "scope": snapshot["scope"], "qualified": passed,
        "quality": quality, "runs": summaries, "comparison": timing,
        "decision": "Eligible for opt-in release confirmation" if passed else "Do not enable from this screen",
        "deployment_changed": False, "script_sha256": snapshot["script_sha256"],
        "protocol_sha256": snapshot["protocol_sha256"],
        "runtime_sha256": binaries, "draft_sha256": draft_hash, "candidate_draft_sha256": candidate_draft_hash,
    }
    with (HERE / f"configuration-{args.candidate}.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"candidate": args.candidate, "qualified": passed, "comparison": timing}), flush=True)


if __name__ == "__main__":
    main()
