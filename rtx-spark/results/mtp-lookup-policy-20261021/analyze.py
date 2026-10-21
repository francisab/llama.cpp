"""Reconcile per-round traces and screen an optimistic lookup selector."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def draft(tokens, sampled, n=12, m=12):
    if len(tokens) <= n + m + 1:
        return []
    pattern = tokens[len(tokens) - n + 1:] + [sampled]
    for start in range(len(tokens) - n - 1, 0, -1):
        if tokens[start:start + n] != pattern:
            continue
        count = min(m, len(tokens) - start - n)
        return tokens[start + n:start + n + count] if count >= n else []
    return []


def accepted_prefix(proposal, future):
    accepted = 0
    for proposed, actual in zip(proposal, future):
        if proposed != actual:
            break
        accepted += 1
    return accepted


def parse_trace(text):
    tasks = {}
    pending = None
    last_time = -1
    for line in text.splitlines():
        if "spec_cost " not in line:
            continue
        match = re.search(r"\bspec_cost (start|select|rollback|end) (.*)", line)
        assert match, "Malformed diagnostic record"
        kind, fields = match.groups()
        event = dict(re.findall(r"(\w+)=([-\w]+)", fields))
        event = {key: value if key == "type" else int(value) for key, value in event.items()}
        assert event["slot"] == 0, "This capture supports one slot only"
        assert event["t_us"] > last_time, "Trace timestamps must increase"
        last_time = event["t_us"]
        if kind == "start":
            tasks.setdefault(event["task"], [])
            if pending is not None:
                assert pending["rollbacks"] and pending["start"]["task"] == event["task"]
                assert pending["start"]["gen"] == event["gen"]
                pending["replay_starts"] += 1
            elif event["n_max"] > 0:
                pending = {"start": event, "rollbacks": [], "replay_starts": 0}
        elif kind == "select":
            assert pending is not None and "select" not in pending
            pending["select"] = event
        elif kind == "rollback":
            assert pending is not None and "select" in pending
            assert pending["start"]["task"] == event["task"]
            pending["rollbacks"].append(event)
        else:
            assert pending is not None and "select" in pending
            start, selected = pending["start"], pending["select"]
            assert event["task"] == start["task"] and event["gen"] == start["gen"]
            assert len(pending["rollbacks"]) == pending["replay_starts"] <= 1
            if pending["rollbacks"]:
                rollback = pending["rollbacks"][0]
                assert rollback["accepted"] == event["accepted"]
                assert rollback["drafted"] == selected["drafted"]
                assert event["verified"] == event["accepted"] + 1
            else:
                assert event["verified"] == selected["drafted"]
            assert event["emitted"] == event["accepted"] + 1 + pending["replay_starts"]
            tasks[event["task"]].append({
                "position": start["gen"] - 1, "hist": selected["hist"], "source": selected["type"],
                "drafted": selected["drafted"], "accepted": event["accepted"], "emitted": event["emitted"],
                "rollback": bool(pending["rollbacks"]),
                "cost_us": event["t_us"] - start["t_us"],
                "proposal_us": selected["t_us"] - start["t_us"],
            })
            pending = None
    assert pending is None, "Incomplete final speculative round"
    return list(tasks.values())


def reconcile(rounds, response, prompt, hybrid):
    tokens = response["tokens"]
    assert len(tokens) == response["timings"]["predicted_n"] == 512
    assert len(prompt) == response["tokens_evaluated"]
    assert sum(row["drafted"] for row in rounds) == response["timings"]["draft_n"]
    assert sum(row["accepted"] for row in rounds) == response["timings"]["draft_n_accepted"]
    position = 0
    for row in rounds:
        assert row["position"] == position
        assert row["hist"] == len(prompt) + position
        assert row["source"] in ("draft-mtp", "ngram-simple")
        proposal = draft(prompt + tokens[:position], tokens[position]) if hybrid else []
        assert (row["source"] == "ngram-simple") == bool(proposal)
        if proposal:
            proposal = proposal[:min(12, 510 - position)]
            assert len(proposal) == row["drafted"]
            assert accepted_prefix(proposal, tokens[position + 1:]) == row["accepted"]
        else:
            assert row["drafted"] == row["accepted"] == min(3, 510 - position)
        position += row["emitted"]
    # A final non-speculative token can have n_max == 0 and no select/end record.
    assert position in (510, 511)
    return {"rounds": len(rounds), "rollback_rounds": sum(row["rollback"] for row in rounds),
            "untraced_terminal_tokens": 511 - position}


def oracle(prompt, tokens, mtp_cost_us, lookup_cost_us):
    """Use perfect acceptance knowledge, free decisions, and zero switching cost."""
    end = len(tokens) - 1
    costs = [0.0] * (end + 1)
    choices = [None] * end
    proposals = []
    for position in range(end - 1, -1, -1):
        step = min(4, end - position)
        costs[position] = mtp_cost_us + costs[position + step]
        choices[position] = ("draft-mtp", step)
        # Keep terminal shortened drafts out of full-length cost extrapolation.
        if position + 13 > end:
            continue
        proposal = draft(prompt + tokens[:position], tokens[position])
        if not proposal:
            continue
        accepted = accepted_prefix(proposal, tokens[position + 1:])
        # The deployed target retains three recurrent rollback states. A larger
        # rejection replays its accepted prefix and samples an additional token.
        step = accepted + 1 + (12 - accepted > 3)
        proposals.append({"position": position, "accepted": accepted, "committed": step})
        alternative = lookup_cost_us + costs[position + step]
        if alternative < costs[position]:
            costs[position] = alternative
            choices[position] = ("ngram-simple", step)
    position = selected = lookup_tokens = 0
    while position < end:
        source, step = choices[position]
        selected += source == "ngram-simple"
        lookup_tokens += step if source == "ngram-simple" else 0
        position += step
    baseline = math.ceil(end / 4) * mtp_cost_us
    return {
        "mtp_model_us": baseline, "oracle_model_us": costs[0],
        "model_gain_percent": 100 * (baseline / costs[0] - 1),
        "lookup_rounds_selected": selected, "lookup_committed_tokens": lookup_tokens,
        "eligible_positions": len(proposals),
        "acceptance_histogram": dict(sorted(Counter(row["accepted"] for row in proposals).items())),
    }


def cost_summary(rows):
    return {
        "rounds": len(rows), "drafted": sum(row["drafted"] for row in rows),
        "accepted": sum(row["accepted"] for row in rows),
        "committed": sum(row["emitted"] for row in rows),
        "rollbacks": sum(row["rollback"] for row in rows),
        "cost_ms_min": min(row["cost_us"] for row in rows) / 1000,
        "cost_ms_median": statistics.median(row["cost_us"] for row in rows) / 1000,
        "cost_ms_mean": statistics.mean(row["cost_us"] for row in rows) / 1000,
        "proposal_ms_median": statistics.median(row["proposal_us"] for row in rows) / 1000,
        "cost_ms_sum": sum(row["cost_us"] for row in rows) / 1000,
    }


def analyze(private):
    manifest = json.loads((private / "before.json").read_text())
    assert digest(HERE / "protocol.json") == manifest["protocol_sha256"]
    assert digest(private / "prompt-tokens.json") == manifest["prompt_tokens_sha256"]
    prompts = json.loads((private / "prompt-tokens.json").read_text())
    prior = private.parent / "copy-edit-ngram" / "mtp" / "base-1" / "server_U1_copy-edit.json"
    assert digest(prior) == manifest["reference_sha256"]["base-1"]
    old = json.loads(prior.read_text())
    published = json.loads((HERE.parent / "copy-edit-ngram-20261021" / "base-1.json").read_text())
    published_rows = {(row["case"], row["repeat"]): row for row in published["rows"]}
    for row in old["rows"]:
        public = published_rows[row["case"], row["repeat"]]
        assert token_digest(row["response"]["tokens"]) == public["tokens_sha256"]
        assert hashlib.sha256(row["response"]["content"].encode()).hexdigest() == public["content_sha256"]
        assert row["response"]["timings"] == public["timings"]
    expected = {row["case"]: row["response"] for row in old["rows"]}
    measured = defaultdict(lambda: defaultdict(list))
    request_summaries = []
    trace_rows = []
    for arm in ("mtp", "hybrid"):
        raw = private / arm / "server_U1_copy-edit.json"
        data = json.loads(raw.read_text())
        assert data["completed"] and data["exit_code"] == 0
        tasks = parse_trace(raw.with_suffix(".log").read_text(encoding="utf-8"))
        requests = data["warmups"] + data["rows"]
        assert len(tasks) == len(requests) == 6
        for index, (rounds, request) in enumerate(zip(tasks, requests)):
            case, response = request["case"], request["response"]
            assert response["tokens"] == expected[case]["tokens"]
            assert response["content"] == expected[case]["content"]
            assert response["prompt"] == expected[case]["prompt"]
            counts = reconcile(rounds, response, prompts[case], arm == "hybrid")
            trace_ms = sum(row["cost_us"] for row in rounds) / 1000
            request_summaries.append({
                "arm": arm, "case": case, "warmup": index < 2, "repeat": request.get("repeat"),
                **counts, "timings": response["timings"], "trace_ms": trace_ms,
                "trace_coverage_percent": 100 * trace_ms / response["timings"]["predicted_ms"],
                "tokens_sha256": token_digest(response["tokens"]),
                "content_sha256": response["content_sha256"],
            })
            if index >= 2:
                measured[case][arm].extend(rounds)
                trace_rows.append({"arm": arm, "case": case, "repeat": request["repeat"], "rounds": rounds})

    cases = {}
    for case, arms in measured.items():
        full_mtp = [row for row in arms["mtp"] if row["drafted"] == 3]
        full_lookup = [row for row in arms["hybrid"] if row["source"] == "ngram-simple" and row["drafted"] == 12]
        assert full_mtp and full_lookup
        mtp_cost = statistics.mean(row["cost_us"] for row in full_mtp)
        lookup_cost = min(row["cost_us"] for row in full_lookup)
        groups = defaultdict(list)
        previous = None
        for row in arms["hybrid"]:
            if row["position"] == 0:
                previous = None
            groups[f'{row["source"]}:accepted={row["accepted"]}:rollback={row["rollback"]}'].append(row)
            if row["source"] == "draft-mtp" and row["drafted"] == 3:
                groups[f"mtp3:previous={previous}"].append(row)
            previous = row["source"]
        cases[case] = {
            "mtp_full_rounds": cost_summary(full_mtp), "lookup_full_rounds": cost_summary(full_lookup),
            "groups": {key: cost_summary(rows) for key, rows in sorted(groups.items())},
            "oracle": oracle(prompts[case], expected[case]["tokens"], mtp_cost, lookup_cost),
            "oracle_costs_us": {"mtp_mean": mtp_cost, "lookup_minimum": lookup_cost},
        }
    passed = all(case["oracle"]["model_gain_percent"] >= 3 for case in cases.values())
    return {
        "completed": True, "request_summaries": request_summaries, "cases": cases,
        "oracle_gate_passed": passed,
        "decision": "Proceed to a causal policy screen" if passed else "Stop before an online policy; optimistic oracle misses the 3% gate on at least one frozen case",
        "measurement_kind": "Instrumented development-set diagnostic; oracle uses counterfactual cost estimates, not measured policy speed",
        "oracle_accounting_note": "Checkpoint replay adds a second bonus token when rejection exceeds three. This corrects the initial protocol's single-bonus shorthand; no threshold or fixture changed.",
        "deployment_changed": False,
    }, trace_rows


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; reconciliation assertions must remain enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    result, rows = analyze(args.private)
    for name, data in (("summary.json", result), ("rounds.json", rows)):
        with (HERE / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
    for case, data in result["cases"].items():
        print(case, json.dumps(data, indent=2))
    print(result["decision"])


if __name__ == "__main__":
    main()
