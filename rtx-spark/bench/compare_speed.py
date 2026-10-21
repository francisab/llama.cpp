"""Compare two run_rebaseline.py server results: per-case median decode speed, geomean ratio, draft acceptance."""
import argparse
import json
import math
import statistics
from pathlib import Path

from ab_llamabench import screen_decision


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("completed"):
        raise ValueError(f"{path} is not a completed run")
    cases = {}
    for row in data["rows"]:
        t = row["response"]["timings"]
        c = cases.setdefault(row["case"], {"tg": [], "pp": [], "draft": 0, "accepted": 0, "n": 0})
        c["tg"].append(t["predicted_per_second"])
        c["pp"].append(t["prompt_per_second"])
        c["draft"] += t.get("draft_n", 0)
        c["accepted"] += t.get("draft_n_accepted", 0)
        c["n"] += t["predicted_n"]
    return data, cases


def check_screen_pair(base, cand):
    if base.get("allow_eos") or cand.get("allow_eos"):
        raise ValueError("Speed screens require fixed output lengths, not natural-EOS runs")
    if not base.get("prompts") or base["prompts"] != cand.get("prompts"):
        raise ValueError("Screen prompt hashes differ or are missing")
    samples = []
    for data in (base, cand):
        keys = {}
        for row in data["rows"]:
            key = (row["case"], row["repeat"])
            timing = row["response"]["timings"]
            count = timing["predicted_n"]
            speed = timing["predicted_per_second"]
            if key in keys or count <= 0 or len(row["response"]["tokens"]) != count:
                raise ValueError(f"Duplicate or incomplete screen sample: {key}")
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError(f"Invalid screen speed: {key}")
            if row["response"]["request"]["n_predict"] != count:
                raise ValueError(f"Screen output hit a different token count: {key}")
            keys[key] = count
        samples.append(keys)
    if not samples[0] or samples[0] != samples[1]:
        raise ValueError("Screen case/repeat sets or output lengths differ")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("cand")
    p.add_argument("--out", type=Path)
    p.add_argument("--screen", action="store_true", help="Triage a short fixed-length pair; not a quality gate")
    p.add_argument("--min-gain", type=float, default=3.0)
    p.add_argument("--max-regression", type=float, default=3.0)
    a = p.parse_args(argv)
    bd, b = load(a.base)
    cd, c = load(a.cand)
    if a.screen:
        check_screen_pair(bd, cd)
    if set(b) != set(c):
        raise ValueError(f"case mismatch: {sorted(b)} vs {sorted(c)}")
    rows, ratios = [], []
    for case in sorted(b):
        mb, mc = statistics.median(b[case]["tg"]), statistics.median(c[case]["tg"])
        ratios.append(mc / mb)
        rows.append({
            "case": case,
            "base_tg_median": mb, "cand_tg_median": mc, "ratio": mc / mb,
            "base_tg_range": [min(b[case]["tg"]), max(b[case]["tg"])],
            "cand_tg_range": [min(c[case]["tg"]), max(c[case]["tg"])],
            "base_accept": b[case]["accepted"] / max(1, b[case]["draft"]),
            "cand_accept": c[case]["accepted"] / max(1, c[case]["draft"]),
        })
        print(f"{case:>10}: {mb:7.2f} -> {mc:7.2f} tok/s ({(mc / mb - 1) * 100:+.2f}%), "
              f"accept {rows[-1]['base_accept']:.3f} -> {rows[-1]['cand_accept']:.3f}")
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    bt = {(r["case"], r["repeat"]): r["response"]["tokens"] for r in bd["rows"]}
    same = sum(1 for r in cd["rows"] if bt.get((r["case"], r["repeat"])) == r["response"]["tokens"])
    print(f"geomean: {(geo - 1) * 100:+.2f}%  load_s {bd.get('load_s', 0):.1f} -> {cd.get('load_s', 0):.1f}  "
          f"gpu_ready_mib {bd.get('gpu_memory_ready_mib')} -> {cd.get('gpu_memory_ready_mib')}  "
          f"identical outputs {same}/{len(cd['rows'])}")
    outcome = None
    if a.screen:
        tests = [{"test": row["case"], "ratio": row["ratio"],
                  "base_range": row["base_tg_range"], "cand_range": row["cand_tg_range"],
                  "n_base": len(b[row["case"]]["tg"]), "n_cand": len(c[row["case"]]["tg"])} for row in rows]
        outcome = screen_decision(tests, None, a.min_gain, a.max_regression)
        outcome["quality_checked"] = False
        outcome["min_gain_pct"] = a.min_gain
        outcome["max_regression_pct"] = a.max_regression
        print(f"screen: {outcome['decision']} - {outcome['reason']}")
    if a.out:
        def side(d):
            # keep file names only, local paths stay private
            return {"name": d["name"], "args": [Path(x).name if ("\\" in x or "/" in x) else x for x in d["command"]],
                    "load_s": d.get("load_s"), "gpu_memory_ready_mib": d.get("gpu_memory_ready_mib")}
        summary = {"base": side(bd), "cand": side(cd), "cases": rows, "geomean_ratio": geo,
                   "identical_outputs": [same, len(cd["rows"])]}
        if outcome is not None:
            summary["screen"] = outcome
        a.out.write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    return {"advance": 0, "park": 2, "reject": 3}[outcome["decision"]] if outcome is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
