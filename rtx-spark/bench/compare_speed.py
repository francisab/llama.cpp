"""Compare two run_rebaseline.py server results: per-case median decode speed, geomean ratio, draft acceptance."""
import argparse
import json
import math
import statistics
from pathlib import Path


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("cand")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    bd, b = load(a.base)
    cd, c = load(a.cand)
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
    print(f"geomean: {(geo - 1) * 100:+.2f}%  load_s {bd.get('load_s', 0):.1f} -> {cd.get('load_s', 0):.1f}  "
          f"gpu_ready_mib {bd.get('gpu_memory_ready_mib')} -> {cd.get('gpu_memory_ready_mib')}")
    if a.out:
        def side(d):
            # keep file names only, local paths stay private
            return {"name": d["name"], "args": [Path(x).name if ("\\" in x or "/" in x) else x for x in d["command"]],
                    "load_s": d.get("load_s"), "gpu_memory_ready_mib": d.get("gpu_memory_ready_mib")}
        a.out.write_text(json.dumps({"base": side(bd), "cand": side(cd), "cases": rows, "geomean_ratio": geo},
                                    indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
