"""Same-binary paired A/B of two llama-bench argument sets, alternating order every round."""
import argparse
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess


def run(exe, model, args):
    out = subprocess.run([exe, "-m", model, *shlex.split(args, posix=False), "-o", "json"], capture_output=True, text=True,
                         timeout=3600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    rows = json.loads(out.stdout)
    return {f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}@d{r.get('n_depth', 0)}": r["avg_ts"] for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-args", required=True)
    p.add_argument("--cand-args", required=True)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rounds = []
    for i in range(a.rounds):
        order = [("base", a.base_args), ("cand", a.cand_args)]
        res = {side: run(a.exe, a.model, args) for side, args in (order if i % 2 == 0 else order[::-1])}
        rounds.append(res)
        print(i + 1, json.dumps(res), flush=True)
    summary = {"base_args": a.base_args, "cand_args": a.cand_args, "model": Path(a.model).name, "rounds": rounds, "paired": {}}
    for test in rounds[0]["base"]:
        ratios = [r["cand"][test] / r["base"][test] for r in rounds]
        summary["paired"][test] = {"geomean": math.exp(statistics.mean(map(math.log, ratios))),
                                   "median": statistics.median(ratios), "wins": sum(x > 1 for x in ratios), "rounds": len(ratios)}
        s = summary["paired"][test]
        print(f"paired {test}: geomean {(s['geomean'] - 1) * 100:+.2f}% median {(s['median'] - 1) * 100:+.2f}% wins {s['wins']}/{s['rounds']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
