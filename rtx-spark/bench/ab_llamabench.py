"""Matched A/B wrapper for llama-bench, with an optional fail-fast speed screen."""
import argparse, json, math, os, shlex, statistics, subprocess, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
def resolve_file(value):
    # Split GGUFs need the symlink filename to locate the other shards.
    path = Path(value).absolute()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    return str(path)

def sanitize(value):
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if key == "model_filename" and isinstance(item, str):
                clean[key] = os.path.basename(item)
            elif isinstance(item, str):
                if "\\" not in item:
                    clean[key] = item
            else:
                clean[key] = sanitize(item)
        return clean
    if isinstance(value, list):
        return [sanitize(item) for item in value if not (isinstance(item, str) and "\\" in item)]
    return None if isinstance(value, str) and "\\" in value else value

def collect(rows):
    if not rows:
        raise ValueError("No benchmark rows")
    grouped = {}
    for row in rows:
        key = (int(row.get("n_prompt", 0) or 0), int(row.get("n_gen", 0) or 0),
               int(row.get("n_depth", 0) or 0))
        samples = [float(v) for v in row.get("samples_ts", []) or []]
        if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
            raise ValueError(f"Missing, non-finite or non-positive samples for {test_label(key)}")
        grouped.setdefault(key, []).extend(samples)
    return grouped

def test_label(key):
    prompt, generated, depth = key
    if generated == 0:
        label = f"pp{prompt}"
    elif prompt == 0:
        label = f"tg{generated}"
    else:
        label = f"pp{prompt}+tg{generated}"
    return label + (f"@d{depth}" if depth > 0 else "")

def summarize(base_rows, cand_rows, bench_args="", rounds=2):
    base, cand = collect(base_rows), collect(cand_rows)
    if set(base) != set(cand):
        raise ValueError(f"Test sets differ: base={sorted(base)}, cand={sorted(cand)}")
    tests = []
    for key in sorted(base):
        bv, cv = base[key], cand[key]
        if len(bv) != len(cv):
            raise ValueError(f"Sample counts differ for {test_label(key)}")
        bm, cm = statistics.median(bv), statistics.median(cv)
        if bm == 0:
            raise ValueError(f"Base median is zero for {test_label(key)}")
        ratio = cm / bm
        tests.append({
            "test": test_label(key), "base_median": bm, "cand_median": cm, "ratio": ratio,
            "base_range": [min(bv), max(bv)], "cand_range": [min(cv), max(cv)],
            "n_base": len(bv), "n_cand": len(cv),
        })
    summary = {
        "base_commit": next((str(r["build_commit"]) for r in base_rows if r.get("build_commit")), "unknown"),
        "cand_commit": next((str(r["build_commit"]) for r in cand_rows if r.get("build_commit")), "unknown"),
        "model_type": next((str(r["model_type"]) for r in base_rows + cand_rows if r.get("model_type")), "unknown"),
        "bench_args": bench_args, "rounds": rounds, "tests": tests,
    }
    print(f"base_commit: {summary['base_commit']}", flush=True)
    print(f"cand_commit: {summary['cand_commit']}", flush=True)
    print(f"{'test':14} {'base median':>14} {'cand median':>14} {'delta %':>10}", flush=True)
    for test in tests:
        print(f"{test['test']:14} {test['base_median']:14.2f} {test['cand_median']:14.2f} "
              f"{(test['ratio'] - 1.0) * 100:+9.2f}%", flush=True)
    return summary

def screen_decision(tests, goal, min_gain, max_regression):
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError("Minimum gain must be finite and non-negative")
    if not math.isfinite(max_regression) or not 0 <= max_regression < 100:
        raise ValueError("Maximum regression must be finite and in [0, 100)")
    targets = [test for test in tests if goal is None or test["test"].startswith(goal)]
    if not targets:
        raise ValueError(f"No tests match the screen goal: {goal}")
    if any(min(test["n_base"], test["n_cand"]) < 2 for test in tests):
        raise ValueError("A screen needs at least two samples per arm and test")
    for test in tests:
        if test["ratio"] < 1 - max_regression / 100:
            return {"decision": "reject", "reason": f"{test['test']} exceeds the regression limit"}
    for test in targets:
        if test["ratio"] < 1 + min_gain / 100:
            return {"decision": "park", "reason": f"{test['test']} is below the minimum gain"}
        # This is a cheap noise filter, not a confidence interval.
        if test["cand_range"][0] <= test["base_range"][1]:
            return {"decision": "park", "reason": f"{test['test']} sample ranges overlap"}
    return {"decision": "advance", "reason": "Speed screen only; correctness and quality are not established"}

def paired(round_rows):
    """Per-round candidate/base ratios; adjacent arms share drift, so this is steadier than pooled medians."""
    result = []
    keys = sorted(set.intersection(*(set(collect(b)) & set(collect(c)) for b, c in round_rows)))
    for key in keys:
        ratios = [statistics.median(collect(c)[key]) / statistics.median(collect(b)[key]) for b, c in round_rows]
        result.append({"test": test_label(key), "ratios": ratios,
                       "geomean": math.exp(statistics.mean(math.log(r) for r in ratios)),
                       "median": statistics.median(ratios), "wins": sum(r > 1 for r in ratios),
                       "rounds": len(ratios)})
    return result

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["q27", "qfn", "nem"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cand", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, help="Control GGUF instead of the local model map")
    parser.add_argument("--cand-model", type=Path, help="Candidate GGUF; defaults to the control model")
    parser.add_argument("--bench-args", help="Shared llama-bench options")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--screen", choices=["prefill", "decode"], help="Stop weak candidates after one pair; confirm survivors in reverse order")
    parser.add_argument("--min-gain", type=float, default=3.0, help="Screen minimum gain in percent (default: 3)")
    parser.add_argument("--max-regression", type=float, default=3.0, help="Screen regression limit in percent on any test (default: 3)")
    parser.add_argument("--budget", type=float, default=600, help="Total screen benchmark budget in seconds (default: 600)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.screen and args.rounds != 2:
        parser.error("--screen requires --rounds 2; it stops early when the first pair is weak")
    if not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("--budget must be finite and positive")
    if not math.isfinite(args.min_gain) or args.min_gain < 0:
        parser.error("--min-gain must be finite and non-negative")
    if not math.isfinite(args.max_regression) or not 0 <= args.max_regression < 100:
        parser.error("--max-regression must be finite and in [0, 100)")
    base, cand = resolve_file(args.base), resolve_file(args.cand)
    if args.base_model is None:
        with open(os.path.join(RESULTS, "model_paths.json"), encoding="utf-8") as file:
            paths = json.load(file)
        args.base_model = Path(paths[{"q27": "d27", "qfn": "qfn", "nem": "nem"}[args.model]])
    base_model = resolve_file(args.base_model)
    cand_model = resolve_file(args.cand_model) if args.cand_model is not None else base_model
    if args.screen and not os.path.samefile(base, cand) and not os.path.samefile(base_model, cand_model):
        parser.error("Screen one change at a time: change the binary OR the model, not both")
    if args.bench_args is None:
        args.bench_args = ("-fa 1 -ngl 99 -ub 512 -p 2048 -n 128 -r 2" if args.screen else
                           "-fa 1 -ngl 99 -ub 512 -p 512,2048 -n 128 -r 3")
    split_args = shlex.split(args.bench_args, posix=False)
    if any(option.split("=")[0] in ("-m", "--model", "-o", "--output") for option in split_args):
        parser.error("Use --base-model/--cand-model for GGUF paths; output is always JSON")
    arms = [("base", base, base_model), ("cand", cand, cand_model)]
    commands = [(side, i, [exe, "-m", model] + (["-lzm", "on"] if args.model == "qfn" else []) + split_args + ["-o", "json"])
                for i in range(1, args.rounds + 1) for side, exe, model in (arms if i % 2 else arms[::-1])]
    if args.dry_run:
        for _, _, command in commands:
            print(subprocess.list2cmdline(command), flush=True)
        return 0
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error(f"Refusing to overwrite output path: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = {"base": [], "cand": []}
    pair_rows = {}
    round_rows = []
    deadline = time.monotonic() + args.budget if args.screen else None
    outcome = {"decision": "running", "reason": "No complete pair"}
    if args.screen:
        save(args.out / "screen.json", outcome)
    for side, i, command in commands:
        try:
            timeout = 3600 if deadline is None else deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("Screen budget exhausted before the next benchmark")
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            print(f"[{side} r{i}] rc={result.returncode}", flush=True)
            if result.returncode:
                tail = "\n".join(result.stderr.splitlines()[-15:])
                (args.out / f"{side}_r{i}.err.txt").write_text(tail + "\n", encoding="utf-8")
                raise RuntimeError(f"{side} round {i} failed with rc={result.returncode}")
            rows = json.loads(result.stdout)
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError(f"{side} round {i} JSON is not a list of objects")
            collect(rows)
            save(args.out / f"{side}_r{i}.json", sanitize(rows))
            all_rows[side].extend(rows)
            pair_rows[side] = rows
            if len(pair_rows) == 2:
                round_rows.append((pair_rows["base"], pair_rows["cand"]))
                summary = summarize(all_rows["base"], all_rows["cand"], args.bench_args, i)
                summary["base_model_filename"] = Path(base_model).name
                summary["cand_model_filename"] = Path(cand_model).name
                summary["paired"] = paired(round_rows)
                for row in summary["paired"]:
                    print(f"paired {row['test']:10} geomean {(row['geomean'] - 1) * 100:+.2f}% "
                          f"median {(row['median'] - 1) * 100:+.2f}% wins {row['wins']}/{row['rounds']}", flush=True)
                save(args.out / "summary.json", summary)
                if args.screen:
                    pair = summarize(pair_rows["base"], pair_rows["cand"], args.bench_args, 1)
                    goal = "pp" if args.screen == "prefill" else "tg"
                    outcome = screen_decision(pair["tests"], goal, args.min_gain, args.max_regression)
                    if outcome["decision"] == "advance" and i < args.rounds:
                        outcome = {"decision": "confirm", "reason": "Repeat the pair in reverse order"}
                    outcome.update({"rounds_completed": i, "goal": args.screen, "min_gain_pct": args.min_gain,
                                    "max_regression_pct": args.max_regression, "quality_checked": False})
                    save(args.out / "screen.json", outcome)
                    print(f"screen: {outcome['decision']} - {outcome['reason']}", flush=True)
                    if outcome["decision"] != "confirm":
                        break
                pair_rows.clear()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            if args.screen:
                save(args.out / "screen.json", {"decision": "error", "side": side, "round": i,
                                                "reason": type(error).__name__, "quality_checked": False})
            raise
    return {"advance": 0, "park": 2, "reject": 3}[outcome["decision"]] if args.screen else 0
if __name__ == "__main__":
    raise SystemExit(main())
