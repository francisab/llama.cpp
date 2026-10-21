"""Matched A/B wrapper for llama-bench."""
import argparse, json, os, shlex, statistics, subprocess
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
with open(os.path.join(RESULTS, "model_paths.json"), encoding="utf-8") as file:
    P = json.load(file)
RUNS = {"q27": ["-m", P["d27"]],
        "qfn": ["-m", P["qfn"], "-lzm", "on"],
        "nem": ["-m", P["nem"]]}

def resolve_exe(value):
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"Executable is not a file: {path}")
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
    grouped = {}
    for row in rows:
        key = (int(row.get("n_prompt", 0) or 0), int(row.get("n_gen", 0) or 0),
               int(row.get("n_depth", 0) or 0))
        grouped.setdefault(key, []).extend(float(v) for v in row.get("samples_ts", []) or [])
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
        if not bv or not cv:
            raise ValueError(f"No samples for {test_label(key)}")
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["q27", "qfn", "nem"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cand", type=Path, required=True)
    parser.add_argument("--bench-args", default="-fa 1 -ngl 99 -ub 512 -p 512,2048 -n 128 -r 3")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    base, cand = resolve_exe(args.base), resolve_exe(args.cand)
    split_args = shlex.split(args.bench_args, posix=False)
    commands = [(side, i, [exe] + RUNS[args.model] + split_args + ["-o", "json"])
                for i in range(1, args.rounds + 1) for side, exe in (("base", base), ("cand", cand))]
    if args.dry_run:
        for _, _, command in commands:
            print(subprocess.list2cmdline(command), flush=True)
        return
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error(f"Refusing to overwrite output path: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    base_rows, cand_rows = [], []
    for side, i, command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        print(f"[{side} r{i}] rc={result.returncode}", flush=True)
        if result.returncode:
            tail = "\n".join(result.stderr.splitlines()[-15:])
            (args.out / f"{side}_r{i}.err.txt").write_text(tail + "\n", encoding="utf-8")
            raise RuntimeError(f"{side} round {i} failed with rc={result.returncode}")
        rows = json.loads(result.stdout)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"{side} round {i} JSON is not a list of objects")
        with (args.out / f"{side}_r{i}.json").open("w", encoding="utf-8", newline="\n") as file:
            json.dump(sanitize(rows), file, indent=2)
            file.write("\n")
        (base_rows if side == "base" else cand_rows).extend(rows)
    summary = summarize(base_rows, cand_rows, args.bench_args, args.rounds)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
if __name__ == "__main__":
    main()
