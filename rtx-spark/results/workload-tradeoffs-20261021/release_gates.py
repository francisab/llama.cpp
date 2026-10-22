"""Release gates for opt-in profiles that already passed their speed gate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "rtx-spark" / "bench"))
sys.path.insert(0, str(HERE))
import en_us_quality
import prose_profile
from config_profiles import compare
from precision import runtime_manifest, sha, save

ORDER = [7, 4, 4, 7]
REPETITIONS = 3
SEED = 1234
OUTSIDE_CASES = ["code256", "novel2K"]
LAUNCHER_SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                     "presence_penalty": 0.0, "repeat_penalty": 1.0}
OUTPUT = {"27b-dflash4": "27b-release-gates.json", "nemotron-nospec": "nemotron-release-gates.json"}
REVISED_OUTPUT = {"27b-dflash4": "27b-release-gates-non-inferiority.json",
                  "nemotron-nospec": "nemotron-release-gates-non-inferiority.json"}
REVISION = {
    "rule": "non-inferiority",
    "post_hoc": True,
    "reason": "The control arm fails the locale_json spelling check with identical output, so the strict rule "
              "cannot separate the candidate from the control.",
    "approval": "Approved by the campaign owner after the strict results were recorded and before any rescoring "
                "or further run.",
    "quality": "The candidate passes every English/code quality check that the control passes. The control's own "
               "failures, output identity and the strict result are reported.",
}
CRITERIA = {
    "27b-dflash4": {
        "quality": "The depth 4 arm passes every English/code quality check. The depth 7 control result and "
                   "output identity are reported.",
        "launcher_sampling": "Requests leave sampling to the launcher's server-side defaults (temperature 1.0, "
                             "top-p 0.95, top-k 20, min-p 0, no penalties) and use seed 1234. The score is seconds "
                             "per generated token over both natural-EOS prose tasks. Depth 4 must pass the frozen "
                             "configuration-profile gate (minimum reduction in both launch orders, separated "
                             "ranges, gain above control reload drift) and every response must stop naturally.",
        "outside_profile": "Code and short-generation results are disclosed and do not veto an opt-in profile.",
        "integrity": "Installed runtime and launcher catalog hashes are unchanged.",
        "qualified": "quality and launcher_sampling pass and integrity holds.",
    },
    "nemotron-nospec": {
        "quality": "The no-speculation arm passes every English/code quality check. The MTP3 control result and "
                   "output identity are reported.",
        "integrity": "Installed runtime hashes are unchanged.",
        "qualified": "quality passes and integrity holds.",
    },
}


def sampling_matches(settings):
    return all(abs(settings[key] - value) < 1e-6 for key, value in LAUNCHER_SAMPLING.items())


def identical_fraction(left, right):
    return sum(a == b for a, b in zip(left, right)) / len(left)


def seconds_per_token(responses):
    return sum(item["wall_s"] for item in responses) / sum(item["timings"]["predicted_n"] for item in responses)


def reduction(base, candidate):
    return 100 * (1 - candidate / base)


def require_complete(data):
    if not data["completed"] or data["exit_code"] != 0 or data["stop"] != "ctrl+c":
        raise ValueError("Incomplete server run")


def make_sampled_completion(port):
    """Like run_rebaseline.completion, but sampling stays at the server's launcher defaults."""
    def sampled_completion(prompt, count, allow_eos=False):
        body = {"prompt": prompt, "n_predict": count, "seed": SEED, "cache_prompt": False,
                "stream": False, "ignore_eos": not allow_eos, "return_tokens": True}
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
        result["wall_s"] = time.perf_counter() - start
        result["request"] = body
        result["content_sha256"] = hashlib.sha256(result["content"].encode()).hexdigest()
        settings = result["generation_settings"]
        if not sampling_matches(settings) or settings["seed"] != SEED:
            raise ValueError("Server did not apply the launcher sampling defaults")
        return result
    return sampled_completion


def launcher_configs(runner, catalog):
    prior = HERE.parent / "27b-profiles-20261021"
    spec = importlib.util.spec_from_file_location("launcher_profile", prior / "validate.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    data = json.loads(catalog.read_text(encoding="utf-8-sig"))
    model = next(item for item in data["models"] if item["id"] == "qwen/qwen3.8-27b")
    recipe_id = json.loads((prior / "protocol.json").read_text())["control_recipe"]
    recipe = next(item for item in model["recipes"] if item["id"] == recipe_id)
    if recipe["serve"] != json.loads((prior / "launcher-source.json").read_text())["serve"]:
        raise ValueError("Actual 27B launcher command changed")
    return [(f"P{depth} q27 launcher dflash{depth}",
             launcher.resolve_command(recipe, runner.matrix.P, runner.matrix.PORT, depth), "decode")
            for depth in (7, 4)], recipe_id


def quality_stage(runner, work, control, candidate):
    reports, hashes = {}, {}
    for tag in (control, candidate):
        out = work / f"quality-{tag}"
        out.mkdir()
        runner.run_server(tag, out, 1, "en-us-quality", allow_eos=True, extra_args=["--reasoning", "off"])
        path = out / f"server_{tag}_en-us-quality.json"
        data = json.loads(path.read_text())
        require_complete(data)
        report = en_us_quality.analyze(path)
        report["path"] = path.name
        reports[tag] = report
        hashes[tag] = {row["case"]: row["response"]["content_sha256"] for row in data["rows"]}
    return {"control": control, "candidate": candidate, "reports": reports,
            "control_passed": reports[control]["passed"], "candidate_passed": reports[candidate]["passed"],
            "passed": reports[candidate]["passed"],
            "identical_outputs": hashes[control] == hashes[candidate], "content_sha256": hashes}


def check_results(report):
    return {(row["phase"], row["case"]): row["passed"] for row in report["scores"]}


def non_inferiority(quality):
    control = check_results(quality["reports"][quality["control"]])
    candidate = check_results(quality["reports"][quality["candidate"]])
    if not control or control.keys() != candidate.keys():
        raise ValueError("Control and candidate quality checks differ")

    def names(keys):
        return sorted(f"{phase}/{case}" for phase, case in keys)

    lost = [key for key, passed in control.items() if passed and not candidate[key]]
    return {"control_failed_checks": names(key for key, passed in control.items() if not passed),
            "candidate_failed_checks": names(key for key, passed in candidate.items() if not passed),
            "lost_control_passes": names(lost), "passed": not lost}


def revised_quality(strict_file):
    strict = json.loads(strict_file.read_text())
    quality = strict["quality"]
    return {"rule": REVISION["rule"], "strict_result": strict_file.name, "strict_result_sha256": sha(strict_file),
            "strict_passed": quality["passed"], "control": quality["control"], "candidate": quality["candidate"],
            "control_passed": quality["control_passed"], "candidate_passed": quality["candidate_passed"],
            "identical_outputs": quality["identical_outputs"], **non_inferiority(quality)}


def case_stats(data, case):
    rows = [row["response"] for row in data["rows"] if row["case"] == case]
    if len(rows) != REPETITIONS:
        raise ValueError(f"Wrong response count for {case}")
    return {"wall_seconds": [item["wall_s"] for item in rows],
            "median_wall_seconds": statistics.median(item["wall_s"] for item in rows),
            "tokens_per_second": [item["timings"]["predicted_per_second"] for item in rows],
            "draft_tokens": [item["timings"].get("draft_n", 0) for item in rows],
            "accepted_tokens": [item["timings"].get("draft_n_accepted", 0) for item in rows],
            "content_sha256": [item["content_sha256"] for item in rows]}


def outside_stage(runner, work):
    arms = []
    for index, depth in enumerate(ORDER):
        out = work / f"outside-{index}-depth{depth}"
        out.mkdir()
        runner.run_server(f"P{depth}", out, REPETITIONS, "decode", capture_metrics=True,
                          case_names=OUTSIDE_CASES, extra_args=["--reasoning", "off"])
        data = json.loads((out / f"server_P{depth}_decode.json").read_text())
        require_complete(data)
        arms.append({"depth": depth, "ready_memory_mib": data["gpu_memory_ready_mib"],
                     "cases": {case: case_stats(data, case) for case in OUTSIDE_CASES}})
    pairs = []
    for base, candidate in ((0, 1), (3, 2)):
        pairs.append({case: {
            "reduction_percent": reduction(arms[base]["cases"][case]["median_wall_seconds"],
                                           arms[candidate]["cases"][case]["median_wall_seconds"]),
            "separated": max(arms[candidate]["cases"][case]["wall_seconds"]) < min(arms[base]["cases"][case]["wall_seconds"]),
        } for case in OUTSIDE_CASES})
    return {"arms": arms, "pairs": pairs}


def literal_check(case, response):
    try:
        return prose_profile.check_literals(case, response)
    except ValueError as error:
        return {"passed": False, "error": str(error), "content_sha256": response["content_sha256"]}


def sampling_stage(runner, work, gate):
    names = list(prose_profile.FACTS)
    runs = []
    for index, depth in enumerate(ORDER):
        out = work / f"sampling-{index}-depth{depth}"
        out.mkdir()
        runner.run_server(f"P{depth}", out, REPETITIONS, "decode", allow_eos=True, capture_metrics=True,
                          extra_args=["--reasoning", "off"])
        data = json.loads((out / f"server_P{depth}_decode.json").read_text())
        require_complete(data)
        responses = {(row["case"], row["repeat"]): row["response"] for row in data["rows"]}
        if not data["allow_eos"] or len(data["rows"]) != REPETITIONS * len(names) or len(responses) != len(data["rows"]):
            raise ValueError("Incomplete sampled prose run")
        scores = [seconds_per_token([responses[(case, repeat)] for case in names])
                  for repeat in range(1, REPETITIONS + 1)]
        summary = [{
            "case": case, "repeat": repeat, "wall_s": response["wall_s"],
            "generated_tokens": response["timings"]["predicted_n"], "stop_type": response["stop_type"],
            "draft_tokens": response["timings"].get("draft_n", 0),
            "accepted_tokens": response["timings"].get("draft_n_accepted", 0),
            "content_sha256": response["content_sha256"], "literals": literal_check(case, response),
        } for (case, repeat), response in sorted(responses.items())]
        drafted = sum(item["draft_tokens"] for item in summary)
        runs.append({"depth": depth, "total_request_seconds": scores, "median_total_seconds": statistics.median(scores),
                     "ready_memory_mib": data["gpu_memory_ready_mib"],
                     "acceptance": sum(item["accepted_tokens"] for item in summary) / drafted if drafted else None,
                     "responses": summary})
    comparison = compare(runs, gate)
    natural = all(item["stop_type"] == "eos" for run in runs for item in run["responses"])
    texts = [[item["content_sha256"] for item in run["responses"]] for run in runs]
    return {"metric": "seconds per generated token over both prose tasks, prompt processing included",
            "seed": SEED, "runs": runs, "comparison": comparison, "all_natural_eos": natural,
            "identical_text_fraction": {
                "control_reloads": identical_fraction(texts[0], texts[3]),
                "candidate_reloads": identical_fraction(texts[1], texts[2]),
                "between_depths": [identical_fraction(texts[0], texts[1]), identical_fraction(texts[3], texts[2])]},
            "passed": comparison["passed"] and natural}


def gates_27b(runner, args, quality=None):
    configs, recipe = launcher_configs(runner, args.catalog)
    runner.matrix.CONFIGS = configs
    gate = json.loads((HERE / "protocol.json").read_text())["configuration_profiles"]["minimum_request_latency_reduction_percent"]
    if quality is None:
        quality = quality_stage(runner, args.work, "P7", "P4")
    save(args.work / "stage-quality.json", quality)
    result = {"recipe": recipe, "quality": quality}
    if result["quality"]["passed"]:
        original_suite, original_completion = runner.matrix.SUITES["decode"], runner.completion
        runner.matrix.SUITES["decode"] = prose_profile.cases()
        runner.completion = make_sampled_completion(runner.matrix.PORT)
        try:
            result["launcher_sampling"] = sampling_stage(runner, args.work, gate)
        finally:
            runner.completion = original_completion
            runner.matrix.SUITES["decode"] = original_suite
        save(args.work / "stage-launcher-sampling.json", result["launcher_sampling"])
        if result["launcher_sampling"]["passed"]:
            result["outside_profile"] = outside_stage(runner, args.work)
            save(args.work / "stage-outside-profile.json", result["outside_profile"])
    result["qualified"] = result["quality"]["passed"] and result.get("launcher_sampling", {}).get("passed", False)
    return result


def gates_nemotron(runner, args, quality=None):
    if quality is None:
        quality = quality_stage(runner, args.work, "N2", "N1")
    save(args.work / "stage-quality.json", quality)
    return {"quality": quality, "qualified": quality["passed"]}


def main():
    import run_rebaseline as runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=sorted(OUTPUT))
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--revised", action="store_true",
                        help="Rescore the saved strict quality stage with the non-inferiority rule, "
                             "then run only the remaining stages")
    args = parser.parse_args()
    if (args.candidate == "27b-dflash4") != (args.catalog is not None):
        parser.error("Only 27b-dflash4 takes --catalog")
    strict_file = HERE / OUTPUT[args.candidate]
    output = HERE / (REVISED_OUTPUT if args.revised else OUTPUT)[args.candidate]
    if output.exists() or (args.revised and not strict_file.exists()):
        parser.error("The result file already exists, or the strict result to revise is missing")
    args.work.mkdir(parents=True, exist_ok=False)
    baseline = runtime_manifest(Path(runner.matrix.EXE).parent)
    catalog_hash = sha(args.catalog) if args.catalog else None
    protocol = {"script_sha256": sha(Path(__file__)), "parent_protocol_sha256": sha(HERE / "protocol.json"),
                "candidate": args.candidate, "catalog_sha256": catalog_hash, "criteria": CRITERIA[args.candidate],
                "quality_order": ["P7", "P4"] if args.candidate == "27b-dflash4" else ["N2", "N1"]}
    if args.candidate == "27b-dflash4":
        protocol.update({"order": ORDER, "repetitions": REPETITIONS, "outside_cases": OUTSIDE_CASES,
                         "launcher_sampling": LAUNCHER_SAMPLING, "seed": SEED})
    quality = None
    if args.revised:
        strict = json.loads(strict_file.read_text())
        if strict["runtime_sha256"] != baseline or strict["protocol"]["catalog_sha256"] != catalog_hash:
            raise ValueError("Installed runtime or launcher differs from the strict-gate evidence")
        quality = revised_quality(strict_file)
        protocol.update({"revision": REVISION, "revised_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "criteria": {**CRITERIA[args.candidate], "quality": REVISION["quality"]}})
    save(args.work / "protocol.json", protocol)
    run = gates_27b if args.candidate == "27b-dflash4" else gates_nemotron
    result = run(runner, args, quality)
    if runtime_manifest(Path(runner.matrix.EXE).parent) != baseline or (args.catalog and sha(args.catalog) != catalog_hash):
        raise ValueError("Runtime or launcher changed during validation")
    result.update({"candidate": args.candidate, "deployment_changed": False, "protocol": protocol,
                   "runtime_sha256": baseline})
    save(output, result)
    print(json.dumps({"candidate": args.candidate, "qualified": result["qualified"],
                      "quality_passed": result["quality"]["passed"],
                      "identical_outputs": result["quality"]["identical_outputs"],
                      "launcher_sampling": result.get("launcher_sampling", {}).get("comparison")}), flush=True)


if __name__ == "__main__":
    main()
