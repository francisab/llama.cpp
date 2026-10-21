import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ab_llamabench as bench
import compare_speed
import copy_edit_suite


def rows(speed):
    return [{"n_prompt": prompt, "n_gen": generated, "n_depth": 0,
             "samples_ts": [speed, speed + 1], "build_commit": "test", "model_type": "test"}
            for prompt, generated in [(2048, 0), (0, 128)]]


class ScreenDecisionTests(unittest.TestCase):
    def decide(self, base=100, cand=105):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = bench.summarize(rows(base), rows(cand))
        return bench.screen_decision(summary["tests"], "tg", 3, 3)["decision"]

    def test_clear_gain_advances(self):
        self.assertEqual(self.decide(), "advance")

    def test_small_gain_is_parked(self):
        self.assertEqual(self.decide(cand=102), "park")

    def test_regression_is_rejected(self):
        self.assertEqual(self.decide(cand=95), "reject")

    def test_other_metric_can_reject(self):
        tests = [{"test": name, "ratio": ratio, "base_range": [99, 101], "cand_range": [104, 106],
                  "n_base": 2, "n_cand": 2} for name, ratio in [("tg128", 1.05), ("pp2048", 0.94)]]
        self.assertEqual(bench.screen_decision(tests, "tg", 3, 3)["decision"], "reject")

    def test_overlapping_ranges_are_parked(self):
        tests = [{"test": "tg128", "ratio": 1.05, "base_range": [90, 110], "cand_range": [100, 110],
                  "n_base": 2, "n_cand": 2}]
        self.assertEqual(bench.screen_decision(tests, "tg", 3, 3)["decision"], "park")

    def test_empty_and_invalid_samples_fail(self):
        for samples in [[], [float("nan")], [float("inf")], [0], [-1]]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                bench.collect([{"samples_ts": samples}])
        with self.assertRaises(ValueError):
            bench.collect([])

    def test_mismatched_tests_or_sample_counts_fail(self):
        for candidate in [rows(105)[:1], [dict(row, samples_ts=[105]) for row in rows(105)]]:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                bench.summarize(rows(100), candidate)

    def test_one_sample_is_not_a_screen(self):
        data = [dict(row, samples_ts=[100]) for row in rows(100)]
        with contextlib.redirect_stdout(io.StringIO()):
            tests = bench.summarize(data, data)["tests"]
        with self.assertRaises(ValueError):
            bench.screen_decision(tests, "tg", 3, 3)

    def test_missing_goal_fails(self):
        with self.assertRaises(ValueError):
            bench.screen_decision([], "tg", 3, 3)


class BenchExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base.exe"
        self.cand = self.root / "cand.exe"
        self.model = self.root / "base model.gguf"
        self.cand_model = self.root / "candidate model.gguf"
        for path in (self.base, self.cand, self.model, self.cand_model):
            path.touch()
        self.out = self.root / "results"
        self.args = ["q27", "--base", str(self.base), "--cand", str(self.cand),
                     "--base-model", str(self.model), "--out", str(self.out)]

    def run_bench(self, args, effects):
        with patch.object(bench.subprocess, "run", side_effect=effects) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                code = bench.main(args)
        return code, run.call_args_list

    def result(self, speed):
        return subprocess.CompletedProcess([], 0, json.dumps(rows(speed)), "")

    def test_weak_candidate_stops_after_first_pair(self):
        code, calls = self.run_bench(self.args + ["--screen", "decode"], [self.result(100), self.result(102)])
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["rounds"], 1)
        outcome = json.loads((self.out / "screen.json").read_text())
        self.assertEqual(outcome["decision"], "park")
        self.assertFalse(outcome["quality_checked"])

    def test_winner_is_confirmed_in_reverse_order(self):
        code, calls = self.run_bench(self.args + ["--screen", "decode"],
                                    [self.result(100), self.result(105), self.result(106), self.result(100)])
        self.assertEqual(code, 0)
        self.assertEqual([call.args[0][0] for call in calls],
                         [str(self.base), str(self.cand), str(self.cand), str(self.base)])
        self.assertTrue(all(0 < call.kwargs["timeout"] <= 600 for call in calls))
        self.assertEqual(json.loads((self.out / "screen.json").read_text())["decision"], "advance")

    def test_second_pair_can_park_first_pair_winner(self):
        code, _ = self.run_bench(self.args + ["--screen", "decode"],
                                 [self.result(100), self.result(105), self.result(101), self.result(100)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads((self.out / "screen.json").read_text())["rounds_completed"], 2)

    def test_model_only_change_uses_same_binary(self):
        self.args[self.args.index("--cand") + 1] = str(self.base)
        code, calls = self.run_bench(self.args + ["--screen", "decode", "--cand-model", str(self.cand_model)],
                                    [self.result(100), self.result(102)])
        self.assertEqual(code, 2)
        self.assertEqual(calls[0].args[0][2], str(self.model))
        self.assertEqual(calls[1].args[0][2], str(self.cand_model))
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["cand_model_filename"], self.cand_model.name)
        self.assertNotIn(str(self.root), (self.out / "summary.json").read_text())

    def test_gguf_shard_filename_is_not_resolved_to_cache_blob(self):
        shard = self.root / "model-00001-of-00003.gguf"
        shard.touch()
        with patch.object(Path, "resolve", return_value=self.root / "cache-blob") as resolve:
            self.assertEqual(bench.resolve_file(shard), str(shard))
            resolve.assert_not_called()

    def test_invalid_options_fail_before_execution(self):
        for options in [["--screen", "decode", "--cand-model", str(self.cand_model)],
                        ["--screen", "decode", "--rounds", "1"], ["--budget", "nan"],
                        ["--min-gain", "inf"], ["--max-regression", "100"],
                        ["--bench-args=-m hidden.gguf"]]:
            with self.subTest(options=options), patch.object(bench.subprocess, "run") as run:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    bench.main(self.args + options)
                run.assert_not_called()

    def test_timeout_crash_and_bad_json_record_errors(self):
        failures = [subprocess.TimeoutExpired("benchmark", 600),
                    subprocess.CompletedProcess([], 1, "", "test failure"),
                    subprocess.CompletedProcess([], 0, "invalid JSON", ""),
                    subprocess.CompletedProcess([], 0, "[]", "")]
        for i, failure in enumerate(failures):
            args = self.args.copy()
            output = self.root / f"failure{i}"
            args[args.index("--out") + 1] = str(output)
            with self.subTest(failure=failure), self.assertRaises((subprocess.TimeoutExpired, ValueError, RuntimeError)):
                self.run_bench(args + ["--screen", "decode"], [failure])
            self.assertEqual(json.loads((output / "screen.json").read_text())["decision"], "error")

    def test_total_budget_stops_before_next_arm(self):
        with patch.object(bench.time, "monotonic", side_effect=[0, 0, 601]):
            with self.assertRaises(TimeoutError):
                self.run_bench(self.args + ["--screen", "decode"], [self.result(100)])
        self.assertEqual(json.loads((self.out / "screen.json").read_text())["reason"], "TimeoutError")

    def test_regular_run_still_completes_all_rounds(self):
        code, calls = self.run_bench(self.args, [self.result(100)] * 4)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 4)
        self.assertFalse((self.out / "screen.json").exists())
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["rounds"], 2)

    def test_dry_run_leaves_no_output(self):
        code, calls = self.run_bench(self.args + ["--screen", "decode", "--dry-run"], [])
        self.assertEqual(code, 0)
        self.assertFalse(calls)
        self.assertFalse(self.out.exists())

    def test_existing_results_are_not_overwritten(self):
        self.out.mkdir()
        (self.out / "keep.txt").write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_bench(self.args, [])
        self.assertEqual((self.out / "keep.txt").read_text(), "keep")


class ServerScreenTests(unittest.TestCase):
    def setUp(self):
        self.base = {"completed": True, "prompts": {"code": {"sha256": "same", "characters": 1}},
                     "name": "test", "command": ["llama-server.exe"],
                     "allow_eos": False, "rows": [
                         {"case": "code", "repeat": i, "response": {"tokens": [1, 2],
                          "request": {"n_predict": 2},
                          "timings": {"predicted_n": 2, "predicted_per_second": 100, "prompt_per_second": 200}}}
                         for i in (1, 2)]}

    def test_valid_fixed_length_pair(self):
        compare_speed.check_screen_pair(self.base, copy.deepcopy(self.base))

    def test_server_screen_exit_codes_and_evidence(self):
        for speed, expected, decision in [(105, 0, "advance"), (101, 2, "park"), (95, 3, "reject")]:
            with self.subTest(speed=speed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate = copy.deepcopy(self.base)
                for row in candidate["rows"]:
                    row["response"]["timings"]["predicted_per_second"] = speed
                bench.save(root / "base.json", self.base)
                bench.save(root / "cand.json", candidate)
                with contextlib.redirect_stdout(io.StringIO()):
                    code = compare_speed.main([str(root / "base.json"), str(root / "cand.json"),
                                               "--screen", "--out", str(root / "comparison.json")])
                self.assertEqual(code, expected)
                result = json.loads((root / "comparison.json").read_text())
                self.assertEqual(result["screen"]["decision"], decision)
                self.assertFalse(result["screen"]["quality_checked"])
                self.assertEqual(result["identical_outputs"], [2, 2])

    def test_unmatched_or_invalid_runs_fail(self):
        candidates = []
        for change in [{"prompts": {}}, {"allow_eos": True}, {"rows": self.base["rows"][:1]},
                       {"rows": self.base["rows"] * 2}]:
            candidates.append({**copy.deepcopy(self.base), **change})
        for field, value in [("tokens", []), ("request", {"n_predict": 3}),
                             ("timings", {"predicted_n": 2, "predicted_per_second": float("nan")})]:
            candidate = copy.deepcopy(self.base)
            candidate["rows"][0]["response"][field] = value
            candidates.append(candidate)
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare_speed.check_screen_pair(self.base, candidate)


class CopyEditFixtureTests(unittest.TestCase):
    def response(self, text, stop="eos"):
        return {"content": text, "stop_type": stop, "truncated": False, "request": {"prompt": "fixture"}}

    def test_fixed_prefix_and_full_edit_limits(self):
        for count in (512, 2048):
            cases = copy_edit_suite.cases(count)
            self.assertEqual(tuple(case[0] for case in cases), copy_edit_suite.CASE_NAMES)
            self.assertTrue(all(case[2] == count for case in cases))
            self.assertTrue(all(case[1]["messages"][0]["role"] == "user" for case in cases))

    def test_exact_edits_pass_and_unchanged_inputs_fail(self):
        for case, expected, unchanged in (
            (copy_edit_suite.CASE_NAMES[0], copy_edit_suite.CODE_EXPECTED, copy_edit_suite.CODE_SOURCE),
            (copy_edit_suite.CASE_NAMES[1], json.dumps(copy_edit_suite.JSON_EXPECTED), json.dumps(copy_edit_suite.JSON_SOURCE)),
        ):
            with self.subTest(case=case):
                self.assertTrue(copy_edit_suite.score(case, self.response(expected))["passed"])
                self.assertFalse(copy_edit_suite.score(case, self.response(unchanged))["passed"])
                self.assertFalse(copy_edit_suite.score(case, self.response(expected, "limit"))["passed"])

    def test_extra_python_changes_fail_without_execution(self):
        case = copy_edit_suite.CASE_NAMES[0]
        for text in (copy_edit_suite.CODE_EXPECTED.replace(", 100)", ", 101)"),
                     copy_edit_suite.CODE_EXPECTED + "\nimport os\n"):
            with self.subTest(text=text):
                self.assertFalse(copy_edit_suite.score(case, self.response(text))["passed"])

    def test_duplicate_or_missing_json_keys_fail(self):
        case = copy_edit_suite.CASE_NAMES[1]
        duplicate = json.dumps(copy_edit_suite.JSON_EXPECTED).replace(
            '{"schema_version": 3,', '{"schema_version": 3, "schema_version": 3,')
        for text in (duplicate, '{"schema_version":3}'):
            with self.subTest(text=text):
                self.assertFalse(copy_edit_suite.score(case, self.response(text))["passed"])


if __name__ == "__main__":
    unittest.main()
