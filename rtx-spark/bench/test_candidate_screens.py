import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_module(name, folder, filename):
    spec = importlib.util.spec_from_file_location(name, RESULTS / folder / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mmq = load_module("mmq_copy_screen", "mmq-copy-20261021", "screen.py")
nvfp4 = load_module("nvfp4_screen", "27b-nvfp4-20261021", "screen.py")
profiles = load_module("profile_screen", "27b-profiles-20261021", "validate.py")
ssm = load_module("ssm_capture", "ssm-ssd-dispatch-20261021", "capture.py")
nemotron = load_module("nemotron_nvfp4_screen", "nemotron-nvfp4-20261021", "screen.py")
precision = load_module("tradeoff_precision", "workload-tradeoffs-20261021", "precision.py")
tradeoffs = load_module("tradeoff_profiles", "workload-tradeoffs-20261021", "config_profiles.py")
long_profiles = load_module("long_tradeoff_profiles", "workload-tradeoffs-20261021", "long_profiles.py")
lookup_profile = load_module("lookup_tradeoff_profile", "workload-tradeoffs-20261021", "lookup_profile.py")
prose_profile = load_module("prose_tradeoff_profile", "workload-tradeoffs-20261021", "prose_profile.py")


class CompleteProseTests(unittest.TestCase):
    def test_literal_checks_require_the_exact_identifiers(self):
        for case, (facts, _) in prose_profile.FACTS.items():
            response = {"content": facts + "\n\n" + facts, "content_sha256": "test",
                        "stop_type": "eos", "request": {"prompt": "explain"}}
            with self.subTest(case=case):
                self.assertTrue(prose_profile.check_literals(case, response)["passed"])
                response["content"] = response["content"].replace(prose_profile.FACTS[case][1][0], "wrong-id")
                self.assertFalse(prose_profile.check_literals(case, response)["passed"])


class FullEditTests(unittest.TestCase):
    def test_every_worker_and_type_is_checked(self):
        for case, fixture in lookup_profile.fixtures().items():
            expected = fixture["expected"]
            response = {"content": json.dumps(expected), "stop_type": "eos", "request": {"prompt": "edit"}}
            with self.subTest(case=case):
                self.assertTrue(lookup_profile.score(response, expected))
                changed = copy.deepcopy(expected)
                changed["workers"][-1]["enabled"] = int(changed["workers"][-1]["enabled"])
                response["content"] = json.dumps(changed)
                self.assertFalse(lookup_profile.score(response, expected))

    def test_prefix_is_not_a_complete_edit(self):
        fixture = next(iter(lookup_profile.fixtures().values()))
        response = {"content": json.dumps(fixture["expected"]), "stop_type": "limit", "request": {"prompt": "edit"}}
        with self.assertRaises(ValueError):
            lookup_profile.score(response, fixture["expected"])


class StreamTimingTests(unittest.TestCase):
    def test_cached_guard_is_primed_in_every_rotated_order(self):
        cases = long_profiles.cases()
        for offset in range(3):
            rows = cases[offset:] + cases[:offset]
            index = next(index for index, row in enumerate(rows) if row[0] == "cached_prime")
            self.assertEqual(rows[index + 1][0], "cached_guard")
            self.assertEqual(rows[index][1], rows[index + 1][1])

    def event(self, content, tokens, stop=False):
        return ("data: " + json.dumps({"content": content, "tokens": tokens, "stop": stop}) + "\n").encode()

    def test_first_content_is_observed_not_inferred_from_prompt_time(self):
        events = [self.event("", []), self.event("hello", [11]), self.event(" world", [12]), self.event("", [], True)]
        times = iter([12.5, 14.0])
        result = long_profiles.read_stream(events, 10, lambda: next(times))
        self.assertEqual(result["ttft_s"], 2.5)
        self.assertEqual(result["wall_s"], 4.0)
        self.assertEqual(result["content"], "hello world")
        self.assertEqual(result["tokens"], [11, 12])

    def test_empty_incomplete_or_duplicate_terminal_stream_fails(self):
        for events in (
            [self.event("", [], True)],
            [self.event("x", [1])],
            [self.event("x", [1]), self.event("", [], True), self.event("", [], True)],
            [self.event("x", [1]), self.event("unexpected", [1], True)],
        ):
            with self.subTest(events=events), self.assertRaises(ValueError):
                long_profiles.read_stream(events, 1, lambda: 2)


class TradeoffTaskTests(unittest.TestCase):
    def test_sql_is_scored_against_independent_tables(self):
        query = ("SELECT customer, SUM(amount_cents) AS total_cents FROM orders WHERE paid=1 "
                 "GROUP BY customer HAVING SUM(amount_cents)>=1000 ORDER BY total_cents DESC, customer ASC")
        self.assertTrue(tradeoffs.score_sql(query)["passed"])
        with self.assertRaises(ValueError):
            tradeoffs.score_sql(query.replace("paid=1", "paid=0"))

    def test_sql_cannot_modify_or_access_external_data(self):
        import sqlite3
        for query in ("DROP TABLE orders", "ATTACH DATABASE 'x' AS x",
                      "SELECT load_extension('x')", "SELECT zeroblob(1000000000)"):
            with self.subTest(query=query), self.assertRaises((ValueError, sqlite3.Error)):
                tradeoffs.score_sql(query)

    def test_latency_gate_requires_both_orders_and_beats_drift(self):
        def row(values):
            return {"total_request_seconds": values, "median_total_seconds": sorted(values)[1]}
        runs = [row([10, 10.1, 10.2]), row([9, 9.1, 9.2]), row([9, 9.1, 9.2]), row([10, 10.1, 10.2])]
        self.assertTrue(tradeoffs.compare(runs, 1)["passed"])
        runs[2] = row([10, 10.1, 10.2])
        self.assertFalse(tradeoffs.compare(runs, 1)["passed"])
        runs[2] = row([9, 9.1, 9.2])
        runs[3] = row([13, 13.1, 13.2])
        self.assertFalse(tradeoffs.compare(runs, 1)["passed"])


class TradeoffPrecisionTests(unittest.TestCase):
    def log(self):
        rows = "\n".join(f"{i} 2.5 \u00b1 0.1" for i in range(1, 17))
        return rows + "\nMean PPL(Q) : 2.5\nMean PPL(Q)/PPL(base) : 1.002\nMean KLD: 0.007\nSame top p: 98.5\n"

    def test_complete_precision_metrics(self):
        self.assertEqual(precision.parse_metrics(self.log()),
                         {"ppl": 2.5, "ppl_ratio": 1.002, "mean_kl_nats": 0.007, "same_top_percent": 98.5})

    def test_missing_duplicate_or_nonfinite_scores_fail(self):
        text = self.log()
        for invalid in (
            text.replace("16 2.5", "17 2.5"),
            text + "Mean KLD: 0.008\n",
            text.replace("0.007", "nan"),
            text.replace("0.007", "1e309"),
            text.replace("98.5", "101"),
            text.replace("1.002", "0"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                precision.parse_metrics(invalid)

    def test_split_names_preserve_source(self):
        self.assertEqual(precision.shards(Path("model.gguf")), [Path("model.gguf")])
        self.assertEqual(precision.shards(Path("model-00001-of-00002.gguf")),
                         [Path("model-00001-of-00002.gguf"), Path("model-00002-of-00002.gguf")])

    def test_clipped_reference_does_not_replace_actual_perplexity(self):
        protocol = json.loads((RESULTS / "workload-tradeoffs-20261021" / "protocol.json").read_text())["nvfp4_precision_preflight"]
        calibration = {"ppl": 3.086314, "ppl_ratio": 1.001971, "mean_kl_nats": 0}
        self.assertEqual(precision.validate_calibration(calibration, "Final estimate: PPL = 3.0863", protocol), 3.0863)
        candidate = {"ppl": 3.086314 * 1.009, "ppl_ratio": 1.011, "mean_kl_nats": 0.005}
        self.assertTrue(precision.precision_passes(candidate, calibration, protocol)[0])
        candidate["mean_kl_nats"] = 0.02
        self.assertFalse(precision.precision_passes(candidate, calibration, protocol)[0])
        with self.assertRaises(ValueError):
            precision.validate_calibration(calibration, "Final estimate: PPL = 3.0", protocol)

    def test_inventory_has_unique_ids_and_no_new_enablement(self):
        assessment = json.loads((RESULTS / "workload-tradeoffs-20261021" / "assessment.json").read_text())
        items = assessment["candidates"]
        self.assertEqual(len(items), len({item["id"] for item in items}))
        self.assertNotIn("enabled", {item["decision"] for item in items})


class MMQFixtureTests(unittest.TestCase):
    def test_complete_frozen_fixtures(self):
        protocol = json.loads((mmq.HERE / "protocol.json").read_text())
        generated = mmq.fixtures(protocol)
        self.assertEqual(generated, (mmq.HERE / "test-ops.txt").read_text())
        self.assertEqual(len(set(generated.splitlines())), 64)
        self.assertEqual(len({line.split()[-1] for line in generated.splitlines()}), 64)

    def test_bad_count_or_unaligned_inner_dimension_fails(self):
        for field, value in (("cases_per_arm", 63), ("inner", 255)):
            protocol = json.loads((mmq.HERE / "protocol.json").read_text())
            protocol["correctness"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                mmq.fixtures(protocol)


class NVFP4ScreenTests(unittest.TestCase):
    def setUp(self):
        self.counts = {("q4_k", 17408, 5120): 2, ("q5_k", 17408, 5120): 1}
        self.base = [{"type": kind, "m": m, "n": n, "k": k, "time_us": 110.0}
                     for kind, m, k in self.counts for n in (1, 8, 512)]
        self.candidate = [{"type": "nvfp4", "m": 17408, "n": n, "k": 5120, "time_us": 100.0}
                          for n in (1, 8, 512)]

    def test_clear_prefill_gain_advances(self):
        result = nvfp4.compare(self.base, self.candidate, self.counts)
        self.assertEqual(result["decision"], "advance")
        self.assertAlmostEqual(result["weighted_geomean_ratios_by_columns"]["512"], 1.1)
        self.assertFalse(result["model_quality_checked"])
        self.assertFalse(result["deployed"])

    def test_small_gain_parks(self):
        for row in self.base:
            row["time_us"] = 102.0
        self.assertEqual(nvfp4.compare(self.base, self.candidate, self.counts)["decision"], "park")

    def test_one_regression_rejects_large_prefill_gain(self):
        self.base[0]["time_us"] = 96.0
        self.assertEqual(nvfp4.compare(self.base, self.candidate, self.counts)["decision"], "reject")

    def test_weighted_geomean_uses_all_tensor_counts(self):
        for row in self.base:
            row["time_us"] = 200.0 if row["type"] == "q4_k" else 100.0
        result = nvfp4.compare(self.base, self.candidate, self.counts)
        self.assertAlmostEqual(result["weighted_geomean_ratios_by_columns"]["512"], 2 ** (2 / 3))

    def test_missing_duplicate_mislabeled_or_nonfinite_cases_fail(self):
        invalid = [self.candidate[:-1], self.candidate + self.candidate[:1],
                   [{**row, "type": "q4_k"} for row in self.candidate]]
        for value in (0, -1, float("nan"), float("inf")):
            rows = copy.deepcopy(self.candidate)
            rows[0]["time_us"] = value
            invalid.append(rows)
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                nvfp4.compare(self.base, candidate, self.counts)

    def test_recorded_rejection_reproduces_without_new_timing(self):
        inventory = json.loads((nvfp4.HERE / "inventory.json").read_text())
        counts = {(row["type"], row["m"], row["k"]): row["count"] for row in inventory["tensors"]}
        base = json.loads((nvfp4.HERE / "original-r1.json").read_text())["cases"]
        candidate = json.loads((nvfp4.HERE / "nvfp4-r1.json").read_text())["cases"]
        recorded = json.loads((nvfp4.HERE / "comparison-r1.json").read_text())
        self.assertEqual(nvfp4.compare(base, candidate, counts), recorded)


class NemotronNVFP4Tests(unittest.TestCase):
    def setUp(self):
        self.base = [
            {"op": "NEMOTRON_FFN", "type": kind, "b": b, "m": m, "n": n, "k": k, "route": route,
             "n_mats": 512, "n_used": 22, "ids_offset": 1, "runs": 100, "time_us": 110.0}
            for b, (kind, m, k) in nemotron.SHAPES.items() for n in (1, 4, 512) for route in range(3)
        ]
        self.candidate = [{**row, "type": "nvfp4", "time_us": 100.0} for row in self.base]

    def test_full_inventory_and_routes_advance(self):
        result = nemotron.compare(self.base, self.candidate)
        self.assertEqual(result["decision"], "advance")
        self.assertEqual(len(result["comparisons"]), 18)
        self.assertAlmostEqual(result["geomean_ratios_by_columns"]["512"], 1.1)
        self.assertFalse(result["weights_converted"])
        self.assertFalse(result["deployed"])

    def test_one_guard_regression_rejects_prefill_win(self):
        self.base[0]["time_us"] = 96.0
        self.assertEqual(nemotron.compare(self.base, self.candidate)["decision"], "reject")

    def test_below_five_percent_parks(self):
        for row in self.base:
            row["time_us"] = 104.9
        self.assertEqual(nemotron.compare(self.base, self.candidate)["decision"], "park")

    def test_recorded_rejection_reproduces_without_new_timing(self):
        base = json.loads((nemotron.HERE / "original-r1.json").read_text())["cases"]
        candidate = json.loads((nemotron.HERE / "nvfp4-r1.json").read_text())["cases"]
        recorded = json.loads((nemotron.HERE / "comparison-r1.json").read_text())
        self.assertEqual(nemotron.compare(base, candidate), recorded)
        self.assertEqual(recorded["decision"], "reject")

    def test_missing_duplicate_or_wrong_routing_fails(self):
        invalid = [self.candidate[:-1], self.candidate + self.candidate[:1]]
        for key, value in (("n_mats", 32), ("n_used", 10), ("ids_offset", 0), ("m", 2048),
                           ("route", 4), ("runs", 0), ("time_us", 0), ("time_us", float("nan"))):
            rows = copy.deepcopy(self.candidate)
            rows[0][key] = value
            invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                nemotron.compare(self.base, rows)

    def test_reverse_prefill_ranges_must_separate(self):
        first = nemotron.compare(self.base, self.candidate)
        second = copy.deepcopy(first)
        self.assertTrue(nemotron.separated_prefill_ranges(first, second))
        prefill = next(row for row in second["comparisons"] if row["n"] == 512)
        prefill["nvfp4_us"] = 110.0
        self.assertFalse(nemotron.separated_prefill_ranges(first, second))

    def test_parse_complete_graph_and_measurement_units(self):
        text = ("\x1b[32mNEMOTRON_FFN(type_a=nvfp4,type_b=f32,n_mats=512,n_used=22,b=1,"
                "m=2688,n=512,k=1024,amax=1.000000,m_v=0,ids_offset=1,nem_route=2): "
                "200 runs - 123.45 us/run\x1b[0m\n")
        row, = nemotron.parse_perf(text)
        self.assertEqual((row["b"], row["n"], row["route"]), (1, 512, 2))
        self.assertEqual(row["time_us"], 123.45)
        native_q4, = nemotron.parse_perf(text.replace("type_a=nvfp4", "type_a=q4_K"))
        self.assertEqual(native_q4["type"], "q4_k")
        with self.assertRaises(ValueError):
            nemotron.parse_perf(text.replace("type_b=f32", "type_b=f16"))

    def test_split_metadata_and_mtp_exclusion(self):
        def field(value):
            return SimpleNamespace(contents=lambda: value)

        metadata = {"general.architecture": "nemotron_h_moe", "nemotron_h_moe.block_count": 89,
                    "nemotron_h_moe.nextn_predict_layers": 1, "nemotron_h_moe.expert_count": 512,
                    "nemotron_h_moe.expert_used_count": 22}
        tensors = []
        for layer in list(range(40)) + [88]:
            for role, kind, shape in (
                ("up_exps", "Q4_K", (1024, 2688, 512)),
                ("down_exps", "IQ4_NL", (2688, 1024, 512)),
                ("up_shexp", "Q4_K", (4096, 5376)),
                ("down_shexp", "Q4_K" if layer < 20 else "Q6_K", (5376, 4096)),
                ("latent_down", "Q4_K", (4096, 1024)),
                ("latent_up", "Q4_K", (1024, 4096)),
            ):
                tensors.append(SimpleNamespace(name=f"blk.{layer}.ffn_{role}.weight",
                                               tensor_type=SimpleNamespace(name=kind), shape=shape))
        readers = []
        for index in (0, 1):
            values = {"split.no": index, "split.count": 2, "split.tensors.count": len(tensors)}
            if index == 0:
                values.update(metadata)
            readers.append(SimpleNamespace(fields={k: field(v) for k, v in values.items()},
                                           tensors=tensors[index::2]))
        first = Path("model-00001-of-00002.gguf")
        reader = lambda path, mode: readers[0 if "-00001-of-" in path.name else 1]
        with patch.dict(sys.modules, {"gguf": SimpleNamespace(GGUFReader=reader)}):
            result = nemotron.inventory(first)
            self.assertEqual(result["source_tensors"], 246)
            self.assertEqual(result["target_ffn_tensors"], 240)
            self.assertEqual(result["selected_expert_tensors"], 80)
            readers[1].fields["general.architecture"] = field("wrong_model")
            with self.assertRaisesRegex(ValueError, "Shard metadata differs"):
                nemotron.inventory(first)
            del readers[1].fields["general.architecture"]
            readers[1].tensors = readers[1].tensors[:-1]
            with self.assertRaisesRegex(ValueError, "Incomplete shard-wide"):
                nemotron.inventory(first)


class LauncherProfileTests(unittest.TestCase):
    def setUp(self):
        source = json.loads((profiles.HERE / "launcher-source.json").read_text())
        self.recipe = {"serve": source["serve"]}
        self.paths = {"d27": r"C:\model cache\target.gguf",
                      "d27_dfl": r"C:\model cache\draft.gguf",
                      "d27_mmproj": r"C:\model cache\projector.gguf"}

    def test_only_draft_depth_differs(self):
        control = profiles.resolve_command(self.recipe, self.paths, 9949, 7)
        candidate = profiles.resolve_command(self.recipe, self.paths, 9949, 4)
        differences = [(a, b) for a, b in zip(control, candidate) if a != b]
        self.assertEqual(differences, [("7", "4")])
        self.assertEqual(len(control), len(candidate))
        for flag, key in (("-m", "d27"), ("-md", "d27_dfl"), ("--mmproj", "d27_mmproj")):
            self.assertEqual(control[control.index(flag) + 1], self.paths[key])
        self.assertEqual(control[control.index("--port") + 1], "9949")
        self.assertNotIn("-hf", control)
        self.assertNotIn("-hfrd", control)
        self.assertIn("--spec-draft-backend-sampling", control)

    def test_windows_continuation_newlines_are_equivalent(self):
        unix = profiles.resolve_command(self.recipe, self.paths, 9949, 4)
        self.recipe["serve"]["command"] = self.recipe["serve"]["command"].replace("\n", "\r\n")
        self.assertEqual(profiles.resolve_command(self.recipe, self.paths, 9949, 4), unix)

    def test_changed_geometry_is_not_silently_accepted(self):
        self.recipe["serve"]["command"] = self.recipe["serve"]["command"].replace("--ubatch-size 512", "--ubatch-size 2048")
        with self.assertRaises(ValueError):
            profiles.resolve_command(self.recipe, self.paths, 9949, 4)


class SSMDispatchTests(unittest.TestCase):
    def result(self):
        row = {"response": {"timings": {"prompt_n": 1565}}}
        return {"warmups": [row], "rows": [row], "completed": True, "stop": "ctrl+c",
                "exit_code": 0, "gpu_memory_ready_mib": 70000, "gpu_memory_done_mib": 70000,
                "gpu_memory_released_mib": 3000}

    def traces(self, k, ssd):
        return ssm.parse_trace("\n".join(
            f"SPARK_SSM name=node_{i} n_t=1561 K={k} d_state=128 head_dim=64 "
            f"n_head=128 n_seq=1 scalar_a=1 ssd={ssd}" for i in range(40)))

    def test_prompt_total_can_include_short_checkpoint_tail(self):
        for tag, k, selected in (("N1", 1, 1), ("N4", 4, 0)):
            with self.subTest(tag=tag):
                summary = ssm.summarize(tag, self.result(), self.traces(k, selected))
                self.assertTrue(summary["matched_hypothesis"])
                self.assertFalse(summary["timings_are_performance_evidence"])

    def test_missing_trace_wrong_dispatch_or_failed_shutdown_fails(self):
        result = self.result()
        self.assertFalse(ssm.summarize("N4", result, [])["matched_hypothesis"])
        self.assertFalse(ssm.summarize("N4", result, self.traces(1, 1))["matched_hypothesis"])
        result["exit_code"] = 1
        self.assertFalse(ssm.summarize("N4", result, self.traces(4, 0))["matched_hypothesis"])


if __name__ == "__main__":
    unittest.main()
