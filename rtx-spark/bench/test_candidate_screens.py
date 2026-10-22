import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
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
release_gates = load_module("release_gates_profile", "workload-tradeoffs-20261021", "release_gates.py")


class CompleteProseTests(unittest.TestCase):
    def test_literal_checks_require_the_exact_identifiers(self):
        for case, (facts, _) in prose_profile.FACTS.items():
            response = {"content": facts + "\n\n" + facts, "content_sha256": "test",
                        "stop_type": "eos", "request": {"prompt": "explain"}}
            with self.subTest(case=case):
                self.assertTrue(prose_profile.check_literals(case, response)["passed"])
                response["content"] = response["content"].replace(prose_profile.FACTS[case][1][0], "wrong-id")
                self.assertFalse(prose_profile.check_literals(case, response)["passed"])


class ProseReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = RESULTS / "workload-tradeoffs-20261021"
        cls.review = json.loads((base / "27b-prose-review.json").read_text(encoding="utf-8"))
        cls.result = json.loads((base / "27b-complete-prose.json").read_text(encoding="utf-8"))

    def test_reviewed_texts_are_the_measured_responses(self):
        self.assertEqual(set(self.review["cases"]), set(prose_profile.FACTS))
        for case, body in self.review["cases"].items():
            recorded = [check for run in self.result["runs"] for check in run["checks"] if check["case"] == case]
            digest = hashlib.sha256(body["text"].encode("utf-8")).hexdigest()
            with self.subTest(case=case):
                self.assertEqual(len(recorded), 16)
                self.assertEqual({check["content_sha256"] for check in recorded}, {digest})
                self.assertEqual(body["content_sha256"], digest)
                self.assertEqual({check["words"] for check in recorded}, {body["words"]})
                self.assertEqual(len(re.findall(r"\b[\w'-]+\b", body["text"])), body["words"])

    def test_review_quotes_are_verbatim_and_literals_present(self):
        for case, body in self.review["cases"].items():
            response = {"content": body["text"], "content_sha256": body["content_sha256"],
                        "stop_type": "eos", "request": {"prompt": "explain"}}
            quotes = [quote for item in body["required_items"] for quote in item["evidence"]]
            quotes += [item["quote"] for item in body["unsupported_additions"] + body["borderline"]]
            with self.subTest(case=case):
                self.assertTrue(prose_profile.check_literals(case, response)["passed"])
                self.assertTrue(quotes)
                for quote in quotes:
                    self.assertIn(quote, body["text"])

    def test_review_does_not_hide_the_instruction_violations(self):
        self.assertFalse(self.review["promotion_blocking"])
        self.assertIn("Do not invent events, causes or benefits", self.review["findings"]["constraint_adherence"])
        for case, body in self.review["cases"].items():
            with self.subTest(case=case):
                self.assertTrue(body["unsupported_additions"])
                self.assertTrue(all(item["verdict"].startswith(("faithful", "wording_drift"))
                                    for item in body["required_items"]))


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

    def timed(self, case, prompt, counts, repeat=None):
        cache_n, prompt_n = counts
        row = {"case": case, "response": {
            "content": json.dumps(long_profiles.EXPECTED), "stop_type": "eos", "ttft_s": 0.5, "wall_s": 2.0,
            "content_sha256": "x", "request": {"prompt": prompt, "cache_prompt": cache_n > 0},
            "timings": {"cache_n": cache_n, "prompt_n": prompt_n}}}
        if repeat is not None:
            row["repeat"] = repeat
        return row

    def run_data(self, prime, guard):
        names = [case[0] for case in long_profiles.cases()]
        counts = {"cached_prime": prime, "cached_guard": guard}

        def rows(repeat):
            return [self.timed(name, "same" if name in counts else name, counts.get(name, (0, 9000)), repeat)
                    for name in names]
        return {"completed": True, "exit_code": 0, "stop": "ctrl+c", "load_s": 1.0, "gpu_memory_ready_mib": 1,
                "warmups": rows(None), "rows": [row for repeat in (1, 2, 3) for row in rows(repeat)]}

    def test_cached_guard_accepts_a_prime_that_the_warmup_already_cached(self):
        for prime in ((0, 1306), (1302, 4)):
            with self.subTest(prime=prime):
                self.assertTrue(long_profiles.analyze(self.run_data(prime, (1302, 4)))["quality_passed"])

    def test_cached_guard_must_serve_its_prompt_from_cache(self):
        for guard in ((0, 1306), (600, 706)):
            with self.subTest(guard=guard), self.assertRaises(ValueError):
                long_profiles.analyze(self.run_data((1302, 4), guard))

    def test_cached_guard_requires_the_identical_prompt(self):
        prime = self.timed("cached_prime", "a", (0, 1306))["response"]
        guard = self.timed("cached_guard", "b", (1302, 4))["response"]
        self.assertFalse(long_profiles.guard_reused_prime(prime, guard))

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

    def test_inventory_outcomes_are_resolved_and_evidence_exists(self):
        folder = RESULTS / "workload-tradeoffs-20261021"
        assessment = json.loads((folder / "assessment.json").read_text())
        self.assertEqual(assessment["status"], "validation_complete")
        self.assertNotIn("validate_profile", {item["decision"] for item in assessment["candidates"]})
        for item in assessment["candidates"]:
            evidence = item.get("evidence", [])
            for name in [evidence] if isinstance(evidence, str) else evidence:
                self.assertTrue((folder / name).is_file(), name)

    def test_only_the_shipped_opt_in_profile_is_enabled(self):
        assessment = json.loads((RESULTS / "workload-tradeoffs-20261021" / "assessment.json").read_text())
        opted = [item["profile_id"] for item in assessment["candidates"] if item["decision"] == "opt_in_profile"]
        shipped = json.loads((RESULTS.parent / "profiles" / "opt-in-profiles.json").read_text())["profiles"]
        self.assertEqual(opted, [profile["id"] for profile in shipped])

    def test_decode_rate_record_is_self_consistent(self):
        record = json.loads((RESULTS / "workload-tradeoffs-20261021" / "27b-quality-decode-rates.json").read_text())
        changes = []
        for name, case in record["cases"].items():
            change = 100 * (case["P4"]["tokens_per_second"] / case["P7"]["tokens_per_second"] - 1)
            self.assertAlmostEqual(change, case["p4_vs_p7_percent"], delta=0.05, msg=name)
            self.assertEqual(case["P4"]["generated_tokens"], case["P7"]["generated_tokens"], name)
            changes.append(change)
        self.assertAlmostEqual(sum(changes) / len(changes), record["mean_p4_vs_p7_percent"], delta=0.05)


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


class FakeServerRunner:
    """Stands in for run_rebaseline and writes server files in the real format."""

    def __init__(self, candidate_speedup=0.04, stop_type="eos"):
        self.matrix = SimpleNamespace(
            PORT=9940, P={}, CONFIGS=[],
            SUITES={"decode": [(name, "p", 128) for name in ("code256", "prose256", "prose512", "novel2K")]})
        self.completion = "greedy"
        self.candidate_speedup = candidate_speedup
        self.stop_type = stop_type
        self.calls = []

    def run_server(self, tag, out, repetitions, suite, allow_eos=False, draft_model=None,
                   capture_metrics=False, case_names=None, exe=None, extra_args=()):
        names = list(case_names or [case[0] for case in self.matrix.SUITES["decode"]])
        if suite == "en-us-quality":
            names = list(release_gates.en_us_quality.CASE_NAMES)
        self.calls.append({"tag": tag, "suite": suite, "names": names, "completion": self.completion,
                           "allow_eos": allow_eos, "extra_args": list(extra_args)})
        scale = 1.0 if tag in ("P7", "N2") else 1.0 - self.candidate_speedup
        rows = [{"case": case, "repeat": repeat, "response": self.response(case, scale)}
                for repeat in range(1, repetitions + 1) for case in names]
        data = {"completed": True, "exit_code": 0, "stop": "ctrl+c", "allow_eos": allow_eos,
                "repetitions": repetitions, "gpu_memory_ready_mib": 30000, "warmups": [], "rows": rows}
        (out / f"server_{tag}_{suite}.json").write_text(json.dumps(data), encoding="utf-8")

    def response(self, case, scale):
        facts = prose_profile.FACTS.get(case)
        content = f"{facts[0]}\n\n{facts[0]}" if facts else "text"
        return {"content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "stop_type": self.stop_type, "request": {"prompt": "p"}, "wall_s": 10.0 * scale,
                "timings": {"predicted_n": 400, "predicted_per_second": 40.0 / scale,
                            "draft_n": 100, "draft_n_accepted": 60}}


class ReleaseGateTests(unittest.TestCase):
    CHECKS = [("warmups", "code"), ("warmups", "locale_json"), ("rows", "code"), ("rows", "locale_json")]
    LOCALE = [("warmups", "locale_json"), ("rows", "locale_json")]

    def run_gates(self, function, runner, failing=(), quality=None):
        def analyze(path):
            return {"path": str(path), "cases": [], "scores": [], "passed": path.name.split("_")[1] not in failing}

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(catalog=Path("catalog.json"), work=Path(folder))
            with patch.object(release_gates, "launcher_configs", return_value=([], "recipe-id")), \
                    patch.object(release_gates.en_us_quality, "analyze", side_effect=analyze):
                result = function(runner, args) if quality is None else function(runner, args, quality)
            return result, sorted(path.name for path in Path(folder).glob("stage-*.json"))

    def outcomes(self, failing):
        return {key: key not in failing for key in self.CHECKS}

    def saved_quality(self, control, candidate, identical=True):
        def report(outcomes):
            return {"scores": [{"phase": phase, "case": case, "passed": passed}
                               for (phase, case), passed in outcomes.items()],
                    "passed": all(outcomes.values())}

        reports = {"P7": report(control), "P4": report(candidate)}
        return {"control": "P7", "candidate": "P4", "reports": reports,
                "control_passed": reports["P7"]["passed"], "candidate_passed": reports["P4"]["passed"],
                "passed": reports["P4"]["passed"], "identical_outputs": identical}

    def test_27b_stages_run_in_the_declared_order_and_restore_the_runner(self):
        runner = FakeServerRunner()
        result, stage_files = self.run_gates(release_gates.gates_27b, runner)
        self.assertEqual([call["tag"] for call in runner.calls],
                         ["P7", "P4", "P7", "P4", "P4", "P7", "P7", "P4", "P4", "P7"])
        self.assertEqual([call["suite"] for call in runner.calls], ["en-us-quality"] * 2 + ["decode"] * 8)
        self.assertTrue(all(call["allow_eos"] for call in runner.calls[:6]))
        self.assertFalse(any(call["allow_eos"] for call in runner.calls[6:]))
        self.assertTrue(all(call["extra_args"] == ["--reasoning", "off"] for call in runner.calls))
        sampled = {call["completion"].__name__ for call in runner.calls[2:6]}
        self.assertEqual(sampled, {"sampled_completion"})
        self.assertEqual({call["completion"] for call in runner.calls[6:]}, {"greedy"})
        self.assertEqual(runner.calls[2]["names"], list(prose_profile.FACTS))
        self.assertEqual(runner.calls[6]["names"], ["code256", "novel2K"])
        self.assertEqual(runner.completion, "greedy")
        self.assertEqual(len(runner.matrix.SUITES["decode"]), 4)
        self.assertTrue(result["qualified"])
        self.assertTrue(result["launcher_sampling"]["comparison"]["passed"])
        self.assertAlmostEqual(result["launcher_sampling"]["comparison"]["pairs"][0]["latency_reduction_percent"], 4)
        self.assertEqual(result["launcher_sampling"]["identical_text_fraction"]["between_depths"], [1.0, 1.0])
        self.assertAlmostEqual(result["launcher_sampling"]["runs"][0]["acceptance"], 0.6)
        self.assertAlmostEqual(result["outside_profile"]["pairs"][1]["code256"]["reduction_percent"], 4)
        self.assertTrue(result["outside_profile"]["pairs"][0]["novel2K"]["separated"])
        self.assertEqual(stage_files, ["stage-launcher-sampling.json", "stage-outside-profile.json",
                                       "stage-quality.json"])

    def test_slower_candidate_fails_the_sampling_gate_and_skips_the_outside_stage(self):
        runner = FakeServerRunner(candidate_speedup=-0.05)
        result, stage_files = self.run_gates(release_gates.gates_27b, runner)
        self.assertEqual(len(runner.calls), 6)
        self.assertFalse(result["launcher_sampling"]["passed"])
        self.assertFalse(result["qualified"])
        self.assertNotIn("outside_profile", result)
        self.assertEqual(stage_files, ["stage-launcher-sampling.json", "stage-quality.json"])

    def test_gain_below_the_frozen_minimum_is_not_enough(self):
        result, _ = self.run_gates(release_gates.gates_27b, FakeServerRunner(candidate_speedup=0.005))
        self.assertFalse(result["launcher_sampling"]["comparison"]["passed"])
        self.assertFalse(result["qualified"])

    def test_candidate_quality_failure_skips_every_speed_stage(self):
        runner = FakeServerRunner()
        result, stage_files = self.run_gates(release_gates.gates_27b, runner, failing=("P4",))
        self.assertEqual([call["tag"] for call in runner.calls], ["P7", "P4"])
        self.assertTrue(result["quality"]["control_passed"])
        self.assertFalse(result["quality"]["candidate_passed"])
        self.assertFalse(result["qualified"])
        self.assertEqual(stage_files, ["stage-quality.json"])

    def test_control_quality_failure_is_reported_without_blaming_the_candidate(self):
        result, _ = self.run_gates(release_gates.gates_27b, FakeServerRunner(), failing=("P7",))
        self.assertFalse(result["quality"]["control_passed"])
        self.assertTrue(result["quality"]["passed"])

    def test_unfinished_sampled_response_fails_the_gate(self):
        result, _ = self.run_gates(release_gates.gates_27b, FakeServerRunner(stop_type="limit"))
        sampling = result["launcher_sampling"]
        self.assertFalse(sampling["all_natural_eos"])
        self.assertFalse(sampling["passed"])
        self.assertIn("naturally", sampling["runs"][0]["responses"][0]["literals"]["error"])

    def test_nemotron_gate_is_the_candidate_quality_check(self):
        runner = FakeServerRunner()
        result, stage_files = self.run_gates(release_gates.gates_nemotron, runner)
        self.assertEqual([call["tag"] for call in runner.calls], ["N2", "N1"])
        self.assertTrue(result["qualified"])
        self.assertTrue(result["quality"]["identical_outputs"])
        self.assertEqual(stage_files, ["stage-quality.json"])
        result, _ = self.run_gates(release_gates.gates_nemotron, FakeServerRunner(), failing=("N1",))
        self.assertFalse(result["qualified"])

    def test_non_inferiority_accepts_a_candidate_that_repeats_the_control_failure(self):
        verdict = release_gates.non_inferiority(self.saved_quality(self.outcomes(self.LOCALE), self.outcomes(self.LOCALE)))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["control_failed_checks"], ["rows/locale_json", "warmups/locale_json"])
        self.assertEqual(verdict["candidate_failed_checks"], verdict["control_failed_checks"])
        self.assertEqual(verdict["lost_control_passes"], [])

    def test_non_inferiority_rejects_a_check_the_candidate_loses(self):
        verdict = release_gates.non_inferiority(
            self.saved_quality(self.outcomes(self.LOCALE), self.outcomes(self.LOCALE + [("rows", "code")])))
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["lost_control_passes"], ["rows/code"])

    def test_non_inferiority_does_not_penalize_an_extra_candidate_pass(self):
        verdict = release_gates.non_inferiority(self.saved_quality(self.outcomes(self.LOCALE), self.outcomes([])))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["candidate_failed_checks"], [])

    def test_non_inferiority_needs_the_same_nonempty_check_set(self):
        shorter = self.outcomes([])
        del shorter[("rows", "code")]
        for control, candidate in ((self.outcomes([]), shorter), ({}, {})):
            with self.subTest(candidate_checks=len(candidate)), self.assertRaises(ValueError):
                release_gates.non_inferiority(self.saved_quality(control, candidate))

    def test_revised_quality_binds_the_unchanged_strict_result(self):
        saved = self.saved_quality(self.outcomes(self.LOCALE), self.outcomes(self.LOCALE))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strict.json"
            path.write_text(json.dumps({"quality": saved}), encoding="utf-8")
            before = path.read_bytes()
            revised = release_gates.revised_quality(path)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(revised["strict_result"], "strict.json")
        self.assertEqual(revised["strict_result_sha256"], hashlib.sha256(before).hexdigest())
        self.assertFalse(revised["strict_passed"])
        self.assertTrue(revised["passed"])
        self.assertTrue(revised["identical_outputs"])
        self.assertEqual(revised["rule"], "non-inferiority")

    def test_revised_27b_gate_reuses_the_quality_result_and_still_requires_speed(self):
        quality = {"rule": "non-inferiority", "passed": True, "identical_outputs": True}
        runner = FakeServerRunner()
        result, stage_files = self.run_gates(release_gates.gates_27b, runner, quality=quality)
        self.assertEqual([call["suite"] for call in runner.calls], ["decode"] * 8)
        self.assertEqual(result["quality"], quality)
        self.assertTrue(result["qualified"])
        self.assertEqual(stage_files, ["stage-launcher-sampling.json", "stage-outside-profile.json",
                                       "stage-quality.json"])
        result, _ = self.run_gates(release_gates.gates_27b, FakeServerRunner(candidate_speedup=0.005), quality=quality)
        self.assertFalse(result["qualified"])
        self.assertNotIn("outside_profile", result)

    def test_revised_27b_quality_failure_skips_every_speed_stage(self):
        runner = FakeServerRunner()
        result, stage_files = self.run_gates(release_gates.gates_27b, runner, quality={"passed": False})
        self.assertEqual(runner.calls, [])
        self.assertFalse(result["qualified"])
        self.assertEqual(stage_files, ["stage-quality.json"])

    def test_revised_nemotron_gate_runs_no_server(self):
        runner = FakeServerRunner()
        result, _ = self.run_gates(release_gates.gates_nemotron, runner, quality={"passed": True})
        self.assertEqual(runner.calls, [])
        self.assertTrue(result["qualified"])

    def test_saved_strict_results_rescore_as_recorded(self):
        for candidate, name in release_gates.OUTPUT.items():
            path = release_gates.HERE / name
            strict = json.loads(path.read_text())
            revised = release_gates.revised_quality(path)
            with self.subTest(candidate=candidate):
                self.assertFalse(strict["qualified"])
                self.assertFalse(revised["strict_passed"])
                self.assertTrue(revised["passed"])
                self.assertTrue(revised["identical_outputs"])
                self.assertEqual(revised["control_failed_checks"], ["rows/locale_json", "warmups/locale_json"])
                self.assertEqual(revised["candidate_failed_checks"], revised["control_failed_checks"])
                self.assertEqual(revised["lost_control_passes"], [])

    def test_incomplete_runs_are_rejected(self):
        for change in ({"completed": False}, {"exit_code": 1}, {"stop": "forced"}):
            data = {"completed": True, "exit_code": 0, "stop": "ctrl+c", **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                release_gates.require_complete(data)

    def test_launcher_sampling_defaults_are_recognized(self):
        launcher = {"temperature": 1.0, "top_k": 20, "top_p": 0.949999988079071, "min_p": 0.0,
                    "presence_penalty": 0.0, "repeat_penalty": 1.0}
        self.assertTrue(release_gates.sampling_matches(launcher))
        for key, value in (("temperature", 0.0), ("top_k", 1), ("top_p", 0.9), ("repeat_penalty", 1.05)):
            with self.subTest(key=key):
                self.assertFalse(release_gates.sampling_matches({**launcher, key: value}))

    def test_sampled_request_leaves_sampling_to_the_server(self):
        settings = {"temperature": 1.0, "top_k": 20, "top_p": 0.949999988079071, "min_p": 0.0,
                    "presence_penalty": 0.0, "repeat_penalty": 1.0, "seed": 1234}
        sent = []

        def reply(settings):
            def urlopen(request, timeout):
                sent.append(request)
                return io.BytesIO(json.dumps({"content": "hello", "generation_settings": settings}).encode())
            return urlopen

        complete = release_gates.make_sampled_completion(9940)
        with patch.object(release_gates.urllib.request, "urlopen", reply(settings)):
            result = complete("prompt", 64, True)
        body = json.loads(sent[0].data)
        self.assertEqual(sent[0].full_url, "http://127.0.0.1:9940/completion")
        self.assertEqual(set(body), {"prompt", "n_predict", "seed", "cache_prompt", "stream",
                                     "ignore_eos", "return_tokens"})
        self.assertEqual((body["n_predict"], body["seed"], body["ignore_eos"], body["cache_prompt"]),
                         (64, 1234, False, False))
        self.assertEqual(result["content_sha256"], hashlib.sha256(b"hello").hexdigest())
        self.assertGreaterEqual(result["wall_s"], 0)
        for changed in ({**settings, "temperature": 0.0, "top_k": 1}, {**settings, "seed": 7}):
            with self.subTest(changed=changed), patch.object(release_gates.urllib.request, "urlopen", reply(changed)):
                with self.assertRaisesRegex(ValueError, "launcher sampling"):
                    complete("prompt", 64, True)

    def test_launcher_configs_change_only_the_draft_depth(self):
        source = json.loads((profiles.HERE / "launcher-source.json").read_text())
        recipe_id = json.loads((profiles.HERE / "protocol.json").read_text())["control_recipe"]
        paths = {"d27": "target.gguf", "d27_dfl": "draft.gguf", "d27_mmproj": "projector.gguf"}
        runner = SimpleNamespace(matrix=SimpleNamespace(P=paths, PORT=9940))
        catalog = {"models": [{"id": "qwen/qwen3.8-27b", "recipes": [{"id": recipe_id, "serve": source["serve"]}]}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            configs, recipe = release_gates.launcher_configs(runner, path)
            self.assertEqual(recipe, recipe_id)
            self.assertEqual([config[0] for config in configs],
                             ["P7 q27 launcher dflash7", "P4 q27 launcher dflash4"])
            differences = [(a, b) for a, b in zip(configs[0][1], configs[1][1]) if a != b]
            self.assertEqual(differences, [("7", "4")])
            catalog["models"][0]["recipes"][0]["serve"]["command"] += " --extra"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "launcher command changed"):
                release_gates.launcher_configs(runner, path)

    def test_scoring_helpers(self):
        responses = [{"wall_s": 6.0, "timings": {"predicted_n": 100}}, {"wall_s": 4.0, "timings": {"predicted_n": 300}}]
        self.assertAlmostEqual(release_gates.seconds_per_token(responses), 0.025)
        self.assertAlmostEqual(release_gates.reduction(10.0, 9.0), 10.0)
        self.assertAlmostEqual(release_gates.identical_fraction([1, 2, 3], [1, 2, 4]), 2 / 3)

    def test_script_is_ascii(self):
        self.assertTrue(Path(release_gates.__file__).read_text(encoding="utf-8").isascii())


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
