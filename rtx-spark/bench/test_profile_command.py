import contextlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest

RTX = Path(__file__).resolve().parents[1]
EVIDENCE = RTX / "results" / "workload-tradeoffs-20261021"
TOOL_FILE = RTX / "tools" / "profile_command.py"
PROFILE_FILE = RTX / "profiles" / "opt-in-profiles.json"
PROFILE_ID = "nemotron-nospec-long-input"


def load_tool():
    spec = importlib.util.spec_from_file_location("profile_command", TOOL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()
ProfileError = tool.ProfileError

CONTROL_TOKENS = ["llama-server.exe", "-m", "C:\\models\\nemotron super.gguf", "--mmproj", "C:\\models\\mm.gguf",
                  "--fit", "off", "--host", "127.0.0.1", "--port", "9940", "-c", "262144", "-np", "1", "-fa", "on",
                  "-ngl", "999", "-b", "4096", "-ub", "4096", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
CONTROL_TEXT = ('llama-server.exe -m "C:\\models\\nemotron super.gguf" --mmproj C:\\models\\mm.gguf --fit off '
                "--host 127.0.0.1 --port 9940 -c 262144 -np 1 -fa on -ngl 999 -b 4096 -ub 4096 "
                "--spec-type draft-mtp --spec-draft-n-max 3")
CANDIDATE_TOKENS = CONTROL_TOKENS[:-2]
CANDIDATE_TOKENS[CANDIDATE_TOKENS.index("draft-mtp")] = "none"


def body(rendered):
    return "\n".join(line for line in rendered.splitlines() if not line.startswith("#"))


class SplitCommandTests(unittest.TestCase):
    def test_quotes_continuations_and_crlf(self):
        text = "& llama-server.exe `\r\n  -m \"a b.gguf\" `\r\n  --name 'it''s' --empty '' -ngl 999\r\n"
        self.assertEqual(tool.split_command(text),
                         ["llama-server.exe", "-m", "a b.gguf", "--name", "it's", "--empty", "", "-ngl", "999"])

    def test_plain_control_command(self):
        self.assertEqual(tool.split_command(CONTROL_TEXT), CONTROL_TOKENS)

    def test_refuses_syntax_it_cannot_reproduce(self):
        for suffix in ("-m $env:MODEL", '-m "a$b"', "-m a;b", "# comment", '-m a"b"', '-m "a`b"', "| more",
                       "-m @args", "-m a,b", "-m (x)", "-m {x}", "-m a>b", "-m a&b"):
            with self.subTest(suffix=suffix), self.assertRaises(ProfileError):
                tool.split_command("llama-server.exe " + suffix)

    def test_refuses_a_second_line(self):
        with self.assertRaises(ProfileError):
            tool.split_command("llama-server.exe -m a\nRemove-Item x")

    def test_requires_llama_server(self):
        for text in ("", "python -m a", "other.exe -m a"):
            with self.subTest(text=text), self.assertRaises(ProfileError):
                tool.split_command(text)
        self.assertEqual(tool.split_command("C:\\bin\\LLAMA-SERVER.EXE -m a")[1:], ["-m", "a"])


class RenderCommandTests(unittest.TestCase):
    def test_round_trip(self):
        for tokens in (CONTROL_TOKENS,
                       ["llama-server.exe", "--name", "it's", "--empty", "", "--seed", "-1", "-m", "a b.gguf"],
                       ["C:\\Program Files\\llama\\llama-server.exe", "-ngl", "999"]):
            with self.subTest(tokens=tokens):
                self.assertEqual(tool.split_command(tool.render_command(tokens)), tokens)

    def test_executable_with_spaces_needs_the_call_operator(self):
        rendered = tool.render_command(["C:\\Program Files\\llama\\llama-server.exe", "-ngl", "999"])
        self.assertTrue(rendered.startswith("& 'C:\\Program Files\\llama\\llama-server.exe'"))
        self.assertFalse(tool.render_command(CONTROL_TOKENS).startswith("&"))

    def test_flag_and_value_stay_together_and_negative_values_are_values(self):
        lines = tool.render_command(["llama-server.exe", "--seed", "-1", "-ngl", "999"]).splitlines()
        self.assertEqual([line.strip() for line in lines], ["llama-server.exe `", "--seed -1 `", "-ngl 999"])


class SyntheticProfileTests(unittest.TestCase):
    TOKENS = ["llama-server.exe", "-a", "1", "-b", "2", "-c", "3"]

    def profile(self, requires=(), edits=()):
        return {"control": {"requires": [list(item) for item in requires]}, "edits": list(edits)}

    def test_edit_changes_only_the_named_value(self):
        profile = self.profile(edits=[{"flag": "-b", "from": "2", "to": "9"}])
        self.assertEqual(tool.apply_edits(profile, self.TOKENS), ["llama-server.exe", "-a", "1", "-b", "9", "-c", "3"])
        self.assertEqual(self.TOKENS[4], "2")

    def test_removal_edit_drops_flag_and_value(self):
        profile = self.profile(edits=[{"flag": "-b", "from": "2"}])
        self.assertEqual(tool.apply_edits(profile, self.TOKENS), ["llama-server.exe", "-a", "1", "-c", "3"])

    def test_edit_refuses_an_unexpected_value(self):
        with self.assertRaises(ProfileError):
            tool.apply_edits(self.profile(edits=[{"flag": "-b", "from": "7", "to": "9"}]), self.TOKENS)

    def test_flag_must_appear_exactly_once_with_a_value(self):
        for tokens in (self.TOKENS + ["-b", "4"], ["llama-server.exe", "-a", "1"], self.TOKENS[:-1] + ["-b"]):
            with self.subTest(tokens=tokens), self.assertRaises(ProfileError):
                tool.value_index(tokens, "-b")

    def test_control_check_reports_the_mismatch(self):
        profile = self.profile(requires=[("-b", "2")])
        tool.check_control(profile, self.TOKENS)
        with self.assertRaises(ProfileError):
            tool.check_control(profile, ["llama-server.exe", "-b", "5"])


class ShippedProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = tool.load_profiles()[PROFILE_ID]

    def test_renders_the_no_speculation_command(self):
        rendered = tool.render_profile(self.profile, CONTROL_TEXT)
        self.assertEqual(tool.split_command(body(rendered)), CANDIDATE_TOKENS)
        self.assertIn("--spec-type draft-mtp -> none", rendered)
        self.assertIn("remove --spec-draft-n-max 3", rendered)
        self.assertEqual(sum(line.startswith("# Avoid when:") for line in rendered.splitlines()),
                         len(self.profile["avoid_when"]))

    def test_refuses_a_command_that_is_not_the_validated_control(self):
        variants = [CONTROL_TEXT.replace("-ub 4096", "-ub 512"),
                    CONTROL_TEXT.replace("-b 4096 ", "-b 2048 "),
                    CONTROL_TEXT.replace("--spec-draft-n-max 3", "--spec-draft-n-max 4"),
                    CONTROL_TEXT.replace(" --spec-draft-n-max 3", ""),
                    CONTROL_TEXT.replace("draft-mtp", "draft-dflash"),
                    CONTROL_TEXT + " -b 4096",
                    CONTROL_TEXT.replace("--spec-type draft-mtp", "--spec-type none")]
        for text in variants:
            with self.subTest(text=text[-60:]), self.assertRaises(ProfileError):
                tool.render_profile(self.profile, text)

    def test_refuses_the_already_edited_command(self):
        with self.assertRaises(ProfileError):
            tool.render_profile(self.profile, tool.render_command(CANDIDATE_TOKENS))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_list_and_show(self):
        code, out, _ = self.run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn(PROFILE_ID, out)
        code, out, _ = self.run_cli("show", PROFILE_ID)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), tool.load_profiles()[PROFILE_ID])

    def test_render_prints_the_command_and_leaves_the_input_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            command = Path(folder) / "control.txt"
            command.write_text(CONTROL_TEXT, encoding="utf-8")
            before = command.read_bytes()
            code, out, err = self.run_cli("render", PROFILE_ID, "--command-file", str(command))
            self.assertEqual((code, err), (0, ""))
            self.assertEqual(tool.split_command(body(out)), CANDIDATE_TOKENS)
            self.assertEqual(command.read_bytes(), before)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ["control.txt"])

    def test_failures_exit_with_code_two_and_a_message(self):
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "changed.txt"
            changed.write_text(CONTROL_TEXT.replace("-ub 4096", "-ub 512"), encoding="utf-8")
            cases = [("render", PROFILE_ID, "--command-file", str(changed)),
                     ("render", PROFILE_ID, "--command-file", str(Path(folder) / "missing.txt")),
                     ("render", "unknown-profile", "--command-file", str(changed)),
                     ("show", "unknown-profile")]
            for args in cases:
                with self.subTest(args=args[:2]):
                    code, out, err = self.run_cli(*args)
                    self.assertEqual((code, out), (2, ""))
                    self.assertTrue(err.startswith("profile_command: "))
            self.assertIn(PROFILE_ID, self.run_cli("show", "unknown-profile")[2])

    def test_profiles_file_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profiles.json"
            path.write_text(json.dumps({"schema": 1, "profiles": [{"id": "only", "title": "Only one"}]}), encoding="utf-8")
            code, out, _ = self.run_cli("--profiles", str(path), "list")
            self.assertEqual((code, out), (0, "only: Only one\n"))


class ShippedDataTests(unittest.TestCase):
    METRICS = {
        "total_s": lambda run: run["median_total_seconds"],
        "ready_memory_mib": lambda run: run["ready_memory_mib"],
        "cached_short_wall_s": lambda run: statistics.median(item["wall_s"] for item in run["cases"]["cached_guard"]),
        "generation_512_wall_s": lambda run: statistics.median(item["wall_s"] for item in run["cases"]["generation_guard"]),
    }

    def setUp(self):
        self.profile = tool.load_profiles()[PROFILE_ID]
        self.speed = json.loads((EVIDENCE / "long-nemotron-nospec.json").read_text())

    def test_measured_numbers_match_the_speed_evidence(self):
        measured, comparison = self.profile["measured"], self.speed["comparison"]
        self.assertTrue(self.speed["qualified"] and comparison["passed"])
        self.assertAlmostEqual(measured["control_reload_drift_percent"], comparison["control_reload_drift_percent"], delta=0.0006)
        self.assertEqual([(pair["control_run"], pair["candidate_run"]) for pair in measured["pairs"]], [(0, 1), (3, 2)])
        for pair, reference in zip(measured["pairs"], comparison["pairs"]):
            self.assertTrue(reference["separated"])
            self.assertAlmostEqual(pair["latency_reduction_percent"], reference["latency_reduction_percent"], delta=0.0006)
            runs = (self.speed["runs"][pair["control_run"]], self.speed["runs"][pair["candidate_run"]])
            self.assertEqual([self.speed["protocol"]["order"][pair["control_run"]], self.speed["protocol"]["order"][pair["candidate_run"]]],
                             ["control", "candidate"])
            for key, metric in self.METRICS.items():
                for stated, run in zip(pair[key], runs):
                    with self.subTest(key=key):
                        self.assertAlmostEqual(stated, metric(run), delta=0.0006)

    def test_stated_tradeoffs_follow_from_the_numbers(self):
        for pair in self.profile["measured"]["pairs"]:
            self.assertGreaterEqual(pair["latency_reduction_percent"], 3.0)
            self.assertGreater(pair["cached_short_wall_s"][1], 1.5 * pair["cached_short_wall_s"][0])
            self.assertGreater(pair["generation_512_wall_s"][1], 1.4 * pair["generation_512_wall_s"][0])
            self.assertGreater(pair["ready_memory_mib"][0] - pair["ready_memory_mib"][1], 4500)

    def test_quality_statements_match_the_release_records(self):
        strict = json.loads((EVIDENCE / "nemotron-release-gates.json").read_text())
        revised = json.loads((EVIDENCE / "nemotron-release-gates-non-inferiority.json").read_text())
        quality = revised["quality"]
        self.assertFalse(strict["qualified"])
        self.assertTrue(revised["qualified"])
        self.assertFalse(revised["deployment_changed"])
        self.assertTrue(quality["identical_outputs"])
        self.assertEqual(quality["lost_control_passes"], [])
        self.assertEqual(quality["control_failed_checks"], quality["candidate_failed_checks"])
        self.assertTrue(quality["control_failed_checks"])
        self.assertTrue(all(name.endswith("locale_json") for name in quality["control_failed_checks"]))

    def test_evidence_files_exist(self):
        for path in self.profile["evidence"]:
            with self.subTest(path=path):
                self.assertTrue((RTX / path).is_file())

    def test_shipped_files_are_ascii(self):
        for path in (PROFILE_FILE, TOOL_FILE, Path(__file__)):
            with self.subTest(path=path.name):
                path.read_bytes().decode("ascii")


if __name__ == "__main__":
    unittest.main()
