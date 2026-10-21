import copy
import unittest

from analyze import accepted_prefix, draft, oracle, parse_trace, reconcile


NORMAL = """other server output
spec_cost start slot=0 task=1 t_us=10 gen=1 n_max=510
spec_cost select slot=0 t_us=20 type=draft-mtp hist=896 drafted=3
spec_cost end slot=0 task=1 t_us=70 gen=1 accepted=3 verified=3 emitted=4
"""
REPLAY = """spec_cost start slot=0 task=1 t_us=10 gen=1 n_max=510
spec_cost select slot=0 t_us=20 type=ngram-simple hist=896 drafted=12
spec_cost rollback slot=0 task=1 t_us=30 accepted=6 drafted=12
spec_cost start slot=0 task=1 t_us=40 gen=1 n_max=510
spec_cost end slot=0 task=1 t_us=70 gen=1 accepted=6 verified=7 emitted=8
"""


class LookupTests(unittest.TestCase):
    def test_exact_draft_and_minimum(self):
        history = [999] + list(range(12)) + list(range(100, 112)) + list(range(11))
        self.assertEqual(draft(history, 11), list(range(100, 112)))
        self.assertEqual(draft(history, 11, m=3), [])
        self.assertEqual(draft(list(range(10)), 10), [])

    def test_position_zero_is_ignored(self):
        history = list(range(12)) + list(range(100, 112)) + list(range(11))
        self.assertEqual(draft(history, 11), [])

    def test_short_latest_match_stops_search(self):
        self.assertEqual(draft([999] + [1] * 40, 1), [])

    def test_prefix_stops_at_first_mismatch(self):
        self.assertEqual(accepted_prefix([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(accepted_prefix([1, 2, 3], [1, 2]), 2)


class TraceTests(unittest.TestCase):
    def test_normal_round(self):
        tasks = parse_trace(NORMAL)
        self.assertEqual(len(tasks), 1)
        row = tasks[0][0]
        self.assertEqual((row["cost_us"], row["proposal_us"], row["position"]), (60, 10, 0))
        self.assertFalse(row["rollback"])

    def test_replay_keeps_original_start_and_bonus(self):
        row = parse_trace(REPLAY)[0][0]
        self.assertEqual((row["cost_us"], row["accepted"], row["emitted"]), (60, 6, 8))
        self.assertTrue(row["rollback"])

    def test_invalid_records_fail(self):
        for text in (
            NORMAL.replace("t_us=70", "t_us=19"),
            NORMAL.replace("gen=1 accepted", "gen=2 accepted"),
            NORMAL.replace("slot=0", "slot=1"),
            NORMAL + "spec_cost broken x=1",
            "\n".join(NORMAL.splitlines()[:-1]),
            REPLAY.replace("emitted=8", "emitted=7"),
            REPLAY.replace("verified=7", "verified=12"),
            REPLAY.replace("t_us=30 accepted=6", "t_us=30 accepted=5"),
        ):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                parse_trace(text)

    def test_reconcile_complete_request_and_fail_counter_mismatch(self):
        prompt = list(range(896))
        response = {
            "tokens": list(range(10000, 10512)), "tokens_evaluated": 896,
            "timings": {"predicted_n": 512, "draft_n": 383, "draft_n_accepted": 383},
        }
        rounds = []
        for position in range(0, 511, 4):
            count = min(3, 510 - position)
            rounds.append({"position": position, "hist": 896 + position, "source": "draft-mtp",
                           "drafted": count, "accepted": count, "emitted": count + 1, "rollback": False})
        self.assertEqual(reconcile(rounds, response, prompt, False)["rounds"], 128)
        bad = copy.deepcopy(response)
        bad["timings"]["draft_n_accepted"] -= 1
        with self.assertRaises(AssertionError):
            reconcile(rounds, bad, prompt, False)


class OracleTests(unittest.TestCase):
    def test_expensive_lookup_never_improves(self):
        prompt = [999] + list(range(17)) * 5
        tokens = (list(range(17)) * 4)[:54]
        result = oracle(prompt, tokens, 40, 10000)
        self.assertGreater(result["eligible_positions"], 0)
        self.assertEqual(result["lookup_rounds_selected"], 0)
        self.assertEqual(result["model_gain_percent"], 0)

    def test_cheap_lookup_can_improve(self):
        prompt = [999] + list(range(17)) * 5
        tokens = (list(range(17)) * 4)[:54]
        result = oracle(prompt, tokens, 40, 20)
        self.assertGreater(result["model_gain_percent"], 0)
        self.assertLessEqual(result["lookup_committed_tokens"], len(tokens) - 1)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O")
    unittest.main()
