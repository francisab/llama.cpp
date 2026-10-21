import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from analyze_trace import analyze, union_ns


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "capture.sqlite"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, streamId INTEGER, globalPid INTEGER,
                    correlationId INTEGER, demangledName INTEGER, gridX INTEGER,
                    blockX INTEGER, dynamicSharedMemory INTEGER, graphId INTEGER, graphNodeId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                    start INTEGER, end INTEGER, correlationId INTEGER,
                    nameId INTEGER, globalTid INTEGER, returnValue INTEGER);
            """)
            symbols = [
                "void mm_ids_helper<(int)10>",
                "void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, (bool)1>",
                "void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, (bool)0>",
                "void mul_mat_q<(ggml_type)21,",
                "void mul_mat_q<(ggml_type)23,",
                "void mul_mat_q<(ggml_type)20,",
                "void mul_mat_q<(ggml_type)8,",
                "void unary_gated_op_kernel<&op_silu, float>",
                "cudaLaunchKernel_v7000", "cudaStreamSynchronize_v3020"
            ]
            db.executemany("INSERT INTO StringIds VALUES (?,?)", enumerate(symbols))
            clock, correlation = 1000, 1
            for tokens in (512, 512, 512, 512, 170, 512):
                for layer in range(48):
                    for slot in range(3):
                        weight = (4 if layer == 2 else 3) if slot < 2 else (6 if layer < 5 else 5)
                        for symbol, duration in ((0, 10), (1 if slot < 2 else 2, 20), (weight, 1000)):
                            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                       (clock, clock + duration, 18, 123, correlation, symbol,
                                        512, 32, tokens * 4 if symbol == 0 else 0, None, None))
                            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,?)",
                                       (clock - 3, clock - 1, correlation, 8, 456, 0))
                            clock += duration + 10
                            correlation += 1
                        if slot == 1:
                            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                       (clock, clock + 30, 18, 123, correlation, 7, 1, 32, 0, None, None))
                            clock += 40
                            correlation += 1
            # A malformed helper in the draft stream must not enter target accounting.
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (1050, 1060, 20, 123, 99999, 0, 512, 32, 2048, None, None))

    def test_counts_and_disjoint_accounting(self):
        result = analyze(self.path)
        self.assertEqual(result["mmq_type_counts"], {"IQ3_S": 564, "IQ4_XS": 12, "IQ4_NL": 258, "Q8_0": 30})
        self.assertEqual(result["categories"]["routing"]["count"], 864)
        self.assertEqual(result["categories"]["glu"]["count"], 288)
        self.assertAlmostEqual(result["selection"]["gpu_busy_ms"], (864 * 1030 + 288 * 30) / 1e6)
        self.assertAlmostEqual(result["optimistic_preparation_reuse"]["removed_gpu_ms"], 288 * 40 / 1e6)
        self.assertEqual(result["sync_or_copy_calls_within_triplet_submission"], {})

    def test_sync_call_is_reported(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1001,1002,99998,9,456,0)")
        result = analyze(self.path)
        self.assertEqual(result["sync_or_copy_calls_within_triplet_submission"],
                         {"cudaStreamSynchronize_v3020": 1})

    def test_missing_helper_fails(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1")
        with self.assertRaises(AssertionError):
            analyze(self.path)

    def test_wrong_quantization_mapping_fails(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=2 WHERE correlationId=2")
        with self.assertRaises(AssertionError):
            analyze(self.path)

    def test_failed_launch_fails(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE correlationId=2")
        with self.assertRaises(AssertionError):
            analyze(self.path)

    def test_interval_union(self):
        self.assertEqual(union_ns([(10, 20), (0, 15), (3, 5), (21, 23)]), 22)
        self.assertEqual(union_ns([]), 0)
        with self.assertRaises(ValueError):
            union_ns([(5, 4)])


if __name__ == "__main__":
    unittest.main()
