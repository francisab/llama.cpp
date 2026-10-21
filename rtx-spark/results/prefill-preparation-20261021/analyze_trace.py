"""Reanalyse the frozen prefill capture; fail if its expected execution shape changes."""

import argparse
import bisect
import collections
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def union_ns(intervals):
    total, end = 0, 0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative trace duration")
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(path):
    if not __debug__:
        raise RuntimeError("run without -O: capture-shape assertions are required")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        kernels = list(db.execute(
            "SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"))
        runtime = list(db.execute(
            "SELECT r.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"))

    target = [r for r in kernels if r["streamId"] == 18]
    helpers = [i for i, r in enumerate(target) if r["name"].startswith("void mm_ids_helper<")]
    assert len(helpers) == 6 * 48 * 3, len(helpers)
    assert len({target[i]["globalPid"] for i in helpers}) == 1
    launches = collections.defaultdict(list)
    for row in runtime:
        if row["name"].startswith("cudaLaunchKernel"):
            launches[row["correlationId"]].append(row)
    runtime_starts = [row["start"] for row in runtime]

    categories = collections.defaultdict(list)
    types = collections.Counter()
    batches = []
    within_submission = collections.Counter()
    removable = []
    glu = []
    for batch in range(6):
        indices = helpers[batch * 144:(batch + 1) * 144]
        counts = {target[i]["dynamicSharedMemory"] // 4 for i in indices}
        assert len(counts) == 1
        tokens = counts.pop()
        assert tokens in (512, 170)
        batches.append({"tokens": tokens, "projections": len(indices)})
        for slot, i in enumerate(indices):
            route, quant, matmul = target[i:i + 3]
            assert route["gridX"] == 512 and route["blockX"] == 32
            assert route["dynamicSharedMemory"] == tokens * 4
            assert quant["name"].startswith("void quantize_mmq_q8_1<")
            assert ("(bool)1>" in quant["name"]) == (slot % 3 != 2)
            match = re.match(r"void mul_mat_q<\(ggml_type\)(\d+),", matmul["name"])
            assert match, matmul["name"]
            types[int(match[1])] += 1
            for name, row in (("routing", route), ("quantization", quant), ("routed_mmq", matmul)):
                assert row["graphId"] is None and row["graphNodeId"] is None
                categories[name].append((row["start"], row["end"]))
            # Zero-cost dual-map generation is optimistic: down needs a forward map, gate/up an inverse map.
            if slot % 3 != 0:
                removable.append((route["start"], route["end"]))
            if slot % 3 == 1:
                removable.append((quant["start"], quant["end"]))
                epilogue = target[i + 3]
                assert epilogue["name"].startswith("void unary_gated_op_kernel<&op_silu,")
                glu.append((epilogue["start"], epilogue["end"]))

            calls = [launches[r["correlationId"]] for r in (route, quant, matmul)]
            assert all(len(c) == 1 for c in calls)
            first, middle, last = [c[0] for c in calls]
            assert first["globalTid"] == middle["globalTid"] == last["globalTid"]
            assert all(c[0]["returnValue"] == 0 for c in calls)
            a = bisect.bisect_left(runtime_starts, first["start"])
            b = bisect.bisect_right(runtime_starts, last["end"])
            for call in runtime[a:b]:
                if call["globalTid"] == first["globalTid"]:
                    within_submission[call["name"]] += 1

    assert sum(b["tokens"] for b in batches) == 2730
    assert types == {21: 564, 23: 12, 20: 258, 8: 30}, types
    start = target[helpers[0]]["start"]
    end = target[helpers[-1] + 2]["end"]
    selected = [r for r in target if r["start"] >= start and r["end"] <= end]
    busy = union_ns((r["start"], r["end"]) for r in selected)
    assert busy > 0 and busy <= end - start
    categories["glu"] = glu
    prep = union_ns(categories["routing"] + categories["quantization"])
    saved = union_ns(removable)
    sync_or_copy = {k: v for k, v in within_submission.items()
                    if any(s in k for s in ("Memcpy", "Synchronize", "WaitEvent", "EventRecord"))}
    with path.open("rb") as source:
        trace_hash = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "trace_sha256": trace_hash,
        "target_stream": 18,
        "batches": batches,
        "mmq_type_counts": {"IQ3_S": types[21], "IQ4_XS": types[23], "IQ4_NL": types[20], "Q8_0": types[8]},
        "selection": {"first_routing_start_ns": start, "last_routed_mmq_end_ns": end,
                      "target_kernel_count": len(selected), "gpu_busy_ms": busy / 1e6,
                      "elapsed_span_ms": (end - start) / 1e6},
        "categories": {name: {"count": len(intervals), "gpu_ms": union_ns(intervals) / 1e6,
                              "busy_share": union_ns(intervals) / busy}
                       for name, intervals in categories.items()},
        "cpu_api_calls_within_triplet_submission": dict(sorted(within_submission.items())),
        "sync_or_copy_calls_within_triplet_submission": sync_or_copy,
        "optimistic_preparation_reuse": {
            "removed_gpu_ms": saved / 1e6, "throughput_equivalent": busy / (busy - saved) - 1,
            "all_preparation_removed_equivalent": busy / (busy - prep) - 1,
            "passes_3pct_triage": busy / (busy - saved) - 1 >= 0.03},
        "limits": [
            "Historical software-instrumented capture, not an independent new sample.",
            "Selection starts at first routed helper and ends at last routed MMQ; it omits input and final tail work.",
            "Target stream only; draft, startup, small-batch decode, CPU work and GPU idle time are excluded from the busy denominator.",
            "GPU-work removal is an optimistic accounting exercise, not measured end-to-end acceleration.",
            "The GPU-only estimate does not bound possible host-launch savings, nor gains from changing the matrix arithmetic.",
            "Zero sync/copy calls within MMQ submission does not mean the full model has no synchronization."
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.trace)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))
