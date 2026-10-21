import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_server import RESULTS, call, text_of

prompts = [
    "Write a complete, well-commented C++ implementation of a thread-safe LRU cache.\n\n",
    "Explain in detail how a CPU cache hierarchy works, from L1 to DRAM.\n\n",
    "List the planets of the solar system with one fact each.\n\n",
    "def fibonacci(n):\n",
    "The quick brown fox",
    "Translate to French: The weather is nice today and we are going to the beach.\n",
]
rows = []
for depth_words in (0, 6000, 20000):
    step_ms = []
    for i, p in enumerate(prompts):
        prefix = (text_of(depth_words, 1000 + i) + "\n") if depth_words else ""
        body_extra = {}
        r = call(prefix + p, 2, body_extra)
        # predicted_ms is not returned by call(); recompute from tg_tps and gen_n
        if r["tg_tps"]:
            step_ms.append(1000.0 * r["gen_n"] / r["tg_tps"])
        rows.append({"depth_words": depth_words, "prompt_n": r["prompt_n"], "gen_n": r["gen_n"],
                     "draft_n": r["draft_n"], "tg_tps_reported": r["tg_tps"]})
        print(json.dumps(rows[-1]), flush=True)
        if depth_words >= 20000 and i >= 2:
            break
    print(f"depth~{depth_words} words: reported gen window ms median={statistics.median(step_ms):.1f} "
          f"(n={len(step_ms)}), implied no-spec decode ~{1000.0/statistics.median(step_ms):.1f} tok/s "
          f"(one batch-1 step per request)", flush=True)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "nospec_step.json"), "w"), indent=1)

