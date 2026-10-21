import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_server import RESULTS, call, text_of

code_prompt = ("Write a complete, well-commented C++ implementation of a thread-safe LRU cache with "
               "templated key and value types, including unit tests using only the standard library.\n\n")
prose_prompt = "Explain in detail how a CPU cache hierarchy works, from L1 to DRAM.\n\n"
rand_prompt = text_of(1500, 1500) + "\nSummarize:"

variants = [
    ("n_max=0", {"speculative.n_max": 0}),
    ("n_max=3", {"speculative.n_max": 3}),
    ("p_min=0.5", {"speculative.p_min": 0.5}),
    ("n_max=3,p_min=0.5", {"speculative.n_max": 3, "speculative.p_min": 0.5}),
    ("n_max=2,p_min=0.75", {"speculative.n_max": 2, "speculative.p_min": 0.75}),
]
results = []
for vname, extra in variants:
    for name, prompt, n in [("code", code_prompt, 256), ("prose", prose_prompt, 256), ("random2K", rand_prompt, 128)]:
        r = call(prompt, n, extra)
        r["case"] = f"{name}/{vname}"
        results.append(r)
        print(json.dumps(r), flush=True)
json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "spec_tune.json"), "w"), indent=1)

