import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_server import RESULTS, call, text_of

code_prompt = ("Write a complete, well-commented C++ implementation of a thread-safe LRU cache with "
               "templated key and value types, including unit tests using only the standard library.\n\n")
prose_prompt = "Explain in detail how a CPU cache hierarchy works, from L1 to DRAM.\n\n"
rand_prompt = text_of(1500, 1500) + "\nSummarize:"

results = []
for spec in ["none", "draft-dflash"]:
    for name, prompt, n in [("code", code_prompt, 256), ("prose", prose_prompt, 256), ("random2K", rand_prompt, 128)]:
        r = call(prompt, n, {"speculative.types": spec})
        r["case"] = f"{name}/spec={spec}"
        results.append(r)
        print(json.dumps(r), flush=True)
json.dump(results, open(os.path.join(RESULTS, "spec_ab.json"), "w"), indent=1)
