import json, os, sys, time, random, urllib.request

URL = "http://127.0.0.1:9931/completion"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
random.seed(1234)
WORDS = ("alpha beta gamma delta kernel tensor memory bandwidth cache expert router token prefill decode "
         "latency throughput matrix vector quant block scale warp thread shared global unified page fault "
         "graph stream launch fusion attention softmax norm rope embed vocab sample draft accept reject").split()


def text_of(n_words, salt):
    r = random.Random(salt)
    return " ".join(r.choice(WORDS) for _ in range(n_words))


def call(prompt, n_predict, extra=None):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "cache_prompt": False, "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    t = out.get("timings", {})
    return {
        "prompt_n": t.get("prompt_n"), "pp_tps": round(t.get("prompt_per_second", 0), 1),
        "gen_n": t.get("predicted_n"), "tg_tps": round(t.get("predicted_per_second", 0), 2),
        "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
        "wall_s": round(wall, 2),
    }


def main():
    results = []
    code_prompt = ("Write a complete, well-commented C++ implementation of a thread-safe LRU cache with "
                   "templated key and value types, including unit tests using only the standard library.\n\n")
    prose_prompt = "Explain in detail how a CPU cache hierarchy works, from L1 to DRAM.\n\n"
    # warmup
    call("Hello", 8)
    for name, prompt in [("decode-code", code_prompt), ("decode-prose", prose_prompt)]:
        for i in range(2):
            r = call(prompt, 256)
            r["case"] = f"{name}#{i}"
            results.append(r)
            print(json.dumps(r), flush=True)
    for n_words, label in [(1500, "pp~2K"), (6000, "pp~8K"), (24000, "pp~32K")]:
        r = call(text_of(n_words, n_words) + "\nSummarize:", 64)
        r["case"] = label
        results.append(r)
        print(json.dumps(r), flush=True)
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "bench_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
