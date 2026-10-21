import json, os, sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "matrix_results.json")))
cases = ["code256", "prose256", "prose512", "novel2K", "pp2K", "pp8K", "pp25K"]
print(f"{'config':24s} {'load_s':>6s} {'VRAM GiB':>8s} | " + " ".join(f"{c:>15s}" for c in cases))
for name, e in R.items():
    if "error" in e:
        print(f"{name:24s} FAILED: {' | '.join(e.get('log_tail', [])[-3:])[:200]}")
        continue
    rows = {r["case"]: r for r in e["rows"]}
    cells = []
    for c in cases:
        r = rows.get(c)
        if not r or "error" in r:
            cells.append(f"{'-':>15s}")
        elif c.startswith("pp"):
            cells.append(f"{r['pp_tps']:>9.0f} pp/s ")
        else:
            acc = f"{r['draft_acc']}/{r['draft_n']}" if r.get("draft_n") else "-"
            cells.append(f"{r['tg_tps']:>5.1f} {acc:>9s}")
    print(f"{name:24s} {e['load_s']:>6.0f} {e['vram_mib']/1024:>8.1f} | " + " ".join(cells))
