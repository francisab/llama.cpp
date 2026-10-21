import sys, os, glob, re, collections
from gguf import GGUFReader

def resolve(p):
    return os.path.realpath(p)

def analyze(paths, label):
    kv = {}
    tensors = []
    for p in paths:
        r = GGUFReader(resolve(p), "r")
        if not kv:
            for f in r.fields.values():
                if f.name.startswith("tokenizer.") or f.name.startswith("general.") and f.name not in ("general.architecture",):
                    continue
                try:
                    v = f.contents()
                except Exception:
                    continue
                if isinstance(v, list) and len(v) > 8:
                    v = f"list[{len(v)}] first={v[:4]}"
                kv[f.name] = v
        for t in r.tensors:
            tensors.append((t.name, int(t.n_bytes), t.tensor_type.name, [int(x) for x in t.shape]))
    print(f"===== {label} =====")
    for k in sorted(kv):
        print(f"  {k} = {kv[k]}")
    total = sum(b for _, b, _, _ in tensors)
    print(f"  tensors={len(tensors)} total_bytes={total/1e9:.2f} GB")
    groups = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for n, b, ty, sh in tensors:
        key = re.sub(r"blk\.\d+\.", "blk.N.", n)
        g = groups[key]; g[0] += 1; g[1] += b; g[2][ty] += 1
    for key, (cnt, b, tys) in sorted(groups.items(), key=lambda x: -x[1][1])[:30]:
        print(f"  {key:45s} n={cnt:3d} {b/1e9:8.3f} GB  {dict(tys)}")
    return kv, tensors

if __name__ == "__main__":
    hub = os.path.expanduser(r"~\.cache\huggingface\hub")
    nem = sorted(glob.glob(hub + r"\models--lmstudio-community--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF\snapshots\*\*.gguf"))
    qfn = sorted(glob.glob(hub + r"\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\*\UD-IQ4_XS\*.gguf"))
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("nem", "both"):
        analyze(nem, "Nemotron-3.5-Super-VL-120B-A12B Q4_K_M")
    if which in ("qfn", "both"):
        analyze(qfn, "Qwen3.8-Flash-Next UD-IQ4_XS")
