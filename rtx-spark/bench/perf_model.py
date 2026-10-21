"""Bandwidth / speculative-decoding cost model for MoE decode on NVIDIA N1X (unified LPDDR5X).

All inputs come from the local GGUF geometry (gguf_geometry.py) and live llama-server timings.
"""

BW_THEORETICAL = 9400e6 * 256 / 8  # LPDDR5X-9400 x 256-bit (GB10-class bus width; inferred) = 300.8 GB/s

# Nemotron-3.5-Super-VL-120B-A12B Q4_K_M (decode-path bytes per token, MTP layer excluded)
E, K = 512, 22
MOE_LAYERS = 40
EXPERT_BYTES = (792_723_456 * 2) / E          # up_exps(Q4_K) + down_exps(IQ4_NL) per expert per layer
ROUTED_1 = K * EXPERT_BYTES * MOE_LAYERS     # routed experts for 1 token
DENSE = (1.135 + 2.465 + 0.16 + 0.336 + 0.19 + 0.44) * 1e9  # shared exp + mamba in/out + attn + router + latent + lm_head
BYTES_1 = DENSE + ROUTED_1


def unique_experts(batch, corr=0.0):
    """Expected distinct experts per layer for `batch` tokens. corr in [0,1) shrinks diversity
    (0 = independent routing = worst case; real routing is correlated between adjacent tokens)."""
    eff = 1 + (batch - 1) * (1 - corr)
    return E * (1 - (1 - K / E) ** eff)


def step_bytes(batch, corr):
    return DENSE + unique_experts(batch, corr) * EXPERT_BYTES * MOE_LAYERS


def tokens_per_step(alpha, n):
    return (1 - alpha ** (n + 1)) / (1 - alpha)


def main():
    print(f"theoretical BW            : {BW_THEORETICAL/1e9:6.1f} GB/s")
    print(f"dense bytes/token         : {DENSE/1e9:6.2f} GB")
    print(f"routed bytes/token        : {ROUTED_1/1e9:6.2f} GB  ({K} of {E} experts, {EXPERT_BYTES/1e6:.2f} MB each, {MOE_LAYERS} layers)")
    print(f"total bytes/token (no spec): {BYTES_1/1e9:6.2f} GB -> ceiling {BW_THEORETICAL/BYTES_1:5.1f} tok/s at 100% BW")

    observed_step_ms = 90.0   # measured: n_max=5 (verify batch 6), code/prose/random all ~89-90 ms per step
    t_draft_ms = 8.0          # DFlash 1.2 GB NVFP4 block draft (inferred)
    print("\nInferred no-spec decode from observed 90 ms spec step:")
    for corr in (0.0, 0.3, 0.5):
        ratio = step_bytes(6, corr) / BYTES_1
        t_base = (observed_step_ms - t_draft_ms) / ratio
        bw_eff = BYTES_1 / (t_base / 1e3)
        print(f"  routing corr={corr:.1f}: uniq experts@6={unique_experts(6, corr):5.1f}  cost ratio={ratio:4.2f}  "
              f"t_base={t_base:5.1f} ms -> {1e3/t_base:5.1f} tok/s  (eff BW {bw_eff/1e9:5.0f} GB/s = {bw_eff/BW_THEORETICAL:4.0%})")

    print("\nThroughput vs draft length n_max (tok/s), corr=0.3 model:")
    corr = 0.3
    t_base = (observed_step_ms - t_draft_ms) / (step_bytes(6, corr) / BYTES_1)
    alphas = {"random/novel text a=0.35": 0.35, "code a=0.73": 0.73, "prose a=0.79": 0.79, "agent turns a=0.86": 0.86}
    header = "  n_max  " + "  ".join(f"{k:>24s}" for k in alphas)
    print(header)
    print(f"  {0:5d}  " + "  ".join(f"{1e3/t_base:24.1f}" for _ in alphas))
    for n in range(1, 8):
        cost = t_base * step_bytes(n + 1, corr) / BYTES_1 + t_draft_ms
        row = [tokens_per_step(a, n) / cost * 1e3 for a in alphas.values()]
        print(f"  {n:5d}  " + "  ".join(f"{v:24.1f}" for v in row))


if __name__ == "__main__":
    main()
