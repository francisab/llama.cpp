# RTX Spark engine work — context handoff

This folder carries everything from the chat "Strata llama.cpp performance study" (2026-10-19/20). Start here, then read `PLAN.md`.

## Goal
Build an inference engine specialized for **NVIDIA RTX Spark (N1X)** from this llama.cpp fork. Learn from Niko1221/Strata, robertkarl/UnbiasedStrata and GenerelSchwerz/llama.cpp, but keep **upstream llama.cpp as the base**: pinned, patches as commits, shipped as a drop-in `ggml-cuda.dll` + `llama-server.exe`.

**Priority: en-US chat/prose, coding, structured output and English documents.** Other languages are informative, not promotion gates. Preserve full target vocabulary and fallback; do not make the engine English-only.

## Machine (verified)
- **Device:** NVIDIA RTX Spark N1X: 20-core ARM64 CPU (dotprod/i8mm/bf16, no SVE), 6144-core Blackwell GPU, **sm_121**.
- **Memory:** 128 GB LPDDR5X-9400 on a 256-bit bus, **≈300 GB/s**, unified.
  - Boot carve-out: **96 GB GPU / 30.5 GB Windows**. Bandwidth is the same either way; only capacity moves.
- **OS/driver:** Windows 11 ARM64, CUDA 13.4 UMD; 45–80 W SoC (SW power-cap events under prefill).
- **llama.cpp install:** `C:\llama.cpp` = official `win-cuda-13.4-arm64` release, now **b11756** (`d80619397`); its `ggml-cuda.dll` already has native `sm_121a`.
- **Local tool availability:** `cmake`, `nvcc`, `nsys`, and `ncu` were not found on PATH; no compiler or profiler was installed for this work. Upstream CI cross-compiles `ggml-cuda` on an x64 host: `cmake/arm64-windows-msvc-cuda.cmake`, `-DGGML_BACKEND_DL=ON -DGGML_CPU=OFF -DGGML_CUDA=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3`.
- **Models are launched by a local launcher app**, normally on port **9931**, with restart-on-crash. Benchmarks here use their own servers on port **9940** and stop them with a real Ctrl+C.

## Models (local HF cache; paths in untracked `results/model_paths.json`, template in `results/model_paths.example.json`)

| Model | Type | File | Bytes per decoded token |
|---|---|---|---|
| Qwen3.8-Flash-Next UD-IQ4_XS (+ `mtp-…-Q8_0` MTP draft) | qwen4exp MoE, 512 experts, 10 active; hyper-connections; 28.8 GB PLE table (lazy, `-lzm on`) | 87 GiB | 5.83 GB |
| NVIDIA Nemotron-3.5-Super-VL-120B-A12B Q4_K_M (+ DFlash NVFP4 draft) | nemotron_h_moe: Mamba-2 + 9 attention layers + latent MoE, 512 experts, 22 active; built-in MTP layer | 65 GiB | 7.45 GB |
| Qwen3.8-27B UD-Q4_K_M (+ DFlash2 draft) | qwen35 dense hybrid (Gated-DeltaNet + full attention every 4th layer) | 15.3 GiB | ≈15.6 GB |

## Initial repeated results (b11756, five-repeat medians, tok/s)

Server decode uses fixed output lengths and the real-use context/projector settings. Prefill is the separate text-only native pp2048 test at depth zero. Exact settings and the complete tables are in `PLAN.md` section 2.7.

| Model | Measured config | Server plain | Code256 | Prose256 / 512 | Random-word128 | Native pp2048 |
|---|---|---:|---:|---:|---:|---:|
| Flash-Next | MTP3, Q8_0 draft, ub512, lazy PLE | 31.39-32.14 | 47.42 | 44.81 / 47.25 | 38.88 | 871.76 |
| Nemotron | Native MTP3, **default p_min=0**, ub4096 | 30.08-30.41 | 46.06 | 45.01 / 46.18 | 42.50 | 891.78 |
| Qwen3.8-27B | DFlash7 for code; DFlash4 for other columns | 15.52-15.73 | 44.44 | 35.06 / 38.33 | 34.85 | 922.23 |

**The earlier blanket Nemotron `p_min=0.5` recommendation is withdrawn.** It was slower than default MTP on all four fixed-length cases. The natural-EOS follow-up also favored default MTP (42.99 versus 32.15 tok/s), though one cut-off response stopped after 36 tokens. The original low-acceptance failure did not reproduce on this build and protocol.

The 27B's two draft depths produced identical tokens on all 20 paired requests: depth four was 5.71% slower for code, 2.66-3.45% faster for prose, and 20.07% faster on the random-word case. Plain versus speculative outputs are not generally identical on any of the three models, so these timings are not quality validation.

**Initial model-only improvement:** Flash-Next's external MTP output head at Q4_K reaches 49.65 / 47.02 / 48.26 / 41.58 tok/s on the four cases, 2.71-6.50% above a repeated Q8_0 control. It saves 303.125 MiB and preserves all 33 other tensors byte-for-byte. The broader validation below supersedes any general speed or quality claim based only on these four prompts.

## Execution findings and limits

**Executed English follow-up:** both heads pass complete Python code (231 assertions per answer), exact US-format JSON, rare identifier/number copying, and known-answer retrieval from a chat-templated 17,225-token English document. The initial Q8 code answer hit the token cap; only code was retried on both heads with a larger matched budget. The original failed result is preserved.

The subsequent five-repeat English/code timing pair gives **Q4_K +4.50% geometric-mean decode throughput**, with per-case gains of 3.21-6.24% and identical tokens in all 40 measured outputs. The smaller draft saves 303.125 MiB and is supported as an **opt-in candidate at MTP3/ub512**, not a universal/default replacement. Earlier reload gains were smaller; no guaranteed uplift, general writing-quality pass or 262K-context claim follows from these checks.

`PLAN.md` section 2.11 and `results/en-us-quality-20261020/final_summary.json` contain the final scores, timing table and audit; `final_manifest.json` records provenance. Six launches completed (including targeted retries), with 50 measured requests and 18 warm-ups. No more GPU jobs are queued and no deployed settings changed.

**Draft-vocabulary screen:** the proxy shortlist study in `PLAN.md` section 2.12 shows that 65,536 ordinary tokens plus all 276 non-normal tokenizer entries cover 99.12% of captured timing tokens and 99.53% of the compact quality-gate tokens. This would retain about 179 MB of the 675 MB Q8_0 output head by row-size estimate. The selection corpus is local llama.cpp code/docs, not a representative independent en-US dataset; coverage is not speculative acceptance or speed. No `d2t` implementation is justified yet. The result files in `results/draft-vocab-feasibility-20261020/` include the ranked candidate IDs and source hashes.

**Prior scope-only reanalysis:** reusing the captured Rust, SQL/Python, JSON and chat cases gave Q4_K +1.73% at fixed length and +3.42% in natural-EOS mode. MTP2 was 4.80% slower than MTP3; MTP4 gained only 0.88% overall while hurting chat. Keep MTP3 rather than an adaptive controller. Details are in `PLAN.md` section 2.10 and `en_us_summary.json`.

The following preserves the original multilingual validation context:

The Q8/Q4/Q8/Q4 reload comparison gives a smaller, workload-dependent Q4 gain: **1.70% equal-weight geometric mean across the six short cases**, with individual changes from -0.41% to +3.56%. All 120 short-case outputs match across those four launches. A separate natural-EOS pair gives +3.46% across the six short cases. Launch-to-launch variation is comparable to some gains; Q4 remains an experimental alternative, not a blanket replacement.

The original raw 17K-token documentation prompt is **not a passing long-context quality test**: Flash-Next immediately emits EOS in natural mode and varies its output in forced mode. It remains excluded. The bounded follow-up above addresses selected code, identifiers and chat-templated document retrieval; an uninterrupted long soak and broader quality remain outside that scope.

Nemotron's default MTP3 beats `p_min=0.5` on all seven held-out cases in both length modes. The cut-off loses 1.2-12.9% in fixed-length tests and does not preserve every output. Keep default MTP3.

The low-ID vocabulary-prefix experiment rules out a naive universal prefix: IDs below 32768 cover only about 40% of Chinese output tokens. That is no longer a blocker for the English-first goal. The English/code slice itself has only 87.50-96.88% coverage at 32K: prefer an English/code-selected draft shortlist with required control tokens and full-target fallback, not an ID cutoff. No actual reduced-head acceptance or speedup has been measured.

Full measurements, the depth comparison, and targeted diagnostics are in `PLAN.md` section 2.9 and `results/plan-20261020/`. Operator timing and routed-expert overlap remain unavailable; do not claim a grouped-kernel gain from aggregate timings.

The completed depth sweep has no universal winner: MTP4 improves Rust by 6.41% over MTP3 but loses 7.52% on chat; MTP2 wins the Chinese case with a different continuation. A hindsight per-case selector offers only 2.87% over the best fixed depth before switching overhead. Keep MTP3 as the conservative default; no adaptive controller was implemented.

The bounded Flash-Next diagnostic shows that existing CUDA graphs and fusion are already valuable: plain tg128 is 30.26 tok/s, versus 15.01 without graphs and 25.58 without fusion. Increasing prefill ub512 to ub2048 gives only +1.82%, within observed variation. Keep graphs/fusion enabled and ub512. These checks do not identify a new kernel optimization.

This pass finished 420 measured server requests and 25 native samples. `execution_summary.json` contains the audited comparisons; `execution_manifest.json` records the validation-pass snapshot. The later `en_us_summary.json`, `en_us_tokens.json` and `en_us_manifest.json` preserve the English/code reanalysis and updated source provenance. The wider verify/lazy/all-model proxy sweep was cancelled rather than extending the benchmark campaign.

**Facts the plan depends on:**
- **Historical speculation cost model, not an operator profile:**
  - MoE: a speculative step costs **≈1.18 + 0.30·n_draft** plain steps.
  - Dense: **≈1.1 + 0.06·n_draft**.
  - Causes on MoE: each MTP draft pass reads a full-vocabulary head (Flash-Next 0.675 GB Q8_0; Nemotron 0.44 GB Q6_K), and each verified token pulls its own experts.
- **Confidence policies need held-out evidence.** The synthetic random-word prompt now has about 72% draft acceptance under Nemotron's default MTP; it is not a reliable general low-acceptance test.
- **`-ub 4096` with Flash-Next at 262K context fails** (`decode() failed: bad allocation`). `-ub 2048` vs 512 makes no consistent difference.
- **Lazy PLE:** the earlier study estimated a roughly 9% prefill penalty on novel text. The new runs do not control cold OS page-cache state and do not revalidate that bound.
- **Per-request `speculative.*` fields are ignored** by `llama-server` (b11681 and b11756); each config needs a relaunch.
- **Measurement pitfall:** the first decode step after a prompt is ~2× slower (graph switch/capture). Use steady-state numbers only.
- **b11681 → b11756:** kernels ±3% (noise); Nemotron MTP code +17%; 27B prose +5–7%.
- **Reference calibration:** upstream on DGX Spark (GB10, Linux) reaches 87% of bandwidth on dense Qwen2.5-7B Q8 (discussion #16578). This is not a direct control for the Windows ARM64 measurements.

## Current status and next steps

The source review and revised gates are in `PLAN.md` sections 3-5. The baseline has now been repeated against the installed binary and hashed model files. The v2 cumulative speed targets and delivery estimates remain withdrawn because they combined unverified gains and included work already upstream.

1. Stop broad benchmarking. Reuse the captured control and choose one specific bottleneck or configuration change.
2. The compact English gate and matched A/B are complete. Keep the Q4_K draft separate as an opt-in MTP3/ub512 candidate; its exact path and rollback hash are in `results/rebaseline-20261020/head-q4k/quantization.json`.
3. The proxy shortlist screen is complete. The additional fork survey found an existing dynamic-head implementation in [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380): evaluate it before designing static MTP `d2t`. It derives a scorer from head weights, so independent selection data is a requirement for our static alternative, not for this dynamic path. A 2026-10-20 readiness check found the native ARM64 host has no detected Visual Studio/CMake/Ninja/NVIDIA CUDA build toolchain; installed b11756 does not expose the feature, and the open PR has only a labeler check visible. Obtain a matching build first; then compare K=0 with K=16,384 using the same Q4_K draft/MTP3/ub512 setup and the compact English gate.
4. Prefer the source-shortlisted QSA-mask fusion for Flash-Next and FFN MMQ fusion for the dense 27B before new kernel designs. Obtain operator timing and routed IDs before grouped-MoE work. No all-fork/all-model benchmark sweep is queued.

The reviewed checkout already includes CUDA graphs, Blackwell NVFP4/MXFP4 MMQ, qwen4exp hyper-connection fusion, grouped MoE top-k, and PLE row prefetch. Do not port or reimplement those without identifying a target-specific gap. No deployed model or launcher configuration was changed by the measurements.

### Additional fork survey

`PLAN.md` section 3.1 records the 2026-10-20 survey with pinned sources and applicability. **Keep upstream as the base.** The most relevant active proposals are dynamic draft-head selection, QSA-mask fusion, and dense/eligible MoE FFN fusion; their GB10 numbers are sourced leads, not measured RTX Spark gains. The dynamic head retains the original weights and adds a scorer, so it is a compute/bandwidth optimization rather than a smaller model file.

`ik_llama.cpp` is useful for CUDA/quantization algorithms; BeeLlama and TheTom are later KV-compression references. GenerelSchwerz and thecodacus expert caches, and RaymondHuang's KV streaming, mainly address capacity/offload problems. The Spark-specific croll83 fork is deprecated in favor of upstream; EngramHalo is explicitly AMD HIP-focused. These are not qualified drop-in Windows ARM64 replacements.

The survey also confirmed `--spec-draft-n-ubatch` in the installed b11756 binary: basic independent draft workspace tuning needs no fork. It is a memory knob, not a demonstrated speed improvement. No fork was installed or benchmarked, and no model, executable, remote or deployment setting changed. The Windows CUDA cross-build still needs an x64 host; API/graph changes require matching executables and DLLs.

## Folder layout
- `PLAN.md`: full analysis, measurements, reference-project evaluation, roadmap, risks, sources.
- `bench/`: harness.
  - `run_rebaseline.py`: repeated native/server benchmarks, raw samples and generated token IDs, isolated logs, and GPU telemetry. Refuses to overwrite an existing result.
  - `heldout_suite.py`: held-out multilingual/code/chat cases, an English/code-only selection, and a documentation prompt pinned to the measured source commit; the raw long prompt has a known EOS limitation.
  - `en_us_quality.py`: four chat-templated natural-EOS cases and explicit scoring: a bounded Python function with 231 assertions, exact US-format JSON, exact identifier/number copying, and distributed facts in pinned English documentation. Both warm-up and measured responses must pass. Generated code is limited to a single function with no imports, attributes, loops or self-reference before execution.
  - `analyze_mtp_tokens.py`: low-ID token coverage and per-position accepted counts divided by verification steps; `--scope en-us` selects the four English/code cases and fails if any are missing. It does not measure router overlap or implement a vocabulary subset.
  - `run_matrix.py`: launch config → benchmark suite → Ctrl+C; `MATRIX_OUT` env var selects the output file.
  - `run_llamabench.py`: DGX Spark methodology; `LB_SUFFIX` env var.
  - `bench_server.py`: request helper; `URL` global.
  - `bench_ple.py`, `bench_nospec.py`, `bench_spec_*.py`, `summarize_matrix.py`, `gguf_geometry.py`, `perf_model.py`.
- `results/`: JSON/markdown summaries, `geometry.txt`, and `model_paths.example.json`.
  - Scripts read `results/model_paths.json` and write to `results/` (server logs to `results/logs/`), via `RESULTS = bench/../results`. They run from any working directory. Copy `model_paths.example.json` to `model_paths.json` and set local paths.
  - Published files omit raw server logs, full generated responses (`server_*.json`) and the local model path file; local paths are replaced by placeholders such as `<hf_home>` and `<launcher_data>`.
  - `rebaseline-20261020/`: initial and final provenance manifests, repeated native samples, summary statistics, and telemetry.
  - `rebaseline-20261020/head-q4k/`: isolated draft-head conversion evidence and candidate measurements; `head-q8-repeat/` is the repeated control. `head_ab_summary.json` compares both with the original control.
  - `plan-20261020/`: held-out head/reload/EOS, Nemotron policy and MTP2/3/4 comparisons. `bench-targeted/` contains only the bounded Flash-Next diagnostics; the broader proxy matrix was cancelled.
  - `en-us-quality-20261020/`: scored natural-EOS tasks, preserved initial token-cap failure, code-only retries, one matched English speed pair, final audit and provenance.

## Repeated benchmarks

Run these sequentially from the repository root, with no other inference process active. Choose a new output directory for a rerun.

```powershell
python rtx-spark\bench\run_rebaseline.py plain qfn nem q27 --out rtx-spark\results\new-run --repetitions 5
python rtx-spark\bench\run_rebaseline.py server Q0 U1 N1 N2 N4 D1 D2 D3 --out rtx-spark\results\new-run --repetitions 5
python rtx-spark\bench\run_rebaseline.py server N1 N2 N4 --out rtx-spark\results\new-run --repetitions 5 --suite novel --allow-eos
```

The native run records pp2048, tg32, and tg128 at depths 0 and 16384. Server runs use the original code/prose/random-word prompts with `cache_prompt=false`, greedy sampling, `ignore_eos=true`, and exact output-length checks. Each case gets a warm-up, followed by five measured requests with rotated case order. Forced-length throughput is not a quality score and is not directly comparable to the earlier early-EOS low-predictability runs. Raw responses retain generated tokens and effective settings so differences can be inspected.

`--allow-eos` measures natural completion lengths. `--draft-model <path>` replaces an external draft model for an isolated A/B; use a separate output directory and preserve the original model. Neither option changes the deployer's configuration.

For a targeted English/code follow-up, use `--suite en-us --spec-metrics`; it runs only Rust, SQL/Python, JSON and the chat-templated arithmetic question. It skips Chinese, Spanish and the invalid raw long fixture, including its source-loading cost. This small timing suite does not replace complete-task or repaired long-context quality checks. The original `--suite heldout` remains available but is not the default workload for this optimization effort.

`U1`, `U2`, and `U4` mean Flash-Next MTP depths 3, 2, and 4 respectively; there is no `U3` CLI tag. Plain-mode `--bench decode`, `nographs`, and `nofusion` test tg128; `--bench ubatch` tests pp2048 at ub512 and ub2048. `verify` and `lazy` profiles exist but were not run in this execution pass. Do not run every profile for every change.

The compact correctness gate is separate from capped throughput:

```powershell
python rtx-spark\bench\run_rebaseline.py server U1 --out rtx-spark\results\quality-control --suite en-us-quality --allow-eos --spec-metrics --repetitions 1
python rtx-spark\bench\en_us_quality.py rtx-spark\results\quality-control\server_U1_en-us-quality.json --out rtx-spark\results\quality-control\scores.json
```

Use the same command with a fresh output directory and `--draft-model <candidate.gguf>` for the candidate. The gate requires natural EOS, exact JSON fields/values, and complete code behavior; it does not equate token parity with correctness. The long fixture inserts synthetic deployment facts across 64000 characters of pinned documentation and applies the model's chat template. It tests retrieval across approximately 17K tokens, not the configured 262K capacity or general prose quality.

For a narrowly scoped retry, both runner and scorer accept `--cases code` (or other explicit case names). Use a new output directory, keep the failed evidence, and report which source supplies each final score; a partial retry alone is not the full quality gate.

## Conventions
- Never run a test server while the user's deployer model is up (VRAM). Check `Get-Process llama-server` first; stop test servers with Ctrl+C (`run_matrix.ctrl_c`).
- Report steady-state numbers, label each claim measured, sourced or inferred, and keep 5-run medians for gates.
