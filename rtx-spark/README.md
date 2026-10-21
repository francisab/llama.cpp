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
- **Local tools:** the initial environment lacked a compiler and profiler. Native ARM64 build tools and portable Nsight Systems are now installed; normal-exit CLI CUDA tracing passes the diagnostic gate (section 2.26). Bounded server traces still report incomplete events. Upstream CI cross-compiles `ggml-cuda` on an x64 host: `cmake/arm64-windows-msvc-cuda.cmake`, `-DGGML_BACKEND_DL=ON -DGGML_CPU=OFF -DGGML_CUDA=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3`.
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

Full measurements, the depth comparison, and targeted diagnostics are in `PLAN.md` section 2.9 and `results/plan-20261020/`. Those runs did not collect operator timing or routed-expert overlap. Later full-lifetime CUDA kernel evidence is in section 2.26, and bounded actual-MTP routed IDs are analyzed in section 2.31. Neither aggregate kernel shares nor duplicate selections establish a grouped-kernel speedup.

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

1. Screen one binary, model or configuration change at a time using the fail-fast protocol below. Stop weak candidates before full validation; do not exhaust the upstream-PR queue before trying our own model or code changes.
2. The compact English gate and matched A/B are complete. The Q4_K draft is deployed only on the Flash-Next launcher recipe (`-md <launcher_data>\custom-models\mtp-...-Q4_K.gguf`); a timestamped `catalog.json` backup is kept beside it. Rollback: restore that backup. Its exact path and hash are in `results/rebaseline-20261020/head-q4k/quantization.json`.
3. The proxy shortlist screen is complete. The fork survey found an existing dynamic-head implementation in [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380). A native ARM64 toolchain is now installed (VS 2022 Build Tools ARM64-hosted MSVC, ARM64 CUDA 13.4 redistributables, ARM64 CMake/Ninja; no x64 emulation) and `tools/build_native_cuda.ps1` builds matched control/candidate binaries. The matched A/B (`PLAN.md` section 2.13) passes the compact English gate with token-identical timing outputs and gains +2.63% (geomean; chat flat) at K=16,384 over its merge-base control.
4. **Deployed:** the current release-like build is b11756 + #30380 + #30640 (Clang 20.1.8 CPU side and tools, MSVC CUDA `ggml-cuda.dll`, same 42-file set). The Flash-Next recipe keeps `--spec-draft-vocab-dynamic 16384`; the #30640 deployment did not change the catalog. Against the prior #30380 build, U1 improves +2.43% geomean with 20/20 token-identical outputs and 8/8 quality checks. A timestamped backup and local rollback script are retained. Remove the dynamic-head flag if the install is replaced by an official build without #30380.
5. **Candidate status:** #30640 is deployed (see `PLAN.md` section 2.15). #29948 passed backend tests but did not show a stable 27B gain in repeated runs and does not apply cleanly to b11756 + #30380; it is not deployed (section 2.16). [ggml-org/llama.cpp#30489](https://github.com/ggml-org/llama.cpp/pull/30489) passed 266 CUDA correctness cases, but its opposite-order Flash-Next prefill gains were only 1.73% and 0.40%. [ggml-org/llama.cpp#30804](https://github.com/ggml-org/llama.cpp/pull/30804) passed 1,334 whole-graph cases with one-node chunks; default 512-node chunks gained 2.87% and 1.60% in decode. Neither clears the predeclared 3% gate in both orders, so both are parked without integration or deployment (sections 2.18-2.19). Two Q6 revisions also failed the precision gate (section 2.21). Continue independent model/framework hypotheses, not marginal-patch repetitions or a subset sweep against the same text. Routed-ID evidence is now available, but grouped-MoE work still needs a fusion-preserving design and measured end-to-end benefit.
6. **Local model experiments:** Q3_K and Q2_K draft heads were rejected for lower acceptance and slower decode. A mixed-Q4 draft FFN saved 1,200 MiB but its initial gain did not survive reverse order. Selective Q6_K target projections gained 6.45% on a five-repeat MTP comparison across Rust and JSON and passed all 8 bounded English tasks, with changed timing outputs. The subsequent precision gate passes its perplexity ratio but fails KL divergence: 0.038966 nats versus the predeclared 0.01 limit; unchanged-model calibration is approximately zero. Training-split importance weighting reduced KL to 0.035893; additionally preserving all 36 recurrent QKV projections at original Q8_0 reduced it to 0.033887. Both revisions still fail, so neither advances to speed confirmation or deployment. See `PLAN.md` sections 2.17 and 2.20-2.21.
7. **Runtime/model follow-up:** four generation threads measured -0.70% geomean; keep the defaults. Upstream GDN alpha/beta fusion [ggml-org/llama.cpp#29187](https://github.com/ggml-org/llama.cpp/pull/29187) passes 14 CUDA cases but its source-base MTP result is mixed: Rust +5.01%, JSON -2.81%, with changed outputs. A local Q5_K draft head measured +2.51% geomean with identical outputs, but JSON improves only 1.85%, below the 3% gate. All stopped after one pair, without deployment (sections 2.23-2.25).
8. **Native profiling:** a normally exiting MTP CLI run yields 145,262 CUDA kernels without diagnostic warnings/errors. The largest Q8_0 MMVQ variant accounts for 31.08% of summed kernel duration; RHS quantization is 2.23%. These are full-lifetime, instrumented kernel shares, not pure-decode or end-to-end speed gains. The rejected server traces remain excluded; routed-expert IDs were captured separately later (sections 2.26 and 2.31).
9. **Local kernel screen:** a three-line Q8_0 four-column change from four warps to two passes 21 supported correctness cases per arm and its changed launch is traced. The MTP screen gives Rust -0.10%, JSON +1.14%, and only 2/4 identical outputs, so it stops without deployment or a parameter sweep. The following hardware-counter preflight is permission-blocked (sections 2.27-2.28).
10. **Hardware-counter blocker:** the signed native Nsight Compute CLI runs, but a one-kernel original-runtime probe returns `ERR_NVGPUCTRPERM`. No counters were collected and no driver/access policy was changed. An elevated capture or counter-access change needs human authorization before this diagnostic can proceed (section 2.28).
11. **Unprivileged alternative:** the blanket Q8_0 four-column MMQ switch passes correctness but slows one tested shape by 38.26%, despite a +9.79% operator geomean. Its preregistered row-coverage revision passes 45 correctness cases and boundary dispatch checks, but gives -0.77% geomean on fresh timing shapes, with a worst regression of 12.99%. Both stop before full-model testing, without deployment or further cutoff tuning. These cache-reuse timings are not model-speed claims (sections 2.29-2.30).
12. **Routed overlap:** complete actual-MTP route records show 38.20% Rust / 32.25% JSON duplicate expert selections across four-token verification batches, but only 9.49% / 7.04% at the same rank. Startup and the four-token prompt tail are excluded, with exact token-accounting checks. This supports a bounded screen of the existing expert-packed MMQ path, not a speed claim or a new cache subsystem. The temporary diagnostic DLL is not deployed (section 2.31).
13. **Routed-MMQ result:** the corrected IQ3_S/IQ4_NL candidate gives Rust -13.55%, JSON -16.06%, geomean -14.81%, with 2/4 identical outputs. It is rejected after one pair, without deployment or cutoff/type tuning. The preflight also corrected an earlier label: kernel enum 21 is IQ3_S, not IQ4_XS; raw profile durations were unaffected. Further grouped work needs to preserve fusion or remove other overhead (section 2.32).
14. **Fusion-preserving ordering triage:** ascending expert-ID ordering raises same-rank overlap to 18.34% Rust / 15.23% JSON on the existing routes, still below the 20% opportunity gate. Stop before a CUDA build, without ordering/subset sweeps. This is logical alignment, not measured memory savings or speed (section 2.33).
15. **PDL runtime screen:** disabling the existing programmatic dependent launch setting gives Rust +2.09%, JSON -0.49%, geomean +0.79%, with 4/4 identical outputs and overlapping ranges. Stop after one pair and keep PDL enabled. Only child-process environment values differed; no persistent setting or deployment changed (section 2.34).
16. **Prompt-lookup triage:** the existing stateless 12-token n-gram search finds no proposal in the recorded Rust output and only three weak proposals in JSON, with no fully correct 12-token proposal. Stop this standard-case hybrid screen before a model run; copy-heavy workloads are evaluated separately below. No lookup-length tuning or runtime change follows (section 2.35).
17. **Copy-heavy follow-up:** on fresh frozen Python rename and JSON unit-conversion fixtures, MTP plus stateless 12/12 prompt lookup loses 15.02% and 9.21% respectively (-12.16% geomean), despite 4/4 identical prefixes. MTP alone already accepts every draft token in these samples. Reject after one pair without full-edit validation or deployment. The reproducible fixtures are available as `run_rebaseline.py --suite copy-edit` (section 2.36).
18. **Target GPU sampling:** the existing option is active in a bounded diagnostic, then gives Rust +1.73%, JSON +2.43%, geomean +2.08%, with 4/4 identical outputs. Both miss the 3% promotion gate despite nonoverlapping ranges. Stop after one pair; draft GPU sampling was already enabled and no deployment setting changes (section 2.37).
19. **Fusion-preserving row tiling:** a four-row IQ3_S/IQ4_NL candidate passes numerical and dispatch checks but gains only 0.49% across the operator set, with an 8.02% shared-fusion regression. Stop before a model run. Corrected whole-graph performance fixtures and all preflight limitations are retained; no kernel or deployment change survives (section 2.38).
20. **OpenMP worker waits:** verified `KMP_BLOCKTIME=0` versus the 200 ms default gives Rust +1.64%, JSON +0.22%, geomean +0.93%, with 4/4 identical outputs. Stop after one pair below the 3% gate. This is child-process-only; no global environment or deployment setting changes (section 2.39).
21. **Benchmark framework correction:** `test-backend-ops perf` now repeats every node for whole-graph cases, with graph-capacity limits in both run-budget paths; ordinary cases still repeat only the output node. A native ARM64 numerical regression fails the old loop and passes the fix across four combinations, with existing SNAKE and Q8 operator checks also passing. This improves measurement validity, not deployed token throughput (section 2.40).
22. **Fail-fast benchmark exits:** performance execution errors and unmatched selectors on a selected backend now return failure instead of success. Unknown explicit backend names fail, while case-insensitive names resolve to the actual device. Native red/green checks cover console, CSV and SQL, including controlled warmup/timed aborts and preserved unsupported-operation skips. Five focused CTests pass; no live runtime or model setting changes (section 2.41).
23. **Routed prefill scheduling:** two correct IQ3_S/IQ4_NL prototypes replace the unsafe timing-only grid cap. The looping version loses 20.14% across six operator cases. Compact small/wide tile lists gain 64.95%, then reach +9.21% pp2048 and +14.77% pp8192 in the first full-model pair. pp2048 misses the fixed 10% gate. A fresh rerun with a relaxed, predeclared 5% gate gives pp2048 +6.16% then +2.48% in reverse order (pp8192 +15.71%, +8.19%), so it stays parked without MTP/quality checks or deployment. Original source and runtime hashes are restored; reproducible patches and evidence are retained (section 2.42).
24. **Already-covered fork leads:** incremental QSA pooled keys already exist in deployed b11756 via upstream [ggml-org/llama.cpp#29751](https://github.com/ggml-org/llama.cpp/pull/29751), so the older pooled-cache draft is not a new port. Our FA scheduler also already uses the actual SM count rather than the power-of-two count fixed in [ikawrakow/ik_llama.cpp#2699](https://github.com/ikawrakow/ik_llama.cpp/pull/2699). Neither is a new local gain (section 2.43).
25. **NVFP4 expert screen:** the unchanged CUDA backend gains +39.57% geometric mean on six real-shape prefill operators, but the shuffled four-token gate/up case loses 43.36% and some down cases also regress. Reject wholesale expert-only conversion at the predeclared regression guard. These are standalone synthetic operators, not model throughput or quality results; no model or runtime changes are deployed. The fixture, opposite-order results and hashes are retained in `results/nvfp4-expert-screen-20261021/` (section 2.44).

26. **MMVQ row-tail correctness:** fixed expert-boundary writes and invalid tail reads, with 64 permanent regression cases and 100 additional checks passing against CPU. Actual-MTP Rust/JSON outputs match across both launch orders; no speed gain is claimed. Operator timing regressions remain unresolved, so the installed runtime is unchanged. The separate one-warp candidate is still unpromoted (section 2.45).

27. **IQ4_NL one-warp MoE CTAs:** rejected. With the row-tail correction in both arms, both pass 100 CPU-reference checks and traces confirm the intended geometry, but the changed operators lose 16.03% geometric mean, worst case -39.84%. No actual-MTP run or deployment follows (section 2.46).
28. **Expert-first verify reuse (Strata-style):** stopped before a kernel. Realistic 4-token route overlap already runs within 7-10% of perfect same-rank sharing in the current MMVQ, so caches capture the reuse; projected headroom is negative (section 2.47).
29. **Whole-batch PLE prefetch (Strata-style):** rejected. Lazy PLE costs 10.3% (2K) and 8.5% (8K) of cold prefill, but hinting the later ubatches' rows at decode start changes cold prefill by +0.15% and -1.34%, within noise; outputs match (section 2.48).

The reviewed checkout already includes CUDA graphs, Blackwell NVFP4/MXFP4 MMQ, qwen4exp hyper-connection fusion, grouped MoE top-k, and PLE row prefetch. Do not port or reimplement those without identifying a target-specific gap.

### Reproducing the unweighted Q6_K experiment

This recipe is retained for reproducibility, not deployment: the unweighted candidate failed the later KL precision gate. The failed importance-weighted and recurrent-QKV-protected revisions have separate protocols and results in `results/target-q6-imatrix-20261021/` and `results/target-q6-protected-qkv-20261021/`.

Use the three `UD-IQ4_XS` shards from `unsloth/Qwen3.8-Flash-Next-GGUF` revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`, with their original filenames together in one directory. `results/target-q6-attention-20261021/tensor-types.txt` pins all 1,224 tensor types: only the 156 Q8_0 attention/SSM projections become Q6_K. Do not reuse this map for a different model revision or quantization.

```powershell
C:\llama.cpp\llama-quantize.exe --allow-requantize --pure --keep-split --max-buffer-size 512 --tensor-type-file rtx-spark\results\target-q6-attention-20261021\tensor-types.txt <source>\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf <new-output>\Qwen3.8-Flash-Next-UD-IQ4_XS-attn-Q6_K IQ4_XS 8
```

Replace the placeholders and use a new output prefix; never overwrite the source shards. The measured quantizer hash and tensor verification are in `conversion.json`. The three resulting shards total 93,035,006,624 bytes, saving 617.58 MiB. The large GGUF artifacts are not published in Git. The results apply to MTP3/ub512 with the Q4_K draft and dynamic draft vocabulary K=16,384, not arbitrary models or settings.

### Additional fork survey

`PLAN.md` section 3.1 records the 2026-10-20 survey with pinned sources and applicability. **Keep upstream as the base.** The most relevant active proposals are dynamic draft-head selection, QSA-mask fusion, and dense/eligible MoE FFN fusion; their GB10 numbers are sourced leads, not measured RTX Spark gains. The dynamic head retains the original weights and adds a scorer, so it is a compute/bandwidth optimization rather than a smaller model file.

`ik_llama.cpp` is useful for CUDA/quantization algorithms; BeeLlama and TheTom are later KV-compression references. GenerelSchwerz and thecodacus expert caches, and RaymondHuang's KV streaming, mainly address capacity/offload problems. The Spark-specific croll83 fork is deprecated in favor of upstream; EngramHalo is explicitly AMD HIP-focused. These are not qualified drop-in Windows ARM64 replacements.

The survey also confirmed `--spec-draft-n-ubatch` in the installed b11756 binary: basic independent draft workspace tuning needs no fork. It is a memory knob, not a demonstrated speed improvement. Pinned PRs are built locally from source on the native ARM64 toolchain, one candidate against its own merge-base control; API/graph changes require matching executables and DLLs.

## Folder layout
- `PLAN.md`: full analysis, measurements, reference-project evaluation, roadmap, risks, sources.
- `bench/`: harness.
  - `run_rebaseline.py`: repeated native/server benchmarks, raw samples and generated token IDs, isolated logs, and GPU telemetry. Refuses to overwrite an existing result. Server mode accepts `--exe PATH` (custom build) and `--extra-args="..."` (one-variable A/B flags).
  - `compare_speed.py`: matched server speed comparison of two `server_*.json` files: per-case median tok/s, ratio, acceptance, geomean, load time, GPU memory and the count of token-identical outputs. `--out` keeps file names only. `--screen` checks matching fixed-length samples and emits an advance/park/reject speed decision.
  - `ab_llamabench.py`: paired `llama-bench` A/B for `q27`, `qfn`, or `nem`, reversing base/candidate order every round. Accepts `--base-model`/`--cand-model` for model-only comparisons, `--bench-args="..."`, `--rounds`, and `--dry-run`. `--screen prefill|decode` adds early stopping, a time budget and `screen.json`. Writes per-round JSON and `summary.json` with paths reduced to file names.
  - `test_fast_screen.py`: dependency-free tests for triage, reverse-order confirmation, model overrides, early stopping, timeouts and malformed/incomplete results.
  - `heldout_suite.py`: held-out multilingual/code/chat cases, an English/code-only selection, and a documentation prompt pinned to the measured source commit; the raw long prompt has a known EOS limitation.
  - `en_us_quality.py`: four chat-templated natural-EOS cases and explicit scoring: a bounded Python function with 231 assertions, exact US-format JSON, exact identifier/number copying, and distributed facts in pinned English documentation. Both warm-up and measured responses must pass. Generated code is limited to a single function with no imports, attributes, loops or self-reference before execution.
  - `analyze_mtp_tokens.py`: low-ID token coverage and per-position accepted counts divided by verification steps; `--scope en-us` selects the four English/code cases and fails if any are missing. It does not measure router overlap or implement a vocabulary subset.
  - `run_matrix.py`: launch config → benchmark suite → Ctrl+C; `MATRIX_OUT` env var selects the output file.
  - `run_llamabench.py`: DGX Spark methodology; `LB_SUFFIX` env var.
  - `bench_server.py`: request helper; `URL` global.
  - `bench_ple.py`, `bench_nospec.py`, `bench_spec_*.py`, `summarize_matrix.py`, `gguf_geometry.py`, `perf_model.py`.
- `tools/build_native_cuda.ps1`: native ARM64 CUDA build (VS ARM64 MSVC, Ninja, `sm_121a`). `-Src`, `-Out`, optional `-Targets` and `-Extra` CMake flags. Copies CUDA runtime DLLs into `bin`. Use for matched A/B pairs.
- `tools/build_release_like.ps1`: deployable build in the official release layout: Clang (LLVM ARM64 release tarball in `-Tools`, major version must match the upstream OpenMP pin) with the release CMake flags for CPU side and tools, plus `ggml-cuda.dll` from `build_native_cuda.ps1`. Seeds the OpenMP fetch cache from the LLVM tarball (no 7-Zip needed). Output in `<Out>\bin`.
- `results/`: JSON/markdown summaries, `geometry.txt`, and `model_paths.example.json`.
  - Scripts read `results/model_paths.json` and write to `results/` (server logs to `results/logs/`), via `RESULTS = bench/../results`. They run from any working directory. Copy `model_paths.example.json` to `model_paths.json` and set local paths.
  - Published files omit raw server logs, full generated responses (`server_*.json`) and the local model path file; local paths are replaced by placeholders such as `<hf_home>` and `<launcher_data>`.
  - `rebaseline-20261020/`: initial and final provenance manifests, repeated native samples, summary statistics, and telemetry.
  - `rebaseline-20261020/head-q4k/`: isolated draft-head conversion evidence and candidate measurements; `head-q8-repeat/` is the repeated control. `head_ab_summary.json` compares both with the original control.
  - `plan-20261020/`: held-out head/reload/EOS, Nemotron policy and MTP2/3/4 comparisons. `bench-targeted/` contains only the bounded Flash-Next diagnostics; the broader proxy matrix was cancelled.
  - `en-us-quality-20261020/`: scored natural-EOS tasks, preserved initial token-cap failure, code-only retries, one matched English speed pair, final audit and provenance.
  - `dyn-head-20261021/`: #30380 matched A/B (control, K=0, K=16,384): quality scores, telemetry and `compare_*.json`.
  - `deploy-b11756-20261021/`: b11756 + #30380 deploy validation: official/all-MSVC/release-like/hybrid builds on U1, N2 and D2, official drift pairs, quality scores, `compare_*.json` and `summary.json`.
  - `ab-30640-20261021/`: #30640 correctness, paired `llama-bench`, server A/B, release parity, quality and deployment evidence.
  - `ab-29948-20261021/`, `ab-29948-20261021-r4/`, `ab-29948-20261021-pp2048-r4/`: #29948 backend tests and repeated 27B A/B results; follow-up summary records why it was not deployed.
  - `draft-head-screen-20261021/`: Q3_K/Q2_K draft-head and mixed-Q4 routed-FFN conversion proofs, short server screens and rejection/parking decisions.
  - `target-q6-attention-20261021/`: selective target-projection conversion proof, native screens, actual MTP A/B and bounded natural-EOS quality gate; no perplexity or deployment validation.
  - `screen-aa-20261021/` and `screen-aa-confirm-20261021/`: unchanged-binary/model calibration; the two-pair run measured tg128 +0.89% and pp2048 -7.20% when pooled. Individual prefill pairs differed by -9.67% and -4.25%.

## Fail-fast candidate screens

Use one primary goal and one changed variable. Start with the relevant kernel correctness check or model-load smoke, then screen speed before spending time on full quality suites or release builds. Keep all candidates separate from the live install and original GGUFs. Model changes are first-class candidates, not just upstream patches.

For a binary change, use matching source/toolchain control and candidate executables:

```powershell
python rtx-spark\bench\ab_llamabench.py q27 --base <control>\llama-bench.exe --cand <candidate>\llama-bench.exe --screen prefill --out rtx-spark\results\screen-binary
```

For a target-model change, use the same executable and explicit GGUFs:

```powershell
python rtx-spark\bench\ab_llamabench.py q27 --base <build>\llama-bench.exe --cand <build>\llama-bench.exe --base-model <original.gguf> --cand-model <candidate.gguf> --screen decode --out rtx-spark\results\screen-model
```

The native screen defaults to pp2048 and tg128, two samples each. It runs control/candidate once and stops immediately for a weak pair. A survivor gets candidate/control once more; both pairs must survive independently. The total benchmark budget is 600 seconds including model loads (`--budget`), excluding build/conversion time. Normal runs without `--screen` retain their full sample defaults. `--dry-run` prints commands without running or creating results.

| Decision | Default rule | Next action |
|---|---|---|
| `reject` (exit 3) | Any measured case regresses more than 3% | Stop; preserve evidence and move to another candidate |
| `park` (exit 2) | A target case gains less than 3%, or target sample ranges overlap | Stop, not repeated retries to hunt for a win |
| `advance` (exit 0) | All target cases clear both filters and no case exceeds the regression limit | Continue to confirmation/quality, never directly to deployment |
| `error` (nonzero) | Crash, timeout, malformed or incomplete output | Stop and diagnose; not a completed speed comparison |

`--min-gain` and `--max-regression` set the percentage thresholds on both comparison tools. Set them before running, not after seeing results. Range separation is only a cheap noise filter, not a confidence interval; a parked result does not prove no improvement. A native screen records its decision in `screen.json`, and completed-pair measurements in `summary.json`.

A same-binary/model calibration on this device measured pooled tg128 +0.89% and pp2048 -7.20% across opposite launch orders (individual prefill pairs: -9.67% and -4.25%). Treat a short native prefill result as a triage signal, not a standalone rejection or promotion; confirm material prefill changes with repeated measurements and GPU power/clock telemetry.

For external draft heads or speculation changes, reuse the server path rather than target-only `llama-bench`. Run only two fixed-length cases initially, with one warm-up and two measured requests per case:

```powershell
python rtx-spark\bench\run_rebaseline.py server U1 --exe <build>\llama-server.exe --draft-model <control-draft.gguf> --extra-args="--spec-draft-vocab-dynamic 16384" --suite en-us --cases rust256 json256 --repetitions 2 --out rtx-spark\results\screen-draft-control
python rtx-spark\bench\run_rebaseline.py server U1 --exe <build>\llama-server.exe --draft-model <candidate-draft.gguf> --extra-args="--spec-draft-vocab-dynamic 16384" --suite en-us --cases rust256 json256 --repetitions 2 --out rtx-spark\results\screen-draft-candidate
python rtx-spark\bench\compare_speed.py rtx-spark\results\screen-draft-control\server_U1_en-us.json rtx-spark\results\screen-draft-candidate\server_U1_en-us.json --screen --out rtx-spark\results\screen-draft-comparison.json
```

These placeholders must be replaced with real paths; retain the same flags in both arms. The dynamic-head flag requires a build containing that feature. Server screens use the existing request/startup timeouts, not the native screen's total budget. Only if the first server comparison advances, run a second pair in reverse launch order in fresh directories and compare it separately. Token identity is reported but is not a quality pass, particularly for changed target weights.

Only confirmed survivors get the repeated performance and natural-EOS quality gates below, plus relevant precision/long-context checks for changed weights. Publish sanitized evidence and identify whether the candidate is our own code, a model/config change, or another author's upstream patch. No screen modifies the deployment or provides a cumulative speedup estimate.

Harness checks (no GPU or local model map required):

```powershell
python -m unittest discover -s rtx-spark\bench -p test_fast_screen.py -v
```

## Repeated benchmarks (survivors only)

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
- Report steady-state numbers, label each claim measured, sourced or inferred, and keep 5-run medians for promotion gates. Two-sample screens are triage only.
