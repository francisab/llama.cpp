# RTX Spark Inference Engine - en-US Optimization and Fork Survey (v9)

**Machine:** NVIDIA RTX Spark N1X (20-core ARM64 CPU, 6144-core Blackwell GPU, **sm_121**) · 128 GB LPDDR5X-9400 on a 256-bit bus (**≈300 GB/s**, shared CPU/GPU) · boot carve-out **96 GB GPU / 30.5 GB Windows** · Windows 11 ARM64 · CUDA 13.4 · 45–80 W SoC.
**Measured software:** llama.cpp b11756 (`d80619397`, official `win-cuda-13.4-arm64` build). The source review is against this repository checkout at `4a45de4f2`; it includes the upstream changes cited below.
**Models tested (2026-10-20):**
- NVIDIA Nemotron-3.5-Super-VL-120B-A12B Q4_K_M — hybrid Mamba-2/attention/latent-MoE, 512 experts with 22 active.
- Qwen3.8-Flash-Next UD-IQ4_XS — linear-attention MoE, 512 experts with 10 active, hyper-connections, 28.8 GB per-layer-embedding (PLE) table.
- Qwen3.8-27B UD-Q4_K_M — dense, Gated-DeltaNet linear attention plus full attention every 4th layer.

**Goal:** optimize US-English chat, prose, coding, structured output, and English document handling on RTX Spark. Keep upstream llama.cpp as the base and reuse relevant lessons from GenerelSchwerz/llama.cpp, Niko1221/Strata and robertkarl/UnbiasedStrata.

**User-directed scope:** en-US is the priority; other languages are not critical. Multilingual speed and task-quality results are informative, not promotion gates for this work. Preserve the full target vocabulary, fallback, and runtime safety; this is not permission to make the engine English-only.

> **Labels:** **[M]** measured on this machine · **[S]** sourced (link) · **[I]** inference, not measured.
> **v7 executed follow-up:** section 2.11 completes the compact English quality gate and one matched timing pair. Both draft heads pass complete-code, US-format JSON, exact identifier and 17,225-token document checks. The new Q4_K pair gains +4.50% across four English/code cases with identical measured tokens. The smaller draft is supported as an opt-in candidate for these workloads, not a universal replacement. Keep MTP3 and ub512; no deployment change or reduced-vocabulary engine patch was made.
> **v8 source follow-up:** section 3.1 surveys additional llama.cpp forks and active upstream proposals. Dynamic draft-head selection already has a Flash-Next/GB10 prototype in [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380); evaluate that before inventing a static MTP vocabulary format. QSA-mask and FFN prefill fusion are the next concrete kernel leads. These are sourced results, not new Windows measurements. No engine, model, deployment, commit or remote changed during this survey.
> **v9 native build follow-up:** a native ARM64 toolchain now builds llama.cpp with CUDA on this host (section 3.1). The matched #30380 A/B passes: +2.63% geomean at K=16,384 over its merge-base control, token-identical outputs, all quality checks pass, +238 MiB and +4 s load (section 2.13). The Q4_K draft is deployed on the Flash-Next launcher recipe, with a catalog backup. QSA-mask and FFN-fusion PRs are reviewed and staged for the same A/B.

---

## 1. Executive summary

1. **Plain decode has less obvious headroom on two of the three models.**
   - Qwen3.8-27B (dense): **82%** of 300 GB/s.
   - Nemotron: **75%**.
   - Qwen3.8-Flash-Next: **60–63%**.
   - These are bandwidth estimates from the recorded benchmark, not proof that the remaining gap is kernel overhead. Re-profile before choosing a kernel project.
2. **Speculative decoding remains the main measured lever [M].** On the fixed-length cases, Flash-Next MTP3 improves decode by 24-48%, Nemotron MTP3 by 41-52%, and 27B DFlash by 87-184%, depending on depth and prompt. These are end-to-end decode rates, not operator-level attribution. The earlier per-step cost model is historical.
3. **Withdraw the blanket Nemotron `p_min=0.5` recommendation.** Default MTP3 also wins all seven held-out cases in fixed-length and natural-EOS modes. The cut-off loses 1.2-12.9% in the fixed-length pass. Adaptive control still needs a demonstrated net benefit, not merely a higher draft acceptance ratio.
4. **Current configuration choices, without changing the deployed settings:**
   - **Nemotron:** native MTP n_max=3, default `p_min=0`; code 46.06, prose 45.01-46.18 tok/s.
   - **Flash-Next:** retain MTP n_max=3 and `-ub 512`; code 47.42, prose 44.81-47.25 tok/s with the original Q8_0 draft.
   - **27B:** DFlash n_max=7 wins code (44.44 versus 41.90); n_max=4 improves prose modestly and the random-word case substantially (34.85 versus 29.02). Both depths generated identical token sequences in all 20 paired requests.
5. **The Q4_K head is now a tested opt-in English/code candidate, not a universal speedup.** It saves 303.125 MiB with all 33 other draft tensors byte-identical. The compact English correctness gate passes, including a chat-templated 17,225-token document. The new matched speed pair gains 3.21-6.24% (+4.50% geometric mean) with identical measured tokens. Earlier English/code reloads gave only +1.73%, so the fresh pair does not establish a guaranteed gain or justify replacing the default for every workload.
6. **Existing graph/fusion work already matters.** In the bounded Flash-Next diagnostic, disabling CUDA graphs reduces tg128 from 30.26 to 15.01 tok/s; disabling fusion gives 25.58. These losses are not untapped speedup estimates. Changing prefill ub512 to ub2048 improves the median only 1.82%, within the observed variance; keep ub512 rather than trade more workspace for an unproven gain.
7. **Keep upstream llama.cpp as the base.** CUDA graphs, Blackwell NVFP4 MMQ, qwen4exp hyper-connection fusion, grouped MoE top-k, and PLE row prefetch already exist here. Prototype only gaps confirmed by profiling; do not port a separate engine or cache subsystem wholesale.
8. **The fork survey narrows the next implementation candidates.** Prioritize the existing dynamic draft-head proposal, QSA-mask fusion for Flash-Next, and MMQ FFN fusion for the dense 27B. Treat KV compression/streaming and CPU expert caches as capacity tools, not automatic speedups when the relevant weights and cache fit. The Spark-specific croll83 fork itself now recommends upstream. See section 3.1 for pinned sources, applicability and build blockers.

### Per-model measured baseline and next question

| Model | Fresh case medians [M] | Current question |
|---|---|---|
| **Flash-Next** | Server plain 31.39-32.14 tok/s; original Q8_0 MTP3 38.88-47.42; native pp2048 871.76 | Q4_K passes the narrow English gate and new matched pair; retain as an opt-in candidate, with operator evidence required before larger engine changes |
| **Nemotron** | Server plain 30.08-30.41 tok/s; default MTP3 42.50-46.18; native pp2048 891.78 | Keep default MTP3; the held-out policy test does not justify p_min=0.5 |
| **Qwen3.8-27B** | Server plain 15.52-15.73 tok/s; DFlash7 code 44.44, DFlash4 prose 35.06-38.33; native pp2048 922.23 | Validate workload-specific depth selection; target-weight quantization still needs quality gates |

---

## 2. Measurements

**Current repeated results are in sections 2.7-2.9; section 2.10 re-scores them for en-US and section 2.11 records the executed English quality/speed follow-up. Sections 2.1-2.6 preserve the earlier study, including b11681 results and the initial b11756 spot checks. They are not matched controls for the new forced-length protocol.**

### 2.1 Method [M]
- **Server suite** (`run_matrix.py`): launches `llama-server` per config on port 9940 with the 262K context and vision projector, as in real use.
  - Greedy completions: code 256 tokens, prose 256 and 512, low-predictability text (random-word 2K prompt, 128 tokens), prefill 2K/8K/25K.
  - Each server is stopped with a real **Ctrl+C** (console control event) and VRAM release is confirmed before the next launch.
  - Results: `results/matrix_results.json`; per-config server logs `results/logs/server_*.log`.
- **llama-bench** (`run_llamabench.py`): the official DGX Spark methodology (`-fa 1 -ub 2048 -p 2048 -n 32 -d 0,16384 -r 2`). Output `results/llamabench_*.md`.
- **Measurement pitfall (behind the v1 correction):** "time of the first decode step after a prompt" is inflated by about 2× (66–84 ms vs a steady 31 ms on Flash-Next). Only steady-state numbers are used below.
- **This build ignores per-request `speculative.*` overrides** (only echoed back), so every speculation variant needs its own server launch.

### 2.2 Plain decode efficiency [M]

| Model | Bytes read per token | llama-bench tg32 (d0 / d16K) | Server decode | Effective bandwidth | % of 300.8 GB/s |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B dense (Q4_K_M mix) | ≈15.6 GB | **15.84 / 15.14** | 15.6–15.8 | ≈247 GB/s | **82%** |
| Nemotron-3.5-Super (Q4_K_M) | 7.45 GB (dense 4.73 + routed 22×3.10 MB×40) | **30.18 / 30.28** | 30.1–30.6 | ≈226 GB/s | **75%** |
| Qwen3.8-Flash-Next (UD-IQ4_XS) | 5.83 GB (dense 4.67 + routed 10×2.42 MB×48) | **30.72 / 29.52** | 31.5–32.5 | ≈179–189 GB/s | **60–63%** |
| *Reference: DGX Spark/GB10, Qwen2.5-7B Q8_0 dense, Linux [S]* | 8.1 GB | 29.4 | — | ≈238 GB/s | 87% of 273 |

Decode speed barely changes with context depth (16K) on any of the three, because they are hybrid linear-attention or SSM designs with few full-attention layers.

### 2.3 Speculative decoding matrix [M]

| Config | Code 256 | Prose 256 | Prose 512 | Low-predictability | Step cost vs plain | VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Flash-Next, none | 32.5 | 31.8 | — | 30.8 | 1.00 | 73.7 GiB* |
| Flash-Next MTP n=2 | 46.4 | 44.0 | 46.1 | **42.3** | 1.78 | 81.5 |
| Flash-Next MTP n=3 | **47.1** | 43.5 | 45.9 | 41.3 | 2.08 | 81.6 |
| Flash-Next MTP n=4 | 45.9 | 43.5 | **47.0** | 38.2 | 2.40–2.53 | 81.8 |
| Nemotron, none | 30.6 | 30.2 | 30.1 | **30.0** | 1.00 | 73.1 |
| Nemotron MTP n=3 | **39.6** | **46.0** | **46.0** | 22.1 (−26%) | 2.11–2.22 | 77.7 |
| Nemotron DFlash n=3 | 39.4 | 44.5 | 45.0 | 19.2 (−36%) | 2.06–2.21 | 79.0 |
| Nemotron DFlash n=5 (your original) | 35.9 | 41.9 | — | 16.6 (−45%) | 2.64–2.75 | ~81 |
| 27B, none | 15.8 | 15.6 | 15.8 | 15.8 | 1.00 | 35.5 |
| 27B DFlash n=4 | 44.3 | **36.0** | **37.8** | **34.8** | 1.31–1.35 | 39.4 |
| 27B DFlash n=7 (your original) | **47.5** | 35.1 | 35.9 | 33.4 | 1.50–1.56 | 39.8 |
| *p-min sweep (`--spec-draft-p-min`):* | | | | | | |
| Nemotron MTP n=3, **p-min 0.5** | 39.5 | 44.6 | 44.5 | **32.4** (+8% vs none) | — | 77.7 |
| Nemotron MTP n=3, p-min 0.75 | 34.3 | 41.0 | 42.1 | 30.0 | — | 77.7 |
| Flash-Next MTP n=3, p-min 0.5 | 35.8 (−24%) | 35.2 | 39.9 | 34.2 | — | 76.7 |
| 27B DFlash n=7, p-min 0.5 | 42.4 (−11%) | 32.5 | 34.1 | 26.4 | — | 39.8 |

**Historical p-min result, not a current recommendation:** the original Nemotron random-word case generated 118 tokens without speculation, 21 with default MTP, and 66 with `p_min=0.5`. These were unequal-length outputs, not a controlled fixed-length comparison. The new b11756 runs favor default MTP, including the natural-EOS follow-up (section 2.7). The earlier cut-off results for Flash-Next and the 27B were not repeated here.

\* The Flash-Next no-spec server used `-ub 512`; the speculative rows use `-ub 2048`. That difference doesn't affect decode (MTP n=3 gave 47.6 tok/s at `-ub 512` too).

**Economics.** Speedup = (tokens per step) ÷ (step cost).
- **MoE models:** step cost ≈ 1.18 + 0.30·n_draft (in plain steps).
  - About half of each extra draft's cost is the MTP draft pass. On Flash-Next that pass reads the **full 248K-vocabulary output head, 0.675 GB at Q8_0**; on Nemotron it reads the shared head, 0.44 GB at Q6_K.
  - The other half is verifying one more token, which needs **its own 10–22 experts in every MoE layer**.
- **Dense model:** the verify batch re-uses the same weights, so a draft token costs only ~0.06×. That's why DFlash gives 2–3× on the 27B but only 1.3–1.5× on the MoE models.
- **Break-even:** on MoE, speculation only pays when each step accepts more than ~1.15 extra tokens. Nemotron on low-predictability text accepts ~0.6 → net loss.

### 2.4 Prefill [M]

| Model | llama-bench pp2048 (d0 / d16K) | Server prefill 2K / 8K / 25K | Effective compute [I] |
|---|---:|---:|---:|
| Qwen3.8-27B | **950 / 806** | 801 / 834 / 807 | ≈52 TFLOPS (27.3B × 2 × 950) |
| Nemotron | **895 / 831** | 772 / 874 / 875 (`-ub 4096`) | ≈22 TFLOPS (≈12B active) |
| Flash-Next | **904 / 787** | 975 / 840 / 867 (`-ub 2048`) · 753 / 898 / 822 (`-ub 512`) | **≈5 TFLOPS (≈3B active)**, rough active-parameter estimate; not an operator profile |

- **Prompt batch size:** `-ub 2048` vs 512 on Flash-Next shows no consistent difference (±6%) and costs 5 GB of VRAM. `-ub 4096` **fails** at 262K context (`decode() failed: bad allocation`, 88 GiB used).
- **Speculation lowers prefill by 5–12%:** the MTP/draft layers also process the prompt.
- **Lazy PLE (`-lzm on`, Flash-Next):** prefill on never-seen code is 873–877 tok/s vs 958 tok/s for the same text a second time and 974 for low-diversity text. The page-cache cost is **≤ ~9% of prefill**; decode is unaffected. *(Process I/O counters can't see memory-mapped page faults; this is inferred from throughput.)*

### 2.5 System [M]
- **Load times:** Nemotron 35 s, Flash-Next 33 s, 27B 15 s (models are read from NVMe; Windows' 30 GB RAM can't cache them).
- **Telemetry:**
  - Decode: 45–49 W, SM ≈2.25 GHz.
  - Prefill: 56–74 W at 2.2–2.4 GHz, settling to ≈2.17 GHz after ~15 s with intermittent SW power-cap events.
  - Power plan: *Balanced*.
- **CPU:** ARMv8.2+ with dotprod, i8mm, bf16 and fp16; no SVE/SME. **The installed `ggml-cuda.dll` already contains native `sm_121a` code**, so rebuilding for the "right arch" gains nothing.
- **Reliability:** one flash-attention startup crash (`cudaFuncSetAttribute`, shared object initialization failed) on 2026-10-19. It did not recur in 13 launches today.
- **Tool availability:** the rebaseline found no `cmake`, `nvcc`, `nsys`, or `ncu` on PATH. No compiler or profiler was installed for these measurements.

### 2.6 Update b11681 → b11756 (2026-10-20 09:00) [M]

| Test | b11681 | b11756 | Change |
|---|---:|---:|---:|
| llama-bench tg32 (27B / Flash-Next / Nemotron) | 15.84 / 30.72 / 30.18 | 15.59 / 29.79 / 30.09 | −2% / −3% / 0% (within noise) |
| llama-bench pp2048 (27B / Flash-Next / Nemotron) | 950 / 904 / 895 | 962 / 888 / 899 | +1% / −2% / 0% |
| Flash-Next MTP3, `-ub 512` (your config): code / prose256 / prose512 | 47.6 / 43.8 / 45.9* | **48.6** / 43.9 / **46.3** | +1–2% |
| Flash-Next prefill 2K / 8K / 25K (`-ub 512`) | 753 / 898 / 822 | 938 / 890 / 822 | ≈ (2K noisy) |
| **Nemotron MTP3 + p-min 0.5:** code / prose256 / prose512 / low-predictability | 39.5 / 44.6 / 44.5 / 32.4 | **46.1** / 45.5 / 44.9 / 33.0 | **+17%** / +2% / +1% / +2% |
| 27B DFlash7: code / prose256 / prose512 / low-predictability | 47.5 / 35.1 / 35.9 / 33.4 | 48.2 / **36.9** / **38.6** / 30.9 | +1% / +5% / +7% / −7% |

\* The b11681 prose512 figure is from the `-ub 2048` run.

- **Kernels:** no measurable change in plain decode or prefill.
- **Speculation improved:** Nemotron MTP acceptance on code went from 159/204 to 177/208 drafts (78% → 85%). Greedy outputs can differ between builds, so the gain is partly prompt-specific.
- **Per-request `speculative.*` overrides are still ignored** in b11756 (`n_max=0` and `types=none` gave identical draft counts). Keep the update.

### 2.7 Five-repeat rebaseline, 2026-10-20 [M]

Evidence is in `results/rebaseline-20261020/`: `manifest.json` contains actual binary, DLL, model-shard, and projector SHA-256 hashes; `plain_summary.json` and `server_summary.json` contain medians, sample ranges, and coefficients of variation (CV). Raw native samples, full server responses, token IDs, effective settings, logs, and GPU telemetry are retained.

The installed binary is still b11756 (`d80619397`), with the same driver, a 98304 MiB GPU carve-out, and the Balanced power plan. No deployed model, binary, power setting, or launcher configuration was changed. Model hashing ran before measurement, so cold OS page-cache state was not controlled.

**Native text-only tests:** `-fa on -ub 2048 -p 2048 -n 32,128 -d 0,16384 -r 5 -ngl 99`, plus `-lzm on` for Flash-Next. These do not include the server's vision projector or fixed 262K context. Depth-prefill state is cached/restored outside the timed section. Each cell is the median of five samples in one process, not five independent model loads.

| Model | pp2048 d0 | tg32 d0 | tg128 d0 | pp2048 d16K | tg32 d16K | tg128 d16K |
|---|---:|---:|---:|---:|---:|---:|
| Flash-Next | 871.76 | 29.78 | 31.42 | 776.64 | 29.31 | 30.73 |
| Nemotron | 891.78 | 29.68 | 30.77 | 839.35 | 30.48 | 29.52 |
| Qwen 27B | 922.23 | 15.67 | 15.85 | 789.21 | 15.01 | 14.96 |

Shallow tg128 CVs are 0.49%, 0.58%, and 0.16%, respectively. Prefill varies more; these results reproduce the broad baseline, not a small prefill optimization.

**Server tests:** the same model/projector files, 262144 context, one slot, full GPU offload, and per-configuration batch settings. Raw `/completion` requests use greedy sampling, seed 1234, `cache_prompt=false`, and `ignore_eos=true`. The runner checks both timing and returned-token counts. One warm-up per case precedes five measured requests with rotated case order; configurations run sequentially, not in a randomized crossover.

| Config | Code256 | Prose256 | Prose512 | Random-word128 |
|---|---:|---:|---:|---:|
| Q0: Flash-Next, none, ub512 | 31.96 | 31.90 | 32.14 | 31.39 |
| U1: Flash-Next, MTP3, ub512 | 47.42 | 44.81 | 47.25 | 38.88 |
| N1: Nemotron, none, ub4096 | 30.25 | 30.31 | 30.41 | 30.08 |
| N2: Nemotron, MTP3, default p_min | 46.06 | 45.01 | 46.18 | 42.50 |
| N4: Nemotron, MTP3, p_min=0.5 | 44.05 | 44.23 | 43.63 | 31.90 |
| D1: Qwen 27B, none, ub2048 | 15.64 | 15.68 | 15.73 | 15.52 |
| D2: Qwen 27B, DFlash7, ub2048 | 44.44 | 34.15 | 37.05 | 29.02 |
| D3: Qwen 27B, DFlash4, ub2048 | 41.90 | 35.06 | 38.33 | 34.85 |

The fixed-length decode CVs range from 0.27% to 2.77%. Compared with DFlash7, DFlash4 is 5.71% slower on code, 2.66% and 3.45% faster on prose, and 20.07% faster on the random-word case. All five token sequences match between the two depths for every case. The small prose differences should still be checked on more prompts.

**Nemotron natural-EOS follow-up:** the same random-word prompt, with `ignore_eos=false` and a maximum of 128 tokens.

| Config | Median decode tok/s | Generated lengths over five requests |
|---|---:|---|
| N1: no speculation | 30.06 | 128, 128, 128, 128, 128 |
| N2: default MTP3 | 42.99 | 128, 128, 128, 128, 128 |
| N4: MTP3, p_min=0.5 | 32.15 | 36, 128, 128, 128, 128 |

Default MTP accepted 86 of 120 drafted tokens on each natural-EOS request. This is not a consistently low-acceptance workload; call it a random-word prompt, not a general low-predictability test. The cut-off's short completion also makes its request wall time incomparable without accounting for length. The earlier blanket recommendation is withdrawn, not replaced with a claim that confidence cut-offs can never help.

**Quality limits:** fixed lengths and repeatability do not prove output equivalence. For Flash-Next, plain versus MTP shares only 111 initial code tokens, 10 prose tokens, and 52 random-word tokens before diverging. Nemotron and 27B also have plain/speculative differences. Full outputs and pairwise prefix lengths are preserved; no task-quality, perplexity/KL, multimodal, cold-PLE, or operator-level performance claim follows from these timings.

### 2.8 Flash-Next external draft head: Q8_0 to Q4_K [M]

Only `output.weight` in the external MTP GGUF changed. The target model and original cached draft were not modified. The head shrank from 675430400 to 357580800 bytes, saving 317849600 bytes (303.125 MiB). All 33 other tensors have identical shapes, types, and SHA-256 hashes, including token embeddings, router weights, and both BF16 indexer projections. All metadata is unchanged, and the original draft's whole-file hash still matches the initial manifest.

The existing quantizer supports this isolated conversion:

```powershell
C:\llama.cpp\llama-quantize.exe --allow-requantize --output-tensor-type q4_k --pure --max-buffer-size 512 `
  <original-MTP-Q8_0.gguf> <new-MTP-head-Q4_K.gguf> Q8_0 8
```

A `--dry-run` first confirmed only one tensor would change. Do not substitute `COPY`: it bypasses quantization before tensor overrides. Actual commands, conversion logs, the derived model path/hash, and every tensor comparison are in `results/rebaseline-20261020/head-q4k/quantization.json`. The 3.56 GiB derived GGUF is in this session's `files/benchmark-models/` artifacts, not Git or the original cache.

The candidate used U1's unchanged settings and requests, followed immediately by another Q8_0 control run. The only command difference is `-md`. Each case has five measured requests after warm-up; the earlier U1 run provides a second Q8_0 reference.

| Case | Repeated Q8_0 tok/s | Q4_K head tok/s | Decode change | Q8_0 request seconds | Q4_K request seconds |
|---|---:|---:|---:|---:|---:|
| Code256 | 47.08 | 49.65 | +5.46% | 5.748 | 5.473 |
| Prose256 | 44.61 | 47.02 | +5.39% | 5.957 | 5.674 |
| Prose512 | 46.99 | 48.26 | +2.71% | 11.091 | 10.826 |
| Random-word128 | 39.04 | 41.58 | +6.50% | 5.031 | 4.882 |

All values are medians. Candidate decode CVs are 0.88-1.64%; the repeated Q8_0 medians differ from the initial U1 control by less than 0.8%. All 20 candidate token sequences exactly match the repeated Q8_0 control. Acceptance is not uniformly better: on Prose512 it falls from 70.02% to 67.81%, yet net throughput improves. This is why draft acceptance alone is not the optimization objective.

**Initial decision (superseded by sections 2.9-2.11):** retain the candidate for a broader validation pass; do not deploy it yet. This establishes a small model-only gain on four raw-completion workloads, not the withdrawn 15-20% prediction or general quality parity. The original follow-up called for held-out code, rare identifiers, multilingual text, natural EOS, long context, and sustained/reloaded runs. The executed checks and resulting English-only disposition are recorded below. No kernel or serving changes are needed for the measured candidate. `head_ab_summary.json` records variance, wall time, acceptance, output parity, and control drift.

---

### 2.9 Plan execution: held-out validation and decision gates [M unless noted]

Evidence is in `results/plan-20261020/`. The new corpus contains Rust channel code, SQL/Python, JSON, Chinese, Spanish, a chat-templated arithmetic question, and 64000 characters of documentation pinned to `d80619397`. The long prompt actually tokenizes to 17044 tokens on Flash-Next and 17475 on Nemotron; the 262144 configured context capacity is not a tested context length.

Each configuration gets one warm-up and five measured requests per case. Q8/Q4/Q8/Q4 independently reloads the Flash-Next model; a separate Q8/Q4 pair allows natural EOS. `/metrics` counter deltas are attributed to each request and checked against response draft counts. These are per-request aggregates, not per-step traces. No engine code, cached model, deployment setting, or power policy changed.

Completed: 12 server launches, 420 measured requests, 84 warm-ups, and 25 native diagnostic samples. `execution_summary.json` derives the comparisons from complete results with clean exits, token-count/hash checks, and matching speculative counters. `execution_manifest.json` records the validation-pass snapshot before the later en-US scope update.

#### Draft head: held-out results and promotion decision

| Case | Q8_0 pooled median | Q4_K pooled median | Change | Output agreement |
|---|---:|---:|---:|---|
| Rust256 | 48.66 | 48.46 | -0.41% | Identical across both heads and reloads |
| SQL/Python256 | 47.25 | 47.43 | +0.36% | Identical |
| JSON256 | 44.54 | 46.13 | +3.56% | Identical |
| Chinese256 | 39.23 | 39.21 | -0.07% | Identical |
| Spanish256 | 44.92 | 46.43 | +3.37% | Identical |
| Chat256 | 48.64 | 50.33 | +3.48% | Identical |
| Raw documentation, 128 generated | 36.76 | 39.67 | +7.92% | Four distinct outputs per head, three shared; exclude from headline gain |

The six short cases have a +1.70% equal-weight geometric mean of per-case speed ratios. A median pooled across unrelated cases is not the workload speedup and must not replace that statistic. The first B/A launch pair ranges from about +2% to +8%, while the second ranges from about -2% to +2%; some apparent gains are within reload/run variation.

Natural EOS gives +1.47% to +5.66% on the six short cases (+3.46% geometric mean); the arithmetic chat ends at 140 tokens for both heads. The raw long prompt instead emits only EOS for both heads (empty content, one returned token, zero decode rate). Its natural-EOS speed ratio is undefined, not zero or a quality pass. Forced output after EOS is unsuitable as a long-context correctness gate.

**Decision at this stage: retain, do not deploy.** All 120 short-case outputs match across the four fixed-length launches, but output parity with a control is not an independent task-quality score. The capped Rust, SQL and JSON responses were not executed or scored as complete tasks; rare-identifier extraction was not separately tested. Section 2.11 subsequently executes the compact English gate with a repaired long fixture and known answers, without repeating the full matrix.

Each head/reload process ran for roughly seven minutes; the repeated sequence exercises loading and mixed-request stability, not an uninterrupted ten-minute decode or one-hour soak. The earlier Q8 runs had unexplained late-repeat slowdowns. Five-second GPU telemetry cannot exclude transient throttling or host scheduling interference; do not label this a thermal diagnosis.

#### Nemotron confidence policy

| Case | Default MTP3 | MTP3 p_min=0.5 | Cut-off change |
|---|---:|---:|---:|
| Rust256 | 44.54 | 40.10 | -9.96% |
| SQL/Python256 | 43.95 | 38.30 | -12.85% |
| JSON256 | 46.94 | 44.56 | -5.07% |
| Chinese256 | 44.89 | 43.05 | -4.10% |
| Spanish256 | 46.05 | 43.70 | -5.11% |
| Chat256 | 48.82 | 48.23 | -1.20% |
| Documentation128 | 44.84 | 43.78 | -2.36% |

Natural-EOS medians also favor default MTP3 on all seven cases (the cut-off loses 1.1-14.4%). All Nemotron requests reach their requested cap in this pass, so it is not evidence about shorter natural completions. Default and cut-off outputs match for the chat and documentation cases, not for every case. Keep default MTP3; a higher acceptance percentage under a cut-off does not establish a throughput or quality benefit.

#### Flash-Next MTP depth and offline policy screen

All three depths use the Q4_K draft head and ub512. Depth 3 pools the two earlier Q4 launches; depths 2 and 4 each have one separate launch. The comparison is not an interleaved same-output replay.

| Case | MTP2 | MTP3 | MTP4 |
|---|---:|---:|---:|
| Rust256 | 46.79 | 48.46 | 51.56 |
| SQL/Python256 | 44.60 | 47.43 | 48.04 |
| JSON256 | 44.36 | 46.13 | 47.93 |
| Chinese256 | 44.64 | 39.21 | 42.04 |
| Spanish256 | 44.53 | 46.43 | 45.22 |
| Chat256 | 47.34 | 50.33 | 46.55 |
| Raw documentation128 | 41.00 | 39.67 | 39.23 |

Depth 4 gains 6.41% on Rust but loses 7.52% on chat relative to depth 3; depth 2 gains 13.86% on Chinese but loses on every other short case. Only Spanish matches token-for-token across all three depths; JSON also matches at depths 2 and 3. Timing differences on different continuations are not proof that switching depths improves an unchanged task.

A hindsight selector choosing the fastest depth for each of the six short cases offers an estimated 2.87% throughput advantage over the best fixed depth (4), before any switching/capture overhead. This uses summed generation-time estimates for one equal-length request per case, not a pooled median across cases. It is an optimistic in-sample screen, not a held-out controller result or a promised speedup. **Keep MTP3 as the conservative default; do not implement an adaptive controller from this dataset.**

For depth 4, accepted counts per verification step at positions 1-4 are 0.935/0.823/0.742/0.597 on Rust, 0.821/0.628/0.474/0.346 on Chinese, and 0.857/0.714/0.614/0.457 on chat. Later positions contribute differently across workloads, but acceptance alone does not pay for their draft and verify cost.

#### Bounded Flash-Next diagnostics

| Diagnostic | Median tok/s | Sample CV | Interpretation |
|---|---:|---:|---|
| Plain tg128, graphs/fusion enabled | 30.26 | 0.49% | Matched diagnostic control |
| Same, CUDA graphs disabled | 15.01 | 0.91% | Existing graphs are important; keep enabled |
| Same, fusion disabled | 25.58 | 4.24% | Existing fusion is useful; not evidence for a particular new fusion |
| pp2048, ub512 | 787.48 | 4.32% | Smaller workspace control |
| pp2048, ub2048 | 801.81 | 5.95% | +1.82%, not separated from run noise |

Each row has five samples. Decode ablations use the same native text-only tg128 configuration with `GGML_CUDA_DISABLE_GRAPHS=1` or `GGML_CUDA_DISABLE_FUSION=1` as the single explicit change. Both switches exist in the installed build's source commit. These are aggregate ablations: they change scheduling, kernels and memory behavior, not just launch time. They cannot apportion latency to individual operators or measure expert reuse. The native prefill check also does not test server allocation at 262K context.

#### Reduced vocabulary and grouped MoE feasibility

`analyze_mtp_tokens.py` measures how often generated target IDs fall below an ID threshold. A 32768-ID prefix covers only about 40% of Chinese output, versus about 84% of Spanish and 97% of chat output. A 65536-ID prefix barely improves Chinese coverage. This rejects a naive low-ID prefix as a general multilingual draft subset; it does **not** evaluate a learned frequency shortlist, Strata's actual selection algorithm, or acceptance with any reduced head. Missing a draft candidate can cause rejection and target fallback without invalidating output when full target verification is preserved.

Per-position metrics are accepted tokens at that position divided by verification steps, not conditional acceptance given that the position was attempted. There is no implementable adaptive policy from these aggregates alone: per-step cost, switching/capture overhead, and a genuinely held-out policy evaluation are missing.

The checkout's quantized `MUL_MAT_ID` dispatch selects MMVQ for small batches on this GPU class (`ggml-cuda.cu:2078-2085`, `mmvq.cu:285-290`). In `mul_mat_vec_q_moe`, each token warp independently indexes its expert (`mmvq.cu:880-946`); explicit cross-token grouping is absent from that path, while cache reuse can still occur. This is a source observation, not a runtime bandwidth measurement.

**Grouped-kernel gate: blocked.** The installed tools do not expose actual routed IDs or per-operator GPU timing, and no compatible local CUDA profiling/build toolchain was found. Independent uniform routing would yield only about 2.9% duplicate expert selections for Flash-Next and 6.3% for Nemotron across four tokens [I]. Those are model assumptions, not measured overlap or bounds on DRAM savings. Neither small-batch target timings nor graphs-off deltas can establish the payoff of a grouped expert kernel. Require actual route overlap and operator costs before a prototype.

After the user questioned excessive benchmarking, the queued all-model verify/lazy/batch matrix was cancelled before launch. Only a bounded Flash-Next graph/fusion and ub512/2048 diagnostic was retained. No lazy-off, controlled cold-PLE, or all-model verify-cost result is claimed for this execution pass.

### 2.10 en-US prioritization [M reanalysis, not new GPU measurements]

The user subsequently prioritized en-US over other languages. `en_us_summary.json` re-scores the captured Rust, SQL/Python, JSON and arithmetic-chat cases with equal case weights. These account for 240 of the 420 measured requests. Chinese and Spanish are excluded for scope; the raw English documentation prompt is excluded because it fails the quality gate, not because of language. This is a code-heavy proxy slice with one chat question, not representative English prose, locale-formatting or complete-task validation.

| Comparison on English/code | Geometric mean change | Decision |
|---|---:|---|
| Flash-Next Q4_K head vs Q8_0, fixed length | +1.73% | Retain the smaller candidate; all 80 selected outputs match across four reloads |
| Flash-Next Q4_K head vs Q8_0, natural EOS | +3.42% | Encouraging but separate launches; do not treat as a guaranteed gain |
| Nemotron p_min=0.5 vs default, fixed length | -7.38% | Keep default p_min=0 and MTP3 |
| Nemotron p_min=0.5 vs default, natural EOS | -7.33% | Same decision |
| Flash-Next MTP2 vs MTP3 with Q4 head | -4.80% | Remove MTP2 from the English-first tuning shortlist |
| Flash-Next MTP4 vs MTP3 with Q4 head | +0.88% | Too small and mixed to change the default; chat loses 7.52% |

The hindsight per-case depth selector offers only 2.00% over fixed MTP4 for this slice, before switching overhead and with different outputs. Removing the Chinese case removes MTP2's only short-case win. This strengthens the decision not to build an adaptive controller now.

**English-specific draft vocabulary is a candidate, not ruled out by Chinese coverage.** However, token IDs are not language labels. Even the existing English/code outputs use IDs outside naive low-ID prefixes:

| Flash-Next fixed-length case | IDs below 32768 | IDs below 65536 |
|---|---:|---:|
| Rust256 | 92.58% | 98.44% |
| SQL/Python256 | 87.50% | 94.14% |
| JSON256 | 94.53% | 99.22% |
| Chat256 | 96.88% | 98.83% |

Prefer a draft-only shortlist selected from separate English prose/code training data, including required control tokens, rather than truncating the vocabulary by ID or fitting a list to these evaluation outputs. Keep full-vocabulary target verification/sampling and fallback, plus an off switch. A missed draft candidate must not become a forbidden target token. Coverage alone does not measure draft acceptance, task correctness or speed.

**Scope decision at this stage:** finish the English correctness checks for the existing Q4_K head before considering an English-selected reduced draft head. Section 2.11 now records those executed checks, including US formatting. Broader English prose quality remains outside this small gate. Other languages are optional diagnostics, not a reason for more sweeps.

The harness now provides `--suite en-us` for this four-case timing slice; the token analyzer accepts `--scope en-us` and rejects inputs missing a selected case. The original `heldout` suite remains unchanged. `en_us_tokens.json` contains the scoped token/acceptance analysis; `en_us_manifest.json` records the source snapshot and derived evidence. No engine binary, deployed configuration or target model was changed.

### 2.11 Executed English quality gate and matched speed check [M]

Evidence is in `results/en-us-quality-20261020/`. `en_us_quality.py` supplies four chat-templated tasks, scores natural-EOS answers, and checks both warm-ups and measured responses:

| Quality check | Q8_0 | Q4_K | Scope |
|---|---|---|---|
| Complete Python interval-overlap function | Pass | Pass | 231 executable assertions per answer: 225 valid interval pairs and six invalid inputs |
| US-format JSON | Pass | Pass | Exact keys/string values, US date/currency/time and canceled/aluminum/gray spelling |
| Rare identifiers and numbers | Pass | Pass | Exact case, punctuation, leading/trailing zeros and primary-vs-decoy selection |
| English document retrieval | Pass | Pass | Three facts distributed across pinned documentation; 17,225 actual prompt tokens and a known JSON answer |

The original Q8 measured code answer exhausted its 1536-token budget during the final function; the Q4 answer and all other checks passed. This was not silently scored as correct. Both heads were retried on **code only**, at the same 2048-token budget, with the prompt and scorer unchanged. All retry answers ended naturally and passed. The original `quality_scores.json` failure remains intact; `final_summary.json` records the resolved gate and the source used for each score.

Only after correctness passed, one Q8/Q4 timing pair ran the four English/code cases at five measured requests per case plus one warm-up. MTP3, ub512, prompt hashes, greedy sampling, fixed 256-token output and all other command/request settings match; only the external draft path changes.

| Case | Q8_0 median tok/s | Q4_K median tok/s | Change | Q8 / Q4 sample CV |
|---|---:|---:|---:|---:|
| Rust256 | 49.49 | 51.08 | +3.21% | 1.16% / 1.18% |
| SQL/Python256 | 47.39 | 49.22 | +3.86% | 0.31% / 1.01% |
| JSON256 | 44.90 | 47.70 | +6.24% | 1.13% / 1.11% |
| Chat256 | 48.94 | 51.26 | +4.72% | 1.20% / 0.90% |

The equal-case geometric mean gain is **4.50%**; all 40 measured timing outputs match token-for-token across heads and repeats. This is a separate fixed-length timing slice, not throughput of the scored natural-EOS tasks. Earlier reload comparisons gave smaller gains; one sequential pair cannot rule out launch-order or sustained-power effects.

**Disposition:** the Q4_K external draft is supported as an **opt-in en-US/code candidate** at MTP3/ub512, saving 303.125 MiB. Keep Q8_0 available for rollback and do not overwrite the original. This pass supports the four scored tasks and the recorded timing cases, not general writing quality, arbitrary code, a thermal soak or correctness at 262K context.

The follow-up finished six launches, 50 measured requests and 18 warm-ups, including the targeted code retries. `final_summary.json` audits responses, hashes, speculative counters and one-variable controls; `final_manifest.json` records source/model provenance and cleanup. No deployer settings, target weights or engine binaries changed. The bounded work is complete. Reduced-vocabulary mapping and grouped kernels remain deferred: token-prefix coverage and this head A/B do not establish their operator costs or justify a larger graph/export change.

### 2.12 Draft-vocabulary feasibility screen [M/I]

Evidence is in `results/draft-vocab-feasibility-20261020/`. A local, separate selection corpus was tokenized with the Flash-Next draft tokenizer: 1,311 code files (25.4 MB, 8.24M tokens) and 53 prose files (0.48 MB, 143K tokens). Evaluation prompts, outputs, `docs/build.md`, and `docs/speculative.md` were excluded from shortlist selection. Ordinary tokens were ranked by equal-weight normalized frequency across code and prose; all 276 non-normal tokenizer entries were retained in every candidate.

The screen evaluated 40 held-out timing responses (10,240 generated tokens) and eight completed quality-gate responses (5,084 tokens), including the matched code retries. The candidate sets cover the following observed output tokens:

| Ordinary-token budget | Total rows incl. 276 controls | Q8_0 head bytes [I] | Row-byte reduction [I] | Timing-output coverage [M] | Quality-output coverage [M] |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 16,660 | 45.3 MB | 93.3% | 82.52% | 92.37% |
| 32,768 | 33,044 | 89.9 MB | 86.7% | 92.87% | 97.29% |
| 65,536 | 65,812 | 179.0 MB | 73.5% | 99.12% | 99.53% |

The 65,536-token candidate is promising for a follow-up prototype: it covers nearly all tokens observed in these English/code outputs and would retain about 26.5% of output-head rows. The estimated Q8_0 row bytes assume 2,720 bytes per row; they are not a measured reduced-head GGUF size. At the measured 300.8 GB/s peak, reading the current 675,430,400-byte head has a 2.25 ms theoretical bandwidth floor; the shortlist floors are 0.15, 0.30 and 0.60 ms. These are not operator timings. The existing Q4_K A/B changed only the output-head tensor and measured +4.50% end-to-end decode on the English/code timing slice, but does not isolate head latency.

**Decision: promising, but not sufficient to implement generic MTP `d2t`.** The source corpus is an in-repo proxy, with a small prose portion; output-token coverage is not draft acceptance, exactness, quality or speed. No representative independent English/code selection corpus or compatible per-operator profiler/build is available, and no reduced-vocabulary MTP acceptance or fallback has been tested. Do not restrict target verification or sampling. The next gate is a representative selection corpus plus a prototype that preserves full-target verification/fallback, then held-out acceptance, correctness and matched speed checks. `summary.json`, `shortlists.json` and `manifest.json` retain coverage, candidate IDs and source hashes.

**v8 update:** that corpus requirement applies to our static shortlist proposal. The dynamic alternative in [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380) derives a low-rank scorer from the existing head weights and selects additional rows per token. It does not require a frequency corpus or a new `d2t` GGUF. Evaluate this existing implementation first once a compatible build is available; the static screen remains evidence, not an implementation commitment.

### 2.13 Dynamic draft head, PR #30380, matched native build A/B [M]

Evidence is in `results/dyn-head-20261021/`. Both binaries were built on this machine with the native ARM64 toolchain (`tools/build_native_cuda.ps1`, MSVC 14.44, CUDA 13.4, `sm_121a`): control at merge-base `ccdbe2905`, candidate at PR head `da03be647`. Same Flash-Next UD-IQ4_XS target, Q4_K external draft, MTP3, ub512, 262K context, prompts and greedy sampling. Only the binary and `--spec-draft-vocab-dynamic` change.

- **Engaged:** with `-lv 4` the candidate logs `vocab_subset: lowest 32768 of 248320 ids + 16384 picked per output, 1 LM head(s), 3.42 s`. The default log level hides this line; no fallback warning appeared. The PR unit test `test-draft-vocab` passes on this build.
- **Exactness:** all 20 fixed-length timing outputs are token-identical across control, K=0 and K=16,384. Draft acceptance is unchanged except chat (0.778 to 0.749).
- **Quality gate:** control and K=16,384 pass all four en-US checks (code, US JSON, identifiers, 17K-token document) in both warm-up and measured responses.

| Case | Control tok/s | K=0 tok/s | K=16,384 tok/s | K=16,384 vs control |
|---|---:|---:|---:|---:|
| Rust256 | 50.98 | 51.46 | 53.06 | +4.08% |
| SQL/Python256 | 50.13 | 49.88 | 51.87 | +3.47% |
| JSON256 | 49.55 | 49.34 | 50.86 | +2.64% |
| Chat256 | 49.29 | 49.82 | 49.48 | +0.39% |

Medians of five measured requests; sample CV 0.9-2.3%. Geometric mean: K=16,384 **+2.63%** over control and +2.36% over K=0 in the same binary; K=0 versus control is +0.27% (noise), so the off path is neutral. Cost: load 32.8 to 36.8 s (scorer setup) and +238 MiB GPU memory at ready.

**Disposition: passes; smaller than the GB10 report (+5.9%).** The author's draft file, OS, prompts and control differ; this pass does not isolate why. Do not add this +2.6% to the Q4_K +4.50%; they were measured against different controls and binaries. These are one sequential pair per variant; chat shows no gain.

---

## 3. Source-verified feature status and reference lessons

The following status is verified in this checkout. This table supersedes the porting assumptions in the initial reference inventory below.

| Technique | Status | Next action |
|---|---|---|
| CUDA graphs, including MoE `MUL_MAT_ID` handling | Already implemented upstream; see `ggml/src/ggml-cuda/ggml-cuda.cu` and [#14753](https://github.com/ggml-org/llama.cpp/pull/14753) | Do not add generic graph support. Investigate only a reproducible issue on the current binary. |
| Qwen4exp hyper-connection gate/post chain | Fused by [#30602](https://github.com/ggml-org/llama.cpp/pull/30602) | Remove this operation chain from K1. The reported ~3% gain was on RTX Pro 6000, not this device. |
| Grouped-expert top-k routing | Added by [#29181](https://github.com/ggml-org/llama.cpp/pull/29181) | Verify its target-model effect; do not plan another routing-fusion port. |
| Blackwell NVFP4/MXFP4 MMQ | Implemented in `ggml/src/ggml-cuda/mmq.cu`; see [#24364](https://github.com/ggml-org/llama.cpp/pull/24364) and [#29857](https://github.com/ggml-org/llama.cpp/pull/29857) | Test the model quantization, dispatch path, quality, and prefill speed separately. Kernel presence alone predicts no gain. |
| Native MTP | Present for Nemotron-H and qwen4exp; see [#29018](https://github.com/ggml-org/llama.cpp/pull/29018), [#27836](https://github.com/ggml-org/llama.cpp/pull/27836), and [#29761](https://github.com/ggml-org/llama.cpp/pull/29761) | Already used in the recorded model results. |
| DFlash/DSpark reduced vocabulary | DFlash/DSpark are implemented; `src/models/dflash.cpp` optionally reads `d2t` and scatters reduced-vocabulary logits. EAGLE-3 also has `d2t`. Generic MTP has no equivalent mapping in this checkout. See [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) and [#25173](https://github.com/ggml-org/llama.cpp/pull/25173). | Do not describe `d2t` itself as missing from DFlash. The open dynamic-head proposal in [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380) is now the first alternative to assess before designing a static MTP export/mapping path. |
| PLE row prefetch | `llama_prefetch_rows` landed in [#29599](https://github.com/ggml-org/llama.cpp/pull/29599). The direct-read proposal [#30772](https://github.com/ggml-org/llama.cpp/pull/30772) was closed in favor of prefetch. | Profile cold/warm rows before changing the existing implementation. Do not port raw disk reads without contrary local evidence. |
| Confidence cut-off and n-gram draft modes | Existing CLI/config controls include `n_max` and `p_min`; multiple n-gram modes are documented. Per-request speculative fields are still disabled under `#if 0` in `tools/server/server-schema.cpp`. | The measured `p_min` behavior is available now. Per-request overrides are a server/API feature, not a decode-kernel optimization. |
| Adaptive draft depth | No general runtime controller found. Related open research issues: [#27821](https://github.com/ggml-org/llama.cpp/issues/27821), [#29875](https://github.com/ggml-org/llama.cpp/issues/29875). | First log acceptance and measured cost by draft depth; compare an offline policy on held-out prompts. Implement only if it beats fixed settings without quality regressions. |
| Grouped MoE prefill | Concrete CUDA reference in [PolyStrata `moe_fused.hpp`](https://github.com/VecSzn/PolyStrata/blob/v0.1.45-poly.1/include/strata/prefill/moe_fused.hpp); upstream has a related SYCL path in [#29245](https://github.com/ggml-org/llama.cpp/pull/29245), but no matching qwen4exp CUDA implementation was found. | A prototype lead, not a drop-in port or a proven speedup. Profile expert GEMMs and token distribution first. |
| Grouped expert decode/verify | GenerelSchwerz has CUDA expert grouping and tests in its `moe-cache` branch. It groups shared expert IDs across routed rows; it is not an upstream speculative-verify feature. | Measure routed-expert overlap and verify tensor layouts before designing a compatible upstream experiment. |
| Independent draft microbatch | Already in `common/arg.cpp`, `common/speculative.cpp` and the installed b11756 help as `--spec-draft-n-ubatch`. Zero inherits the target microbatch. | No BeeLlama port is needed for the basic control. A smaller draft workspace is a configuration experiment only if memory pressure warrants it; verify effective settings and prefill cost rather than assuming a speedup. |

### Reference-project signal quality

| Source | Transferable evidence | Limits |
|---|---|---|
| Niko1221/Strata | Technique inventory: custom kernels, draft-vocabulary selection, and acceptance-aware drafting. | Speed tables are self-reported on other GPUs and configurations; they are not Spark forecasts. |
| robertkarl/UnbiasedStrata and VecSzn/PolyStrata | Public source, concrete grouped-expert prefill and PLE kernels, and a useful pinned-source/correctness process. | Separate engine and build stack; source availability does not imply performance or Windows ARM64 compatibility. |
| GenerelSchwerz/llama.cpp | Real CUDA expert grouping/staging code and tests that clarify data structures and edge cases. | Substantial fork divergence. Cache residency is not valuable merely because it exists, and grouped decode may not match upstream verify graph layouts. |

The inventory below is retained as background from v2; use the source-verified status above for current decisions.

| Source | Technique | Relevant here? | Evidence |
|---|---|---|---|
| Strata, GenerelSchwerz | VRAM expert cache, CPU expert pool, KV streaming, PCIe expert pipelining, route tail-skip | **Low relevance for expert offload.** Expert weights fit in the 96 GB carve-out; Flash-Next's lazy PLE table is a separate residency question. | Strata's own unified-memory docs disable these [S] |
| Strata | Fused / thread-block-cluster decode kernels, EXPDEC shared-memory expert staging | **Flash-Next only** (60–63% bandwidth efficiency); little room on Nemotron (75%) or the 27B (82%) | §2.2 |
| Strata | **Draft-vocabulary subsetting** (`--draft-vocab`) | **Yes.** Each MTP pass reads a 0.44–0.68 GB full-vocabulary head. | §2.3 |
| Strata | Acceptance-gated drafting (n-gram drafting enabled only when its measured acceptance pays) | **Yes.** Nemotron loses 26–45% on low-predictability text. | §2.3 |
| Strata | PLE handling (`--ple-io ram`) | Minor. The lazy-PLE penalty is ≤9% of prefill; the table can't sit in Windows RAM at this carve-out. A GPU-resident PLE lookup is an engine option. | §2.4 |
| Strata | Purpose-built Flash-Next prefill (2.0–2.6K tok/s on an RTX 3090) | **Yes**, as evidence of headroom: llama.cpp prefill here is ≈5 effective TFLOPS. | [UnbiasedStrata README](https://github.com/robertkarl/UnbiasedStrata) |
| GenerelSchwerz | Grouped, expert-deduplicated execution for decode/verify batches; graph "certificates" for MTP/draft | **Partly.** Deduplication only saves experts that adjacent tokens share. | §2.3 economics |
| UnbiasedStrata | Process: pinned commit, patches as commits, one-command build, correctness (teacher-forced KL) and ±5% speed-parity gates | **Yes** | — |
| Upstream llama.cpp | Native Nemotron MTP (#29018), DFlash, EAGLE-3 with a reduced draft vocabulary (`d2t`), NVFP4 W4A4 matmuls on Blackwell (#24364, #29857), official Windows-ARM64 CUDA builds | **Base** | [S] |

### 3.1 Additional fork survey, 2026-10-20 [S/I]

**Scope and provenance.** Queried 100 star-ranked and 100 newest entries from the upstream fork network, supplemented by targeted Spark/GB10/ARM64 searches, then read selected READMEs, source files, PR diffs and review discussions. This is a bounded survey, not an audit of every fork. The upstream comparison is pinned to `339f3242c761f16c27e5ffcde1d35dcc31933a55`, nine commits after our source baseline `4a45de4f2`. None of the open proposals below is in that comparison. Source/API snapshots are retained in the session's `files/fork-survey-20261020/`; the links below identify public revisions independently of those local artifacts.

**Hardware filter.** Our 128 GB is shared system memory, with a 96 GB GPU carve-out, not 128 GB of separate VRAM. Linux DGX Spark GB10 is a useful nearby architecture, not a control for Windows RTX Spark N1X. CPU/GPU copies still consume bandwidth and memory even on a shared-memory device; neither a Linux container nor `GGML_CUDA_ENABLE_UNIFIED_MEMORY` establishes zero-copy access across the Windows carve-out. Do not transfer x86 AVX, AMD HIP/Vulkan, discrete-GPU PCIe, or multi-GPU gains to this machine.

#### Highest-value source candidates

| Priority | Source and pinned proposal | What the source actually changes | Evidence and decision for this machine |
|---|---|---|---|
| 1 | [praneshgo/llama.cpp at da03be6474](https://github.com/praneshgo/llama.cpp/tree/da03be64749b994d6dc2aeedff0a3b3da9f2b6fe), [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380), open | `--spec-draft-vocab-dynamic K`: compute the lowest 32,768 rows plus K context-selected rows. A rank-512 scorer is constructed from head weights; selected logits use the original head rows and are scattered into full-vocabulary output. Only the draft context receives the option. | Author reports Flash-Next UD-IQ4_XS with an MTP Q4_K_M draft on GB10: 41.51 to 43.97 tok/s (+5.9%), acceptance 58.0% to 57.8%, K=16,384, MTP3. Two rounds over 165 greedy cases, not our draft artifact or Windows control. **First build candidate, ahead of a new static `d2t` design.** |
| 2 | Upstream development branch at [8a86212329](https://github.com/ggml-org/llama.cpp/tree/8a86212329f2d07f76fb3ba3799436e19b4d2205), [ggml-org/llama.cpp#30640](https://github.com/ggml-org/llama.cpp/pull/30640), open | Adds `KPOOL_MASK` and a CUDA kernel to build the QSA selected-pool/tail mask without the existing multi-op mask chain. It does not replace the whole attention calculation or implement a new expert cache. | Flash-Next GB10 author results: prefill +3.9% to +5.9%, decode +0.6% to +1.3% at tested depths, best of two alternating rounds, ub2048. **Concrete Flash-Next prefill candidate.** CPU/CUDA implementations and backend tests exist, but the inspected checks are not all green and do not qualify Windows ARM64. |
| 3 | [anujj/llama.cpp at 0ba2b5f2ad](https://github.com/anujj/llama.cpp/tree/0ba2b5f2ada4bedc85b38a819f17a58d7daee5a5), [ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948), open | Fuses dense FFN gate/up matmuls and GLU on the larger-batch MMQ path. Interleaves row loading inside the kernel; the model file layout is unchanged. This is distinct from existing small-batch fusion and routed-expert grouping. | Author reports Qwen3.8-27B Q4_K_M prefill +7.1% on GB10. A [follow-up using UD-Q4_K_M](https://github.com/ggml-org/llama.cpp/pull/29948#issuecomment-6165544748) reports +3.9% for this patch alone; 30/64 FFN layers have mixed gate/up types. **Use the UD result as the closer lead, not a promised 7.1%.** |
| 3, separate A/B | [SongXiaoXi/llama.cpp at 784fd01678](https://github.com/SongXiaoXi/llama.cpp/tree/784fd01678cb91227c8de2fa1566308e7e3e2244), [ggml-org/llama.cpp#30489](https://github.com/ggml-org/llama.cpp/pull/30489), open | Fuses GLU into Q8_1 activation quantization before a dense or eligible routed-MoE down projection, removing an F32 intermediate write/read. The broadcast scatter path remains unfused. | Small author-reported prefill gains on RTX 4090; the GB10 UD-Q4_K_M follow-up reports +5.6% for both patches versus +3.9% for the preceding gate/up fusion alone. This does not isolate the activation-quantization fusion's contribution. **Check actual down-projection dispatch before considering it for Flash-Next or Nemotron.** |
| 4 | Upstream development branch at [ed93210d3b](https://github.com/ggml-org/llama.cpp/tree/ed93210d3b8639a23b5024ad5561e34fc69db681), [ggml-org/llama.cpp#30804](https://github.com/ggml-org/llama.cpp/pull/30804), open | Splits CUDA graph capture/replay into chunks, default 512 GGML nodes; zero restores a whole graph. Cuts avoid splitting fused operations or active concurrent regions. | Author reports 1-2% token-generation gains. Discussion still requests Windows measurements and graph/fusion regression coverage. **Lower priority:** small enough to require a careful same-build A/B; our graph-disable slowdown is not its potential gain. |

**Dynamic-head caveats from the code.** This is not our fixed 65K shortlist: tokens outside the low-ID base remain candidates through the dynamic scorer. The full draft head stays resident and new scorer/basis buffers are allocated, so it reduces per-step head work, not GGUF size or necessarily peak memory. Startup constructs the scorer with repeated matrix operations and CPU orthogonalization; record load time as well as steady decode. EOG IDs are forced into the dynamic candidates, not every one of our 276 non-normal IDs. Repacked/split heads are skipped, non-quantized CUDA heads may be slow, and the reduced path explicitly omits head LoRA. Oversized K or initialization exceptions log a full-head fallback. Confirm engagement in the log; a fallback is not a valid optimized sample.

The added unit test uses a tiny random model, compares selected finite logits with a tolerance, and checks EOG, row counts and an oversized-K fallback. Its small embedding makes the scorer full-rank there; it does not establish selection accuracy for Flash-Next's 2,560-wide head, our Q4_K tensor, or stochastic sampling. Full-target verification remains required, but is not by itself evidence that every generated sequence is identical. Use the existing English/code task checks, including identifiers, JSON/control tokens and document retrieval, before promotion.

#### Broader forks: useful references, not replacement engines

| Fork / inspected revision | Transferable work | Disposition |
|---|---|---|
| [ikawrakow/ik_llama.cpp, 565f08d1ea](https://github.com/ikawrakow/ik_llama.cpp/tree/565f08d1eac9b73bd39fa369b3ffc154e8394b27) | Real CUDA fused MoE execution, ARM_NEON/CUDA row-interleaved quants and attention scheduling. [ikawrakow/ik_llama.cpp#2699](https://github.com/ikawrakow/ik_llama.cpp/pull/2699) changes its new MMA FA grid to use all SMs on eligible large-query shapes, with GB10 measurements. | **High-value algorithm reference, high integration cost.** Its README says the last full upstream sync was August 2024; GitHub now exposes it as a standalone repo. Its quant formats, graphs and model compatibility are not a drop-in DLL contract. The FA result is for DeepSeek test/full models, not our three models. Its [PLE prefetch](https://github.com/ikawrakow/ik_llama.cpp/pull/2692) is Linux/macOS-only and overlaps a mechanism already upstream here. |
| [Anbeeld/beellama.cpp, ce1a9994fa](https://github.com/Anbeeld/beellama.cpp/tree/ce1a9994fad8cf7706749666249398bbb7c600b1) | KVarN cache quantization, high-precision recent-token tails, speculative rollback/state work and CUDA attention routes. The source gates routes by shape and device resources. | **Capacity/long-context research, not first decode work.** Published quality comparisons are Qwen3.6/Gemma on other hardware; cache precision changes need task-quality gates. The inspected release offers Windows x64 CUDA, not Windows ARM64 CUDA. The basic separate draft-microbatch control already exists in our source and binary. |
| [GenerelSchwerz/llama.cpp, f3f31a788f](https://github.com/GenerelSchwerz/llama.cpp/tree/f3f31a788f05d99f9404afe90f9de4baaa2f6e6f) | Grouped expert execution, staging, ownership/capture contracts and memory accounting. The remote tip still matches the source inspected earlier. | **Keep grouped-algorithm lessons, defer the cache subsystem.** Actual routed-ID overlap and verify layouts remain missing. Its private backend procedures require matching executables/libraries; do not mix its CUDA DLL with the installed upstream server. |
| [thecodacus/llama.cpp, 906247a58f](https://github.com/thecodacus/llama.cpp/tree/906247a58f0d28cd8537999f1feceb441029392d) | Host registration, asynchronous full-expert-tensor prefetch and hot-expert GPU/CPU splitting. The scheduler uses additional backend streams/events and staging slots. | **Low priority while experts fit.** Its +64% prefill example uses an RTX 3060 12 GB with CPU-offloaded experts, not an all-GPU Spark workload. Extra staging is a cost here, not free shared-memory bandwidth. Routing diagnostics are more transferable than its headline cache gain. |
| [RaymondHuang210129/llama.cpp-adaptive-kv-streaming, db8fd5d898](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming/tree/db8fd5d898b0d19be74a4269c126c23d2f62c38d) | Full encoded KV history in host RAM, a GPU working set/ring, and phase-aware prefill/decode/draft workspace reuse. Source admission checks geometry, compiled readers and launch resources. | **Capacity fallback only.** The headline contrast is against UVM oversubscription on a 16 GB RTX 5070 Ti. The qualified target is one serial Qwen3.8-27B request on Linux CUDA; Windows DLL work is not Windows qualification. Host history plus GPU copies must fit the shared 128 GB budget. |
| [TheTom/llama-cpp-turboquant, ca28b4bc0e](https://github.com/TheTom/llama-cpp-turboquant/tree/ca28b4bc0e531635d83fe530c6e6e8d694c4330d) | CUDA low-bit/rotated KV kernels, including direct compressed-block reads inside MMA attention rather than whole-cache F16 materialization. | **Later KV research.** It is a broad integration branch, not just one TurboQuant patch. Rotation, cache state and attention kernels must agree; lower cache bytes do not establish end-to-end gains or English long-context quality. |
| [croll83/llama.cpp-dgx, 9131f2e6ec](https://github.com/croll83/llama.cpp-dgx/blob/9131f2e6ec8f34a733fe9b141a2c5b07c16b7645/README.md) | Historical GB10-specific GDN tiling, NVFP4, DFlash and TurboQuant integration. | **Do not adopt.** The maintainer's notice says it is deprecated in favor of upstream, citing convergence, draft-cache growth and quality problems in experimental compressed-KV paths. The SM-specific name is not evidence it beats the current release. |
| [Aristo94/EngramHalo.cpp, ed1eab898e](https://github.com/Aristo94/EngramHalo.cpp/blob/ed1eab898e8582931a42f01585aa007a65f1ded3/docs/strix-halo/README.md) | Same Flash-Next model: sparse QSA gather, lazy PLE residency and long-context diagnostics. | **Architectural reference only.** Explicitly ROCm/HIP gfx1151-qualified; even its Vulkan results do not transfer cleanly. Linux page-cache advice and AMD kernel selection are not Windows CUDA patches. Prefer the narrower current QSA proposal before a gather rewrite. |

Additional screen-outs: [quimmedes/cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp/tree/986650e9847246e5750f1c4d9d6aa0acab469268) emphasizes host/disk expert capacity and serving/model additions; its README does not establish a faster Windows ARM64 control. [straightdot/LlamaBwell](https://github.com/straightdot/LlamaBwell/blob/942569a13853b67c2bb96cc1ba7ad4288121c8af/README.md) publishes Windows x64/RTX 5080+4060 Ti results on an older base; source is offered as a release archive, not reviewed in this pass. Neither is shortlisted as an ARM64 binary replacement. [stew675's adaptive-MTP proposal, ggml-org/llama.cpp#27210](https://github.com/ggml-org/llama.cpp/pull/27210), is real prior art, but recommends an upper depth of at least seven; it does not overturn our measured MTP3 decision or justify another broad depth sweep.

**Already-upstream exclusions.** Low-bit MMVQ [ggml-org/llama.cpp#30517](https://github.com/ggml-org/llama.cpp/pull/30517) is an ancestor of this checkout at `2744e48d1e`; chunked CUDA/HIP GDN is present at `62df1815b` ([ggml-org/llama.cpp#29353](https://github.com/ggml-org/llama.cpp/pull/29353)). Keep these, native sm_121a, CUDA graphs, MTP/DFlash, NVFP4, routing/HC fusion and PLE prefetch out of the new-port list. An impressive fork comparison against an older baseline does not make them new work.

**Changed next step (v9).** The toolchain blocker is resolved. A native ARM64 toolchain now builds full CUDA binaries on this host with no x64 emulation: VS 2022 Build Tools (ARM64-hosted MSVC 14.44), ARM64 CUDA 13.4 redistributables, CMake 4.x and Ninja (`tools/build_native_cuda.ps1`, about 4-5 min per build). The #30380 matched A/B is complete and passes (section 2.13). Do not build a static MTP vocabulary subsystem; the dynamic path covers it.

Build differences from the official release: the release cross-compiles only `ggml-cuda.dll` with MSVC and builds everything else with Clang 20; our build is all MSVC. GPU kernels are the same compiler; CPU-side code differs. Before deploying a custom build, run a parity pair (official b11756 versus our unmodified b11756 build) on the same harness.

Remaining fusion candidates, each against its own merge-base control:

- **QSA k-pool mask, [ggml-org/llama.cpp#30640](https://github.com/ggml-org/llama.cpp/pull/30640).** Adds `GGML_OP_KPOOL_MASK` (CPU and CUDA). On by default when the model has `indexer_kpool>0` (Flash-Next); no runtime toggle, so the control is the merge-base build. Mask-only change; attention numerics do not change. Gate: `test-backend-ops -o KPOOL_MASK`, then Flash-Next prefill at 2K and 8K plus decode at fixed ubatch.
- **Dense FFN MMQ fusion, [ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948).** Fuses up, gate and GLU in one MMQ pass for prefill and large batches; decode (MMVQ) is not affected. Needs same type for gate and up, type in {Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS} and `ne01 % 128 == 0`. In the 27B UD file, 37 of 65 layers have matching gate/up types and 35 use a supported type (IQ4_XS 20, Q5_K 9, Q4_K 5, Q6_K 1). Gate: backend-ops MUL_MAT/GLU fusion tests, then 27B `llama-bench -p 512,2048 -n 128`.

Keep the optional 64 GB machine idle unless a specific fitting job is assigned, and confirm any final performance decision on the 128 GB machine.

---

## 4. Base selection

| Criterion | **Upstream llama.cpp** | GenerelSchwerz | Strata | UnbiasedStrata |
|---|---|---|---|---|
| Runs on Windows ARM64 + CUDA | **Yes, official builds** | Untested (Linux-qualified) | No (GB10 ports unmerged) | No (x86-64 Linux only) |
| Your three models | **All three** | Inherits upstream | Flash-Next only | Subset |
| Divergence / rebase cost | 0 | 487 ahead / 258 behind, core scheduler rewrites | Separate engine | Separate engine |
| **Decision** | **Keep as base** | Inspect isolated algorithms only | Benchmark isolated algorithms only | Reuse pinned-source and correctness-gate practices |

**Recommended development shape:** keep the official upstream build as the control. Do not start a long-lived engine fork or add `cc == 1210` branches yet. For each proposed change, first build a repeatable benchmark and correctness test; keep experiments small enough to upstream or discard. A forked DLL is justified only after a change shows repeatable value on this hardware and has a clear Windows ARM64 maintenance owner.

---

## 5. Revised roadmap

The v2 step estimates and cumulative tok/s projections below remain withdrawn. They combined unverified gains, double-counted upstream work, and lacked repeated A/B evidence. Sections 2.7-2.11 provide controls and validation decisions, not a cumulative forecast. Prioritize en-US and code; keep other languages non-gating. The bounded English follow-up is complete; broad benchmarking remains stopped.

| Order | Work | Evidence needed to proceed | Stop condition |
|---|---|---|---|
| 0 | Completed the installed-binary control in section 2.7. Refresh it when binary, model, settings, or workload changes. | Five samples per case, actual hashes, exact commands, full outputs, logs, and telemetry. Cold cache and operator profiles remain unmeasured. | A patched-binary claim needs an unmodified build from the same source/toolchain, not only this release control. |
| 1 | Completed the English quality gate and matched speed pair: Q4_K +4.50% in the new pair; smaller +1.73% in earlier reloads. Retain as an opt-in English/code candidate. | The recorded scope passes; general writing, arbitrary code and much longer context require their own evidence before broader claims. No further sweep is queued. | Do not replace the deployed default or advertise a guaranteed gain from one pair. Router F32 to BF16 is not a CLI-only experiment here. |
| 2 | Dynamic draft head [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380): matched native A/B passed (section 2.13), +2.63% geomean at K=16,384, token-identical, +238 MiB, +4 s load. | Toolchain parity pair, then a b11756+PR build with the same gain before deployment; enable only on the Flash-Next recipe. | No claim beyond the four en-US cases; chat is flat. Do not add to the Q4_K gain without a combined measurement. |
| 3 | QSA k-pool mask [ggml-org/llama.cpp#30640](https://github.com/ggml-org/llama.cpp/pull/30640) reviewed safe; worktrees ready (section 3.1). CUDA graph-chunk proposal not started. | Same-source control/candidate builds, `test-backend-ops -o KPOOL_MASK`, Flash-Next prefill 2K/8K and decode A/B. | Do not port whole forks or interpret existing graph/fusion ablation losses as remaining headroom. |
| 4 | MTP2/3/4 and Nemotron p_min comparisons executed. Keep MTP3; the hindsight depth selector has only 2.00% headroom on the English/code slice before overhead. | Only revisit if a representative English workload shows a larger opportunity, with per-step cost traces and measured switching/capture cost. | No adaptive implementation is justified by the current per-request aggregates or different-output throughput alone. |
| 5 | Source gate assessed; grouped MoE verify prototype blocked, not implemented. | Actual per-layer routed-ID overlap, bytes saved, launch/packing cost, and correctness under rollback/multi-sequence batches. | Do not replace missing route measurements with uniform-routing assumptions or target-only timing proxies. |
| 6 | Dense FFN MMQ fusion [ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948) reviewed safe; 35 of 65 27B layers eligible (section 3.1). Assess before new grouped-MoE work. Grouped prefill still needs evidence that expert GEMMs dominate; NVFP4 remains a separate format/quality experiment. | Actual quant/dispatch coverage, same-build speed and correctness; routed distributions for a grouped kernel; stable Windows ARM64 execution. | No assumed 1.5-2x improvement, no double-counting overlapping fusions, and no swapping model precision inside a kernel-only comparison. |
| 7 | Treat per-request speculative server controls as a separate serving/API decision, not an engine speed step. | Confirm the desired request semantics and server-scope fit; implement tests proving parameters are honored rather than merely echoed. | Do not combine this with speculation algorithm work or claim it improves single-request throughput. |

There is no defensible cumulative target or delivery timeline yet. Report each prototype's measured delta independently; add a combined projection only after combined measurements confirm interactions and no double counting.

### Historical v2 estimates (withdrawn; retained for audit only)

Everything below this notice in section 5 records the original v2 roadmap. It is not the current implementation plan, and its effort estimates and speed projections should not be used for prioritization.

### 5.0 Execution order and cumulative projections (historical, withdrawn)

| # | Step | Code changes? | Effort | Why this position |
|---|---|---|---|---|
| 0 | Toolchain + speed/correctness harness | — | 1 wk | Prerequisite for everything |
| 1 | **Quant/config tuning:** Flash-Next dense Q8_0 → Q6_K remix; MTP head Q8_0 → Q4_K; MoE routers F32 → BF16 (`llama-quantize --tensor-type`) | No (model files only) | 1 wk | Cheapest, biggest certain win for the default model; only quality must be gated |
| 2 | **Per-request speculation params** in `llama-server` (parse `speculative.n_max/p_min/types`; currently echoed only) | Small server patch, upstreamable | 1 wk | Lets each workload use its best measured config (27B n=4 for prose, Nemotron p-min) and allows A/B tests without restarts |
| 3 | **Draft-vocabulary subsetting** for MTP/DFlash heads (`d2t`-style id map) | `common/speculative` + converter | 2–3 wk | Removes the measured largest waste: full-vocabulary head reads on every draft pass |
| 4 | **Adaptive speculation controller** (rolling acceptance → n_draft/p_min per slot, uses the §2.3 cost model) | `common/speculative` + server | 1–2 wk | Robustness: no content type slower than plain decode |
| 5 | **Decode kernel fusion** — Flash-Next first (hyper-connections, router+top-k, 10+1 experts, linear-attention), then Nemotron | `ggml-cuda` | 5–7 wk | Largest single gain, but biggest effort and rebase risk; only Flash-Next has wide headroom (60–63% bandwidth) |
| 6 | **MoE verify expert deduplication** — *gate: first measure expert overlap in verify batches; skip if < 20%* | `ggml-cuda` (`mul_mat_vec_q_moe`) | 3–4 wk | Payoff depends on routing overlap, which isn't measured yet |
| 7 | **Prefill track** (can run in parallel): NVFP4 FP4 tensor cores for 27B/Nemotron (upstream kernels exist; mostly conversion + validation); Flash-Next grouped GEMM for 512 small experts; Nemotron chunked SSD scan | Model files + `ggml-cuda` | 2–3 wk (FP4) + 5–6 wk (FN) + 3 wk (SSD) | Independent of decode work; matters for first-turn and large tool-output latency |

**Cumulative projection after all steps** (tok/s; range = low…high estimate; prefill = prompt tok/s):

| Model | Plain decode | Spec: code | Spec: prose | Spec: low-predictability | Prefill |
|---|---|---|---|---|---|
| **Flash-Next** (today 31 / 48.6 / 45.1 / 39.0 / 890) | **41–47** (+32–52%) | **68–93** (+41–91%) | **64–88** (+42–94%) | **54–75** (+40–91%) | **1,340–1,870** (+50–110%) |
| **Nemotron** (today 30.1 / 46.1 / 45.2 / 33.0 / 880) | 32–34 (+5–14%) | **52–64** (+12–40%) | **52–66** (+15–45%) | 37–48 (+13–44%) | **1,100–1,410** (+25–60%) |
| **Qwen3.8-27B** (today 15.6 / 48.2 / 37.8 / 30.9 / 962) | 16–17 (+2–10%) | 50–56 (+3–16%) | 41–47 (+7–25%) | 33–39 (+7–25%) | **1,300–1,730** (+35–80%) |

**Per step, where the gains land:**
- **Flash-Next:** steps 1 (+12–20% speculative), 3 (+6–10%) and 5 (+20–30% plain, +15–25% speculative). Prefill comes from step 7.
- **Nemotron:** steps 3 (+4–8%), 5 (+4–10%), 6 (+4–12%) and 7 (prefill).
- **27B:** steps 2 (+3–6% prose via n=4), 3 (+2–5%) and 7 (FP4 prefill +35–80%). Its plain decode is already at 82% of bandwidth.
- **Physical ceilings** (85% of 300 GB/s): Flash-Next plain ≈ 51 tok/s after the step-1 remix (4.96 GB/token); Nemotron ≈ 34; 27B ≈ 16.8.

### Phase 0 — Toolchain and harness (≈1 week)
- **v9 note:** superseded. Builds now run natively on the ARM64 host (`tools/build_native_cuda.ps1`); see section 3.1.
- **Build:** cross-compile `ggml-cuda` like upstream CI. On an x64 Windows host (or a GitHub Actions `windows-2022` runner): `vcvarsall amd64_arm64`, toolchain `cmake/arm64-windows-msvc-cuda.cmake`, flags `-DGGML_BACKEND_DL=ON -DGGML_CPU=OFF -DGGML_CUDA=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3`, CUDA 13.4.
  - Dev builds use `-DCMAKE_CUDA_ARCHITECTURES=121a-real` only, for faster compiles.
- **Deploy:** copy `ggml-cuda.dll` into `C:\llama.cpp`, keeping the stock DLL for rollback.
- **Harness:** `bench/run_matrix.py` + `bench/run_llamabench.py` are the speed gate (median of 3, ±3%).
- **Correctness gates:** `test-backend-ops`, `llama-perplexity` within 1% on a 64K-token text, teacher-forced KL against stock.

### Phase 1 — Configuration only (done or ready, ~1 day)

| Change | Measured / expected |
|---|---|
| **Nemotron:** DFlash n=5 → **MTP n=3 + `--spec-draft-p-min 0.5`** (no external draft) | **+10% code, +6% prose, low-predictability 16.6 → 32.4 tok/s [M]**; −1.3 GiB VRAM vs DFlash |
| **27B:** DFlash n=7 for coding agents; n=4 for prose/mixed | n=4: prose +3–5%, low-predictability +4%, code −7% [M] |
| **Flash-Next:** keep MTP n=3; `-ub 512` (≥ 2048 only costs VRAM); never 4096 at 262K | [M] |
| Update to the current build (b11756) | **Done 09:00 [M]:** kernels ±3% (noise); Nemotron MTP code +17%, 27B prose +5–7% (§2.6) |
| Windows *Best performance* power mode on AC | Prefill +5–10% [I] (power-cap events observed) |
| Carve-out: 96 GB for Nemotron/Flash-Next; the **27B needs only ~40 GiB**, so a 64 GB carve-out would give Windows room to cache models (reloads in seconds) | Load time only |

### Phase 2 — Speculation engine work (5–7 weeks) — biggest win for all three models

| # | Change | Idea from | Expected [I] |
|---|---|---|---|
| S1 | **Adaptive speculation controller:** per slot, track rolling acceptance; pick n_draft 0–7 from the measured cost model (MoE ≈ 1.18 + 0.30·n, dense ≈ 1.1 + 0.06·n); turn speculation off below break-even; per-request override support | Strata acceptance gating | Nemotron low-predictability text 16–22 → **30** (+36–80%); +3–8% average elsewhere |
| S2 | **Draft-vocabulary subsetting for MTP heads:** score a ~32K-token subset and map back to full ids, as EAGLE-3's `d2t` does. Interim: requantize the MTP head to Q4_K/NVFP4. | Strata `--draft-vocab` | Flash-Next speculative **+15–20%** (draft pass 0.79 → ~0.21 GB); Nemotron +6–9% |
| S3 | **Expert-deduplicated grouped GEMV for verify batches**: read each unique expert once per layer for all k+1 tokens | Strata EXPDEC, GenerelSchwerz grouped decode | MoE verify +5–12% (only shared experts are saved) |
| S4 | Content-aware draft source: n-gram drafting for code edits and repeated context; MTP/DFlash otherwise | Strata prompt-lookup; upstream `ngram-mod` | Code-edit turns +10–30% |

### Phase 3 — Flash-Next decode and prefill kernels (5–7 weeks)

| # | Change | Expected [I] |
|---|---|---|
| K1 | **Decode fusion** for qwen4exp: 4-way hyper-connection mixing fused into the preceding/following GEMVs; RMSNorm + q8_1 quantization; router + top-k; 10 routed + shared experts in one launch; linear-attention update; CUDA graphs + PDL throughout | Plain decode 31.8 → **38–42** (+20–30%); carries into speculative decode |
| K2 | **Prefill: grouped GEMM for 512 small experts** (640-wide FFN) with per-expert token bucketing; fused hyper-connections | Prefill 900 → **1,400–1,900** |
| K3 | GPU-resident PLE lookup (or pinned host with async prefetch) instead of lazy SSD/page-cache reads | Prefill +3–9% on new text; less first-token jitter |
| K4 | Optional quant mix: the 3.80 GB/token of Q8_0 dense weights at Q6_K | Plain decode +15% if quality gates pass |

### Phase 4 — Prefill on tensor cores for Nemotron and 27B (4–5 weeks)

| # | Change | Expected [I] |
|---|---|---|
| P1 | **NVFP4 weights** using upstream's Blackwell W4A4/W4A8 MMQ (#24364/#29857), keeping sensitive tensors at Q6_K/Q8_0 | 27B prefill 950 → **1,300–1,800**; Nemotron 895 → 1,100–1,300 |
| P2 | Chunked Mamba-2 SSD scan (Nemotron) | Nemotron prefill +10–20% |
| P3 | 40K first-turn time to first token (Nemotron) | ≈47 s → **30–36 s** |

### Phase 5 — Hardening (ongoing ≈20%)
- Flash-attention startup-crash retry/fallback; Windows TDR limit (<2 s per kernel); 1-hour soak test with RSS/VRAM flat.
- Weekly rebase; UNBIASED-style `UPSTREAM.md` changelog per patch.

**Total:** ≈3–4 months. The Phase 1 gains are available now. S1/S2 deliver the next visible step, about 3–5 weeks in.

---

## 6. Why (explanation)

- **Decode is bound by memory bandwidth.** Tokens/s ≈ effective bandwidth ÷ bytes read per token.
  - At 300 GB/s: 27B → ~19 tok/s at the limit (15.8 measured); Nemotron → 40 (30.4 measured); Flash-Next → 52 (31.8 measured).
  - The 27B and Nemotron have less measured bandwidth headroom than Flash-Next, but this is not a complete latency attribution.
  - Flash-Next's lower effective bandwidth is consistent with many small experts, hyper-connections, PLE and the attention indexer. It does not prove launch overhead dominates. The hyper-connection gate chain and grouped-expert top-k already have upstream fusions; use profiling to find remaining work.
- **Speculation multiplies decode** by tokens per step ÷ step cost.
  - Dense models verify k drafts for nearly the price of one token (same weights).
  - MoE models pay again for every token's experts and for every MTP draft pass's full-vocabulary head. So the MoE speedup ceiling is lower, and long drafts (n=5) are worse than n=3.
  - A reduced MTP vocabulary and adaptive draft depth remain hypotheses. DFlash/DSpark already support optional `d2t`; the remaining proposed mapping is for MTP.
- **Prefill needs operator-level evidence.** The effective TFLOPS values are rough calculations from active parameter counts, not measured hardware utilization. Blackwell FP4 MMQ is already implemented, but only model-specific format, dispatch, and quality tests can establish whether it helps.
  - A grouped-expert prefill kernel is a source-backed research lead, not a known fix. The RTX 3090 Strata measurements are not comparable enough to set a Spark target.
- **Unified memory with a carve-out:**
  - Bandwidth is identical whatever the split; only capacity moves.
  - With 96 GB for the GPU, Windows keeps 30 GB. That is why models reload from NVMe (15–35 s) and why Flash-Next's 28.8 GB PLE table has to be read lazily.
  - Expert-cache/offload designs are low relevance while the tested model fits. PLE residency is a separate capacity question; upstream row prefetch is already available.

---

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| Raw long documentation prompt stops immediately under natural EOS and varies under forced continuation | Keep it excluded. The repaired chat-templated retrieval fixture passes at 17,225 tokens; do not extrapolate that to general document quality or 262K |
| Reload variation overlaps the small Q4_K gain | Report per-launch and per-case results; keep it experimental rather than advertise a universal improvement |
| Low-predictability results use synthetic random-word text; real RAG/summaries may behave differently | Validate fixed `p_min` and draft-depth policies on held-out real workloads before claiming a robust controller |
| Reduced MTP vocabulary can omit useful draft candidates | Keep full-vocabulary target verification and fallback; validate token mapping, rare identifiers, English/code acceptance and task quality. Other languages are optional diagnostics, not promotion gates |
| FP4 model quality and dispatch are model-specific | Test quantization, selected CUDA kernel, quality, and throughput separately; mixed precision only where measured necessary |
| Proposed gains overlap with upstream work and with one another | Rebase the baseline, list already-present fusions, measure each change independently, then measure combinations |
| Server request overrides are not currently exposed | Treat as an API behavior change; add schema and request-path tests if separately approved |
| Laptop power/thermal limits (45–80 W) | Report sustained (10-minute) numbers; power mode |
| Upstream churn in `ggml-cuda` | Small patch surface; weekly rebase; CI speed gate |
| Windows specifics (WDDM, TDR, no TCC) | Measure all gates on this machine; the DGX Spark discussion is not a direct Windows control |

---

## 8. Appendix

### 8.1 Launch configurations supported by the repeated timing results

The Q8_0 commands remain the unchanged baseline. For the validated opt-in Flash-Next en-US candidate, replace only `-md` with `candidate_path` from `results/rebaseline-20261020/head-q4k/quantization.json`; retain MTP3 and ub512. That JSON also records the original path and both hashes for rollback. The benchmark work did not modify the deployer.

```powershell
# Qwen3.8-Flash-Next (unchanged from your deployer config)
llama-server -m <Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf> --mmproj <mmproj-BF16.gguf> -c 262144 -np 1 -fa on -lzm on `
  -b 2048 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3 -md <mtp-Qwen3.8-Flash-Next-Q8_0.gguf>

# Nemotron-3.5-Super: native MTP, default p_min; the blanket 0.5 recommendation is withdrawn
llama-server -m <Nemotron-...-Q4_K_M-00001-of-00002.gguf> --mmproj <mmproj-BF16.gguf> -c 262144 -np 1 -fa on --fit off `
  -b 4096 -ub 4096 --spec-type draft-mtp --spec-draft-n-max 3

# Qwen3.8-27B: n=7 for coding, n=4 for prose/mixed
llama-server -m <Qwen3.8-27B-UD-Q4_K_M.gguf> --mmproj <mmproj-F16.gguf> -c 262144 -np 1 -fa on -b 2048 -ub 2048 `
  --spec-type draft-dflash --spec-draft-n-max 7 -md <Qwen3.8-27B-DFlash2-Q4_K_M.gguf> --spec-draft-ngl 999
```

### 8.2 Files in `rtx-spark/`
Scripts are in `bench/`; they read `results/model_paths.json` (local and untracked; copy `results/model_paths.example.json`) and write to `results/` (server logs to `results/logs/`). The published branch omits raw server logs, full generated responses (`server_*.json`) and local paths.
- `bench/run_rebaseline.py`: repeated native and server measurements; isolated result directories, exact output-length checks, optional natural EOS and external draft replacement. Current evidence is in `results/rebaseline-20261020/`.
- `bench/heldout_suite.py`, `bench/analyze_mtp_tokens.py`: pinned held-out prompts, low-ID coverage, and per-position acceptance. Plan execution evidence is in `results/plan-20261020/`; `execution_summary.json` and `execution_manifest.json` index the final results and limitations.
- `bench/run_matrix.py`: launch → benchmark → Ctrl+C driver; `results/matrix_results.json`, `results/logs/server_*.log`.
- `bench/run_llamabench.py`: `results/llamabench_q27.md`, `results/llamabench_qfn.md`, `results/llamabench_nem.md`.
- `bench/bench_ple.py`: `results/ple_diversity_qfn.json` (lazy-PLE diversity test).
- `bench/bench_server.py`, `bench/bench_nospec.py`, `bench/bench_spec_*.py`: earlier live-server runs (`results/*_nemotron.json`, `results/baseline_qfn*.json`, `results/depth_qfn_nospec.json`).
- `bench/gguf_geometry.py`: `results/geometry.txt` (per-tensor bytes).
- `bench/perf_model.py`: `results/perf_model.txt` (v1 cost model; superseded by the measured economics in §2.3).
- `bench/summarize_matrix.py`: tables in §2.3.

### 8.3 Key sources
- Strata: [README](https://github.com/Niko1221/Strata), [HOW_IT_WORKS](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/HOW_IT_WORKS.md), [DETAILS](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/DETAILS.md), [MAC](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MAC.md), [STRIX_HALO](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/STRIX_HALO.md), GB10 ports [#409](https://github.com/Niko1221/Strata/pull/409) [#1077](https://github.com/Niko1221/Strata/pull/1077) [#2419](https://github.com/Niko1221/Strata/pull/2419), [#774](https://github.com/Niko1221/Strata/issues/774)
- UnbiasedStrata: [README](https://github.com/robertkarl/UnbiasedStrata), [UPSTREAM.md](https://github.com/robertkarl/UnbiasedStrata/blob/main/UPSTREAM.md), [PLAN.md](https://github.com/robertkarl/UnbiasedStrata/blob/main/PLAN.md)
- GenerelSchwerz: [README](https://github.com/GenerelSchwerz/llama.cpp), [fork-features.md](https://github.com/GenerelSchwerz/llama.cpp/blob/moe-cache/docs/fork-features.md), [benchmarks](https://github.com/GenerelSchwerz/llama.cpp/wiki/Benchmark-Comparison-Showcase)
- llama.cpp: [DGX Spark performance #16578](https://github.com/ggml-org/llama.cpp/discussions/16578), [MoE graph warm-up #14753](https://github.com/ggml-org/llama.cpp/pull/14753), [Nemotron MTP #29018](https://github.com/ggml-org/llama.cpp/pull/29018), [DFlash #22105](https://github.com/ggml-org/llama.cpp/pull/22105), [DSpark #25173](https://github.com/ggml-org/llama.cpp/pull/25173), [NVFP4 #24364](https://github.com/ggml-org/llama.cpp/pull/24364), [#29857](https://github.com/ggml-org/llama.cpp/pull/29857), [grouped MoE top-k #29181](https://github.com/ggml-org/llama.cpp/pull/29181), [qwen4exp hyper-connection fusion #30602](https://github.com/ggml-org/llama.cpp/pull/30602), [PLE row prefetch #29599](https://github.com/ggml-org/llama.cpp/pull/29599), [adaptive speculation research #27821](https://github.com/ggml-org/llama.cpp/issues/27821) and [#29875](https://github.com/ggml-org/llama.cpp/issues/29875), [MTP reduced-vocabulary research #25187](https://github.com/ggml-org/llama.cpp/issues/25187), [release workflow (Windows ARM64 CUDA)](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/release.yml)
- N1X specs: [TechPowerUp](https://www.techpowerup.com/gpu-specs/rtx-spark-n1x-6144-core.c4410), [ComputingForGeeks](https://computingforgeeks.com/nvidia-rtx-spark-linux/)
