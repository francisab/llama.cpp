# RTX Spark Inference Engine - en-US Optimization and Fork Survey (v11)

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
> **v10 deployment:** a b11756 + #30380 build that follows the official release layout (Clang 20 CPU side, MSVC CUDA) now replaces the official install. The Flash-Next recipe enables `--spec-draft-vocab-dynamic 16384`: +3.24% / +3.35% over official, token-identical, 8/8 quality checks. Nemotron and 27B are unchanged within noise (section 2.14). The official install and catalog are backed up for rollback.
> **v11 QSA follow-up:** #30640 passed CUDA backend tests and matched Flash-Next A/B checks. The b11756 + #30380 + #30640 release-like build measured +2.43% over the deployed #30380 build, with 20/20 identical outputs and 8/8 quality checks, and is now deployed (section 2.15). #29948 is not deployed: repeated 27B results are small and noisy, and its patch does not apply cleanly to b11756 + #30380 (section 2.16). The fork branch was fast-forwarded to upstream master merge `96bbbd788`; #30640 remains open, with a test-only update after the measured head.
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
8. **Keep upstream as the base and measure one proposal at a time.** The dynamic draft head and QSA-mask fusion are locally validated; #29948 did not show a stable 27B win. Treat KV compression/streaming and CPU expert caches as capacity tools, not automatic speedups when the relevant weights and cache fit. The Spark-specific croll83 fork itself now recommends upstream. See section 3.1 for pinned sources, applicability and build blockers.

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

### 2.14 Deploy validation: b11756 + #30380 release-like build [M]

Evidence is in `results/deploy-b11756-20261021/` (`summary.json`). The PR diff applies cleanly to b11756 (`d80619397`). Two build styles were compared with the official b11756 release on the same harness: U1 with five repetitions, N2 and D2 with three. Fresh official runs were interleaved to bound drift. All U1 runs use the Q4_K draft in both arms.

| Comparison | Geomean | Outputs |
|---|---:|---|
| Official vs official repeat (U1, two pairs) | -0.09%, +0.11% | identical |
| Official vs all-MSVC b11756, unpatched | -0.98% | identical |
| Official vs all-MSVC + PR, K=16,384 | +2.28% | identical |
| Official vs release-like + PR, flag off | -0.71% | identical |
| Official vs release-like + PR, K=16,384 (two control runs) | +3.35%, +3.24% | identical |
| Release-like vs hybrid with official `ggml-cuda.dll`, K=16,384 | +0.09% | identical |
| N2 official vs release-like + PR, flag off | +0.80% (official drift -0.90%) | identical |
| D2 official vs release-like + PR, flag off | +0.43% (official drift +1.00%) | identical |

The release-like build (`tools/build_release_like.ps1`) mirrors the release jobs:
- CPU side and tools: Clang 20.1.8 from the LLVM ARM64 release tarball with the release CMake flags. The file set matches the official zip (42 files).
- `ggml-cuda.dll`: MSVC with CUDA 13.4, `sm_121a` only.
- The upstream OpenMP fetch needs 7-Zip; the script seeds its cache from the same LLVM 20.1.8 tarball.

The all-MSVC build is about 1% slower than official; the release-like build recovers most of it. The U1 flag-off gap comes mostly from one chat median. With the flag on, the official `ggml-cuda.dll` changes nothing (+0.09%). CUDA runtime DLLs are byte-identical to the official zip. The release-like build also halves scorer setup (1.61 s vs 3.48 s with MSVC). Quality gate at K=16,384: 8/8 checks pass, and `-lv 4` shows `vocab_subset ... 16384 picked per output`.

**Deployed.** The release-like build replaces the official install directory (SHA-256 verified copy; the official directory is kept as a timestamped backup). Only the Flash-Next launcher recipe gains `--spec-draft-vocab-dynamic 16384` (catalog backed up). A smoke test with the exact recipe command (`-hf`, HTTPS) passes: 52.7 tok/s, draft acceptance 101/144, correct output. In this test `-hf` also fetched the repository's 3.9 GB Q8_0 MTP file once; the recipe still loads the local Q4_K draft. Without symlink privilege the file is stored as a plain file.

Rollback: rename the backup directory back and restore the catalog backup. If the install directory is replaced with an official release that lacks #30380, also remove the flag: `llama-server` stops at startup on unknown arguments. The gain over the original Q8_0 recipe is not measured in one pair.

This deployment was superseded by the b11756 + #30380 + #30640 build in section 2.15. The previous #30380-only build is retained as the rollback backup; the launcher catalog was not changed during the #30640 deployment.

### 2.15 QSA k-pool mask, PR #30640, native ARM64 A/B and deployment [M]

Evidence is in `results/ab-30640-20261021/`. Control and candidate were built at merge-base `5e03a1ff4` and PR head `8a8621232`. Both use the same Flash-Next model and native ARM64 CUDA build. The CUDA `KPOOL_MASK` backend-ops filter passes 12/12 candidate cases.

The paired `llama-bench` run (two rounds, three repetitions per round) gives +0.68% at tg128, +3.98% at pp2048, and +6.07% at pp8192. In the five-repetition en-US server suite, decode improves +1.42% geomean; all 20 outputs are token-identical and draft acceptance is unchanged.

**Release validation:** the release-like b11756 + #30380 + #30640 build has the same 42-file layout as the #30380 control. Against the deployed #30380-only build, the U1 en-US suite improves +2.43% geomean; all 20 outputs are token-identical and draft acceptance is unchanged. The natural-completion quality gate passes 8/8 warm-up and measured checks. A live smoke request using the existing launcher recipe returns the requested marker at 47.2 tok/s with draft acceptance 31/42.

**Deployed.** The current install is b11756 + #30380 + #30640. Its predecessor is kept in a timestamped sibling backup; the catalog was not changed. A local rollback script restores the predecessor. The measured candidate was PR head `8a8621232`. The current upstream head is `2961d5e5c`; its two implementation commits are unchanged and its only new commit adds three lines to initialize test tensors. Runtime code therefore matches the tested candidate, but the updated test fixture itself has not yet been rerun.

### 2.16 Dense FFN MMQ fusion, PR #29948, native ARM64 A/B [M]

Evidence is in `results/ab-29948-20261021/`, `results/ab-29948-20261021-r4/`, and `results/ab-29948-20261021-pp2048-r4/`. The candidate at `0ba2b5f2a` passes 1270/1270 `MUL_MAT_VEC_FUSION` backend-ops tests; the control passes 1265/1265.

Initial two-round results showed tg128 -6.25%, pp512 +8.35%, and pp2048 +4.28%. Follow-up four-round runs reduce the apparent changes: at pp512, tg128 is -1.43% and prefill +1.66%; at pp2048, tg128 is +1.29% and prefill +2.42%. Sample ranges overlap broadly. Do not treat these small changes as a stable performance win.

**Not deployed.** The patch also fails a dry-run application to b11756 + #30380 at `ggml/src/ggml-cuda/mmq.cuh:1443`. Rebase/integration and another matched comparison are required before considering it for a release build.

---

### 2.17 Fail-fast local model-precision experiments, 2026-10-21 [M]

These are our model-artifact experiments, not integrations of upstream engine patches. The deployed b11756 + #30380 + #30640 binary and Q4_K draft are unchanged. Server screens use rust256 and json256, one warm-up and two measured responses per case, MTP3/ub512, and dynamic draft vocabulary K=16,384. The declared thresholds are 3% minimum gain per case and 3% maximum regression; a first-pair survivor receives one reversed-order pair.

| Candidate | Isolated change | Measured screen | Decision |
|---|---|---|---|
| Q3_K draft output head | Converted from the original Q8_0 head; all 33 other tensors unchanged; 80.52 MiB smaller than the Q4_K control | Two-case decode geomean -7.46%; lower draft acceptance; 4/4 outputs identical | Reject after first pair |
| Q2_K draft output head | Converted from the original Q8_0 head; all 33 other tensors unchanged; 142.09 MiB smaller | Decode -10.01%; lower acceptance; 4/4 outputs identical | Reject after first pair |
| Mixed-Q4 routed draft FFN | Gate/up tensors Q8_0 to Q4_K, down tensor to Q4_0 because its 640-wide rows do not fit Q4_K blocks; 31 other tensors unchanged, including head and router; 1,200 MiB smaller | First pair +3.97%, reversed pair +0.94%; 4/4 outputs identical in each pair | Park; first gain did not hold up |
| Q6_K target attention/SSM projections | 156 eligible Q8_0 tensors changed; 1,068 other tensors and non-layout metadata unchanged; 617.58 MiB smaller across three separate shards | Plain tg128 +6.52%; pp2048 -1.01%. Five-repeat actual MTP comparison: Rust +6.29%, JSON +6.61%, geomean +6.45%; 0/10 outputs identical. All 8 bounded natural-EOS English checks passed for the candidate | Revise after the later KL precision gate failed (section 2.20); not deployed |

Evidence: `results/draft-head-screen-20261021/summary.json` and `results/target-q6-attention-20261021/summary.json`, with conversion manifests, tensor hashes and per-comparison data. The target conversion took 163.5 s and conversion plus full tensor verification took 706.5 s; this preparation cost is separate from the native screen budget. All original GGUFs were preserved.

The five-repeat actual MTP comparison used control then Q6, with five fixed-length measurements per case and recorded GPU telemetry. Median throughput rose from 50.96 to 54.33 tok/s for JSON (+6.61%) and from 54.08 to 57.48 tok/s for Rust (+6.29%); the sample ranges did not overlap. The geometric-mean gain was 6.45%. The reversed-order pair showed a misleading +24.29% because the control ran after Q6 and its telemetry fell to 18.62 W / 1,683 MHz; exclude that pair from promotion evidence.

The Q6 candidate passed all four bounded natural-EOS English tasks for both warm-up and measured responses: 8/8 checks, including 231 code assertions, exact JSON, identifier copying, and document retrieval. Three task outputs changed, so this is not output equivalence or a general quality guarantee. A same-binary/model calibration also showed pooled tg128 +0.89% and pp2048 -7.20% across opposite launch orders; individual prefill pairs were -9.67% and -4.25%. Short native prefill screens are not reliable standalone rejection evidence on this device. The later precision check in section 2.20 fails its KL gate; broader long-context and deployment validation remain undone.

### 2.18 GLU activation-quantization fusion, PR #30489, bounded native screen [M]

This is an upstream-authored CUDA optimization, not our own kernel. The candidate `784fd01678cb91227c8de2fa1566308e7e3e2244` was built against its true merge-base control `411d548515017ca75c2a12c39659b6e7e1551dd7`, with identical native ARM64 MSVC/CUDA 13.4 flags. Flash-Next has 43 IQ4_NL and five Q8_0 expert down projections of shape `(640, 2560, 512)`, with ten selected experts; its graph has the relevant SWIGLU-to-MUL_MAT_ID chain. Runtime fusion dispatch was not instrumented.

All 266 new `GLU_MUL_MAT` and `SWIGLU_MUL_MAT_ID` CUDA correctness cases passed against the CPU reference, including layout and fallback cases. These use the upstream correctness shapes; the exact full-model expert geometry appears in the upstream performance cases, not its correctness set.

The original, unchanged target weights were measured with `-fa 1 -ngl 99 -ub 512 -p 2048 -n 128 -r 3 -lzm on`, without MTP. The predeclared protocol completed one control/candidate then candidate/control sequence within 206.75 seconds. This bounded opposite-order check was chosen before measurement because the earlier A/A calibration showed noisy short prefill results; it is not a change to the harness's automatic early-stop behavior.

| Order | pp2048 | tg128 | Decision |
|---|---|---|---|
| Control then candidate | +1.73% | +1.29% | Below the 3% prefill gate; ranges overlap |
| Candidate then control | +0.40% | -0.21% | Below the 3% prefill gate; ranges overlap |
| Pooled medians, six samples per arm and case | +1.51% | +0.26% | No stable local improvement established |

**Park, not a deployment candidate.** No integration or further repetitions are justified without a changed implementation or new bottleneck evidence. The 218 one-second telemetry samples include four software power-cap events and no hardware thermal-slowdown flag; they are not an operator profile or proof that these small deltas are causal. Full-model English quality and release-stack validation were deliberately not run after this weak speed screen. The installed binaries, launcher, and model files remain unchanged.

Evidence, predeclared thresholds, per-order results, build flags, and binary hashes are in `results/pr30489-screen-20261021/`. The subsequent graph-chunking screen is recorded below.

### 2.19 CUDA graph chunking, PR #30804, bounded native screen [M]

The upstream-authored candidate `ed93210d3b8639a23b5024ad5561e34fc69db681` was compared with its true parent/merge base `99db719bdf07f7d537431ca556a0512bf3d995b6`, using identical native ARM64 MSVC/CUDA 13.4 builds. It captures and replays chunks of 512 GGML nodes by default; zero selects a whole graph in the modified implementation, not the original parent implementation.

All 1,334 `MUL_MAT_VEC_FUSION` and `ROPE_SET_ROWS` whole-graph CUDA cases passed against the CPU reference with `GGML_CUDA_GRAPH_CHUNK_SIZE=1`, deliberately stressing short boundaries around fusion and fallback paths. This is not exhaustive graph or concurrency coverage.

The original Flash-Next target, without MTP, used the same pp2048/tg128, ub512, three-sample settings and bounded control/candidate/candidate/control protocol as section 2.18. Performance used `GGML_CUDA_GRAPH_CHUNK_SIZE=512`, not the one-node correctness setting. The complete native screen took 201.67 seconds.

| Order | tg128, primary goal | pp2048 | Decision |
|---|---|---|---|
| Control then candidate | +2.87% | +1.62% | Below the predeclared 3% decode gate |
| Candidate then control | +1.60% | -0.64% | Below the gate; decode sample ranges overlap |
| Pooled medians, six samples per arm and case | +1.95% | +0.61% | Park; insufficient promotion evidence |

**Not integrated or deployed.** The small positive result is not evidence that further improvement is impossible, but it does not justify lowering the threshold or searching chunk sizes until one wins. Telemetry has 219 one-second samples, three software power-cap events, and no hardware thermal-slowdown flag. No actual MTP throughput, full-model English quality, or release-stack validation was run for this weak candidate.

Evidence is in `results/pr30804-screen-20261021/`. Work returns to the surviving selective Q6 target model, starting with a pinned, bounded English perplexity/KL precision gate rather than another marginal upstream-patch repetition.

### 2.20 Q6 target precision gate and revision [M]

The same installed `llama-perplexity.exe` and backend DLLs compared the original and unweighted Q6 targets on the first 16 Wikitext-2 test chunks, with context/batch 1024, ub512, full GPU offload, flash attention, and lazy PLE. Dataset revision, hashes, commands, and the unchanged binary manifest are in `results/target-q6-precision-20261021/`. The predeclared limits were a mean perplexity ratio of 1.01 and mean KL divergence of 0.01 nats.

| Run against the original target's saved logits | Mean PPL | Reported PPL ratio | Mean KL, nats | Same top token |
|---|---|---|---|---|
| Original target A/A calibration | 2.202870 | 1.000358 | Approximately 0 | 100.000% |
| Unweighted Q6 target | 2.208432 | 1.002883 | 0.038966 | 94.569% |

The ratios use the tool's compressed reference logits; the unchanged-model calibration documents that numerical floor. Q6 passes the perplexity-ratio gate but exceeds the KL limit. Its earlier speed and eight English task passes remain valid within their scope, but they do not establish distribution preservation. **Do not deploy this unweighted candidate.** This is a failed numerical promotion gate, not proof of a general language-quality failure.

Both inference processes completed successfully. A console encoding error occurred only while printing the completed candidate log; saved metrics and binary hashes were audited without rerunning the candidate. The separate A/A run was a diagnostic calibration, not a retry of Q6.

The follow-up used an importance-weighted Q6 revision with the existing quantizer, the same exact 156-tensor selection, and the original Q8 source projections. Its protocol uses 64 training-split chunks of 512 tokens, never the test text for importance calibration, and keeps the same precision thresholds. The already-used test prefix is a development gate; the unused validation split remains reserved for independent confirmation. Results follow in section 2.21.

### 2.21 Importance-weighted Q6 revisions [M]

Importance collection completed in 154.172 seconds. All 156 selected projections had correctly sized, finite, nonnegative importance data and positive observations. The existing GGUF metadata-copy helper replaced only the matrix's dataset path with `wiki.train.raw`; all 3,704 importance tensor payloads were verified byte-identical. Quantization runs used the matrix basename, so generated provenance does not contain local absolute paths. Partial calibration coverage of routed experts does not authorize changing those experts.

| Fixed candidate | Original Q8_0 projections changed to Q6_K | Conversion | Precision screen | PPL ratio | Mean KL, nats | Decision |
|---|---|---|---|---|---|---|
| Training-weighted, original scope | 156 | 158.218 s | 47.219 s | 1.001779 | 0.035893 | Fails the unchanged 0.01 KL limit |
| Training-weighted, recurrent QKV protected | 120 | 157.844 s | 46.531 s | 0.998493 | 0.033887 | Fails the same limit; stop this narrowing hypothesis |

Both completed all 16 development chunks using the same saved reference, dataset bytes, executable and backend DLL hashes. Importance weighting lowered the reported KL by 7.89% relative to unweighted Q6, but did not close the precision gap. The second revision preserved all 36 recurrent `attn_qkv.weight` payloads exactly and retained the identical weighted Q6 bytes for the other 120 projections. This fixed selection was motivated by QKV feeding persistent convolution/recurrent state before output gating and projection (`src/models/qwen4exp.cpp:1070-1138`), not by a layer-by-layer score sweep. Its additional 5.59% KL reduction does not establish recurrent QKV as the main source of the discrepancy.

All 1,224 names, shapes and output types, selected-payload hashes and metadata changes were checked before each screen. The protected candidate also passed byte comparisons for all 36 preserved QKV tensors. Full unchanged-payload audits were deliberately deferred until a precision pass and therefore were not performed; do not infer complete byte preservation for either artifact. Neither candidate proceeded to the untouched validation split, task checks, MTP speed confirmation or deployment. Lower perplexity alone is not distribution preservation.

Evidence is in `results/target-q6-imatrix-20261021/` and `results/target-q6-protected-qkv-20261021/`. Only the failed full-scope weighted model's three scratch shards were reclaimed after recording its selected-payload manifest and numerical result; original weights, the earlier unweighted Q6 model, calibration data and deployment are unchanged. Continue independent model/framework hypotheses rather than relaxing the gate or repeatedly narrowing against this prefix.

### 2.22 Follow-up source gates [S]

The refreshed CUDA queue was filtered before spending more build or GPU time. Exact proposal heads and reasons are in `results/candidate-source-gates-20261021.json`.

| Proposal | Source-gate decision |
|---|---|
| [ggml-org/llama.cpp#30918](https://github.com/ggml-org/llama.cpp/pull/30918), fused sampling penalties | Not selected for the current recipe, which does not enable backend sampling. The author's roughly 22 microsecond saving requires active penalties. |
| [ggml-org/llama.cpp#30792](https://github.com/ggml-org/llama.cpp/pull/30792), native Q8 KV tile kernel | Not on Flash-Next's Blackwell dispatch path: its 256-wide heads return from the earlier Turing-MMA branch before the changed generic fallback. Do not transfer the reported gfx1151 gains to N1X. |
| [ggml-org/llama.cpp#30493](https://github.com/ggml-org/llama.cpp/pull/30493), vectorized MMQ copy | Subsequently screened in section 2.53: 64/64 correctness cases per arm, 27B prefill +1.28%, decode -0.82%. Parked at the first pair, not deployed. |

No patch from this source-only triage was built, integrated or deployed. The independent generation-thread experiment is recorded below.

### 2.23 Four generation threads [M]

The installed MTP3 recipe was compared with four target/draft CPU generation threads instead of the automatically resolved 20. Target and draft batch threads remained at 20; original target weights, the Q4_K draft, ub512, lazy PLE, dynamic vocabulary 16384, affinity, priority and polling settings were unchanged.

| Fixed 256-token case, two samples per arm | Control median | Four-thread median | Change |
|---|---|---|---|
| Rust | 55.39 tok/s | 54.54 tok/s | -1.53% |
| JSON | 51.41 tok/s | 51.48 tok/s | +0.14% |

Geometric mean: **-0.70%**, with all four output-token sequences identical and unchanged draft counts/acceptance. The first pair misses the predeclared 3% per-case gain, so no reverse-order run or thread-count sweep was performed. Keep automatic generation threads; no launcher change. The completed pair took 144.266 seconds and retained 130 telemetry samples, seven software power-cap samples and no hardware thermal-slowdown flag.

The first harness attempt completed only a control run, then rejected an outdated startup-log format. That control and its telemetry are retained separately. The parser was corrected and the matched comparison restarted before any candidate measurement; the failed attempt was not silently discarded or used to select a favorable baseline. Evidence is in `results/decode-threads4-20261021/`.

The subsequent GDN alpha/beta producer-fusion screen is recorded below.

### 2.24 GDN alpha/beta producer fusion [M]

Upstream [ggml-org/llama.cpp#29187](https://github.com/ggml-org/llama.cpp/pull/29187), head `e75c82b7750ce2163acacf3b41232f385c758135`, was built against its true merge base `1b43d311698269073fb4827db10f9d6f195d2e7d`. Both native ARM64 Release builds use MSVC 19.44.35229.0, CUDA 13.4.59, CCCL v3.4.3 and `121a-real`; installed binaries were not replaced.

All 14 selected CUDA correctness cases passed, including three local test-only additions for the actual F32 `[2560, 48]` projections, head size 128 and 1/4/8 tokens. Production source remains at the upstream candidate head. The supplemental patch, exact build settings and binary hashes are in `results/pr29187-screen-20261021/`. Source matching and these warmup-enabled tests do not constitute a production fusion-dispatch trace.

The first actual-MTP3 pair used original target weights, the Q4_K draft, ub512 and two fixed-length Rust/JSON samples per arm. Both arms omit the dynamic-vocabulary flag because their matched source base does not include that separately deployed patch. These rates are not a comparison against the installed release stack.

| Case | Control median | Candidate median | Change |
|---|---|---|---|
| Rust | 43.72 tok/s | 45.91 tok/s | +5.01% |
| JSON | 43.03 tok/s | 41.82 tok/s | -2.81% |

Geometric mean: **+1.03%**, failing the unchanged 3% per-case gate. Stop after the first pair; no reverse-order run, integration or deployment. Outputs are repeatable within each arm but **0/4** match across arms, and JSON draft acceptance falls from 76.6% to 71.6%. Thus the mixed timing result is not an isolated kernel-latency measurement or proof of model-quality equivalence. No perplexity/KL or English task campaign was spent on this speed-screen failure.

The pair completed in 153.641 seconds. Its 140 telemetry samples include nine software power-cap flags and no hardware thermal-slowdown flag. Runtime DLL hashes were unchanged after correctness and timing runs. This remains upstream work evaluated locally, not a locally authored engine improvement.

The independent Q5_K draft-head follow-up is recorded below.

### 2.25 Higher-precision Q5_K draft head [M]

One Q5_K `output.weight` revision was rebuilt from the original Q8_0 draft, not from the deployed Q4_K file. All metadata and all 33 non-head tensors match the original and the existing Q4_K control. Both original files and every installed runtime DLL were hash-verified unchanged. The candidate adds 75.78125 MiB to the Q4_K model; conversion and full verification took 108.594 seconds.

The installed b11756 + #30380 + #30640 binaries, original target, MTP3, ub512, dynamic vocabulary 16384 and automatic threads remained unchanged. The first pair used two fixed-length samples per case after warmup:

| Case | Q4_K median | Q5_K median | Change | Draft acceptance |
|---|---|---|---|---|
| Rust | 55.32 tok/s | 57.07 tok/s | +3.17% | 78.1% to 81.1% |
| JSON | 51.88 tok/s | 52.84 tok/s | +1.85% | 71.6% to 71.6% |

Geometric mean: **+2.51%**, with **4/4 identical output-token sequences**. JSON misses the predeclared 3% per-case gate, so stop without a reverse-order run, quality campaign or deployment. Higher precision improved Rust acceptance, but this pair does not establish a robust general improvement. Keep Q4_K and do not extend this into another head-precision sweep. Evidence is in `results/draft-head-q5-20261021/`.

The screen took 157.781 seconds, including post-run binary/model verification. Its 143 telemetry samples contain four software power-cap flags and no hardware thermal-slowdown flag.

The next priority is operator evidence rather than more marginal precision trials. The installed toolkit has no local CUPTI or Nsight CLI, but NVIDIA's current CUDA 13.4.3 redistributable manifest now lists native Windows ARM64 CUPTI, Nsight Systems and Nsight Compute packages. Validate a portable native profiler before treating the earlier local profiling blocker as resolved; do not infer kernel costs from the end-to-end screen results.

### 2.26 Native CUDA profiling and the next kernel hypothesis [M]

NVIDIA's portable Windows ARM64 Nsight Systems 2026.3.2.476 package was verified against its official SHA256 manifest and a valid Authenticode signature. Its actual profiler executable is ARM64, as are the profiled applications; the Python controller is x64. The package bundles the required CUPTI library. No installed model, runtime DLL, launcher recipe, CUDA installation, PATH, driver or power policy changed.

The backend-test smoke capture contains 195 CUDA kernels, including 22 GDN alpha/beta fusion kernels, and all 14 selected cases passed. It also reports 30 graph clone-ID lookup errors, so this proves execution in the backend test, not clean production graph metadata. The initial hardware-mode server capture and subsequent software-mode capture both report `Not all CUDA events might have been collected.` A separate console-launcher attempt did not fix server shutdown. These pilots are retained and excluded from kernel ranking; only their uniquely identified processes were stopped.

The bounded alternative succeeded: the installed MTP-enabled `llama-cli` completed one 128-token Rust chat request and exited normally under software CUDA tracing. It used the same original target, Q4_K draft, MTP3/ub512, dynamic vocabulary 16384 and runtime DLLs. The report contains **145,262 CUDA kernel records**, including **127,304 graph-node records**, from one GPU process and no diagnostic warnings/errors. The application reports 41 prompt tokens, 128 generated tokens and 91 accepted of 107 draft tokens. Runtime hashes and GPU/process cleanup were verified afterward.

This is a **full-lifetime diagnostic**, including model initialization, default warmup, chat prefill, generation and teardown. It is not the earlier cached raw-completion workload, pure decode, an uninstrumented speed comparison or routed-expert overlap evidence. Summed kernel duration is 2,124.425 ms, while the union of kernel execution intervals is 2,031.877 ms; overlap and idle time prevent interpreting the sum as wall time.

| Observed kernel variant | Calls | Summed duration | Share of summed kernel duration |
|---|---:|---:|---:|
| Q8_0 MMVQ, four-column variant | 18,397 | 660.247 ms | 31.08% |
| IQ3_S routed MMVQ, fused variant | 1,785 | 276.381 ms | 13.01% |
| IQ4_NL routed MMVQ, unfused variant | 1,634 | 145.739 ms | 6.86% |
| RHS `quantize_q8_1` | 25,123 | 47.451 ms | 2.23% |

The dominant Q8_0 variant uses 32 x 4 threads, 56 registers per thread and two output rows per block in the inspected source. Its leading observed grid is 5120 x 1 x 1 (4,895 calls, 212.488 ms). Kernel names and grids do not uniquely identify tensors. The next narrow hypothesis is a Q8_0 small-batch launch-geometry screen on this device, with matched source/builds, backend correctness, then the unchanged first-pair Rust/JSON gate. Do not add a cross-call RHS cache on the strength of its 2.23% kernel share; even that share includes work a cache could not remove.

Evidence is in `results/native-profile-20261021/`: package provenance, all rejected pilots, accepted CLI metadata and complete symbol/launch-geometry aggregates. Raw traces, SQLite databases, response text and application logs remain private. Normal-exit CLI tracing is now usable within its stated scope; clean bounded server tracing and routed-ID collection remain open.

### 2.27 Local Q8_0 four-column launch-geometry screen [M]

The first profile-led local kernel experiment changed only three lines in the existing `MMVQ_PARAMETERS_GB10` table: Q8_0 with four output columns uses two warps instead of four. It adds no new kernel, quantization, cache or runtime control. The pinned release CUDA build rebuilt only `mmvq.cu.obj` and relinked `ggml-cuda.dll`; all other runtime binaries match the installed control byte-for-byte. The original source and build DLL were restored, and the candidate remained in a separate runtime directory.

Both arms pass **21/21 supported CUDA correctness cases**: three existing four-column cases plus 18 generic-op cases using six original GGUF hyper-connection/attention/SSM shapes at 1/4/8 columns. Two unsupported Q8_0 x F16 cases are excluded, not counted as passes. A separate normal-exit software trace confirms six four-column Q8 launches with **32 x 2 threads**, versus the control profile's 32 x 4, with no diagnostic warnings/errors. The profiler omitted the final buffered stdout summary; the complete unprofiled run supplies the numerical results.

The actual MTP3/ub512/Q4_K/dynamic-vocabulary-16384 first pair used two fixed-length samples per case:

| Case | Installed control | Two-warp candidate | Change |
|---|---|---|---|
| Rust | 55.00 tok/s | 54.95 tok/s | -0.10% |
| JSON | 51.57 tok/s | 52.16 tok/s | +1.14% |

Geometric mean: **+0.52%**, failing the unchanged 3% per-case gate. Rust sample ranges overlap. Only **2/4** output-token sequences match across arms: JSON matches, while Rust changes despite unchanged aggregate acceptance. JSON acceptance increases from 71.6% to 73.2%. These results do not establish quality equivalence or a robust speed improvement.

Stop after this first pair, without reverse order, task-quality campaign, integration, deployment or a launch-parameter sweep. The screen took 144.188 seconds; 133 telemetry samples contain five software power-cap flags and no hardware thermal-slowdown flag. Evidence and the reproducible three-line patch are in `results/q8-fourcol-warps-20261021/`; apply the patch with `git apply --unidiff-zero`. No production source change was retained.

Next, determine whether the dominant Q8_0 kernels are bandwidth-, occupancy- or instruction-limited with a bounded native hardware-counter capture before proposing another kernel change. Their duration share alone does not explain the bottleneck.

### 2.28 Native hardware-counter preflight [M]

The official CUDA 13.4.3 manifest's Windows ARM64 Nsight Compute package was downloaded, SHA256-verified and extracted portably. The package version is 2026.3.1.2; its signed ARM64 CLI reports 2026.3.1.0, build 38829034. No CUDA, PATH, driver or access-policy change was made.

A bounded single-kernel probe used the original installed CUDA DLL and the Q8_0 `[320, 10240]`, four-column backend case. User configuration loading was disabled, clocks were left unmodified with `--clock-control none`, cache flushing was disabled, and collection was limited to one `mul_mat_vec_q` launch with the basic metric set. The numerical case passed, but the profiler returned **`ERR_NVGPUCTRPERM`**: this process lacks permission to access GPU performance counters. No counters were accepted and no bandwidth, occupancy or instruction bottleneck is inferred.

The probe exited in 6.282 seconds and its process/GPU cleanup was verified. `results/native-counters-20261021/probe.json` preserves tool provenance, settings and the original blocker. At this point, further hardware-counter work required a human-authorized elevated capture or an approved access-policy change following [NVIDIA's guidance](https://developer.nvidia.com/ERR_NVGPUCTRPERM). Section 2.57 records the later explicitly approved capture and its different driver-resource failure. Do not silently change the driver security policy or escalate again.

### 2.29 Unprivileged Q8_0 MMQ dispatch screen [M]

With administrator authorization unavailable, an ordinary backend timing experiment compared existing kernels instead of requesting hardware counters. A two-line dispatch revision selected MMQ rather than MMVQ for dense Q8_0 four-column work on sm_121. All other types, batches, architectures and routed MoE dispatch remained unchanged. The isolated candidate passes 21 supported correctness cases; a clean software trace confirms six MMQ launches and no MMVQ launches for the six selected four-column shapes.

The existing backend perf runner measured each shape twice per arm, using repeated same-tensor graphs. These timings include backend/RHS-quantization work and favor cache reuse; they are not DRAM bandwidth or full-model measurements.

| Input width K | Output rows M | MMVQ median | MMQ median | Operator throughput change |
|---:|---:|---:|---:|---:|
| 320 | 10240 | 29.150 us | 20.090 us | +45.10% |
| 10240 | 320 | 16.025 us | 25.955 us | -38.26% |
| 2560 | 10240 | 108.370 us | 104.410 us | +3.79% |
| 2560 | 6144 | 32.450 us | 21.750 us | +49.20% |
| 6144 | 2560 | 33.350 us | 25.480 us | +30.89% |
| 2560 | 2560 | 18.275 us | 18.945 us | -3.54% |

The +9.79% geometric mean does not rescue the regressions. **Stop after the operator screen**, without a full-model run, deployment or threshold change. The four timing processes completed in 27.594 seconds. The original source/build DLL and the installed runtime remain unchanged. Evidence is in `results/q8-fourcol-mmq-20261021/`.

The source uses 128-row Q8_0 MMQ tiles, and the accepted trace's device metadata reports 48 SMs and 24 MiB L2. One further hypothesis will restrict the switch to output rows sufficient for one SM wave. This is a new geometry-conditioned dispatch experiment, not a second measurement of the rejected blanket switch. Its selection shapes are development data; fresh shapes and threshold-boundary cases must be declared before measurement, followed by the unchanged full-model gate only if it survives.

### 2.30 Geometry-conditioned Q8_0 MMQ revision [M]

The preregistered revision selects MMQ only for sm_121/Q8_0/four columns when output rows reach 128 times the device SM count. It passes 45 supported candidate correctness cases: the original 21 and 24 fresh cases, including columns 1/4/8 and rows 6143/6144/6145. The original runtime also passes all 24 fresh cases. A clean software trace verifies MMVQ below the row boundary and MMQ at and above it.

| Fresh input width K | Fresh output rows M | MMVQ median | Candidate median | Operator throughput change |
|---:|---:|---:|---:|---:|
| 512 | 8192 | 20.700 us | 21.085 us | -1.83% |
| 4096 | 8192 | 139.595 us | 160.440 us | -12.99% |
| 2048 | 12288 | 99.010 us | 91.750 us | +7.91% |
| 2560 | 12288 | 127.070 us | 132.640 us | -4.20% |
| 8192 | 6144 | 203.775 us | 222.605 us | -8.46% |
| 2560 | 6145 | 31.875 us | 26.990 us | +18.10% |

The fresh-shape operator geomean is **-0.77%**; three shapes exceed the 3% regression limit. **Stop after the operator screen.** The four timing processes took 28.516 seconds. There is no full-model run, cutoff search, further geometry-conditioned revision or deployment. This result rejects the proposed row-coverage rule, not every possible MMQ implementation. These remain cache-reuse operator measurements, not a hardware-counter attribution of the regressions.

The candidate patch, preregistered protocol, generated test cases, correctness/dispatch audit and all timing samples are retained in `results/q8-wide-mmq-20261021/`. The owned source patch set and original build DLL were restored; installed binaries are unchanged. Q8 dispatch tuning is closed for this pass. Follow a distinct measured bottleneck rather than tuning against these same development shapes.

### 2.31 Actual-MTP routed expert overlap [M]

A temporary CPU-side diagnostic DLL uses the existing evaluation callback to copy only `ffn_moe_topk` tensors. Two naturally exiting CLI requests use the original target, Q4_K draft, MTP3/ub512 and dynamic vocabulary 16384, generating 128 tokens each. The CUDA DLL is unchanged. This callback segments graphs and synchronizes readback: its timings are not benchmark results.

All 48 target layers and the MTP layer are captured with ten distinct expert IDs per token, in range 0-511. The I32 ID view has strides `[4, 2048]` bytes, not packed 40-byte rows. The analysis excludes the startup MTP probe, the wide prefill batch, and the four-token prompt tail. Target batch sizes reconcile exactly with the reported prompt count, draft tokens, accepted tokens and 128 generated tokens.

| Development case | Four-token verification batches | Layer evaluations | Expert selections | Duplicate selections across tokens | Duplicate selections at the same rank |
|---|---:|---:|---:|---:|---:|
| Rust | 35 | 1680 | 67200 | 38.20% | 9.49% |
| JSON | 32 | 1536 | 61440 | 32.25% | 7.04% |

Each case also ends with one three-token verification batch. Rust accepts 91/107 draft tokens; JSON accepts 94/98. The current multi-token MMVQ kernel assigns one token per warp and one expert rank per block, so most observed overlap is across ranks, not available to a same-rank-only sharing change. Both cases pass the preregistered 20% opportunity gate for investigating global grouping.

These percentages count logical expert selections, not physical memory traffic or available speedup: caches can already reuse weights. Before adding a grouped kernel, screen the existing MMQ expert-packing path at the actual small verify batches. It introduces packing/quantization work and loses the existing MMVQ gate/up fusion, so it can still lose overall. No new grouped implementation or deployment is justified by overlap alone.

`results/routed-overlap-20261021/` contains the diagnostic patch, capture provenance and sanitized per-layer aggregates. Raw IDs and logs remain private. The first logging pilot collected no records because core INFO maps to TRACE verbosity; it is retained as a failure. The same DLL captured both cases at verbosity four. Original source, original build DLL and installed binaries are restored/unchanged.

The following grouped-kernel correctness preflight also caught a quantization-label error in the earlier kernel table: enum 21 is IQ3_S, not IQ4_XS (enum 23). Original GGUF headers confirm 94 IQ3_S gate/up tensors, two IQ4_XS gate/up tensors, 43 IQ4_NL down tensors and five Q8_0 down tensors. The table above is corrected; raw trace symbols and durations were already intact. The misattributed prototype and malformed fixtures are retained as a preflight failure, with no candidate speed measurement. The corrected experiment targets IQ3_S/IQ4_NL and derives fixture types/block sizes from pinned definitions.

### 2.32 Existing routed MMQ screen [M]

A three-line host dispatch change lowers the sm_121 routed-MMVQ cutoff from eight to three tokens for IQ3_S/IQ4_NL. Existing callers also use this helper for fusion and graph-sync eligibility. Thus the candidate uses the existing GPU expert-packing/MMQ path for four-to-eight-token work; one-to-three-token routed work, other types, other architectures and dense dispatch stay unchanged.

Both runtimes pass 22 CPU-reference cases covering columns 1/3/4/5/8, both RHS broadcast layouts, and the actual 512-expert gate/up and down shapes with strided IDs. A clean software trace verifies both types reach MMQ for four tokens. Correctness fixtures have intentionally repeated expert sets and are not used as routing-performance proxies.

The uninstrumented original-model MTP screen uses two warmed samples per case, not the route-capture DLL:

| Case | Original decode median | Routed-MMQ median | Change |
|---|---:|---:|---:|
| Rust | 55.47 tok/s | 47.95 tok/s | -13.55% |
| JSON | 52.14 tok/s | 43.76 tok/s | -16.06% |

The geometric mean is **-14.81%**, with only 2/4 identical token sequences. **Reject after the first pair**, without reverse order, a quality campaign, type/cutoff/subset tuning or deployment. The bounded run completed in 150.188 seconds. The combination of expert packing, small MMQ tiles and lost gate/up fusion loses overall; this does not separately measure each cost.

`results/routed-mmq-20261021/` retains the corrected protocol and patch, preflight mistake, correctness/dispatch proof, timing samples, telemetry and decision. Original source/build DLLs and the installed runtime remain unchanged. Further grouped work must preserve the useful existing fusion or otherwise remove overhead; overlap alone is insufficient.

### 2.33 Fusion-preserving rank-alignment triage [M]

A distinct alternative to MMQ packing is to order each token's expert work by ascending expert ID inside the fused MMVQ kernel, while mapping activations and results back to their original slots. This would retain the existing graph layout and arithmetic order. Before implementing that mapping, an offline replay of the captured routes measures how much same-rank overlap sorting would expose.

| Development case | Original same-rank duplicates | Ascending-ID same-rank duplicates | Additional aligned selections |
|---|---:|---:|---:|
| Rust | 9.49% | 18.34% | 8.85 percentage points |
| JSON | 7.04% | 15.23% | 8.19 percentage points |

Neither case reaches the preregistered 20% logical-reuse triage gate. **Stop this ordering hypothesis before CUDA implementation**, without trying alternate ordering heuristics or layer subsets. The analysis reproduces the published original totals exactly, reconciles every record against the raw logs and phase/token accounting, and checks synthetic disjoint, aligned and permuted expert sets.

These are reused development routes, not new samples. No kernel speed, sorting overhead, physical traffic or cache benefit was measured. Failing this heuristic gate does not establish that every fusion-preserving grouped design would fail. `results/routed-rank-alignment-20261021/` retains the protocol, per-layer counts and input/script hashes; raw routes remain private. No runtime or model changes were made.

### 2.34 PDL-disabled runtime screen [M]

The existing CUDA launch helper enables programmatic dependent launch by default on supported kernels. This screen changes only the child-process environment: control `GGML_CUDA_PDL=1`, candidate `GGML_CUDA_PDL=0`. Executable/DLL hashes, models, graphs, fusion and MTP settings are identical. Disabling the runtime flag does not remove compiled synchronization instructions or restore compile-time `restrict` annotations.

Both settings pass 46 CPU-reference cases: 24 dense Q8 and 22 routed IQ3_S/IQ4_NL cases. Clean software traces of the same Q8 operator show the same kernels but two extended launch calls for control versus two ordinary launch calls for candidate. The first diagnostic filter matched no tests; that empty capture is preserved as a preflight failure, not accepted evidence.

| Case | PDL-on decode median | PDL-off median | Change |
|---|---:|---:|---:|
| Rust | 53.73 tok/s | 54.86 tok/s | +2.09% |
| JSON | 52.10 tok/s | 51.85 tok/s | -0.49% |

The geometric mean is +0.79%, all four token sequences are identical, and both cases have overlapping ranges. **Stop after the first pair**: neither case meets the 3% gate. The run completed in 142.421 seconds. Keep PDL enabled; no reverse-order run, quality campaign, compiled-annotation variant or deployment follows.

`results/pdl-off-20261021/` contains the preregistered protocol, preflight failure, numerical/launch proof, timing samples and telemetry. No user/machine environment, launcher, driver policy, binary or model was changed.

### 2.35 Stateless prompt-lookup replay [M]

The existing `ngram-simple` implementation runs before MTP and searches only the current request's token history, avoiding the previous-answer cache effect of stateful n-gram modes on repeated benchmarks. Its current search rejects a copied suffix shorter than the lookup length: setting maximum draft length three while retaining the default 12-token lookup would always return no draft. This replay therefore fixes both lengths at 12, the smallest usable maximum without a code change.

The installed vocabulary-only tokenizer reproduces the 30-token Rust and 32-token JSON prompts from the PDL control. An offline translation of the pinned search examines every generated position with 12 known future tokens, not just actual MTP drafting boundaries:

| Case, per 256-token completion | Positions checked | Proposals | Fully correct 12-token proposals | Accepted-prefix lengths |
|---|---:|---:|---:|---|
| Rust | 244 | 0 | 0 | None |
| JSON | 244 | 3 | 0 | 0, 1, 2 tokens |

Both measured repeats give the same result because their token sequences are identical; they are not independent evidence. Neither case passes the preregistered requirement of one fully correct proposal. **Stop before a hybrid model run on these standard cases**, without a lookup-length sweep.

This is teacher-forced eligibility replay, not runtime acceptance, speed, quality or proof of hybrid equivalence. Copy-heavy code edits and structured-data transformations remain unmeasured and would need their own frozen fixtures and workload-scoped claims. `results/ngram-simple-replay-20261021/` contains the protocol, counts and source/input hashes. No model weights were loaded for replay and no source, runtime or deployment changed.

### 2.36 Copy-heavy hybrid prompt-lookup screen [M]

Two new frozen fixtures exercise whole-file Python symbol renaming and JSON timeout-unit conversion. Their expected answers contain 842 and 1,253 tokens, so the fixed 512-token timing requests remain prefixes of the edits. Thirteen validator self-checks pass before measurement. Both arms use the original target/Q4_K draft and MTP3/ub512/dynamic vocabulary 16384, with reasoning off. The candidate adds the existing stateless `ngram-simple` mode at lookup/draft lengths 12/12; this is configuration testing, not our implementation of a new speculator.

| Copy-heavy case | MTP-only decode median | Hybrid median | Change |
|---|---:|---:|---:|
| Python rename | 63.86 tok/s | 54.26 tok/s | -15.02% |
| JSON unit conversion | 63.71 tok/s | 57.84 tok/s | -9.21% |

The geometric mean is **-12.16%**, with 4/4 identical measured prefixes. Control MTP acceptance is 100% on both cases; candidate aggregate acceptance falls to 85.60% and 76.39%. Candidate draft counters exceed what three-token MTP alone could generate in both cases, confirming that longer proposals were exercised. The changed draft lengths, acceptance and verification-kernel dispatch are not isolated latency attributions.

**Reject after the first pair**, completed in 208.078 seconds. No reverse order, full-edit quality campaign, lookup/draft-length tuning or deployment follows. Identical 512-token prefixes do not establish correct complete edits.

`results/copy-edit-ngram-20261021/` retains the protocol, fixture hash, preflight, samples and telemetry. The frozen fixture module and validators are in `bench/copy_edit_suite.py`; `run_rebaseline.py --suite copy-edit` selects fixed 512-token requests, or a 2,048-token cap with `--allow-eos`. The public suite selector was wired after the screen; the screen used the same hashed fixture function through the existing custom-suite path. Future use should preserve this distinction between throughput triage and complete-edit validation.

### 2.37 Target backend-sampling screen [M]

Draft GPU sampling is already enabled in the deployed runtime; target GPU sampling is not. This candidate adds only `--backend-sampling` to the original-model MTP3/dynamic-head recipe, with unchanged executables and DLLs. A naturally exiting 64-token CLI probe records 64 backend-token selections. The first logging pilot completed but could not prove engagement because `LOG_DBG` requires verbosity five rather than four; both protocols and the failed pilot are retained.

| Case | CPU target-sampling median | GPU target-sampling median | Change |
|---|---:|---:|---:|
| Rust | 54.80 tok/s | 55.75 tok/s | +1.73% |
| JSON | 51.57 tok/s | 52.82 tok/s | +2.43% |

The geometric mean is +2.08%, with nonoverlapping ranges, 4/4 identical token sequences and unchanged acceptance. Nevertheless, both cases miss the predeclared 3% gain. **Stop after the first pair**, completed in 145.219 seconds, without reverse order, sampler-parameter tuning or deployment. This is insufficient for promotion, not evidence of zero benefit.

All measured responses confirm the intended target backend-sampling state. The verbose probe is not timing evidence. Grammar, reasoning-budget and log-probability compatibility remain outside this screen. `results/target-backend-sampling-20261021/` contains the engagement evidence, preserved logging pilot, samples and telemetry; target sampling remains at the deployed default.

### 2.38 Fusion-preserving routed row-tiling screen [M]

The single candidate computes four output rows per block instead of two, only for sm_121, four-token routed IQ3_S/IQ4_NL operations with row counts divisible by four. Expert order, reduction order and gate/up/shared-expert fusion are unchanged. The divisibility guard matters because the inner loop loads every compile-time row before the bounds-checked store.

Both arms pass 46 CPU-reference cases, including both RHS layouts, original 512-expert ID strides, neighboring row counts and shared/gate fusion. Clean numerical traces verify the four-row fused and unfused kernels, their launch geometry, and the two-row fallback. A balanced control/candidate/candidate/control operator screen gives:

| Case | Throughput change |
|---|---:|
| IQ3_S, original 512-expert layout | +1.59% |
| IQ4_NL, original 512-expert layout | +9.92% |
| IQ3_S, small shared/gate-fusion graph | -0.69% |
| IQ4_NL, small shared/gate-fusion graph | -8.02% |

The geometric mean is only +0.49%, below the predeclared 5% operator gate, with a regression larger than the permitted 3%. **Stop before full-model MTP**, without a type-subset or row-count sweep. The operator screen takes 112.422 seconds; that excludes compilation and diagnostic preflights. These are repeated same-tensor timings, not physical bandwidth or model-speed measurements.

The first test executable lacked shared-fusion cases in its performance registry. Inspection also found that the existing performance helper repeats only the final output node, which would mostly time ADD rather than the fusion graph. The corrected, identical test executables register the frozen cases and repeat whole graphs with capacity accounting. Numerical and dispatch checks pass again after rebuilding. The initial missing-stdout assumption, incomplete operator pilot and an optional bulk-loop profiler timeout are retained in the preflight notes; the ordinary fusion pilot completes in 1.359 seconds, and no bulk-profile counts are claimed.

`results/routed-row4-20261021/` contains both harness revisions, the candidate patch, fixtures, manifests, numerical/dispatch results and operator samples. Apply the zero-context patches with `git apply --unidiff-zero`. Original owned sources, build DLLs and build configuration are restored. No production kernel or deployment change is retained.

### 2.39 OpenMP worker-wait screen [M]

The installed CPU backend is built with `GGML_USE_OPENMP`, which excludes ggml's custom worker-polling loop. Rather than screening an inapplicable `--poll` setting, this candidate sets only child-process `KMP_BLOCKTIME=0`. The hypothesis is that immediately sleeping idle OpenMP workers could reduce CPU contention or shared power pressure during GPU-heavy generation. `KMP_SETTINGS=1`, identical in both arms, confirms the actual default of 200 ms versus zero in the diagnostic and real server logs.

| Case | Default median | Zero-blocktime median | Change |
|---|---:|---:|---:|
| Rust | 54.80 tok/s | 55.70 tok/s | +1.64% |
| JSON | 51.57 tok/s | 51.68 tok/s | +0.22% |

Geometric mean: +0.93%, with unchanged acceptance and 4/4 identical token sequences. Both cases miss the fixed 3% gain; JSON ranges overlap. **Stop after the first pair**, completed in 146.390 seconds, without a wait-policy/blocktime sweep, reverse order, quality campaign or deployment. No causal power or contention reduction is inferred from these timings.

The first diagnostic used a selector absent from its fixture file and ran zero cases; the corrected diagnostic requires one passed case and unambiguous effective settings. `results/openmp-blocktime-zero-20261021/` retains that limitation, the source gate, runtime engagement, measured pair and telemetry. Installed binaries, thread counts, model/draft recipe and global environment remain unchanged. Administrator authorization was unavailable, so this experiment did not retry hardware counters or elevate.

### 2.40 Permanent whole-graph benchmark correction [M]

The performance-loop defect found in section 2.38 is now fixed in `tests/test-backend-ops.cpp`, rather than only in an experiment patch. Cases with `run_whole_graph()` repeat all original graph nodes in order. Ordinary cases retain output-only repetition. Both the FLOP and memory budget paths account for the number of nodes per repetition before appending work to the fixed-capacity graph. Existing FLOP definitions, result formatting and runtime kernels are unchanged.

A numerical regression increments an upstream in-place counter and squares it at the output. Its check and counter reset run outside the timed region. Four combinations cover whole-graph/output-only execution and FLOP/memory budgets, including capacity-limited batches. The original loop passes the two output-only cases, then fails the whole-graph numerical assertion; the corrected loop passes all four. The CPU CTest entry also requires a completed performance-result line so a zero-case invocation cannot pass silently. Assertions do not depend on a speed threshold.

| Native Windows ARM64 CPU check | Result |
|---|---|
| Original repetition logic with the new fixture | Expected assertion failure; CTest exit 8 |
| Corrected logic, four regression combinations | Passed; CTest invocation 4.171 seconds |
| Existing F32 SNAKE whole-graph cases | 8/8 numerical cases and one performance case passed |
| Existing Q8_0 `MUL_MAT` case | 1/1 numerical case and one performance case passed |

Reproduce the regression in a CPU-enabled test build, with its matching OpenMP runtime available:

```powershell
cmake --build "<native-build>" --target test-backend-ops
ctest --test-dir "<native-build>" -R "^test-backend-ops-perf-graph$" --output-on-failure --timeout 45
```

`results/perf-graph-harness-20261021/validation.json` records commands, actual case coverage, return codes and source/binary hashes. The isolated build initially linked an unavailable MSVC OpenMP DLL; selecting the already-installed matching native LLVM library resolved that dependency before the red/green regression. No new runtime dependency was installed. This validation is CPU-only; it does not claim CUDA coverage of the permanent fixture or an inference-speed gain. The live runtime, older owned build/source, model recipe and global settings remain unchanged.

### 2.41 Fail-fast performance-runner exits [M]

The earlier zero-case preflights exposed a second framework defect: `test-backend-ops perf` ignored execution failures, and unmatched selectors could report a successful backend without measuring anything. The runner now uses its existing `test_status_t` to distinguish filtered, unsupported, successful and failed cases. It stops a backend's performance loop on an execution failure and rejects a selection matching no cases. An unknown explicit backend fails before enumeration; a valid name is canonicalized so `-b cpu` actually runs CPU rather than silently skipping it.

| Native CPU comparison | Original runner | Corrected runner |
|---|---|---|
| Missing operation, parameters or backend, each in console/CSV/SQL | Exit 0 in all nine scenarios | Exit 1 with a diagnostic |
| Controlled warmup and timed graph aborts, each in console/CSV/SQL | Exit 0 in all six scenarios | Exit 1 |
| Controlled unsupported operation, each in console/CSV/SQL | Nonfatal skip | Nonfatal skip retained |
| Explicit lower-case `cpu` | Exit 0 without running the requested case | Exit 0 with the requested case executed |

Five focused CTests pass: the whole-graph regression, three negative selections and case-insensitive backend selection. Valid console, CSV and SQL executions still succeed; CSV rows and SQL result fields are parsed, and the existing SNAKE fixture passes eight numerical cases plus one performance case. This is failure-handling validation, not a throughput comparison.

The paired diagnostic builds use a private wrapper around the same runner source and native CPU library. Its abort callback fails either during warmup or after both warmup nodes; recorded graph statuses distinguish these paths. A separate controlled support response exercises the nonfatal skip path. The wrapper is retained in `results/perf-exit-status-20261021/` as evidence, not compiled into the normal runner. The final build is uninstrumented. Allocation failure was not injected, and no CUDA or model-speed claim is made.

Run the focused suite with `ctest --test-dir "<native-build>" -R "^test-backend-ops-perf-" --output-on-failure --timeout 45`. The evidence records the initial diagnostic-parser correction and all final checks. Default backend-skipping behavior and numerical/gradient/support-mode selection semantics are unchanged, except that explicit backend names now resolve consistently or fail. No deployment, model, driver policy or global environment change is made.

### 2.42 Routed-prefill tile scheduling: correct prototypes and full-model screen [M]

The earlier J16/grid-cap probe suggested substantial headroom, but a cap can omit work for hot experts. Its 38-41% lower operator times were a timing-only upper bound, not a valid general implementation or a deployed gain. The serialized generic operator loader also concentrates routes on the same experts; it must not stand in for shuffled routing. This screen uses registered `test_mul_mat_id` fixtures with explicit routing distributions.

Two isolated IQ3_S/IQ4_NL prototypes reuse the existing MMQ tile arithmetic. The first launches four column tiles per expert and loops over additional tiles. The second builds compact tile lists: experts with at most 32 columns use J16; hotter experts use J128. Separate small/wide kernels avoid forcing hot experts through many narrow tiles. Other types, non-Blackwell devices and token counts below 64 keep the original path. These are experimental environment-gated patches, not changes to the compiled repository runtime.

Both candidates and the original DLL pass 12/12 CPU-reference cases. These cover the actual gate/up shape (IQ3_S, m=640, k=2560, broadcast activations) and down shape (IQ4_NL, m=2560, k=640) at 512 tokens, each with shuffled, fully concentrated and mixed routes. Mixed routing sends three quarters of tokens to the same ten experts. Additional m=129, k=256 cases at 63/64/65 tokens cover the activation boundary and row tails. All fixtures use nonzero ID-view offsets, and concentrated routes include the final expert. The tolerance is the existing 5e-4 NMSE limit.

Each operator screen runs base/candidate/candidate/base, with six n=512 cases per launch. The following are throughput deltas from the median of two whole-operator timings per arm, not kernel-only speedups:

| Routing | Projection | Loop prototype | Compact prototype |
|---|---|---:|---:|
| Shuffled | IQ3_S gate/up | +12.42% | +57.59% |
| Shuffled | IQ4_NL down | -13.44% | +87.83% |
| Concentrated | IQ3_S gate/up | -58.16% | +18.25% |
| Concentrated | IQ4_NL down | -16.33% | +105.77% |
| Mixed | IQ3_S gate/up | -5.80% | +57.00% |
| Mixed | IQ4_NL down | -19.18% | +78.11% |

The loop is rejected at -20.14% geomean. The compact prototype clears the prefill operator gate at +64.95%, with no regressions and nonoverlapping ranges in these six cases. Control timings drift between launches, so these are screening results, not precise universal gains.

A fresh full-model A/A calibration gives pp2048 +1.55% and pp8192 -1.48%. Before testing the candidate, the full-model gate is fixed at at least 10% higher throughput on both lengths, nonoverlapping sample ranges, and at most 3% regression. Both arms use the unchanged Flash-Next UD-IQ4_XS target and matched b11756 + dynamic-head + QSA-mask runtime; only `ggml-cuda.dll` differs. This is target-only `llama-bench`, not actual MTP serving.

| First full-model pair | Base median | Compact median | Throughput change |
|---|---:|---:|---:|
| pp2048, ub512 | 761.93 tok/s | 832.09 tok/s | +9.21% |
| pp8192, ub512 | 746.04 tok/s | 856.26 tok/s | +14.77% |

**Park after the first pair.** pp2048 misses the predeclared 10% gate. Do not lower the threshold, select only pp8192, or describe this as a deployed improvement. The wrapper exits 2 with its explicit `park` decision. Reverse-order full-model confirmation, broader operator sizes, actual-MTP regression, English quality and final release promotion are deliberately not run after this gate failure.

**Fresh 5% confirmation.** The 10% gate came from older 7-9% pp2048 A/A noise; the fresh A/A is about 1.5%. A new protocol with a 5% gain gate (same 3% regression limit, nonoverlapping ranges) was written before a fresh rebuilt-candidate run. The first pair is not reinterpreted.

| Fresh pair | pp2048 change | pp8192 change |
|---|---:|---:|
| Round 1, base then compact | +6.16% | +15.71% |
| Round 2, compact then base | +2.48% | +8.19% |

**Park again.** pp2048 misses 5% in reverse order; the control drifts 797 to 845 tok/s between rounds. pp8192 gains in all three pairs (+8.2% to +15.7%), but selecting only long prompts after this result is not allowed. Evidence is in `results/prefill-moe-tiles-20261021/confirm-5pct/`; the candidate runtime is removed and the source and installed DLL are unchanged.

`results/prefill-moe-tiles-20261021/` preserves the original cap probe, both prototype patches, routing-fixture patch, exact case results, binary hashes, A/A and full-model samples, protocol and disposition. The older owned source diff and untracked files, CPU build DLLs, CUDA build DLL and all recorded installed runtime hashes match their pre-experiment state. The two new candidate runtime directories are removed and the previous test executable is restored. No model, launcher, global environment or deployment is changed.

The compact-list idea is not claimed as novel: [ggml-org/llama.cpp#26159](https://github.com/ggml-org/llama.cpp/pull/26159) proposes compact NVFP4 scheduling. At the live check it is open, draft and unmerged at `f1694d2615e4f3fb78aa0d143beea9c1764c6394`; its `build_moe_job_map` and `mul_mat_q_moe_compact` functions are absent from both our worktree and deployed source. Our already-present [ggml-org/llama.cpp#30553](https://github.com/ggml-org/llama.cpp/pull/30553) Stream-K block cap is a different change. This experiment implements small/wide IQ3_S/IQ4_NL scheduling, not that NVFP4 patch.

### 2.43 Two fork leads already covered by the deployed source [S]

The live [ggml-org/llama.cpp#28699](https://github.com/ggml-org/llama.cpp/pull/28699) head `141f3f5646` is still an open, conflicting draft. It is not a new pooled-key-cache opportunity for this installation: upstream `66e0c17ee` ([ggml-org/llama.cpp#29751](https://github.com/ggml-org/llama.cpp/pull/29751)) is an ancestor of deployed b11756. That source already stores raw and pooled indexer keys, marks only touched or newly completed pools, and scatters their results back into the cache. Sequence edits invalidate the affected pools. The older patch fails to apply to both this checkout and the deployed source. No port or duplicate benchmark is justified, and no new local gain is attributed to this existing upstream work.

Likewise, the merged [ikawrakow/ik_llama.cpp#2699](https://github.com/ikawrakow/ik_llama.cpp/pull/2699) removes power-of-two SM rounding in that fork's `fattn-new-mma.cu` for eligible prefill. Our deployed `fattn-common.cuh` already reads the actual `nsm` and uses it for both Stream-K capacity and non-Stream-K wave occupancy. Its optional whole-output-tile rounding, limited to 5% occupancy loss, is a different rule. This fork-specific fix is not an available drop-in gain here. The pinned checks are in `results/nvfp4-expert-screen-20261021/qsa-applicability.json` and `fa-grid-applicability.json`.

### 2.44 NVFP4 expert-format feasibility screen [M]

The next independent experiment compares existing upstream NVFP4 kernels with the actual IQ3_S gate/up and IQ4_NL down shapes, without changing the backend or converting a model. Both formats use the same newly built test executable and the unchanged deployed CUDA DLL. The routing fixture uses 512 experts, ten selections per token, an offset ID view, and three routing patterns: seeded shuffled experts, the last ten experts, and one-quarter spread/three-quarters concentrated. The shuffled IDs use the same seed across formats and change between timing iterations.

Before measurement, the protocol requires at least 5% prefill geometric-mean gain in each launch order, separated prefill launch ranges, and no case slower by more than 3%, including a four-token decode/verification guard. Six actual-shape NVFP4 cases and eighteen boundary/layout/routing cases pass the existing CUDA-versus-CPU checks. The reference uses the same NVFP4 weights, and Blackwell native four-bit activations use the existing 0.02 NMSE tolerance: this is backend correctness, not preserved model precision or task quality.

Original/NVFP4/NVFP4/original measurements give:

| Projection and routing | n512 prefill throughput change | n4 small-batch throughput change |
|---|---:|---:|
| Gate/up, shuffled | +49.42% | -43.36% |
| Gate/up, concentrated | +43.49% | +48.68% |
| Gate/up, mixed | +45.96% | +46.15% |
| Down, shuffled | +55.67% | -1.55% |
| Down, concentrated | +8.37% | -3.43% |
| Down, mixed | +40.03% | -6.02% |

Prefill gains +37.37% and +42.10% geometric mean in the two orders, +39.57% pooled, with all six launch ranges separated. The small-batch aggregate is +1.60%, but it hides the shuffled gate/up regression: -40.83% and -45.88% in the two orders. Several down cases also exceed the regression limit.

**Reject wholesale expert-only NVFP4 at the operator screen.** This provides a concrete prefill lead but not a safe general model-conversion recommendation. These are synthetic, standalone matmuls, not fused gate/up/GLU or end-to-end MTP measurements; the result does not prove that every NVFP4 model is slower. Do not promote a favorable routing subset or count the operator gain as delivered model speed. No model conversion, precision evaluation, whole-model speed claim or deployment follows this failed guard.

`results/nvfp4-expert-screen-20261021/` preserves the predeclared protocol, all case results, both launch orders, source fixture patch, binary hashes and applicability checks. The fixture builds against the owned b11756 source stack, including its existing benchmark repetition/error fixes. Owned test source, CPU build outputs and all previously recorded installed-runtime hashes are restored. NVFP4 kernels remain attributed to upstream; our contribution is the local fixture and feasibility evidence.

### 2.45 MMVQ row-tail correctness correction [M]

The independent IQ4_NL one-warp experiment exposed failures in its original control before candidate timing. Generic routed MMVQ used the output token stride, which spans all selected experts, as the row bound for one expert. An incomplete tile could write into the next expert's output. Generic and multi-token kernels also read unused weight rows past the end of a tile.

The source correction passes the actual row count into generic MMVQ, bounds bias loads and output stores by that count, and maps unused tail reads to a valid row. Bounds-checked specializations are selected only for incomplete row tiles; aligned launches avoid dynamic row-offset checks. The same handling covers fused gates/shared experts and ordinary dense MMVQ. The permanent tests cover IQ4_NL, IQ3_S, Q4_K and Q8_0, odd row counts, dense batches with one/two/eight columns, both routed-input layouts, offset ID views and shared/gate fusion. The original control fails the regression suite; all 64 permanent cases and 100 existing/boundary/fusion checks pass with the correction.

Three simpler implementations passed correctness but retained operator regressions. The specialized implementation also has unresolved operator wall-time regressions: the 22-case opposite-order geometric mean is -7.34%. Software tracing confirms matching launch geometry/register counts; the aligned IQ4_NL four-token kernel median is 57.888 versus 57.984 us, while the one-token median is 20.096 versus 20.832 us. These diagnostics do not clear the operator guard.

A separate actual-MTP diagnostic uses the same target, Q4_K draft head, MTP3/ub512 and dynamic vocabulary 16384, with one warmup and five measured requests per case per launch. Rust median decode changes -0.014% and +0.696% in the two orders; JSON changes +0.333% and +1.388%. All paired generated token sequences match. This is limited evidence that the correctness correction preserves these workloads, not a speed gain or a substitute for broader release validation.

**Retain the source-level correctness correction, but do not deploy it or claim a throughput improvement.** Operator-cost investigation remains open. The one-warp optimization remains a separate, unpromoted candidate and must use the same corrected baseline in both arms. `results/mmvq-rowtail-20261021/` contains the four implementation patches, measurements, binary hashes and paired output hashes. Original owned source/build files and the installed runtime remain unchanged. Superseded experimental binaries were removed during the requested disk cleanup; their source and measurement evidence were retained.

### 2.46 IQ4_NL one-warp MoE CTAs: rejected [M]

Hypothesis: on sm_121, give each token of unfused routed IQ4_NL multi-token MMVQ (2-8 tokens) its own one-warp CTA (token in grid Z) instead of one warp per token inside a shared CTA. Row tiles, dot products, fusion and other types are unchanged. Both arms include the section 2.45 row-tail correction; only `ggml-cuda.dll` differs.

Both arms pass 100 CPU-reference checks (34 serialized, 54 IQ4_NL boundary, 12 shared/gate fusion). Software traces confirm the candidate launches `(64,10,4)` grids of `(32,1,1)` blocks, the baseline `(64,10,1)` grids of `(32,4,1)` blocks, and fused work keeps its original geometry. Nsight truncates test stdout in both arms, so the trace step checks child exit codes and kernel geometry; correctness comes from the separate complete runs.

Base/candidate/candidate/base operator screen (15 changed cases plus fused guard; routing fixture from section 2.42):

| Routing | n2 | n3 | n4 | n8 | m3072 n4 |
|---|---:|---:|---:|---:|---:|
| Shuffled | -3.29% | -2.55% | -2.62% | -2.74% | -0.02% |
| Concentrated | -16.45% | -17.12% | -15.58% | -17.70% | -39.72% |
| Mixed | -13.67% | -20.69% | -11.32% | -23.12% | -39.84% |

Changed-case geometric mean is -16.03% (gate: +5%, no case below -3%). **Reject.** Splitting token warps into separate CTAs hurts most when tokens share experts, which suggests the shared CTA already gets weight reuse through L1/L2. Per protocol, no warp-count, row-count or type sweep and no actual-MTP run follow. Evidence is in `results/iq4-singlewarp-20261021/`; source and installed runtime are unchanged.

### 2.47 Expert-first verify reuse ceiling: stop [M]

Strata decodes each distinct IQ expert once per verify window (`row_dot_multi`, groups of four tokens). Section 2.31 shows real 4-token MTP windows select 40 experts per layer but only 24.7 (Rust) / 27.1 (JSON) distinct ones. Question: does the current routed MMVQ already reuse experts shared across ranks? A diagnostic test build adds two fixtures to the section 2.42 routing test: route 3 uses the same ten experts for every token but rotates their rank, and route 4 draws each token's ten experts from a 32-expert pool per 4-token group (expected 24.85 distinct). The runtime is the committed row-tail build; `ggml-cuda.dll` is unchanged.

All 8 new CPU-reference cases pass. Mean of two runs, us:

| Shape | T0 shuffled (~40 distinct) | T1 same rank (10) | T3 rotated rank (10) | T4 realistic (~25) | Linear ideal at 24.85 | Headroom |
|---|---:|---:|---:|---:|---:|---:|
| IQ3_S gate/up n4 | 77.2 | 60.3 | 64.3 | 64.7 | 68.7 | -6.2% |
| IQ4_NL down n4 | 131.5 | 58.3 | 61.2 | 64.4 | 94.6 | -46.9% |
| IQ3_S gate/up n8 | 203.1 | 115.8 | 127.2 | 140.8 | 159.0 | -12.9% |
| IQ4_NL down n8 | 271.2 | 112.7 | 119.1 | 191.5 | 191.2 | +0.2% |
| IQ4_NL m3072 n4 | 128.1 | 47.1 | 50.1 | 51.4 | 87.2 | -69.5% |

Cross-rank sharing costs only 5-10% more than same-rank sharing (T3/T1 1.05-1.10, gate 1.3), and realistic overlap at n4 is already within 7% (gate/up) and 10% (down) of perfect same-rank sharing. Caches already capture most cross-rank reuse. **Stop** the expert-first reuse candidate (S3) before a kernel prototype: even a perfect grouped kernel could save at most T4-T1 on these operators, and only part of verify time is routed MMVQ. This also explains the section 2.46 result. Evidence is in `results/route-reuse-ceiling-20261021/`; `fixture.patch` is relative to the section 2.42 routing test source, not repository HEAD.

### 2.48 Whole-batch PLE prefetch: rejected [M]

Strata gathers the next prompt chunk's PLE rows while the GPU runs the current chunk. Our qwen4exp already prefetches each ubatch's PLE rows in `set_input`, just before that ubatch runs. First, the stall was measured with the deployed recipe (installed runtime, `-lzm on`, `-b 2048 -ub 512`, MTP3 Q4_K draft head, dynamic vocab 16384). Novel random-word and Python stdlib prompts each ran three times (`cache_prompt=false`). Cold penalty = 1 - warm/cold time:

| Prompt | Median cold penalty | Range by text | Run-order drift |
|---|---:|---|---:|
| 2K | 10.3% | words 14-16%, code 5.5-7% | ~1% |
| 8K | 8.5% | words 12-13%, code 3-5% | ~1% |

The predeclared gate (>= 5% at 2K or 8K) passed. The candidate then hashed the PLE rows of ubatches 2-4 at the start of each multi-ubatch `decode` call (single sequence, token input only) and issued one `llama_prefetch_rows`. Rows are the same as `set_input`, and a prefetch is only a hint. Only `llama.dll` changed; the control was built from the same unmodified source. An ABBA run used fresh novel prompts per phase (4 words + 2 code at each length), each cold then warm:

| Length | Cold control | Cold candidate | Cold gain | Warm gain |
|---|---:|---:|---:|---:|
| 2K | 784.5 tok/s | 785.7 tok/s | +0.15% | +1.45% |
| 8K | 809.0 tok/s | 798.1 tok/s | -1.34% | -1.46% |

A fixed 64-token greedy generation matched in all four phases. The gain (>= +3% at both lengths) was not met, and 8K is below the +1% reject line: **reject**. By text type, the cold gain is within +-1.8% everywhere, below phase-to-phase noise (2K phase medians 756-791 tok/s). The cause was not measured. Possible causes: `PrefetchVirtualMemory` reads are not faster than demand faults on this NVMe path, or the cold cost is not page-fault latency that earlier hints can hide. Do not repeat a deeper prefetch window without a measurement of the cause (for example, ETW hard-fault and disk-queue traces). Strata-style RAM locking of the 28.8 GB table does not fit the 30 GB Windows share. Evidence is in `results/ple-batch-prefetch-20261021/` (`stall_*` = penalty measurement, `candidate.patch` relative to the owned source).

### 2.49 Resident prefill preparation and fused-IQ applicability [M/S]

Reanalysis of the existing native software trace separates the resident CUDA path from Strata's streamed-expert path. It covers all six target MMQ microbatches (512/512/512/512/170/512 tokens), not just the longest busy segment used in the earlier tile investigation. The 864 routing-helper/quantization/MMQ triplets reconcile to 48 layers x three projections x six microbatches. Draft stream 20 and the last four small-batch tokens are excluded. Between the first target routing helper and the last routed MMQ, target stream 18 has 3,689.51 ms of kernel busy time:

| Target operation | Calls | GPU time | Selected target busy time |
|---|---:|---:|---:|
| Expert routing/packing map | 864 | 32.83 ms | 0.89% |
| Routed activation quantization | 864 | 40.97 ms | 1.11% |
| Routed quantized matrix multiplication | 864 | 2,372.81 ms | 64.31% |
| Gate/up SiLU multiplication | 288 | 31.40 ms | 0.85% |

Each triplet submits three kernel launches, with **no copy, synchronization, wait-event or record-event API call between those launches on the submitting thread**. Source agrees: `ggml_cuda_mul_mat_id` takes the quantized MMQ path, whose helper processes all 512 experts in one launch. The host-copy/per-expert synchronization fallback exists but is not this path. Thus Strata's one-copy-dependency/one-release-event-per-expert-group improvement has no matching ring to optimize here. This does not mean the entire model is synchronization-free.

There is duplicate preparation across projections. Optimistically removing two of the three routing helpers plus the second gate/up quantization in every layer saves 36.87 ms of GPU work, equivalent to **1.01%** higher throughput in this selected GPU-only accounting. Even deleting all routed preparation gives only 2.04%. The 3% complexity-triage threshold is not met: **stop the preparation-reuse prototype before implementation**. Down uses a forward map while gate/up uses an inverse map, so the estimate already assumes free generation of both. These figures exclude host-launch savings and are not an end-to-end upper bound or a measured speedup. No claim about physical bandwidth or available matmul arithmetic improvement follows.

Fresh reads of the original GGUF headers confirm 94 IQ3_S and two IQ4_XS gate/up tensors, plus 43 IQ4_NL and five Q8_0 down tensors. The only IQ4_XS gate/up pair is layer 2, whose down tensor is Q8_0. Therefore **zero of 48 layers** fits Strata's complete native-IQ fused path (IQ4_XS gate/up with Q2_0 or IQ4_NL down). An IQ3_S implementation or split epilogue would be a new design, not a drop-in port; its altered arithmetic needs separate quality evidence. Do not reclassify the already parked compact-MMQ scheduling or rejected whole-expert NVFP4 screen as new Strata gains.

`results/prefill-preparation-20261021/` contains the frozen selection/triage protocol, executable analyzer, six synthetic accounting/failure tests, target tensor list, trace hash and source/binary provenance. Reproduction uses `python analyze_trace.py <preserved-prefill.sqlite> <new-summary.json>`; the raw trace remains private. The installed binaries still match the captured manifest. No model execution, CUDA build, engine edit, deployment or measured speed gain accompanies this analysis.

### 2.50 Shared-system-prefix retention: already working [M]

A bounded two-conversation diagnostic uses the unchanged installed server, original target, Q4_K MTP draft, MTP3, dynamic vocabulary 16384, one slot and b2048/ub512. Synthetic inventories A/B are placed in system messages; three different user lookups per inventory run in order A1,A2,B1,B2,A3,B3. All answer fields have exact expected values. Actual chat-template/tokenizer output gives common prefixes of 1,684 tokens for A and 1,780 for B, with full prompts of 1,696 and 1,792 tokens.

The first arm leaves the installed defaults unchanged: 8,192 MiB prompt RAM cache, 32 context checkpoints, minimum spacing 8,192 tokens. The second disables only `--cache-ram`; hybrid checkpoints remain enabled. First-content latency is measured from the HTTP request through the first nonempty SSE content, so it includes prompt-cache save/restore work before the server's prompt timer:

| Request | Default cached tokens | Default first content | RAM-cache-off cached tokens | RAM-cache-off first content |
|---|---:|---:|---:|---:|
| A2, adjacent shared prefix | 1,677 | 333.8 ms | 1,677 | 329.9 ms |
| B2, adjacent shared prefix | 1,773 | 233.9 ms | 1,773 | 218.6 ms |
| A3, return after B | 1,677 | 428.6 ms | 0 | 2,557.4 ms |
| B3, return after A | 1,773 | 393.0 ms | 0 | 2,656.9 ms |

Defaults reuse more than 99.5% of each measured common prefix, above the fixed 95% gate. Four additional uncached default-arm references report zero cached tokens and 2,648-2,923 ms first-content latency. All 16 requests return the exact expected semantic JSON answer; formatting/token identity is not asserted. Logs confirm 112.571 MiB hybrid checkpoint restores and a parked prompt-cache entry of about 505-508 MiB, well below its capacity. They also confirm target and draft state are saved. Source restores the speculative state alongside target/draft checkpoints, rather than treating hybrid state as KV alone.

**Stop the new system-prefix-pinning candidate for this demonstrated workload.** The existing upstream checkpoint and prompt-cache mechanisms already provide adjacent-prefix reuse and interleaved conversation recovery; this is not our engine gain. The measurements are single diagnostic samples with verbose logging, not an opposite-order throughput campaign or a new default recommendation. Eviction under capacity pressure, concurrent clients, long conversation chains and multimodal state remain outside this test. A separate retention-policy experiment needs a measured failure in one of those workloads, not a claim that prefix caching is absent.

`results/prefix-retention-20261021/` preserves the preregistered checks, runnable fixture/SSE runner, 16 response records, selected cache logs and provenance. Reproduce with `python screen.py --draft <Q4_K-draft.gguf> --out <new-private-directory>` from an environment with `results/model_paths.json` configured. Both owned server processes are stopped; installed binaries, launcher settings and model files are unchanged.

### 2.51 Cost-aware MTP versus lookup: causal selector rejected [M/S]

The remaining Strata policy lead was tested without changing the frozen section 2.36 fixtures or the MTP3 / lookup 12/12 lengths. Four compact instrumentation sites record round start, selected source, checkpoint rollback and final acceptance. Both instrumented arms retain the installed runtime except the two rebuilt common/server implementation DLLs. Native prompt token counts reconcile to 896 and 1,272; all twelve diagnostic responses match the earlier saved prefixes. Every proposed and accepted token counter reconciles, including checkpoint replay. The measured intervals cover 99.15-99.69% of response decode time.

Source explains an important cost discontinuity: `need_n_rs_seq()` reserves `draft.n_max` recurrent rollback states, three here, even when stateless lookup proposes twelve tokens. Rejecting more than three forces checkpoint restoration and replay of the accepted prefix. That replay emits a second bonus token, which the trace analyzer accounts for rather than treating it as a new proposal. Each measured unconditional-hybrid request has 13 replay rounds for Python and 15 for JSON.

| Instrumented round cost | Python rename | JSON unit conversion |
|---|---:|---:|
| Pure MTP3, median | 62.59 ms | 62.23 ms |
| Fully accepted 12-token lookup, median | 149.35 ms | 143.49 ms |
| MTP3 immediately after lookup, median | 104.98 ms | 98.39 ms |

Some full-copy spans can pay, but low-acceptance lookup and following MTP rounds are expensive. These conditional timings do not isolate the cause of the following-round slowdown. The optimistic dynamic-programming oracle uses exact future acceptance, minimum measured lookup cost, mean baseline MTP cost, free selection and no switching penalty. It predicts +24.05% / +47.87% on the development streams, or +14.04% / +21.74% when lookup uses its fully accepted median cost. Those are counterfactual model estimates, **not measured policy speed or rigorous end-to-end bounds**.

One causal prototype was frozen before its timing run. It resets per request, warms up with four MTP rounds, samples an eligible lookup proposal, and tracks microseconds per committed token with EWMA alpha 0.25. Lookup must appear at least 3% faster by estimated committed-token rate; otherwise fifteen eligible proposals are bypassed before one new probe. Its timer includes rejection replay, but does not predict downstream transition costs. Only the isolated candidate's `llama-common.dll` changes; the timing run has no per-round diagnostic logging.

| Fixed-prefix screen | MTP control | Cost policy | Change |
|---|---:|---:|---:|
| Python rename, 512 tokens | 62.84 tok/s | 52.93 tok/s | -15.77% |
| JSON unit conversion, 512 tokens | 63.62 tok/s | 55.73 tok/s | -12.40% |
| Equal-weight geometric mean | | | **-14.10%** |

All four paired prefixes are identical. Policy counters prove that both selection and bypass were exercised: Python selects 11/13 lookup proposals and bypasses 56/50; JSON selects eight and bypasses 81 in each repeat. **Reject at the first speed pair.** No cadence, EWMA, threshold, length or fixture tuning follows; reverse order and full natural-EOS edit validation are not justified. This rejects the tested simple controller, not every possible adaptive policy.

`results/mtp-lookup-policy-20261021/` preserves both protocols, diagnostic and candidate patches, the policy header, native policy checks, ten Python analyzer tests, per-round records and hash manifests. `capture.py`, `analyze.py` and `screen.py` reuse the established benchmark runner and require a fresh private artifact directory with the documented original-source snapshots. To apply the experimental policy patch, also copy `policy.h` to the owned source's `common\spark-cost-policy.h`; it is not an engine feature in this checkout. The original owned sources and common/server DLLs are restored, and installed binaries/models/launcher settings are unchanged.

The three-state/twelve-token mismatch is new evidence for a separate rollback-capacity experiment, not permission to retune the failed selector. A capacity change must keep draft lengths fixed, account for additional memory and pass its own correctness/performance gates.

### 2.52 Lookup rollback-capacity match: mixed result, rejected [M/S]

The per-round evidence in section 2.51 justifies a distinct structural experiment, not another selector sweep. The candidate raises target recurrent rollback capacity to the maximum of its existing capacity and `ngram-simple.size_m` when that lookup mode is active. Lookup remains 12/12, MTP remains three, and the draft context still explicitly disables its own recurrent snapshots. Only the isolated `llama-common.dll` changes. A startup diagnostic reads the actual target value through `llama_n_rs_seq`, confirming twelve states; neither draft length nor the frozen editing fixtures changes.

The speed denominator is a **fresh pure-MTP control**, not the slower previously rejected hybrid. With the same target, Q4_K head, dynamic vocabulary 16384 and b2048/ub512, one warmup and two measured requests per case give:

| Fixed-prefix screen | Pure MTP3 | Capacity-matched hybrid | Change |
|---|---:|---:|---:|
| Python rename, 512 tokens | 60.40 tok/s | 58.26 tok/s | **-3.55%** |
| JSON unit conversion, 512 tokens | 58.90 tok/s | 66.73 tok/s | +13.30% |
| Equal-weight geometric mean | | | +4.54% |

All four paired prefixes match. Ready GPU memory increases from 78,558 to 79,660 MiB (+1,102 MiB), below the predeclared 2,048 MiB guard. This is the observed ready-memory difference, not an isolated allocation measurement. The Python control ranges from 58.48 to 62.33 tok/s while the candidate ranges from 58.20 to 58.32; the two-sample medians and drift do not establish a precise general regression estimate.

**Reject at the first pair:** Python exceeds the 3% regression guard, so the positive geomean and JSON result do not authorize promotion. No capacity sweep, post-hoc JSON-only policy, reverse-order run or full natural-EOS quality campaign follows. Matching rollback capacity is insufficient to make this hybrid consistently better than pure MTP on the frozen pair.

`results/lookup-rollback-capacity-20261021/` retains the preregistered protocol, runnable screen, four-line experimental source patch, actual-capacity/memory checks, measured request hashes and provenance. The original source and build DLLs are restored; temporary runtime copies are removed. The installed runtime, models and launcher are unchanged. Together with section 2.51 this finishes the bounded remaining Strata policy investigation, not the wider optimization roadmap or the unresolved MMVQ correctness-fix performance gate.

### 2.53 Vectorized MMQ activation copy: parked at first pair [M/S]

The complete diff of [ggml-org/llama.cpp#30493](https://github.com/ggml-org/llama.cpp/pull/30493), pinned at `9e37a8578f47936fb7f89cbb45f4e98affc134a9`, replaces scalar activation staging with aligned `int2` loads/stores and retains bounds checks when the work partition is not exact. The head commit alone is only an incremental int4-to-int2 change; this screen uses the complete PR diff against the original scalar implementation. This is an upstream integration experiment, not an original kernel optimization.

Only the isolated candidate's `ggml-cuda.dll` changes. Both arms pass all 64 CPU-reference CUDA fixtures: eight quant types present in the 27B model, rows 65/128, columns 63/64/65/512 and inner dimension 256. The native 27B first pair gives:

| Test | Control | Vectorized copy | Change |
|---|---:|---:|---:|
| pp2048 | 852.49 tok/s | 863.36 tok/s | +1.28% |
| tg128 | 15.38 tok/s | 15.25 tok/s | -0.82% |

**Park at the unchanged 3% prefill gate.** No reverse-order run, full quality campaign, other-model expansion or deployment follows. `results/mmq-copy-20261021/` retains the protocol, complete pinned patch, source/runtime hashes, generated fixtures, correctness records and native samples. The original owned source and DLL are restored and control objects rebuilt; installed binaries were not replaced.

### 2.54 Actual 27B prose launcher profile: reverse order misses the gate [M]

The selected 36 GB vision/DFlash launcher recipe uses b4096, ub512 and context 262144, unlike the earlier depth study's b2048/ub2048. This screen parses that actual launcher command, substitutes the corresponding local target/draft/projector paths, and compares depth seven with depth four on a private loopback port. Only draft depth differs between arms. The four existing launcher recipes remain untouched.

| Case | First-pair depth7 / depth4 | First-pair change | Reverse-pair depth7 / depth4 | Reverse-pair change |
|---|---:|---:|---:|---:|
| prose256 | 29.85 / 34.60 tok/s | +15.91% | 32.43 / 33.40 tok/s | +2.98% |
| prose512 | 36.32 / 37.55 tok/s | +3.39% | 35.70 / 36.72 tok/s | +2.88% |
| Geometric mean | | +9.47% | | +2.93% |

Each arm has one warmup and two measured requests per case. All eight paired measured sequences are identical; all four servers exit normally and release GPU memory. The first depth7 prose256 samples vary substantially (26.70-33.00 tok/s), and the reverse-order prose256 ranges overlap.

**Park, without rounding 2.98% up to 3% or rerunning until it wins.** The prepared five-repeat confirmation and natural-EOS quality stages do not run; no separate prose recipe is added. Fixed-length requests use greedy sampling, not the launcher's stochastic defaults, so token agreement is not a general quality claim. `results/27b-profiles-20261021/` preserves the frozen protocol, exact catalog command/hash, runnable validator and both pair summaries.

### 2.55 27B NVFP4 FFN eligibility: prefill gain fails decode/verify guards [M/S]

The actual GGUF has 65 blocks including one appended MTP block. The screen excludes that block and inventories all 192 FFN tensors in the 64 target layers. Seven original quant types occur in both orientations: up/gate m17408/k5120 and down m5120/k17408. The filename does not imply uniform Q4_K weights.

On the unchanged installed CUDA implementation, all nine selected native CPU-reference NVFP4 cases pass, including W4A8 and W4A4 fixtures. The pre-conversion speed screen uses synthetic tensors with the actual original type/shape mix: 42 original cases and six NVFP4 cases at columns 1, 8 and 512. Native operator timings are weighted by target-tensor counts:

| Columns / role | Weighted geometric-mean throughput change | Worst individual change |
|---|---:|---:|
| 1 / decode guard | -0.30% | -24.51%, Q3_K down |
| 8 / verification guard | **-9.69%** | **-29.57%, IQ4_NL up/gate** |
| 512 / prefill | +67.82% | +26.28%, IQ3_S down |

**Reject at the first pair.** The prefill result clears the preregistered 5% operator gate but does not override the 3% per-case regression guard. No reverse pair, favorable type-subset selection, activation-precision tuning, weight conversion or deployment follows. These are synthetic operator results, not model tok/s or quantized-model quality; native FP4 activation tolerances are not permission for model-quality loss.

`results/27b-nvfp4-20261021/` retains the inventory, fixtures, runtime hashes, correctness result, parsed timings and full comparison. This confirms that existing NVFP4 kernels can accelerate dense prefill on this device, but a blanket FFN conversion does not qualify as a safe mixed-workload improvement.

### 2.56 Nemotron SSD dispatch: MTP rollback exclusion observed [M/S]

Chunked Mamba2 SSD is already implemented in `ggml/src/ggml-cuda/ssm-scan.cu`; it is not a missing port. Its long-sequence path requires scalar A, 128 < tokens <= 8192, suitable NVIDIA hardware/alignment and **K=1**. The graph caller in `src/models/mamba-base.cpp` instead passes `K = n_rs_seq + 1` when recurrent rollback is enabled.

A logging-only CUDA build, isolated from the installed runtime, directly observes both paths. Each arm serves two identical synthetic requests, each with 1,565 prompt tokens and one generated token, at context 8192 and b4096/ub4096. Startup warmup is disabled. The logged long SSM batch is 1,561 tokens, not the aggregate prompt total:

| Arm | Long-batch host dispatch records | Rollback K | Existing SSD selected |
|---|---:|---:|---|
| No speculation | 80 | 1 | Yes, all records |
| MTP3, N4 diagnostic configuration | 80 | 4 | No, all records |

Both use d_state128, head_dim64, 128 heads and one sequence. N4 retains its explicit p_min0.5 only as a diagnostic configuration; this is not a renewed recommendation for that previously rejected cutoff, and one-token requests do not measure its acceptance behavior. CUDA graph capture/reuse can alter host dispatch counts, so 80 records must not be interpreted as a GPU kernel count.

**The rollback condition really excludes SSD for the observed long MTP prefill batches.** Timings are instrumentation-perturbed and are not a speed comparison. Removing the K guard or forcing K=1 would discard required recurrent snapshots. A separate rollback-preserving implementation needs CPU-reference and snapshot-equivalence tests, a bounded performance screen and model-quality evidence before adoption; none is implemented here.

`results/ssm-ssd-dispatch-20261021/` contains the protocol, diagnostic patch, source/runtime hashes, trace parser and both observed records. The initial analyzer incorrectly equated aggregate prompt tokens with the long batch size; the corrected parser reuses the completed control capture without rerunning it, and a regression test covers checkpoint-tail accounting. Both servers stop normally and release GPU memory. The original source/DLL are restored after capture; nothing is deployed.

### 2.57 Approved Q8 hardware counters: driver-resource blocker [M]

The user subsequently approved the bounded Q8_0 hardware-counter diagnosis. The signed native ARM64 Nsight Compute executable and original runtime hashes were rechecked; the exact K320/M10240/N4 fixture again passed its one CPU-reference CUDA case before collection. One user-approved Windows elevation launched the same fixture with `Administrator=true`, one `mul_mat_vec_q` launch, kernel replay, six diagnostic sections, configuration files off, and no clock or cache control.

The earlier permissions error was cleared, but collection failed after 3.148 seconds with **`Failed to create counter availability image (error = 20)` / `ResourceUnavailable`**. No counters or `.ncu-rep` report were produced. Read-only follow-up found no active Nsight/DCGM/test/server capture, an empty Nsight Systems session list, and the FrameView SDK service stopped. Normal NVIDIA container services were left running. These checks do not identify the unavailable resource or establish the cause; they do not prove that the platform is unsupported.

**Blocked by the driver resource, not by missing authorization.** The collector exited 1 and cleanup was verified at 2,517 MiB GPU memory. Driver 617.42, access policy, services, clocks, installed binaries and launcher settings were not changed. No additional elevated retry or driver reset followed. `results/native-counters-20261021/elevated-protocol.json`, `capture-elevated.ps1` and `elevated-result.json` preserve the approved scope, runnable wrapper and sanitized outcome; the earlier `probe.json` remains the historical permission failure.

### 2.58 Nemotron NVFP4 experts: prefill gain fails the MTP guard [M/S]

The approved order continues with Nemotron after the Q8 capture attempt; the already-completed 27B result is retained rather than repeated. Both Nemotron GGUF shards contain 781 tensors. Model metadata lives in the first shard; the second contains only split metadata. Excluding the appended MTP block leaves 88 target blocks and 240 FFN tensors. The frozen conversion scope is all 80 routed expert tensors: 40 Q4_K up projections [1024,2688,512] and 40 IQ4_NL down projections [2688,1024,512]. Shared experts, latent projections, router, attention/SSM, output and MTP remain outside the candidate.

This model has no separate gate projection. The up fixture includes `MUL_MAT_ID -> RELU -> SQR`; down includes `MUL_MAT_ID -> MUL` by expert weights. Both arms use the same native test executable, unchanged installed CUDA backend, and the section 2.40 whole-graph repetition fix. Four repetition-regression cases pass. All 18 CPU-reference graph cases per arm pass with actual matrix dimensions, 32 experts/top22, three routing patterns and n1/n4/n512. The reduced expert count bounds CPU-reference setup cost; it is not a full-512-expert correctness or model-quality campaign. Existing NMSE limits are unchanged, including the native FP4 activation tolerance.

Performance uses all 512 experts/top22 and an offset ID view. Routes are seeded shuffled experts, the last 22 experts, and one-quarter spread/three-quarters concentrated. The fixed seed resets outside each timed iteration, preserving the same routes across formats. Complete local graphs are repeated, not only their terminal activation or multiplication. First-pair throughput changes:

| Projection / routing | n1 decode | n4 verification | n512 prefill |
|---|---:|---:|---:|
| Up, shuffled | -2.05% | +1.67% | +38.44% |
| Up, concentrated | +2.47% | -1.07% | +19.70% |
| Up, mixed | +1.12% | +5.36% | +69.14% |
| Down, shuffled | -1.62% | +0.08% | +38.84% |
| Down, concentrated | +5.91% | **-31.97%** | +29.45% |
| Down, mixed | +6.31% | +8.18% | +45.52% |
| Equal-weight geometric mean | +1.97% | **-4.04%** | +39.38% |

**Reject at the first pair.** Concentrated n4 down takes 164.92 us on IQ4_NL versus 242.41 us on NVFP4, exceeding the frozen 3% per-case regression guard. The +39.38% prefill result clears the 5% goal but cannot override that loss. No reverse pair, route/type-subset rescue, activation-precision tuning, conversion, quality campaign or deployment follows. These synthetic graphs omit shared/latent computation, routing-score computation, expert summation and full-model scheduling; they are not model tok/s or proof of preserved model precision.

`results/nemotron-nvfp4-20261021/` retains the protocol, complete shard inventory, source/fixture patches, hashes, 36 reference checks and all 36 performance cases. The initial lowercase `q4_k` selector matched only the nine IQ4_NL cases; the count guard stopped before performance, and the corrected native `q4_K` selector ran the full frozen fixture. `execution-notes.json` preserves that setup error. The original owned test source, its 145 pre-existing added lines, control build objects, original executable and tests-off configuration are restored. Reproduce with the two patches listed in `source.json`, the unchanged CUDA runtime, and `screen.py --test-exe <isolated-test-backend-ops.exe> --model <first-original-shard.gguf> --work <new-private-directory>`. No original weights are converted.

**Disposition of this pass:** no speed candidate qualified for full release/quality/stability promotion or runtime integration. The installed runtime and launcher defaults remain unchanged. Q8 counters remain driver-resource blocked; per-request speculation semantics and broader long-context/soak hardening remain separate work, not silently claimed as completed by these screens.

### 2.59 User-directed workload-specific tradeoff reassessment [M/S]

The user requested reconsidering previously ignored candidates using end-to-end tradeoffs and selected **workload-specific opt-in profiles, with the balanced default unchanged**. The inventory and new protocols are in `results/workload-tradeoffs-20261021/`. Historical rejected/parked screens remain unchanged; a slower verification operator is no longer an automatic veto on a separately validated long-prefill profile. Correctness and model precision remain hard gates. Existing features, unsupported designs, noisy non-wins and driver blockers are distinguished from candidates requiring new measurements.

Long-prefill profiles must show at least 3% lower complete-request latency in both launch orders with separated primary ranges. Low-cost configuration profiles use a 1% minimum, require both orders and separated ranges, and must exceed control reload drift. These are new profile-specific criteria, not retroactive passes. Complete task outputs are scored independently; synthetic operator geometric means do not become model-speed claims.

The first model-level check converts all 192 selected 27B FFN tensors to NVFP4 in a separate file using the actual existing quantized GGUF as input. All 674 unselected tensor payloads are byte-verified unchanged; metadata names, types and values match. This tests the readily available requantization path, not conversion from original BF16.

| Precision check, 16 pinned Wikitext-2 chunks | PPL ratio | Mean KL, nats | Same top token | Decision |
|---|---:|---:|---:|---|
| Original 27B A/A calibration | 1.000015 | Approximately 0 | 100.000% | Valid numerical control |
| 27B FFN NVFP4 requantization | 1.020371 | 0.056200 | 90.484% | Fails the unchanged 1.01 PPL-ratio and 0.01 KL gates |

**Do not enable this requantized 27B candidate.** Its prefill operator lead remains real, but it now has a model-level precision failure independent of the old verification-speed veto. No end-to-end speed or general language-quality claim follows. Original-BF16 conversion has not been evaluated. Nemotron and Flash-Next precision checks and the profile validations are reported below.

The Nemotron conversion also passes full tensor/metadata isolation, but its unchanged-model calibration stops before candidate evaluation: reported compressed-reference PPL ratio is 1.001971, while mean KL is approximately zero and actual PPL 3.086314 matches the original reference pass's printed 3.0863. The initial guard and this blocked attempt are retained in `nvfp4-nemotron-calibration.json`; this is not a failed NVFP4 model-quality result. Reference-format diagnosis is required before continuing.

**Precision-method correction, declared before measuring the Nemotron or Flash-Next candidates:** `tools/perplexity/perplexity.cpp:79-106` clamps saved logits at `max_logit - 16` before uint16 encoding. The reported reference PPL uses those clipped values, while the evaluated model's PPL uses actual logits (`:191-217`). This explains why A/A PPL can differ from the compressed denominator even when actual PPL agrees and KL is near zero. Revision 2 compares candidate actual PPL to the original A/A actual PPL; it requires the A/A actual value to match the original reference pass within 0.01% and keeps the 0.001 calibration-KL guard. The candidate limits remain exactly 1% actual PPL increase and 0.01 mean KL. The initial protocol and script are preserved in commit `af98296ae`; neither failed Q6 decisions nor the 27B precision failure is reversed.

**Nemotron now has a completed model-quality failure, not only a calibration blocker.** After validating and reusing the original full conversion audit, candidate actual PPL is 3.200457 versus original A/A 3.086314: ratio **1.036984**. Mean KL is **0.056827** and top-token agreement is 92.747%. Both quality gates fail. All 80 selected expert tensors were converted, all 701 unselected payloads and all metadata preserved; no favorable subset was substituted. `nvfp4-nemotron-precision.json` records the corrected-method result while the initial calibration record remains unchanged. No speed run or deployment follows this failure. The failed 27B and Nemotron scratch GGUFs were removed after preserving payload manifests, logs and reference logits; original models remain untouched.

**Flash-Next also completes and fails the retained precision gate.** The full three-shard audit verifies all 144 selected expert tensors converted and all 1,080 unselected payloads plus metadata unchanged. Actual PPL changes from 2.202870 to 2.217932: ratio **1.006837**, which passes the 1.01 PPL limit. However, mean KL is **0.130035**, above 0.01, and top-token agreement is 90.521%. A passing PPL ratio alone is insufficient. `nvfp4-flash-next-precision.json` preserves this outcome. No end-to-end speed run, tensor-subset rescue or deployment follows.

All three available-GGUF NVFP4 requantization paths now have completed, negative model-quality decisions. This does not establish the result of quantizing original BF16 weights. It also does not turn the original prefill operator gains into full-model gains.

The profile harnesses measure complete natural-EOS SQL, record extraction, source-grounded prose and full structured edits. Prose literal checks are preliminary; every unique full response requires an independent fact/caveat review before promotion. The long-request runner measures streaming TTFT directly, explicitly primes its cached guard with an identical immediately preceding request, and separates a forced-512-token generation guard from the primary request score. The compact runtime must pass the rebuilt twelve-case CPU-reference fixture before whole-model profiling. Their results follow below.

**Compact-prefill promotion is now blocked by the original control's reference check.** The rebuilt tester with the unchanged installed CUDA DLL passes 11/12 cases. Shuffled IQ3_S, m640/k2560/n512, 512 experts/top10 and an offset ID view reports NMSE 0.000614063 against the existing 0.0005 limit. The candidate was not tested and no compact speed measurements followed; this is not evidence that the candidate caused the failure. The existing random tensor initializer did not serialize the failing inputs, so the exact draw cannot be replayed. Retain the failure rather than raising tolerance or rerunning until a pass. `compact-correctness.json` records the cases, original-runtime/tester hashes and raw-log hash.

All six failed NVFP4 scratch shards have been removed, reclaiming about 175.5 GiB. Original models, reference logits, payload manifests and raw quality logs remain preserved.

**Follow-up reference triage:** chunked GDN from [ggml-org/llama.cpp#29353](https://github.com/ggml-org/llama.cpp/pull/29353) is already present, including the separate `gated-delta-net-mma.cu` implementation. The older [ggml-org/llama.cpp#26001](https://github.com/ggml-org/llama.cpp/pull/26001) is only a differential-review lead if a measured path exposes a gap, not another basic port. The harness already collects and cross-checks per-request speculative metrics. The measured Nemotron SSD/no-speculation tradeoff remains model-specific; it is not a generic SSD recommendation for Flash-Next.

**Flash-Next GDN profiling:** the existing 10,000-character trace contains 2,734 prompt tokens. The `gdn_single` and `ssm_conv_long_token_f32` kernels account for 89.88 ms, or 2.51% of summed CUDA kernel durations and 1.82% of prompt-evaluation wall time. Kernel durations are aggregated work and may overlap; this is not a latency speedup estimate. The profile exposes no material GDN hot-path gap, so no differential review or port of #26001 is justified. The trace hash, query method and limitations are in `results/workload-tradeoffs-20261021/gdn-prefill-share.json`.

**Configuration-profile screens: none qualified.** Each screen used the frozen configuration gate (at least 1% lower median latency in both launch orders, separated ranges, and a gain above control reload drift) on fresh natural-EOS requests with independent task scoring. All quality checks passed in every arm. Arm order was control, candidate, candidate, control. Pair A compares the first control with the next candidate; pair B compares the last control with the preceding candidate. Three screens used greedy SQL and structured-record copying. The lookup screen used two fresh twelve-worker full JSON edits, scored as the complete edited JSON.

| Candidate | Pair A | Pair B | Control reload drift | Ready memory vs control | Result |
|---|---:|---:|---:|---|---|
| Target backend sampling | -3.32% | -10.23% | 1.48% | within 3 MiB | Slower in both orders; the earlier +2.08% geomean did not replicate |
| MTP4 against MTP3 | +7.77% | +5.61% | 6.42% | +99 to +140 MiB | Both pairs separated, but pair B is below the control's own drift; fails |
| Q5_K draft head | -0.26% | -1.95% | 0.77% | +74 to +121 MiB | The earlier +2.51% geomean did not replicate |
| Lookup rollback capacity, 12 states | +1.04% (ranges overlap) | +1.56% | 0.52% | +1,013 to +1,043 MiB | Pair A is not separated; fails the frozen gate |

None is enabled. The earlier per-case regressions (chat -7.52% for MTP4, Python -3.55% for lookup capacity) are unchanged. `configuration-backend-sampling.json`, `configuration-mtp4.json`, `configuration-q5-head.json` and `lookup-full-edits.json` hold the runs, hashes and per-case data.

**27B DFlash4 prose profile (the launcher recipe at depth 4 against its depth 7): the greedy speed gate passed, the frozen launcher-sampling gate failed, not enabled.**

- *Greedy complete prose.* Two source-grounded prose tasks, depth order 7/4/4/7, three repeats, summed complete natural-EOS request time. Depth 4 was 4.80% and 3.93% faster, both separated, with 1.84% control drift and 445-477 MiB less ready memory. Text was byte-identical at both depths. A manual semantic review of all 32 responses (two unique texts) found that both add unsupported rationale against the prompt's "do not invent events, causes or benefits" rule (7 and 6 additions), identically at both depths. That is a model and prompt property, not a depth effect; it is disclosed, not endorsed. The reviewer was the campaign assistant, not a third-party human, and only two prose fixtures exist (`27b-complete-prose.json`, `27b-prose-review.json`).
- *Release quality.* The four English/code checks fail `locale_json` in both arms (the model writes `cancelled`, `aluminium` and `grey`, not the requested en-US spellings), in measured and warm-up rows, with identical output. The strict rule therefore could not separate the candidate from the control (`27b-release-gates.json`, retained). The user then approved a **post-hoc** non-inferiority rule: the candidate must pass every check the control passes. It did (`27b-release-gates-non-inferiority.json`).
- *Launcher sampling (frozen gate).* Depth 4 must also pass the configuration gate under the launcher's server-side sampling (temperature 1.0, top-p 0.95, top-k 20, seed 1234), scored as seconds per generated token over both prose tasks. Medians in run order were 0.03614 (depth 7), 0.03394 (depth 4), 0.03375 (depth 4) and 0.03393 (depth 7). Reductions were +6.09% and +0.55%, both separated, with 6.51% control reload drift (the first control load was slower than the last). Pair B is below 1% and below the drift, so the gate **fails**. Every response ended naturally and text was identical across reloads and depths. Acceptance was 0.301 at depth 7 and 0.434 at depth 4; ready memory was 37,792/37,797 against 37,313/37,346 MiB. The run was not repeated until it passed.
- *Outside-profile observation, not gated.* A single pass of the four quality prompts (64-112 tokens) decoded 20.2-24.2% slower at depth 4 (mean -22.44%, `27b-quality-decode-rates.json`), against 5.71% slower on code in the earlier fixed-length decode table. The outside-profile code and long-generation stage was skipped because the launcher-sampling gate had already failed.

**Nemotron no-speculation profile: qualified as an explicit opt-in.** The comparison is `--spec-type none` against the campaign control MTP3 with `-b 4096 -ub 4096` (`run_matrix.py` N2), not against the launcher's DFlash5 recipe. Workload: two uncached record-extraction requests (three synthetic record facts inserted into tiled, pinned public project documentation; 9,073 and 26,933 prompt tokens; a 50-token JSON answer that ends naturally, cap 128), summed complete-request time per repetition, median of three, order control/no-speculation/no-speculation/control. The gate is at least 3% lower latency in both orders with separated ranges.

| Measure (MTP3 -> no speculation) | Pair A | Pair B |
|---|---:|---:|
| Summed request time | 48.430 -> 45.829 s (-5.37%) | 48.364 -> 46.171 s (-4.53%) |
| Ready GPU memory | 79,758 -> 75,109 MiB | 79,747 -> 75,113 MiB |
| Guard: cached request with a 4-token suffix | 1.095 -> 1.780 s | 1.146 -> 1.775 s |
| Guard: forced 512-token generation | 13.449 -> 18.941 s | 13.481 -> 19.088 s |

Control reload drift was 0.14%, the ranges separated and all task checks passed. Prompt processing was 8.0-10.5% faster, but decode fell from about 49 to about 29 tokens/s on the long requests (about 44 to about 30 in the 512-token guard), so the profile pays only when outputs are short relative to the prompt. The estimated break-even is an output of 0.7-0.9% of the uncached prompt tokens (about 80 tokens at 9k, about 180-210 at 27k), derived from the measured rates and not measured directly. Cached short requests are 54.9-62.5% slower and a forced 512-token generation is 40.8-41.6% slower. The four English/code release checks fail `locale_json` identically under both settings (a model limitation), so the profile passed the post-hoc non-inferiority rule and not the strict rule; both records are retained (`nemotron-release-gates.json`, `nemotron-release-gates-non-inferiority.json`). The 512-token guard text differs between the two settings and is not scored. Evidence: `long-nemotron-nospec.json`; an earlier, invalidated run is described under Disclosures.

**Additional actual-launcher comparison:** no speculation also passes the same 3%/separated-range gate against the launcher's DFlash5 recipe (`-b 4096 -ub 4096`, depth 5) with temperature 1.0, top-p 0.95, seed 1234 and server-default top-k. The summed request medians fell from 49.231 to 44.923 s (-8.75%) and from 47.024 to 43.913 s (-6.62%), with 4.69% control reload drift; both pair ranges are separated. Ready memory fell by 6,363-6,422 MiB (6.2-6.3 GiB). All 16 warmup and measured extraction checks passed in each of the four arms. The cached 4-token-suffix guard was 74-76% slower without speculation, and the forced 512-token generation guard was 17-19% slower; the 512-token guard is timing-only, not quality-scored. This is a narrow short-output result, not a long-generation quality claim. The earlier runner attempt used greedy rather than launcher sampling and stopped on the DFlash5 control; it is excluded. The corrected evidence is `nemotron-dflash5-comparison.json`.

**Enablement.** The launcher catalog is generated and marked not to be edited, so no launcher recipe changed. The only enabled item is the explicit opt-in profile `nemotron-nospec-long-input` in `profiles/opt-in-profiles.json`, with its numbers, limits, evidence and rollback. `tools/profile_command.py render nemotron-nospec-long-input --command-file <file>` reads the operator's own control command (one `llama-server` line, optional backtick continuations), validates the speculation, batch and ubatch flags for either MTP3 or launcher DFlash5, and prints the no-speculation command with the use/avoid notes. The DFlash5 variant also validates draft offload and removes its draft model and draft-only flags. Model identity and sampling are not checked by the renderer. It refuses any other input with exit code 2 and changes no file, catalog or runtime. Rollback is running the unchanged control. At publication the installed `llama-server.exe`, `llama.dll`, `ggml-cuda.dll` and launcher catalog hashes match those recorded in the profile results (the deployed b11756 + #30380 + #30640 runtime), and no server process is running.

**Disclosures.**
- *Revised quality mode.* It was added to `release_gates.py` after the strict runs. The strict records carry the hash of the earlier script; the revised records embed the revision, approval, strict-result hash and time.
- *Standby incident.* The machine entered Modern Standby during an earlier configuration batch and later restarted uncleanly (bugcheck 0x1A, cause not determined). Backend sampling and MTP4 had already finished in that batch and are reported from it. The Q5 draft-head screen was interrupted at its second candidate arm (server not ready within 600 s), so it was rerun from the start as a complete four-arm run; only the complete rerun is reported and the partial data are not scored. Installed runtime and catalog hashes were verified after the restart. Later batches hold a process-scoped keep-awake request and check that no Kernel-Power standby event (IDs 506 and 507) occurred during each stage.
- *Nemotron harness defect.* The first Nemotron run failed in analysis after only its control arm. The cached-guard check required the guard to process fewer prompt tokens than its priming request, which cannot hold when the warmup has already cached the prime. The check now requires the guard to serve at least 90% of its prompt from cache and reports both counts, with three tests added. It was fixed before any candidate arm existed and changed only that non-scored guard and its protocol text; the primary gate, arm order and scoring are unchanged.
- *Excluded Nemotron run.* The next complete run was flagged by the standby check that was written before it. Modern Standby (screen-off) sessions overlapped the loaded time of the four arms (control, candidate, candidate, control) by 0%, 9%, 69% and 95%. GPU telemetry shows the benchmark kept running at 92-96% utilization, so this was not suspension. A screen-off control plausibly runs faster, but that was not tested. The check was not relaxed; the stage was rerun once in a batch that also holds a display-required request and wakes the display. The numbers were visible when the rerun was chosen, so it is a judgment call: scored as measured, that run would have failed the gate in the second launch order (reductions +5.13% separated and +0.84% not separated, control drift 3.51%). Its first pair, whose arms were awake or nearly so, was close to the rerun's first pair (+5.13% against +5.37%). Every arm passed its quality checks, ready memory was 4.55-4.58 GiB lower and prompt rates were +3.31% to +9.11% higher (rerun: +8.05% to +10.52%). The run is retained and not scored (`long-nemotron-nospec-invalid-display-off.json`). The rerun is the only valid complete measurement of this profile.
- *Drift.* Control reload drift ranged from 0.14% to 6.51% between profiles, so each gain is compared with the drift measured in the same batch.
- *Scope.* One machine, one build (b11756 + #30380 + #30640), one model per profile and synthetic workloads. None of these is a general language-quality claim.

**Residual blocked or parked items:**
1. NVFP4 from original BF16 weights was tested and fails the unchanged precision gate for both models. Sources: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) at `de4b8e4d43b917e7706784d8bb445c9af86a3540` (131 safetensors shards, 335.28 GiB) and [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/tree/1f8953f7f0935358e3605150d22bc15df062061c) at `1f8953f7f0935358e3605150d22bc15df062061c` (63 shards, 231.82 GiB); all 195 files matched their pinned LFS SHA-256. The encoder was declared before any score (`bf16-nvfp4-protocol.json`): ggml's NVFP4 has no tensor-level scale, so typical expert weights (amax about 0.03-0.05) put most UE4M3 block scales in the subnormal range. Each expert is therefore divided by `s_e = amax_e / (6 * 240)` before `ggml_quantize_chunk`, and `s_e` is stored in the existing optional `blk.N.ffn_X_exps.scale` tensor. `bf16_nvfp4.py` builds the candidate; `precision.py --candidate-dir` audits it (only the selected tensors and their scale tensors differ; metadata differs only in `split.tensors.count`) and runs the same reference, calibration and candidate passes. Sampled experts correlate at r 0.990-0.997 with the dequantized original GGUF experts. On those samples, relative weight RMSE is 0.095 scaled, 0.107-0.119 unscaled and 0.071-0.146 for the original GGUF types.

| Model | Actual-PPL ratio (limit 1.01) | Mean KL (limit 0.01) | Same top token | Requantized result (ratio / KL) |
|---|---:|---:|---:|---|
| Nemotron (80 tensors) | 1.011206 | 0.060600 | 93.13% | 1.036984 / 0.056827 |
| Flash-Next (144 tensors) | 0.978578 | 0.160775 | 90.50% | 1.006837 / 0.130035 |

Both fail, so neither is enabled and no speed was measured. The BF16 source cut the Nemotron PPL gap from 3.7% to 1.1%, and Flash-Next scores 2.1% lower PPL than its own UD-IQ4_XS reference. Mean KL, however, is measured against the quantized GGUF, not the BF16 model, so it also counts movement toward the BF16 model as divergence. A KL gate against BF16 (or a high-precision proxy) logits would be a new, separately declared protocol; this result does not justify relaxing the frozen gate. The Flash-Next candidate is also larger (NVFP4 replaces 3.44-bpw IQ3_S experts), so it is not a size-matched comparison. Evidence: `nvfp4-bf16-nemotron-precision.json`, `nvfp4-bf16-flash-next-precision.json`.

   **27B against BF16 logits (closes the NVFP4 track).** To remove the quantized-reference objection, `bf16-reference-protocol.json` was declared before any 27B build. 27B is the one model whose BF16 weights fit in memory: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (18 shards, 51.75 GiB, all SHA-256 verified), converted with the repository converter to a 54.66 GB BF16 GGUF. All 192 dense FFN tensors were rebuilt as NVFP4 with one per-tensor `.scale` (sampled tensors correlate at r 0.996-0.9998 with the deployed weights). `bf16_reference_gate.py` scores the deployed GGUF (control) and the candidate against the same BF16 logits; the gate is non-inferiority on both mean KL and actual PPL.

   | Against BF16 (PPL 5.5889) | Mean KL | Actual PPL | Same top token |
   |---|---:|---:|---:|
   | Deployed UD-Q4_K_M | 0.010281 | 5.560534 | 95.96% |
   | NVFP4 FFN candidate | 0.044512 | 5.659747 | 91.43% |

   The candidate is inferior: 4.3x the control's KL and 1.8% higher PPL. The deployed dynamic quant is already close to BF16, and it keeps some FFN tensors at higher precision (for example `blk.63.ffn_up` is Q6_K, relative weight RMSE 0.019 against 0.095 for NVFP4). NVFP4 with the stock ggml encoder is therefore closed for all three models; reopening it would need a better encoder (for example per-block scale search or importance weighting), not another gate. Evidence: `nvfp4-bf16ref-27b-precision.json`.
2. Q8_0 hardware-counter attribution remains blocked by the driver `ResourceUnavailable` error. Do not retry elevation or change driver policy, services or clocks without a separately agreed diagnosis.
3. Compact routed prefill remains parked: the unchanged control failed one shuffled IQ3_S n512 reference case (NMSE 0.000614 against 0.0005), and its random draw cannot be replayed. No candidate speed run is warranted because the compact prototype already missed the reverse-order 5% full-model gate in section 2.42. Do not raise tolerance or promote it.
4. MMVQ row-tail correctness (section 2.45) is resolved: retain the correctness fix in source, but do not deploy or claim a speed gain; the corrected operator screen is -7.34% across 22 cases, while the limited actual-MTP diagnostic is token-identical and does not clear that cost.
5. Flash-Next gated-delta-net prefill is resolved for the measured trace: the identified path accounts for 2.51% of summed prompt-kernel durations; no #26001 differential review or port is justified without a future profile showing a larger gap.
6. 27B DFlash4 stays parked. The frozen launcher-sampling gate already failed; any new profile requires a separate protocol and human review.

### 2.60 Upstream sync, sampled speculation and thermal noise (2026-10-23) [S]

**Measurement noise.** Same-binary 27B `llama-bench` runs (pp2048, tg128, `-r 2`) back to back fell from 765 to 610 to 552 tok/s (-28%) with SM clocks dropping from 2366 to 1625 MHz at 65-70 C GPU temperature and 20-38 W. The only active clock-event reason was "Reliability"; GPU thermal and power caps were inactive, and the machine was on AC. Cooling the GPU to 55 C before each run did not make runs repeatable (747, 867, 746 tok/s). A two-arm native screen on this machine is therefore dominated by run order: the deployed build looked 28% faster when it ran first and 17% slower when it ran second. Comparisons below use alternating order with several rounds or ABBA server pairs, and small differences are treated as noise.

**Power plan (roadmap item `power-mode-and-memory-carveout`).** The active plan had been switched from Balanced to High performance at 12:01 by a 32-bit `powercfg.exe` (not by this campaign). Four alternating blocks of three 27B runs (High performance, Balanced, High performance, Balanced; GPU cooled to 55 C first) gave medians of pp2048 843.58 vs 887.48 tok/s and tg128 14.71 vs 15.57 tok/s for High performance vs Balanced, with overlapping ranges (Balanced 778.84-920.53, High performance 787.17-874.48). Not separated, so no plan is recommended; High performance was restored. Disk idle shutdown and PCIe ASPM were set to off on AC (user approved) after repeated external-drive dropouts.

**Upstream sync (b11845 + the deployed #30380 diff).** Built natively (`C:\src` scratch, release-like layout, 42 files); #30640 and [ggml-org/llama.cpp#30663](https://github.com/ggml-org/llama.cpp/pull/30663) are upstream there. The #30380 diff needed one manual merge with upstream's Hadamard `cur_mm` change in `build_lora_mm`.

| Comparison | Pair A | Pair B | Notes | Decision |
|---|---:|---:|---|---|
| 27B native, 8 alternating rounds | tg128 +2.70%, pp2048 -1.02% | (pooled) | ranges overlap heavily | no change |
| Flash-Next U1 server (MTP3, dynamic vocab, Q4_K head), greedy | +3.66% | +0.35% | identical outputs 4/4; ready memory -490 MiB | park |

**Probabilistic draft sampling under the launcher's sampling** (temperature 1.0, top-p 0.95, top-k 20; `run_rebaseline.py --sampling launcher`, new). One variable, ABBA, three repetitions per case:

| Candidate | json256 | rust256 | Acceptance (json / rust) | Decision |
|---|---|---|---|---|
| Flash-Next MTP3, deployed build, `--spec-draft-sampling probabilistic` | +3.62% / +4.17% | -7.97% / -12.47% | 0.655 -> 0.689 / 0.894 -> 0.736 | reject |
| 27B DFlash2, b11845 build (#30663), `--spec-draft-sampling probabilistic` | +1.36% / -4.88% | +25.81% / +9.18% | 0.503 -> 0.458 / 0.337 -> 0.369 | reject (code-only gain, JSON loss) |

The 27B control itself moved from 24.53 to 29.58 tok/s on rust256 between its two arms, so only the acceptance changes are reliable. The deployed runtime and launcher are unchanged. Evidence: `results/upstream-sync-20261023/`.

**Open upstream CUDA PRs.** Each is another author's patch, screened on b11845 (or its own base) with alternating rounds; per-round ratios pair adjacent arms, so drift cancels.

| PR | Workload | Result | Correctness | Decision |
|---|---|---|---|---|
| [ggml-org/llama.cpp#31047](https://github.com/ggml-org/llama.cpp/pull/31047) MoE layout for MMQ/MMF | Flash-Next pp2048, 8 rounds | candidate faster in 8/8 rounds, paired geomean +6.03% (median about +4.2%; one round +22%); tg32 +0.60% | mean KL 0.000000, 100% same top token | **advance** |
| #31047 | Nemotron pp2048, 6 rounds | +1.55% (3/6 wins), tg32 +0.02% | not separately run | neutral |
| [ggml-org/llama.cpp#31004](https://github.com/ggml-org/llama.cpp/pull/31004) graph shape keying (PR head vs its parent 339f3242c; does not apply to b11845) | 27B DFlash2 server ABBA | +4.27% / -2.01% | identical outputs 6/6 | no gain |
| [ggml-org/llama.cpp#30797](https://github.com/ggml-org/llama.cpp/pull/30797) Q5_K MMA | 27B pp2048, 8 rounds | -1.05% (3/8 wins) | mean KL 0.002783, 97.86% same top token | reject |

**Deployment (b11845 + deployed #30380 diff + #31047).** Release-like 42-file build, version `build 11845, commit 9ff0d7e90`. The English/code quality suite (natural EOS, scored) passed for both the old runtime and the candidate on Flash-Next U1 (MTP3, dynamic vocabulary, Q4_K head), 27B D2 (DFlash2 depth 7) and Nemotron N2 (MTP3). Deployed to the install directory with every file hash-verified, the launcher catalog hash unchanged and the previous b11756 + #30380 + #30640 runtime kept as a renamed backup with a rollback script. A post-deploy server smoke test on the installed runtime completed for U1, D2 and N2. What to expect: Flash-Next prompt processing about 4-6% faster on long prompts and about 490 MiB less ready GPU memory; decode and the other two models unchanged within noise. These gains are upstream work, not ours. Evidence: `results/upstream-sync-20261023/` (PR screens, KL checks, quality scores).

## 3. Source-verified feature status and reference lessons

The following status is verified in this checkout. This table supersedes the porting assumptions in the initial reference inventory below.

| Technique | Status | Next action |
|---|---|---|
| CUDA graphs, including MoE `MUL_MAT_ID` handling | Already implemented upstream; see `ggml/src/ggml-cuda/ggml-cuda.cu` and [#14753](https://github.com/ggml-org/llama.cpp/pull/14753) | Do not add generic graph support. Investigate only a reproducible issue on the current binary. |
| Qwen4exp hyper-connection gate/post chain | Fused by [#30602](https://github.com/ggml-org/llama.cpp/pull/30602) | Remove this operation chain from K1. The reported ~3% gain was on RTX Pro 6000, not this device. |
| Grouped-expert top-k routing | Added by [#29181](https://github.com/ggml-org/llama.cpp/pull/29181) | Verify its target-model effect; do not plan another routing-fusion port. |
| Blackwell NVFP4/MXFP4 MMQ | Implemented in `ggml/src/ggml-cuda/mmq.cu`; see [#24364](https://github.com/ggml-org/llama.cpp/pull/24364) and [#29857](https://github.com/ggml-org/llama.cpp/pull/29857) | Sections 2.55 and 2.58: 27B dense FFN prefill gains 67.82% and Nemotron routed graphs gain 39.38%, but decode/verification guards reject both conversion scopes before model-quality work. Kernel presence alone predicts no mixed-workload gain. |
| Chunked Mamba2 SSD prefill | Implemented in `ggml/src/ggml-cuda/ssm-scan.cu`, with a K=1 restriction. The graph caller requests extra snapshots when rollback is enabled. | Section 2.56 observes SSD selected without speculation and excluded at K=4 under MTP3. Investigate rollback-preserving execution, not another basic SSD port or an unsafe guard removal. |
| Native MTP | Present for Nemotron-H and qwen4exp; see [#29018](https://github.com/ggml-org/llama.cpp/pull/29018), [#27836](https://github.com/ggml-org/llama.cpp/pull/27836), and [#29761](https://github.com/ggml-org/llama.cpp/pull/29761) | Already used in the recorded model results. |
| DFlash/DSpark reduced vocabulary | DFlash/DSpark are implemented; `src/models/dflash.cpp` optionally reads `d2t` and scatters reduced-vocabulary logits. EAGLE-3 also has `d2t`. Generic MTP has no equivalent mapping in this checkout. See [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) and [#25173](https://github.com/ggml-org/llama.cpp/pull/25173). | Do not describe `d2t` itself as missing from DFlash. The open dynamic-head proposal in [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380) is now the first alternative to assess before designing a static MTP export/mapping path. |
| PLE row prefetch | `llama_prefetch_rows` landed in [#29599](https://github.com/ggml-org/llama.cpp/pull/29599). The direct-read proposal [#30772](https://github.com/ggml-org/llama.cpp/pull/30772) was closed in favor of prefetch. | Profile cold/warm rows before changing the existing implementation. Do not port raw disk reads without contrary local evidence. |
| Confidence cut-off and n-gram draft modes | Existing CLI/config controls include `n_max` and `p_min`; multiple n-gram modes are documented. Per-request speculative fields are still disabled under `#if 0` in `tools/server/server-schema.cpp`. | The measured `p_min` behavior is available now. Per-request overrides are a server/API feature, not a decode-kernel optimization. |
| Adaptive draft depth | No general runtime controller found. Related open research issues: [#27821](https://github.com/ggml-org/llama.cpp/issues/27821), [#29875](https://github.com/ggml-org/llama.cpp/issues/29875). | First log acceptance and measured cost by draft depth; compare an offline policy on held-out prompts. Implement only if it beats fixed settings without quality regressions. |
| Grouped MoE prefill | CUDA already has device-side expert packing and quantized grouped MMQ. Strata additionally offers streamed-group dependency batching and an optional fused math path; neither means grouped CUDA GEMM is absent here. | Section 2.49: no per-expert copy ring in our resident path, preparation-reuse GPU-only estimate 1.01%, and zero layers compatible with the complete native-IQ fused path. Do not port without a separately justified gap. |
| Shared-prefix and conversation retention | Existing server checkpoints and RAM prompt cache save/restore target and draft state, with speculative checkpoint restoration. | Section 2.50: adjacent and interleaved synthetic conversations preserve more than 99.5% of their common prefixes. No new pinning mechanism without a measured pressure/eviction or state-restoration gap. |
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
| 2 | Upstream development branch at [8a86212329](https://github.com/ggml-org/llama.cpp/tree/8a86212329f2d07f76fb3ba3799436e19b4d2205), [ggml-org/llama.cpp#30640](https://github.com/ggml-org/llama.cpp/pull/30640), open | Adds `KPOOL_MASK` and a CUDA kernel to build the QSA selected-pool/tail mask without the existing multi-op mask chain. It does not replace the whole attention calculation or implement a new expert cache. | Flash-Next GB10 author results: prefill +3.9% to +5.9%, decode +0.6% to +1.3%. Our native ARM64 CUDA test passes 12/12; server speed is +1.42% over the PR merge-base and +2.43% over the deployed release, with token-identical outputs (section 2.15). Deployed locally. |
| 3 | [anujj/llama.cpp at 0ba2b5f2ad](https://github.com/anujj/llama.cpp/tree/0ba2b5f2ada4bedc85b38a819f17a58d7daee5a5), [ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948), open | Fuses dense FFN gate/up matmuls and GLU on the larger-batch MMQ path. Interleaves row loading inside the kernel; the model file layout is unchanged. This is distinct from existing small-batch fusion and routed-expert grouping. | Author reports Qwen3.8-27B Q4_K_M prefill +7.1% on GB10. A [follow-up using UD-Q4_K_M](https://github.com/ggml-org/llama.cpp/pull/29948#issuecomment-6165544748) reports +3.9% for this patch alone; 30/64 FFN layers have mixed gate/up types. Our 4-round local runs show only +1.66% to +2.42% prefill with overlapping ranges; patch does not apply to b11756 + #30380 (section 2.16). Not selected for deployment. |
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

Build differences from the official release: the release cross-compiles only `ggml-cuda.dll` with MSVC and builds everything else with Clang 20. The all-MSVC build is about 1% slower. For deployment use `tools/build_release_like.ps1`, which follows the release layout (sections 2.14-2.15). For a matched A/B, the all-MSVC script is enough because both arms use the same toolchain.

The QSA k-pool mask A/B and release validation are complete (section 2.15). The remaining fusion candidate was measured and not selected for deployment:

- **Dense FFN MMQ fusion, [ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948).** Fuses up, gate and GLU in one MMQ pass for prefill and large batches; decode (MMVQ) is not affected. Needs same type for gate and up, type in {Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS} and `ne01 % 128 == 0`. In the 27B UD file, 37 of 65 layers have matching gate/up types and 35 use a supported type (IQ4_XS 20, Q5_K 9, Q4_K 5, Q6_K 1). Four-round local results are small/noisy, and the patch does not apply to the current release base (section 2.16); do not deploy without a clean rebase and stronger evidence.

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

The v2 step estimates and cumulative tok/s projections below remain withdrawn. They combined unverified gains, double-counted upstream work, and lacked repeated A/B evidence. Sections 2.7-2.11 provide controls and validation decisions, not a cumulative forecast. Prioritize en-US and code; keep other languages non-gating. The bounded English follow-up is complete.

**Latest user-directed policy (section 2.59):** reassess parked items for explicit workload-specific opt-in profiles while preserving the balanced default. The profile protocols in `results/workload-tradeoffs-20261021/protocol.json` supersede the blanket per-component regression veto below for this new campaign only. Preserve historical failures; do not relax correctness/precision or call an unfinished candidate validated.

**User-directed execution change: screen quickly and fail fast, for binary changes and model changes alike.** The table below is a backlog, not a requirement to spend a full validation campaign on every item. Model/head quantization experiments can be screened without waiting for the upstream-PR queue. Distinguish our own changes, model/configuration experiments, and integrations of other authors' patches in every result.

1. State one change and one primary goal: decode or prefill speed. Check the relevant dispatch or tensor first; skip inapplicable proposals. Keep the control source/toolchain fixed for binary changes, or the executable fixed for model changes. Preserve the original weights and live deployment.
2. Build only the needed native ARM64 targets, or generate one separate model candidate. Run the narrow correctness/load smoke needed for that change, not every backend or model suite. Reuse a control build only when its source, compiler and flags match.
3. Run a short speed screen. `ab_llamabench.py --screen prefill|decode` uses pp2048 and tg128 with two samples each, up to two pairs, and a 600-second total benchmark budget including loads. A weak first pair stops before the second. For draft/speculation changes, use the existing server harness on only two English cases with two measured requests per case, then `compare_speed.py --screen`; native target-only timing is not a proxy for draft acceptance or speculative throughput.
4. Default triage: reject a greater-than-3% regression in any measured case; park gains below 3% or overlapping target sample ranges. Only clear survivors get one reversed-order pair. Both pairs must survive before longer validation. These are configurable screening thresholds, not significance tests or proof that parked candidates have no benefit. Do not keep rerunning a marginal candidate until it wins.
5. Only survivors get the full repeated A/B, English quality gate, relevant model-precision checks, and release-like integration/deployment validation. Model changes may alter tokens; speed alone does not establish equivalent quality. A screen never authorizes deployment. Revisit a parked candidate only with a changed implementation, a better-matched workload, or new bottleneck evidence.

Screen decisions and partial measurements are retained. Errors and timeouts stop the run explicitly. Build/conversion time is separate from the native benchmark budget; server screens use a bounded case count and the existing server timeouts, not that native wall-clock cap. Commands and exit codes are in `README.md`, under "Fail-fast candidate screens".

| Order | Work | Evidence needed to proceed | Stop condition |
|---|---|---|---|
| 0 | Completed the installed-binary control in section 2.7. Refresh it when binary, model, settings, or workload changes. | Five samples per case, actual hashes, exact commands, full outputs, logs, and telemetry. Cold cache and operator profiles remain unmeasured. | A patched-binary claim needs an unmodified build from the same source/toolchain, not only this release control. |
| 1 | Completed the English quality gate and matched speed pair: Q4_K +4.50% in the new pair; smaller +1.73% in earlier reloads. Retain as an opt-in English/code candidate. | The recorded scope passes; general writing, arbitrary code and much longer context require their own evidence before broader claims. No further sweep is queued. | Do not replace the deployed default or advertise a guaranteed gain from one pair. Router F32 to BF16 is not a CLI-only experiment here. |
| 2 | Dynamic draft head [ggml-org/llama.cpp#30380](https://github.com/ggml-org/llama.cpp/pull/30380): matched A/B passed (section 2.13); **deployed** as a b11756+PR release-like build (section 2.14), +3.24% / +3.35% over official on the Flash-Next recipe, token-identical, 8/8 quality, other models unchanged. | Rebuild with the same script on each upstream upgrade until the PR merges; repeat the U1 pair and quality gate. | No claim beyond the four en-US cases. Remove the flag if the install directory reverts to an official build without the PR. |
| 3 | QSA k-pool mask [ggml-org/llama.cpp#30640](https://github.com/ggml-org/llama.cpp/pull/30640) passed native CUDA tests and matched A/B, then deployed with #30380 (section 2.15). The PR remains open; current head adds a test-only tensor initialization after the tested head. | Before rebuilding from the current PR head, rerun its updated `KPOOL_MASK` test and repeat U1 parity after upstream base changes. GLU fusion and graph chunking are parked after bounded screens (sections 2.18-2.19). Weighted Q6 and protected-QKV revisions failed precision (section 2.21); continue independent runtime/model hypotheses after source gates (section 2.22). | Do not port whole forks, sweep model subsets against reused text, or interpret existing graph/fusion ablation losses as remaining headroom. |
| 4 | MTP2/3/4 and Nemotron p_min comparisons executed. Keep MTP3; the hindsight depth selector has only 2.00% headroom on the English/code slice before overhead. The separate MTP/lookup cost selector is rejected at -14.10%; matching rollback capacity also fails its per-case guard (sections 2.51-2.52). | New policy work needs causal acceptance prediction and transition-cost evidence, not the optimistic lookup oracle alone. | No cadence, threshold, capacity, draft-length or fixture tuning to rescue the failed candidates. |
| 5 | Actual-MTP routes captured (section 2.31). Expert-packed MMQ, rank alignment and fusion-preserving row tiling failed their gates (sections 2.32-2.33 and 2.38). | New bottleneck evidence before another routed prototype; preserve fusion, numerical correctness and dispatch proof. Logical duplicates do not measure bytes saved. | No type-subset, row-count or rank-ordering sweep against the failed cases. Do not treat overlap as a speedup. |
| 6 | Dense FFN MMQ fusion [ggml-org/llama.cpp#29948](https://github.com/ggml-org/llama.cpp/pull/29948) passes its backend-ops tests, but repeated 27B speed gains are small/noisy and its patch does not apply to b11756 + #30380 (section 2.16). Not deployed. | Rebase/integrate to a clean release base, then repeat matched prefill and decode tests before reconsidering. Grouped prefill still needs evidence that expert GEMMs dominate; NVFP4 remains separate. | No assumed 1.5-2x improvement, no double-counting overlapping fusions, and no swapping model precision inside a kernel-only comparison. |
| 7 | Treat per-request speculative server controls as a separate serving/API decision, not an engine speed step. | Confirm the desired request semantics and server-scope fit; implement tests proving parameters are honored rather than merely echoed. | Do not combine this with speculation algorithm work or claim it improves single-request throughput. |
| 8 | Fast follow-up completed: upstream MMQ-copy patch and actual 27B prose profile parked; 27B dense FFN and Nemotron expert NVFP4 scopes rejected before conversion (sections 2.53-2.55 and 2.58). | Reconsider only with new implementation/workload evidence. Existing source, model precision and launcher defaults remain the controls. | No marginal reruns, rounding a near miss up to the gate or choosing a favorable type/routing subset. |
| 9 | Nemotron MTP3 long-prefill dispatch excludes the existing SSD path because K=4 (section 2.56). | A separate rollback-preserving design and snapshot-correctness tests, followed by operator and whole-model gates. No speed benefit has been measured. | Never force K=1 or drop snapshots to obtain a faster prefill result. |

The bounded Q8 hardware-counter capture was explicitly approved and attempted after section 2.28; it is now blocked by `ResourceUnavailable` from the driver, not by authorization (section 2.57). Do not change driver policy, services or clocks, or repeat elevation to work around it without a separately agreed diagnosis. Target backend sampling and zero OpenMP blocktime also missed their gates (sections 2.37 and 2.39); the deployed recipe remains unchanged. The permanent whole-graph benchmark fix in section 2.40 protects future operator screens from timing only the terminal node; use it in matched control/candidate test builds.

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
| S3 | **Expert-deduplicated grouped GEMV for verify batches**: read each unique expert once per layer for all k+1 tokens | Strata EXPDEC, GenerelSchwerz grouped decode | MoE verify +5–12% (only shared experts are saved); **stopped, section 2.47**: caches already capture cross-rank reuse |
| S4 | Content-aware draft source: n-gram drafting for code edits and repeated context; MTP/DFlash otherwise | Strata prompt-lookup; upstream `ngram-mod` | Code-edit turns +10–30% |

### Phase 3 — Flash-Next decode and prefill kernels (5–7 weeks)

| # | Change | Expected [I] |
|---|---|---|
| K1 | **Decode fusion** for qwen4exp: 4-way hyper-connection mixing fused into the preceding/following GEMVs; RMSNorm + q8_1 quantization; router + top-k; 10 routed + shared experts in one launch; linear-attention update; CUDA graphs + PDL throughout | Plain decode 31.8 → **38–42** (+20–30%); carries into speculative decode |
| K2 | **Prefill: grouped GEMM for 512 small experts** (640-wide FFN) with per-expert token bucketing; fused hyper-connections | Prefill 900 → **1,400–1,900** |
| K3 | GPU-resident PLE lookup (or pinned host with async prefetch) instead of lazy SSD/page-cache reads | Prefill +3–9% on new text; less first-token jitter measured cold penalty 8.5-10.3% at 2K/8K; **whole-batch prefetch rejected, section 2.48** |
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
