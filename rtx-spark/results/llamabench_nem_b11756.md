| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| nemotron_h_moe 120B.A12B Q4_K - Medium |  65.28 GiB |   123.61 B | CUDA       |  99 |     2048 |   1 |          pp2048 |        899.29 Â± 4.05 |
| nemotron_h_moe 120B.A12B Q4_K - Medium |  65.28 GiB |   123.61 B | CUDA       |  99 |     2048 |   1 |            tg32 |         30.09 Â± 0.81 |
| nemotron_h_moe 120B.A12B Q4_K - Medium |  65.28 GiB |   123.61 B | CUDA       |  99 |     2048 |   1 | pp2048 @ d16384 |        834.86 Â± 9.81 |
| nemotron_h_moe 120B.A12B Q4_K - Medium |  65.28 GiB |   123.61 B | CUDA       |  99 |     2048 |   1 |   tg32 @ d16384 |         29.70 Â± 1.07 |

build: d80619397 (11756)

<!-- stderr tail -->
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 113128 MiB):
  Device 0: NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU), compute capability 12.1, VMM: yes, VRAM: 113128 MiB
load_backend: loaded CUDA backend from C:\llama.cpp\ggml-cuda.dll
load_backend: loaded RPC backend from C:\llama.cpp\ggml-rpc.dll
load_backend: loaded CPU backend from C:\llama.cpp\ggml-cpu.dll